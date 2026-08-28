"""Clear / Review / Flag (spec §1.3, §3.7).

The output number this exists to produce: the share of a moderation queue the
system decides without a human, while holding false positives on authentic
images at the target rate. That is the Impact figure the rubric asks for.

Two independent gates decide an image. The probability gates say *what* the
answer is: `p >= flag_threshold` is AI-generated, `p <= clear_threshold` is
authentic, and the band between them is genuinely undecided. The EQI gate says
whether the probability is worth listening to at all: below `eqi_threshold` the
image goes to a human whatever `p` says, because the evidence the probability
was computed from has been degraded away.

Fitted on internal validation rows only -- `fit_policy` requires a per-row
`split=` column and checks it (see the package docstring). A policy whose
thresholds were chosen on the rows it is later scored on reports the fit, not
the reviewer load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aigcdet.calibrate import check_fit_split
from aigcdet.eval.metrics import threshold_at_fpr

#: Wide enough for the longest decision, "review". A narrower dtype would
#: truncate it silently.
_DECISION_DTYPE = "<U6"


@dataclass(frozen=True)
class Policy:
    """The three numbers a deployed decision rule consists of.

    Frozen: a report and the decisions it describes must come from the same
    thresholds, and a policy mutated between the two desynchronises them
    without any error.
    """

    flag_threshold: float      # p >= this  -> flag as AI-generated
    clear_threshold: float     # p <= this  -> clear as authentic
    eqi_threshold: float       # below this -> always review, whatever p says


def _check_scores(p: np.ndarray, eqi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pr = np.asarray(p, dtype=np.float64)
    ev = np.asarray(eqi, dtype=np.float64)
    if pr.ndim != 1 or ev.ndim != 1:
        raise ValueError(f"p and eqi must be 1-D, got shapes {pr.shape} and {ev.shape}")
    if pr.size == 0:
        raise ValueError("p is empty")
    if pr.shape != ev.shape:
        raise ValueError(
            f"p and eqi must have the same length, got {pr.size} and {ev.size}")
    if not (np.isfinite(pr).all() and np.isfinite(ev).all()):
        raise ValueError("p and eqi must be finite; got NaN or inf")
    if pr.min() < 0.0 or pr.max() > 1.0:
        raise ValueError(
            f"p must be calibrated probabilities in [0, 1], got range "
            f"[{pr.min()!r}, {pr.max()!r}]")
    return pr, ev


def _check_labels(y: np.ndarray, n: int) -> np.ndarray:
    yt = np.asarray(y)
    if yt.shape != (n,):
        raise ValueError(
            f"p and y must have the same length, got {n} and {yt.size}")
    uniq = np.unique(yt)
    if not np.isin(uniq, (0, 1)).all():
        raise ValueError(f"y must be 0/1, got values {uniq.tolist()}")
    if uniq.size < 2:
        raise ValueError(
            "fitting a policy needs both classes in the validation set; a "
            f"false-positive rate over an empty class is undefined (got only "
            f"{uniq.tolist()})")
    return yt.astype(int)


def fit_policy(p: np.ndarray, y: np.ndarray, eqi: np.ndarray,
               target_fpr: float = 0.01, target_coverage: float = 0.85,
               *, split) -> Policy:
    """Choose the three thresholds on internal validation rows.

    `split` is required and holds one split label per row; every row must be
    the internal validation split (spec §6.7).

    `target_fpr` is spent symmetrically: at most that share of authentic images
    may be flagged, and at most that share of AI-generated images may be
    cleared. The EQI threshold then buys back coverage by deferring the
    least-evidenced `1 - target_coverage` of images to a human.

    The clear threshold is the mirror of the flag threshold, obtained by asking
    `threshold_at_fpr` the same question with the labels flipped and the scores
    NEGATED. Negation is used rather than the complement `1 - p`: `-(-p)` is
    exact in binary floating point where `1 - (1 - p)` is not, and a threshold
    landing one ulp off an observed probability silently moves that entire tie
    group between "clear" and "review".

    Both thresholds inherit `threshold_at_fpr`'s conservatism -- it can return
    a stricter threshold than the exact optimum, never a looser one -- so the
    realised rates land at or under the target on both sides.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr!r}")
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in [0, 1], got {target_coverage!r}")
    pr, ev = _check_scores(p, eqi)
    check_fit_split(split, pr.size)
    yt = _check_labels(y, pr.size)

    flag = threshold_at_fpr(yt, pr, target_fpr)
    clear = -threshold_at_fpr(1 - yt, -pr, target_fpr)
    if target_coverage == 0.0:
        # np.quantile(eqi, 1.0) is the maximum EQI, which the `>=` gate still
        # admits. Zero coverage means zero images, not one.
        eqi_thr = float("inf")
    else:
        eqi_thr = float(np.quantile(ev, 1.0 - target_coverage))
    return Policy(flag_threshold=float(flag), clear_threshold=float(clear),
                  eqi_threshold=eqi_thr)


def decide(p: np.ndarray, eqi: np.ndarray, policy: Policy) -> np.ndarray:
    """Route each image to "clear", "review" or "flag".

    An image whose probability satisfies both gates at once -- possible only
    when `clear_threshold >= flag_threshold`, i.e. when the fit was degenerate
    -- goes to review. Picking a winner there would state a confident decision
    the thresholds themselves disagree about.
    """
    pr, ev = _check_scores(p, eqi)
    out = np.full(pr.size, "review", dtype=_DECISION_DTYPE)
    confident = ev >= policy.eqi_threshold
    flag = confident & (pr >= policy.flag_threshold)
    clear = confident & (pr <= policy.clear_threshold)
    out[flag & ~clear] = "flag"
    out[clear & ~flag] = "clear"
    return out


def auto_decided_fraction(decisions: np.ndarray) -> float:
    """Share of the queue decided without a human, over ALL images given."""
    d = np.asarray(decisions)
    if d.size == 0:
        raise ValueError("decisions is empty; there is no fraction to report")
    return float((d != "review").mean())


def policy_report(p: np.ndarray, y: np.ndarray, eqi: np.ndarray,
                  policy: Policy) -> dict:
    """The numbers the write-up quotes, each over a stated population.

    Every rate below names the set it is averaged over, because the same
    quantity over a different denominator is a different claim:

    - `auto_fraction`: over ALL images -- the share decided without a human.
      This is the reviewer-load figure.
    - `review_fraction`: over ALL images -- the complement of `auto_fraction`.
    - `realised_fpr`: over the AUTO-DECIDED AUTHENTIC images only (`n_authentic_auto`)
      -- the share of them flagged as AI-generated. It is NOT the false-positive
      rate over all authentic images: the EQI gate has already removed the
      least-evidenced ones from this population, which is why it is lower.
    - `accuracy_on_auto`: over the AUTO-DECIDED images only (`n_auto`) -- the
      share whose decision matches the label. It is not the accuracy of the
      detector on the full set, which is lower.

    `realised_fpr` and `accuracy_on_auto` are `None` when their population is
    empty, which happens whenever the EQI gate defers everything. An empty
    denominator is not a rate of 0.0 and not a silent NaN; the caller has to
    see that there is no number rather than quote one.

    `n_auto` and `n_authentic_auto` are the denominators, reported so the rates
    can be read with the sample size they rest on.
    """
    pr, ev = _check_scores(p, eqi)
    yt = _check_labels(y, pr.size)

    d = decide(pr, ev, policy)
    auto = d != "review"
    flagged = d == "flag"
    authentic_auto = auto & (yt == 0)
    n_auto = int(auto.sum())
    n_authentic_auto = int(authentic_auto.sum())

    return {
        "auto_fraction": float(auto.mean()),
        "review_fraction": float((~auto).mean()),
        "realised_fpr": (float(flagged[authentic_auto].mean())
                         if n_authentic_auto else None),
        "accuracy_on_auto": (float((flagged[auto].astype(int) == yt[auto]).mean())
                             if n_auto else None),
        "n_auto": n_auto,
        "n_authentic_auto": n_authentic_auto,
    }
