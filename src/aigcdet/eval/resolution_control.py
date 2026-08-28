"""Resolution-controlled evaluation subsets (spec §6.5, alongside `eval.controls`).

Why this module exists
----------------------
Image resolution leaks the label in this project's data, badly enough that a
headline number computed over a whole evaluation frame is not evidence about
the detector.

Measured on the frozen 138,116-row manifest (short side = ``min(width,
height)``, after short-side-512 normalisation): 206 of 224 distinct short
sides contain exactly one class, and 40,982 rows -- 29.7% of the pool -- sit
in one of those single-class strata. Short side 256 is 0 authentic / 25,728
generated; short side 224 is 0 / 7,002; short side 200 is 40,000 / 11,516.
Predicting each stratum's majority class from the short side alone scores
0.769 against a 0.529 majority baseline. A model with no access to pixels at
all clears 76% by reading the image header.

The organisers' demo benchmark is worse, and is the reason this module
refuses rather than caveats. All 4,998 COCO authentic images are exactly
200x200. All 8,843 DALL-E 3 generated images have a short side between 346
and 1,746 (median 1,024). **The two halves share no short side at all**, so a
single `>= 300?` comparison on the file header scores 100% on the scored
benchmark. A benchmark TPR of 0.99 is therefore consistent with a detector
that learned "big image = generated" and consistent with a detector that
learned generation artefacts, and the number alone cannot tell the two apart.

What this module does about it
------------------------------
`resolution_matched_subset` selects a subset in which resolution carries no
label information *by construction*: inside every retained resolution stratum
there are exactly as many authentic rows as generated ones, so the stratum
identity is independent of the label and the best possible resolution-only
classifier scores exactly chance. `resolution_leakage` measures that, before
and after, so the claim is checked rather than asserted.

The cost is rows, and the cost is the point. A stratum containing only one
class cannot be balanced and is dropped whole -- that is what "resolution is
uninformative here" costs on this data, and `ResolutionMatchReport` reports
every dropped row and the stratum it came from. A matched subset that keeps
95% of the frame and one that keeps 5% support very different claims, and a
reader who is only shown the metric cannot tell which they are looking at.

What a number computed on a matched subset does and does not mean
-----------------------------------------------------------------
It DOES mean: on images whose resolution gives away nothing, the detector
scored this. Resolution is controlled, not merely uncorrelated by luck.

It does NOT mean: the detector ignores resolution. Nothing here changes the
model; it changes the population the model is scored on. A detector that
reads resolution and nothing else scores chance here, which is the discovery,
but a detector that reads resolution *and* artefacts still gets credit for the
artefacts.

It does NOT mean the number transfers to the full frame. The matched subset is
a non-random subsample -- it over-weights whatever resolutions both classes
happen to occupy -- so it is a statement about that population, not an
unbiased estimate of full-frame performance. Report both, and report the
retention rate next to the matched number.

It does NOT control anything except the image geometry. Nothing here controls
JPEG history, colour statistics, source dataset or content -- `eval.controls`
is the module for content-blind leakage generally.

Design choices a sceptical reader will ask about
------------------------------------------------
**Why exact width-x-height by default, and not the short side?** Because
`exact_short_side` does not actually finish the job on this data, and the
report says so. Matching on the short side leaves the long side and the
aspect ratio free inside a stratum: 1024x1024 and 1024x1792 are one stratum,
and on the demo benchmark's generated half those two shapes are 7,524 and 57
images. Measured on the frozen manifest, a subset matched on the exact short
side still lets an exact-dimensions rule score **+0.2487 over chance** --
so a number published from it is a number about partially-controlled
resolution. `exact_dimensions` is the only strategy under which the residual
is zero by construction, and on this project's data it is affordable: 34,166
of 138,116 rows survive (24.7%, against 46.2% for the short side), which is
17,083 authentic rows against a floor of 1,000. Strictness that costs a
metric would be a bad trade; strictness that costs half the surplus of an
already-sufficient subset is the right default.

Every strategy therefore reports `residual_exact_advantage` -- the
exact-dimensions leak measured on what came back -- and
`ResolutionMatchReport.describe()` downgrades its own claim to "PARTIAL
CONTROL ONLY" whenever that number is non-zero. This is what stops a coarse
`binned_short_side` from reading as a clean control: bin everything into one
bucket and the *bins* balance perfectly while the exact sizes stay perfectly
separating, and the residual (+0.5000 on the demo benchmark) is what catches
it.

**Why offer bins at all, then?** Because "the exact subset is too small to
support the metric" is a real situation, and the honest response to it is a
weaker claim clearly labelled, not a silent one. `binned_short_side` is a
decision the caller makes explicitly and the report annotates -- it is never
reached by default.

**Why refuse a small subset instead of returning it with a warning?** Because
TPR at 1% FPR on a small authentic set is not a noisy estimate of the right
quantity, it is a different quantity. The threshold is the empirical
(1 - alpha) quantile of the authentic scores; with `n` authentic rows the
achievable false-positive rates are 0, 1/n, 2/n, ..., so at alpha = 0.01 and
n = 100 the only thresholds available are FPR 0 and FPR 0.01, and the FPR
0.01 threshold *is* the single largest authentic score. One observation sets
the operating point, and one unlucky outlier moves the reported TPR by an
arbitrary amount. The floor here is `MIN_EXCEEDANCES` (10) authentic rows
above the threshold, i.e. ``ceil(10 / target_fpr)`` = 1,000 authentic rows at
1% FPR -- the usual rule of thumb for an empirical tail quantile, and the
smallest number at which the threshold is a quantile rather than an
order statistic. The generated floor is smaller (`MIN_GENERATED`, 100)
because TPR is an ordinary proportion once the threshold is fixed. Both are
parameters; a caller who wants to publish under a weaker floor must say so in
code, and `ResolutionMatchTooSmall` carries the full report so the refusal is
as informative as a success.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from aigcdet.operating_point import TARGET_FPR, fpr_label

#: Default seed. Shared with `eval.grid.BENCHMARK_SEED` by value rather than by
#: import so a change to the grid's sampling does not silently re-draw every
#: published resolution-matched subset.
RESOLUTION_MATCH_SEED = 20260827

#: Authentic rows required *above* the threshold for TPR@target_fpr to be a
#: tail quantile rather than a single order statistic. See the module
#: docstring; ten is the customary floor for an empirical tail quantile.
MIN_EXCEEDANCES = 10

#: Generated rows required. TPR is an ordinary proportion once the threshold is
#: fixed, so this floor is about the width of its confidence interval (+/- ~10
#: points at n = 100), not about whether the statistic is defined.
#:
#: Matching equalises the two classes, so the EFFECTIVE floor is
#: ``max(min_authentic, min_generated)`` and this one never binds at its
#: default (100 < 1000). It is here for the caller who deliberately lowers
#: `min_authentic` -- so that doing so does not silently buy a ten-row subset
#: -- and it would bind on its own if a future strategy ever matched with
#: unequal class counts.
MIN_GENERATED = 100

#: Octave bins for `binned_short_side`. The open-ended top bin is deliberate:
#: a fixed upper edge would silently merge 2048 and 8192 into one stratum.
DEFAULT_SHORT_SIDE_BIN_EDGES: tuple[float, ...] = (0, 64, 128, 256, 512, 1024, 2048, math.inf)

#: Columns every function here requires on the frame it is handed. `width` and
#: `height` are `data.manifest.MANIFEST_COLUMNS` members and are copied into
#: every feature bank's `meta.parquet`, so an eval frame derived from either
#: already carries them.
REQUIRED_COLUMNS = ("label", "width", "height")

AUTHENTIC, GENERATED = 0, 1

#: Reasons a row can leave the frame. `no_counterpart` is the finding;
#: `surplus` is ordinary balancing.
DROP_NO_COUNTERPART = "no_counterpart"
DROP_SURPLUS = "surplus"


# --------------------------------------------------------------------------
# stratum definitions
# --------------------------------------------------------------------------

def exact_short_side(df: pd.DataFrame) -> np.ndarray:
    """``min(width, height)``, exactly. The strictest short-side control."""
    return np.minimum(df["width"].to_numpy(), df["height"].to_numpy()).astype(np.int64)


def exact_dimensions(df: pd.DataFrame) -> np.ndarray:
    """``"{width}x{height}"``. Controls aspect ratio and long side as well."""
    return (df["width"].astype(np.int64).astype(str) + "x"
            + df["height"].astype(np.int64).astype(str)).to_numpy()


def binned_short_side(
    edges: Sequence[float] = DEFAULT_SHORT_SIDE_BIN_EDGES,
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Short side bucketed into half-open ``[lo, hi)`` bins.

    Weaker than `exact_short_side` on purpose: within a bin the exact
    resolution still varies, so a subset matched this way supports the claim
    "coarse resolution was controlled" and not "resolution was controlled".
    Use it when the exact subset cannot support the metric, and say which one
    the published number came from -- `ResolutionMatchReport.strategy` records
    it, including the edges.
    """
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or len(e) < 2:
        raise ValueError(f"need at least two bin edges, got {list(edges)}")
    if not np.all(np.diff(e) > 0):
        raise ValueError(f"bin edges must be strictly increasing, got {list(edges)}")

    def _binned(df: pd.DataFrame) -> np.ndarray:
        ss = np.minimum(df["width"].to_numpy(), df["height"].to_numpy()).astype(float)
        # `right=False` -> half-open [lo, hi); index 0 means "below the first
        # edge", which is out of range and must not silently share a bin with
        # the first real bucket.
        idx = np.searchsorted(e, ss, side="right") - 1
        if np.any(idx < 0) or np.any(ss >= e[-1]):
            bad = sorted(set(ss[(idx < 0) | (ss >= e[-1])].tolist()))
            raise ValueError(
                f"short sides {bad} fall outside the bin edges {list(edges)}; "
                "widen the edges rather than letting them share a bucket")
        return np.array([f"[{e[i]:g},{e[i + 1]:g})" for i in idx], dtype=object)

    _binned.label = f"binned_short_side(edges={[float(x) for x in edges]})"  # type: ignore[attr-defined]
    return _binned


#: Name -> stratum function. `strategy` accepts a key of this mapping or any
#: callable taking the frame and returning one stratum key per row, so a
#: caller can control on something this module never anticipated without
#: editing it.
STRATUM_STRATEGIES: Mapping[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "exact_short_side": exact_short_side,
    "exact_dimensions": exact_dimensions,
    "binned_short_side": binned_short_side(),
}

#: The strictest strategy, and the only one whose `residual_exact_advantage`
#: is zero by construction. See "Why exact width-x-height by default" above.
DEFAULT_STRATEGY = "exact_dimensions"


def _resolve_strategy(strategy) -> tuple[Callable[[pd.DataFrame], np.ndarray], str]:
    if callable(strategy):
        label = getattr(strategy, "label", None) or getattr(strategy, "__name__", repr(strategy))
        return strategy, str(label)
    if strategy not in STRATUM_STRATEGIES:
        raise KeyError(
            f"unknown stratum strategy {strategy!r}; known strategies are "
            f"{sorted(STRATUM_STRATEGIES)}, or pass a callable "
            "(frame -> one stratum key per row)")
    return STRATUM_STRATEGIES[strategy], str(strategy)


def _check_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"frame is missing {missing}; resolution control needs "
            f"{list(REQUIRED_COLUMNS)} (they are manifest columns and are "
            "copied into every feature bank's meta.parquet)")
    if len(df) == 0:
        raise ValueError("frame is empty; nothing to match")
    for c in ("width", "height"):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            raise ValueError(f"{c} has {int(v.isna().sum())} non-numeric or missing values")
        if (v <= 0).any():
            raise ValueError(f"{c} has {int((v <= 0).sum())} non-positive values")
    labels = set(pd.unique(df["label"]).tolist())
    if not labels <= {AUTHENTIC, GENERATED}:
        raise ValueError(
            f"label must be 0 (authentic) or 1 (generated); found {sorted(labels)}")


def _identity_order(df: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    """`pos` reordered by a stable per-row identity, so which rows get picked
    does not depend on how the caller happened to sort the frame.

    Without this, sorting an eval frame by score before matching would change
    the matched subset even though the seed and the rows are identical, and two
    reports of "the same" matched number would disagree. Falls back to
    positional order when the frame carries no identity column -- synthetic
    fixtures often do not -- which is still deterministic, just order-sensitive.
    """
    for col in ("row_id", "rel_path", "path"):
        if col in df.columns:
            keys = df[col].to_numpy()[pos]
            if len(set(keys.tolist())) == len(keys):
                return pos[np.argsort(keys.astype(str), kind="stable")]
    return pos


def _sorted_keys(key_str: np.ndarray) -> list[str]:
    """Distinct stratum keys, numerically where they are numbers.

    Purely presentational -- it fixes the row order of `ResolutionMatchReport.strata`
    so a reader scanning it sees 128, 200, 512 rather than the lexicographic
    1024, 128, 200. It must not change which rows are selected, because
    `_stratum_rng` derives each stratum's draw from the stratum key alone.
    """
    keys = {str(k) for k in key_str.tolist()}

    def order(k: str):
        try:
            return (0, float(k), "")
        except ValueError:
            return (1, 0.0, k)

    return sorted(keys, key=order)


def _stratum_rng(seed: int, stratum_key) -> np.random.Generator:
    """One independent generator per stratum.

    Deriving each stratum's draw from ``(seed, stratum)`` rather than walking a
    single generator across strata means adding or removing one stratum does
    not re-draw every other stratum -- so the report's per-stratum counts stay
    comparable between a run on the full frame and a run on a slice of it.
    """
    digest = hashlib.sha256(f"{int(seed)}|{stratum_key}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


# --------------------------------------------------------------------------
# leakage diagnostic
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolutionLeakage:
    """How much of the label a resolution stratum gives away, on one frame."""

    strategy: str
    n_rows: int
    n_authentic: int
    n_generated: int
    n_strata: int
    n_single_class_strata: int
    n_rows_in_single_class_strata: int
    stratum_majority_accuracy: float
    majority_baseline_accuracy: float

    @property
    def advantage_over_majority(self) -> float:
        """Accuracy a resolution-only rule buys over always guessing the
        larger class. Zero means resolution is uninformative *on this frame*.

        In-sample: each stratum is scored by its own majority, so this is the
        best a resolution-only rule could do with perfect knowledge of the
        frame, i.e. an upper bound on the leak rather than a held-out estimate.
        """
        return self.stratum_majority_accuracy - self.majority_baseline_accuracy

    @property
    def share_in_single_class_strata(self) -> float:
        return self.n_rows_in_single_class_strata / self.n_rows if self.n_rows else 0.0

    def describe(self) -> str:
        lines = [
            f"Resolution leakage, strategy={self.strategy}:",
            f"  {self.n_rows} rows ({self.n_authentic} authentic / "
            f"{self.n_generated} generated) in {self.n_strata} strata.",
            f"  {self.n_single_class_strata} strata ({self.n_rows_in_single_class_strata} "
            f"rows, {100 * self.share_in_single_class_strata:.1f}%) contain exactly one class.",
            f"  A resolution-only rule scores {self.stratum_majority_accuracy:.4f} "
            f"against a {self.majority_baseline_accuracy:.4f} majority baseline "
            f"(+{self.advantage_over_majority:.4f}).",
        ]
        if self.advantage_over_majority <= 1e-12:
            lines.append(
                "  Resolution is uninformative on this frame: a number computed "
                "here cannot be explained by the model reading the image header.")
        else:
            lines.append(
                "  Resolution carries label information on this frame, so a "
                "number computed here is NOT evidence that the model read pixels.")
        return "\n".join(lines)


def resolution_leakage(df: pd.DataFrame, strategy=DEFAULT_STRATEGY) -> ResolutionLeakage:
    """Measure how much a resolution-only classifier could score on `df`.

    Run it before and after `resolution_matched_subset` -- before it says how
    bad the problem is, after it should say `advantage_over_majority == 0`,
    which is the check that the matching did what it claims.
    """
    _check_frame(df)
    fn, label = _resolve_strategy(strategy)
    stratum = np.asarray(fn(df))
    y = df["label"].to_numpy()

    n_auth = int((y == AUTHENTIC).sum())
    n_gen = int((y == GENERATED).sum())
    counts = (pd.DataFrame({"stratum": stratum, "label": y})
              .pivot_table(index="stratum", columns="label", aggfunc="size", fill_value=0))
    for c in (AUTHENTIC, GENERATED):
        if c not in counts.columns:
            counts[c] = 0
    a, g = counts[AUTHENTIC].to_numpy(), counts[GENERATED].to_numpy()

    single = (a == 0) | (g == 0)
    return ResolutionLeakage(
        strategy=label,
        n_rows=len(df),
        n_authentic=n_auth,
        n_generated=n_gen,
        n_strata=int(len(counts)),
        n_single_class_strata=int(single.sum()),
        n_rows_in_single_class_strata=int((a + g)[single].sum()),
        stratum_majority_accuracy=float(np.maximum(a, g).sum() / len(df)),
        majority_baseline_accuracy=float(max(n_auth, n_gen) / len(df)),
    )


# --------------------------------------------------------------------------
# the matched subset
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolutionMatchReport:
    """What the matching kept, what it threw away, and what that cost.

    `strata` is one row per stratum of the INPUT frame -- including the strata
    that contributed nothing -- with columns ``stratum``, ``n_authentic_in``,
    ``n_generated_in``, ``n_kept_per_class``, ``n_kept``, ``n_dropped`` and
    ``drop_reason``. A stratum that survived intact still appears, so the table
    sums to the input frame and the accounting can be checked rather than
    trusted.
    """

    strategy: str
    seed: int
    n_in: int
    n_in_authentic: int
    n_in_generated: int
    n_out: int
    n_out_authentic: int
    n_out_generated: int
    n_strata_in: int
    n_strata_kept: int
    n_dropped_no_counterpart: int
    n_dropped_surplus: int
    #: `ResolutionLeakage.advantage_over_majority` of the STRICTEST strategy
    #: (`exact_dimensions`) recomputed on the OUTPUT subset. Zero means
    #: resolution really is uninformative on what came back. It is not zero
    #: automatically: matching on a coarse `binned_short_side` equalises the
    #: bins while leaving the exact sizes inside them as separating as ever,
    #: and this is the number that catches that. Zero rows out scores 0.0 --
    #: read `n_out` first.
    residual_exact_advantage: float
    strata: pd.DataFrame = field(repr=False)

    @property
    def n_dropped(self) -> int:
        return self.n_in - self.n_out

    @property
    def retention(self) -> float:
        return self.n_out / self.n_in if self.n_in else 0.0

    @property
    def generated_share_in(self) -> float:
        return self.n_in_generated / self.n_in if self.n_in else 0.0

    @property
    def generated_share_out(self) -> float:
        return self.n_out_generated / self.n_out if self.n_out else 0.0

    def describe(self) -> str:
        """Plain statement of what a number on this subset means, with the cost.

        Written to be pasted next to the metric it qualifies. A retention rate
        without the claim is a statistic nobody acts on, and the claim without
        the retention rate is the thing this module exists to prevent.
        """
        head = [
            f"Resolution-matched subset (strategy={self.strategy}, seed={self.seed}):",
            f"  in : {self.n_in} rows ({self.n_in_authentic} authentic / "
            f"{self.n_in_generated} generated, "
            f"{100 * self.generated_share_in:.1f}% generated) in "
            f"{self.n_strata_in} strata.",
            f"  out: {self.n_out} rows ({self.n_out_authentic} authentic / "
            f"{self.n_out_generated} generated, "
            f"{100 * self.generated_share_out:.1f}% generated) in "
            f"{self.n_strata_kept} strata.",
            f"  dropped {self.n_dropped} rows "
            f"({100 * (1 - self.retention):.1f}% of the frame): "
            f"{self.n_dropped_no_counterpart} had no counterpart of the other "
            f"class at their resolution, {self.n_dropped_surplus} were surplus "
            "of the majority class within a matched stratum.",
        ]
        if self.n_strata_kept == 0:
            head.append(
                "  NOTHING SURVIVED. The two classes share no resolution "
                "stratum under this strategy, so this frame cannot be "
                "resolution-matched at all -- resolution alone separates the "
                "classes perfectly. That is a finding about the data, not a "
                "result about the model.")
            return "\n".join(head)
        if self.residual_exact_advantage > 1e-12:
            head += [
                f"  PARTIAL CONTROL ONLY. Within every retained stratum the two "
                f"classes are equinumerous, but the strata are coarser than the "
                f"resolution itself: on the matched subset an EXACT-dimensions "
                f"rule still scores {self.residual_exact_advantage:+.4f} over "
                f"chance. What was controlled is coarse resolution, not "
                f"resolution. Say that, or re-match with "
                "strategy='exact_dimensions'.",
                f"  COST: {100 * (1 - self.retention):.1f}% of the frame.",
            ]
            return "\n".join(head)
        head += [
            "  MEANS: on images whose resolution gives away nothing about the "
            "label, the detector scored this. Within every retained stratum "
            "the two classes are equinumerous, so the best possible "
            "resolution-only classifier scores exactly chance here.",
            "  DOES NOT MEAN: (a) that the detector ignores resolution -- the "
            "population changed, the model did not; (b) that the number "
            "estimates full-frame performance -- the subset is a non-random "
            "subsample that over-weights the resolutions both classes occupy; "
            "(c) that anything other than image geometry is controlled -- "
            "content, JPEG history and source dataset are untouched.",
            f"  COST: {100 * (1 - self.retention):.1f}% of the frame. Read the "
            "matched number and the retention rate together.",
        ]
        return "\n".join(head)


class ResolutionMatchTooSmall(ValueError):
    """The matched subset cannot support the metric it was requested for.

    Carries the `report` so a refusal is as informative as a success: the
    caller can still print the per-stratum accounting and see *why* the subset
    collapsed, and can decide between a weaker stratum strategy and not making
    the claim.
    """

    def __init__(self, message: str, report: ResolutionMatchReport):
        super().__init__(message)
        self.report = report


def minimum_authentic_rows(target_fpr: float = TARGET_FPR,
                           min_exceedances: int = MIN_EXCEEDANCES) -> int:
    """Authentic rows needed for a TPR@`target_fpr` threshold to be meaningful.

    ``ceil(min_exceedances / target_fpr)``. With `n` authentic rows the
    empirical FPR can only take the values ``k/n``, so the threshold at
    `target_fpr` is set by the ``ceil(target_fpr * n)``-th largest authentic
    score. Requiring at least `min_exceedances` such scores is what separates
    "a tail quantile" from "one order statistic and its outliers".
    """
    if not 0 < target_fpr <= 1:
        raise ValueError(f"target_fpr must be in (0, 1], got {target_fpr}")
    if min_exceedances < 1:
        raise ValueError(f"min_exceedances must be >= 1, got {min_exceedances}")
    return int(math.ceil(min_exceedances / target_fpr))


def resolution_matched_subset(
    df: pd.DataFrame,
    *,
    strategy=DEFAULT_STRATEGY,
    seed: int = RESOLUTION_MATCH_SEED,
    target_fpr: float = TARGET_FPR,
    min_exceedances: int = MIN_EXCEEDANCES,
    min_authentic: int | None = None,
    min_generated: int = MIN_GENERATED,
    enforce_minimum: bool = True,
) -> tuple[pd.DataFrame, ResolutionMatchReport]:
    """Rows of `df` in which resolution carries no label information.

    Inside every retained stratum the authentic and generated counts are equal,
    so stratum and label are independent and a resolution-only classifier
    scores exactly chance. Strata holding only one class are dropped whole --
    they cannot be balanced, and dropping them is the measurement.

    Returns ``(subset, report)``. The subset preserves the input frame's
    columns and index labels and is ordered as the input frame was.

    Raises `ResolutionMatchTooSmall` when the result cannot support
    TPR@`target_fpr` (see `minimum_authentic_rows` and the module docstring).
    The exception carries the report. Set ``enforce_minimum=False`` only when
    the subset is destined for a metric with no tail-quantile threshold, such
    as AUC -- never to publish a TPR the guard rejected.

    Determinism: the same frame and seed give the same rows, and re-sorting the
    frame's rows does not change the selection when the frame carries a
    `row_id`, `rel_path` or `path` column with unique values.
    """
    _check_frame(df)
    fn, strategy_label = _resolve_strategy(strategy)
    stratum = np.asarray(fn(df))
    if len(stratum) != len(df):
        raise ValueError(
            f"stratum strategy {strategy_label} returned {len(stratum)} keys "
            f"for {len(df)} rows")

    y = df["label"].to_numpy()
    key_str = np.array([str(k) for k in stratum.tolist()], dtype=object)
    keys = _sorted_keys(key_str)

    rows: list[dict] = []
    picked: list[np.ndarray] = []
    for key in keys:
        in_stratum = np.where(key_str == key)[0]
        auth = in_stratum[y[in_stratum] == AUTHENTIC]
        gen = in_stratum[y[in_stratum] == GENERATED]
        # The minority class is the cap: matching means equal counts, and the
        # larger class cannot contribute rows the smaller one cannot pair.
        take = int(min(len(auth), len(gen)))
        n_in_stratum = len(auth) + len(gen)
        rows.append({
            "stratum": key,
            "n_authentic_in": int(len(auth)),
            "n_generated_in": int(len(gen)),
            "n_kept_per_class": take,
            "n_kept": 2 * take,
            "n_dropped": n_in_stratum - 2 * take,
            "drop_reason": ("" if n_in_stratum == 2 * take
                            else DROP_NO_COUNTERPART if take == 0
                            else DROP_SURPLUS),
        })
        if take == 0:
            continue
        rng = _stratum_rng(seed, key)
        for pool in (auth, gen):
            ordered = _identity_order(df, pool)
            picked.append(rng.choice(ordered, size=take, replace=False)
                          if take < len(ordered) else ordered)

    strata = pd.DataFrame(rows, columns=["stratum", "n_authentic_in", "n_generated_in",
                                         "n_kept_per_class", "n_kept", "n_dropped",
                                         "drop_reason"])
    pos = (np.sort(np.concatenate(picked).astype(np.int64)) if picked
           else np.empty(0, dtype=np.int64))
    subset = df.iloc[pos]

    no_counterpart = int(strata.loc[strata["drop_reason"] == DROP_NO_COUNTERPART,
                                    "n_dropped"].sum())
    surplus = int(strata.loc[strata["drop_reason"] == DROP_SURPLUS, "n_dropped"].sum())
    report = ResolutionMatchReport(
        strategy=strategy_label,
        seed=int(seed),
        n_in=len(df),
        n_in_authentic=int((y == AUTHENTIC).sum()),
        n_in_generated=int((y == GENERATED).sum()),
        n_out=len(subset),
        n_out_authentic=int((subset["label"].to_numpy() == AUTHENTIC).sum()),
        n_out_generated=int((subset["label"].to_numpy() == GENERATED).sum()),
        n_strata_in=len(strata),
        n_strata_kept=int((strata["n_kept"] > 0).sum()),
        n_dropped_no_counterpart=no_counterpart,
        n_dropped_surplus=surplus,
        residual_exact_advantage=(
            resolution_leakage(subset, exact_dimensions).advantage_over_majority
            if len(subset) else 0.0),
        strata=strata,
    )
    # Cheap invariants over the accounting itself. These have caught real
    # off-by-one bugs in this file; they cost microseconds and they are the
    # difference between a report and a claim about a report.
    assert report.n_out_authentic == report.n_out_generated, report
    assert int(strata["n_kept"].sum()) == report.n_out
    assert report.n_dropped == no_counterpart + surplus == int(strata["n_dropped"].sum())

    if enforce_minimum:
        floor_auth = (minimum_authentic_rows(target_fpr, min_exceedances)
                      if min_authentic is None else int(min_authentic))
        if report.n_out_authentic < floor_auth or report.n_out_generated < min_generated:
            raise ResolutionMatchTooSmall(
                f"resolution-matched subset is too small to support "
                f"TPR@{fpr_label(target_fpr)}: {report.n_out_authentic} authentic "
                f"and {report.n_out_generated} generated rows survived matching "
                f"(strategy={strategy_label}), against floors of {floor_auth} "
                f"authentic and {min_generated} generated. "
                f"{report.n_dropped_no_counterpart} of {report.n_in} input rows "
                f"had no counterpart of the other class at their resolution. "
                f"With {report.n_out_authentic} authentic rows the "
                f"{fpr_label(target_fpr)} threshold is set by the "
                f"{max(1, math.ceil(target_fpr * report.n_out_authentic))} most "
                f"extreme authentic score(s), so the metric would be noise. "
                f"Use a coarser `strategy`, or pass enforce_minimum=False if "
                f"the metric has no tail-quantile threshold.",
                report)
    return subset, report
