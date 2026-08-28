"""Robustness table, heatmap and degradation-head validation (spec §6.1, §6.4, §3.4).

This module renders the numbers a reader acts on, so its job is as much refusal
as computation: a presentation defect here is indistinguishable from a
measurement defect to whoever reads the table.

Two accuracy columns are reported deliberately. `acc_oracle` re-tunes the
threshold per condition, which most papers do and which implicitly assumes
test-time knowledge of the degradation. `acc_fixed` uses one threshold chosen
on clean validation and frozen, which is the deployment condition. The gap
between them is score drift under degradation.

Four things this module refuses to do rather than do plausibly:

1. **Compare incomparable rungs.** `robustness_table` checks that every rung
   was scored over the same conditions, in the same order, on the same images,
   and routes the rungs' banks through `eval.grid.assert_banks_comparable`.
   Differing view coverage turns a rung comparison into a comparison of
   augmentation budgets, which invalidates the §6.4 model-selection rule.
   `banks` is REQUIRED, because the frame-level checks cannot see the backbone
   and cannot see the manifest either (`image_idx` is a positional index, so
   two different manifests of equal length produce identical sets). A caller
   with no banks passes `BANKS_NOT_VERIFIED`, which does not silence the gap
   but records it: a `banks_verified = False` column and a banner on every
   rendering.
2. **Emit an unlabelled or wrongly-labelled tier.** The project runs two
   publishable evaluation tiers plus `smoke` (`TIER_CONDITIONS`), and quoting a
   selection-tier number as a final-report number is the failure mode. `tier`
   is required, checked against a closed vocabulary, checked against the
   table's actual condition coverage, carried as a real column (not only in the
   lossy `DataFrame.attrs`) and re-checked at render time.
3. **Drop the unseen-severity distinction.** `HELDOUT_SEVERITY_CONDITIONS` is
   the difference between "robust to what we trained on" and "robust to what we
   did not". It is flagged per condition, aggregated into its own summary
   column, marked in the markdown and the heatmap, and its disappearance from a
   reshaped table is an error rather than a silent omission.
4. **Let one rendering path be weaker than another.** `to_markdown` and
   `save_heatmap` run the *same* gate, `_check_renderable`. The figure ships as
   `docs/robustness_table.png` next to the markdown and is what a reader looks
   at first, so it must not be the path where a relabelled tier or a dropped
   column goes unnoticed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aigcdet.augment.recipes import FAMILIES, N_FAMILIES
from aigcdet.augment.scenarios import (
    CORE_CONDITIONS, EVAL_GRID, HELDOUT_SEVERITY_CONDITIONS,
)
from aigcdet.eval.metrics import (
    accuracy_at_threshold, bootstrap_ci, expected_calibration_error, roc_auc,
    tpr_at_fpr,
)
from aigcdet.features.proxies import PROXY_NAMES

if TYPE_CHECKING:  # `features.bank` drags in torch/PIL/cv2; only needed for types
    from aigcdet.features.bank import FeatureBank

#: The two evaluation tiers and the condition coverage each one is defined
#: over (plan global constraints). The ablation/selection tier runs the full
#: twenty-condition grid over a 5k internal-validation split plus a 5k
#: stratified benchmark subsample; the final-report tier runs the complete
#: ~13.8k benchmark over the fifteen core conditions, once. The mapping is the
#: mechanism that makes a *wrongly*-labelled table impossible, not merely a
#: docstring: a twenty-condition table cannot claim `final_report`.
#:
#: `smoke` is the deliberate third entry, and its coverage is `None` meaning
#: "any subset". Plan 4 needs a three-condition smoke run, and the alternative
#: to a first-class tier for it is a bypass around the coverage check -- which
#: is precisely how a three-condition smoke number reaches a results table.
#: Instead it is a tier that renders with `NOT FOR PUBLICATION` stamped on both
#: the markdown and the figure.
TIER_CONDITIONS: dict[str, tuple[str, ...] | None] = {
    "ablation": tuple(EVAL_GRID),
    "final_report": tuple(CORE_CONDITIONS),
    "smoke": None,
}

#: Tiers whose output must never be quoted. Rendered with a banner.
UNPUBLISHABLE_TIERS: frozenset[str] = frozenset({"smoke"})

#: Banner stamped on any rendering of an unpublishable tier.
NOT_FOR_PUBLICATION = "NOT FOR PUBLICATION"

#: Banner stamped on any rendering built without the bank-level check.
UNVERIFIED_BANNER = "bank-level comparability NOT verified"

#: The explicit opt-out for `robustness_table`'s required `banks` argument.
#:
#: `banks` is required because the frame-level checks cannot cover two of
#: `eval.grid._COMPARABLE_KEYS`. They cannot see `backbone` -- a score frame
#: records none -- and they cannot see `manifest_sha256`, because `score_grid`
#: fills `image_idx` from `bank.meta["image_idx"]`, the POSITIONAL manifest
#: index: two banks over two DIFFERENT manifests of equal length produce
#: byte-identical `image_idx` sets, so comparing those sets proves nothing
#: about which images were scored. A rung pair straddling a manifest re-split
#: therefore passes every frame-level check with one rung's labels misaligned.
#:
#: Callers who genuinely have no banks pass this sentinel. It does not silence
#: the concern; it records it, as a `banks_verified = False` column and a
#: banner on every rendering. An unverified comparison must be visibly
#: unverified rather than indistinguishable from a verified one.
BANKS_NOT_VERIFIED = "banks-not-verified"

#: Sentinel default marking `banks` as required while still allowing a helpful
#: error rather than a bare TypeError that never mentions BANKS_NOT_VERIFIED.
_REQUIRED = object()

#: `condition_metrics` columns that may be tabulated as a robustness metric.
#: Validated BEFORE any rung is scored: `heldout_severity` is a flag and `n`,
#: `boot_seed`, `boot_n` are provenance, and none of the four is a metric.
METRIC_COLUMNS: tuple[str, ...] = (
    "auc", "auc_lo", "auc_hi", "tpr_at_1pct", "acc_oracle", "acc_fixed", "ece",
)

#: Bootstrap settings for every reported AUC (spec §6.1: 1000 resamples, 95%
#: CI). The seed is a parameter and is recorded in the table, because a CI that
#: cannot be reproduced is decoration.
DEFAULT_BOOT_SEED: int = 20260827
DEFAULT_N_BOOT: int = 1000

#: Marker appended to a held-out-severity condition wherever the table is
#: rendered for a human (markdown headers, heatmap tick labels).
HELDOUT_MARK: str = " (unseen)"

#: Column-name prefixes and names `robustness_table` reserves for its own
#: summary/provenance columns. A condition may not collide with them, so
#: `_condition_columns` can separate the two from the column names alone --
#: which survives any reshape, unlike `DataFrame.attrs`.
_SUMMARY_PREFIXES = ("robust_", "heldout_", "seen_")
_SUMMARY_NAMES = ("tier", "n_images", "boot_seed", "boot_n", "banks_verified")

#: Which proxy validates which degradation family, and which WAY the proxy is
#: expected to move as the learned severity rises (spec §3.4).
#:
#: The sign is the whole point of this check. `jpeg_quality` FALLS as JPEG
#: severity rises and `laplacian_var` FALLS as blur severity rises, so a
#: correctly trained head produces a strong NEGATIVE Spearman for those two
#: families; `noise_floor` rises with noise severity, so noise is positive.
#: Taking absolute values would report a head that learned the relationship
#: backwards as a healthy one -- exactly the failure this day-4 check exists to
#: catch -- so the correlation is reported signed, alongside the expected sign
#: and their product (`spearman_aligned`, where negative means BACKWARDS).
_PROXY_FOR_FAMILY: dict[str, tuple[str, int]] = {
    "jpeg": ("jpeg_quality", -1),
    "blur": ("laplacian_var", -1),
    "noise": ("noise_floor", +1),
}

#: The families a model-free proxy exists for, exported so callers do not
#: hardcode the triple and drift from `_PROXY_FOR_FAMILY`.
PROXIED_FAMILIES: tuple[str, ...] = tuple(_PROXY_FOR_FAMILY)


# --- thresholds ------------------------------------------------------------

def _best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """The threshold maximising accuracy, computed exactly in O(n log n).

    Exactness is load-bearing rather than a nicety. `acc_fixed <= acc_oracle`
    is an invariant the table is read against -- the gap between the columns is
    reported as score drift -- and it only holds if `acc_oracle` is a true
    maximum over ALL thresholds. Scoring a subsample of candidate thresholds
    (say 512 quantiles, which is tempting once a condition has ~14k distinct
    scores) can miss the optimum and let a frozen clean threshold beat the
    "oracle", inverting the invariant on real-sized data while looking fine on
    a 400-row fixture.

    Sorting once and sweeping gives every distinct threshold's accuracy
    exactly: predicting 1 iff `s >= ss[k]` is correct on
    `(n_pos - cum_pos[k]) + cum_neg[k]` rows. Only split points between
    distinct score values are reachable thresholds. `k == n` is the
    reject-everything threshold, which is a real operating point (it is optimal
    when a condition has destroyed the signal and the majority class is
    authentic) and is otherwise unreachable from the observed scores.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    n = len(y)
    if n == 0:
        raise ValueError("cannot choose a threshold from an empty score set")
    order = np.argsort(s, kind="stable")
    ss = s[order]
    ys = y[order].astype(np.int64)
    cum_pos = np.concatenate(([0], np.cumsum(ys)))
    cum_neg = np.concatenate(([0], np.cumsum(1 - ys)))
    correct = (int(ys.sum()) - cum_pos) + cum_neg
    reachable = np.ones(n + 1, dtype=bool)
    reachable[1:n] = ss[1:] != ss[:-1]
    k = int(np.argmax(np.where(reachable, correct, -1)))
    return float(ss[k]) if k < n else float(ss[-1] + 1.0)


# --- per-condition metrics -------------------------------------------------

def condition_metrics(scores_df: pd.DataFrame, probs: pd.Series | np.ndarray | None = None,
                      clean_threshold: float | None = None,
                      seed: int = DEFAULT_BOOT_SEED,
                      n_boot: int = DEFAULT_N_BOOT) -> pd.DataFrame:
    """One row per condition: AUC with a bootstrap CI, TPR@1%FPR, both
    accuracies, ECE, n, and the unseen-severity flag.

    `seed` and `n_boot` are recorded in the returned frame (`boot_seed`,
    `boot_n`) so a reported interval can be reproduced from the table alone.
    """
    required = {"condition", "label", "score"}
    missing = required - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing column(s) {sorted(missing)}; "
                         "it must be a frame as produced by eval.grid.score_grid")
    df = scores_df.copy()
    if probs is not None:
        p = np.asarray(probs.to_numpy() if hasattr(probs, "to_numpy") else probs,
                       dtype=float)
        if len(p) != len(df):
            raise ValueError(f"probs has {len(p)} entries but scores_df has "
                             f"{len(df)} rows; they must align row for row")
        df["prob"] = p

    if clean_threshold is None:
        clean = df[df["condition"] == "clean"]
        if clean.empty:
            raise ValueError(
                "no rows with condition == 'clean', so the frozen deployment "
                "threshold behind `acc_fixed` cannot be chosen; pass "
                "clean_threshold explicitly (it must come from clean validation "
                "data, never from the condition being scored)")
        clean_threshold = _best_threshold(clean["label"].to_numpy(),
                                          clean["score"].to_numpy())
    clean_threshold = float(clean_threshold)

    rows = []
    for cond, g in df.groupby("condition", sort=False):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        lo, hi = bootstrap_ci(roc_auc, y, s, n=n_boot, seed=seed)
        rows.append({
            "condition": cond,
            "auc": roc_auc(y, s), "auc_lo": lo, "auc_hi": hi,
            "tpr_at_1pct": tpr_at_fpr(y, s, 0.01),
            "acc_oracle": accuracy_at_threshold(y, s, _best_threshold(y, s)),
            "acc_fixed": accuracy_at_threshold(y, s, clean_threshold),
            "ece": (expected_calibration_error(y, g["prob"].to_numpy())
                    if "prob" in g else float("nan")),
            "n": len(y),
            "heldout_severity": cond in HELDOUT_SEVERITY_CONDITIONS,
            "boot_seed": seed,
            "boot_n": n_boot,
        })
    return pd.DataFrame(rows)


# --- the robustness table --------------------------------------------------

def _is_summary_column(name: str) -> bool:
    return name in _SUMMARY_NAMES or name.startswith(_SUMMARY_PREFIXES)


def _condition_columns(table: pd.DataFrame) -> list[str]:
    """The condition columns of a robustness table, from its column names alone.

    Deliberately not read from `DataFrame.attrs`: pandas drops `attrs` through
    most reshapes and every round trip to disk, and a marking that vanishes
    when the table is reshaped is exactly the silent-omission failure this
    module is written to prevent.
    """
    return [c for c in table.columns if not _is_summary_column(str(c))]


def _check_tier(tier: str) -> None:
    if tier not in TIER_CONDITIONS:
        raise ValueError(
            f"unknown evaluation tier {tier!r}; it must be one of "
            f"{sorted(TIER_CONDITIONS)}. Every table states its tier, because a "
            "selection-tier number quoted as a final-report number is the "
            "failure this label exists to prevent.")


def _check_tier_coverage(tier: str, conditions: Sequence[str]) -> None:
    """The tier's claim must match the conditions actually evaluated."""
    _check_tier(tier)
    allowed = TIER_CONDITIONS[tier]
    if allowed is None:
        # `smoke` is defined over whatever was run; its renderings carry a
        # NOT FOR PUBLICATION banner instead of a coverage guarantee.
        return
    expected = set(allowed)
    got = set(map(str, conditions))
    if got != expected:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise ValueError(
            f"tier {tier!r} is defined over {len(expected)} conditions but the "
            f"table covers {len(got)}: missing {missing}, unexpected {extra}. "
            "The tier label must describe the evaluation that was actually run "
            "(condition counts per tier: "
            f"{ {t: ('any' if c is None else len(c)) for t, c in TIER_CONDITIONS.items()} }).")


def _scored_conditions(scores: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(scores["condition"].tolist()))


def _check_rungs_comparable(per_rung: Mapping[str, pd.DataFrame]) -> list[str]:
    """Every rung must have been scored over the same conditions, in the same
    order, on the same images.

    This is the frame-level half of the project's "identical view coverage
    across compared rungs" constraint. It cannot see the backbone -- a score
    frame does not record one -- so callers holding the feature banks should
    pass them to `robustness_table` for the full check.
    """
    ref_rung = next(iter(per_rung))
    ref_conditions = _scored_conditions(per_rung[ref_rung])
    ref_images = np.unique(per_rung[ref_rung]["image_idx"].to_numpy()) \
        if "image_idx" in per_rung[ref_rung].columns else None
    for rung, scores in per_rung.items():
        conditions = _scored_conditions(scores)
        if conditions != ref_conditions:
            raise ValueError(
                f"rungs {ref_rung!r} and {rung!r} were scored over different "
                f"conditions and cannot be compared: only in {ref_rung!r}: "
                f"{sorted(set(ref_conditions) - set(conditions))}, only in "
                f"{rung!r}: {sorted(set(conditions) - set(ref_conditions))}"
                + ("" if set(conditions) != set(ref_conditions)
                   else " (same set, different order)")
                + ". Comparing rungs over different view coverage measures "
                "augmentation budgets, not models.")
        if ref_images is not None:
            if "image_idx" not in scores.columns:
                raise ValueError(
                    f"rung {ref_rung!r} records image_idx but rung {rung!r} does "
                    "not, so the two cannot be shown to have scored the same "
                    "images")
            images = np.unique(scores["image_idx"].to_numpy())
            if images.shape != ref_images.shape or not np.array_equal(images, ref_images):
                raise ValueError(
                    f"rungs {ref_rung!r} and {rung!r} were scored on different "
                    f"images ({len(ref_images)} vs {len(images)} distinct "
                    "image_idx) and cannot be compared")
    return ref_conditions


def _check_banks(banks: Mapping[str, "FeatureBank"], per_rung: Mapping[str, pd.DataFrame],
                 conditions: Sequence[str]) -> None:
    """Route the rungs' banks through the R24 guard, then add what it cannot say.

    `assert_banks_comparable` treats `manifest_sha256 = None` on both sides as
    agreement. For a resume/merge check that is reasonable; for the table that
    decides the headline model it is not -- two banks that both decline to say
    which manifest they indexed are absence of evidence, not evidence of
    sameness, and banks index the manifest POSITIONALLY, so a re-split between
    two extractions misaligns labels without changing any shape. This caller is
    therefore stricter: a fingerprint must be present, not merely equal.

    The same reasoning applies to `n_images`, and for the same reason it is
    applied the same way. That key is what stops a 5k selection-tier bank from
    sitting beside a 13.8k final-report score frame, and a config that omits it
    used to SKIP the check rather than fail it -- so any bank type whose config
    did not carry the key was exempt from the one comparison that catches a
    tier mismatch, and would render with `banks_verified = True` and no banner.
    A bank that will not say how many images it holds is not a bank that agrees
    with the scores; it is one that will not say.
    """
    if set(banks) != set(per_rung):
        raise ValueError(
            f"banks cover rungs {sorted(banks)} but the scores cover "
            f"{sorted(per_rung)}; pass one bank per rung or none at all")
    # Imported here, not at module scope: `eval.grid` (and `features.bank`,
    # hence the TYPE_CHECKING import above) pull in torch, PIL, tqdm and cv2,
    # which a caller that only wants to render an already-computed table
    # should not have to pay for.
    from aigcdet.eval.grid import assert_banks_comparable

    ordered = [banks[rung] for rung in per_rung]
    assert_banks_comparable(ordered)
    for rung, bank in zip(per_rung, ordered):
        config = getattr(bank, "config", {})
        if config.get("manifest_sha256") is None:
            raise ValueError(
                f"the bank for rung {rung!r} at {getattr(bank, 'path', '?')} "
                "records no manifest_sha256, so it cannot be shown to have been "
                "extracted from the same frozen manifest as the rungs it is "
                "compared against. Two unfingerprinted banks agree only in the "
                "sense that neither will say; re-extract with a fingerprint "
                "before building a comparison table.")
        bank_conditions = config.get("conditions")
        if bank_conditions is None:
            raise ValueError(
                f"the bank for rung {rung!r} has no 'conditions' in its config, "
                "so its view axis is not the condition axis; a robustness table "
                "needs banks written by extract_eval_bank")
        if list(bank_conditions) != list(conditions):
            raise ValueError(
                f"the bank for rung {rung!r} was extracted over conditions "
                f"{list(bank_conditions)} but its scores cover "
                f"{list(conditions)}; the banks passed do not belong to these "
                "scores")
        # The condition axis matching is not enough: a 5k bank sits happily
        # beside 13.8k score frames, which is a different evaluation tier
        # wearing the right condition list.
        scores = per_rung[rung]
        n_scored = (int(scores["image_idx"].nunique())
                    if "image_idx" in scores.columns else None)
        n_bank = config.get("n_images")
        if n_bank is None:
            raise ValueError(
                f"the bank for rung {rung!r} at {getattr(bank, 'path', '?')} "
                "records no n_images, so it cannot be shown to hold the rows "
                "its scores cover. Skipping the check for such a bank makes "
                "every bank type whose config omits the key -- a fused A5 row, "
                "for one -- the only row in the table exempt from the check "
                "that separates a 5k selection-tier bank from a 13.8k "
                "final-report score frame, and the table would still report "
                "banks_verified = True.")
        if n_scored is not None and int(n_bank) != n_scored:
            raise ValueError(
                f"the bank for rung {rung!r} holds {int(n_bank)} images but its "
                f"scores cover {n_scored} distinct image_idx; the banks passed "
                "do not belong to these scores (a bank from a different "
                "evaluation tier will match on conditions and not on rows)")


def robustness_table(per_rung: Mapping[str, pd.DataFrame], tier: str,
                     metric: str = "auc", seed: int = DEFAULT_BOOT_SEED,
                     n_boot: int = DEFAULT_N_BOOT,
                     banks: Mapping[str, "FeatureBank"] | str = _REQUIRED,
                     ) -> pd.DataFrame:
    """Rungs x conditions for one metric, plus summary and provenance columns.

    Summary columns, named after the metric so a TPR mean is never labelled as
    an AUC one: `robust_<metric>` (mean over every degraded condition, clean
    excluded, per §6.1's robust-AUC definition), `heldout_<metric>` (mean over
    the unseen-severity conditions only) and `seen_<metric>` (mean over the
    degraded conditions the training sampler could have drawn). The last two
    are the "robust to what we trained on" versus "robust to what we did not"
    split, which is the most informative comparison in the table.

    Provenance columns: `tier`, `n_images`, `boot_seed`, `boot_n`. They are
    real columns rather than `DataFrame.attrs` entries because attrs do not
    survive a reshape or a round trip through CSV, and a table whose tier went
    missing in transit is one whose numbers can be quoted at the wrong tier.

    `banks` (rung -> FeatureBank) is REQUIRED. The frame-level checks cover
    the condition axis and therefore `n_views`, but they cannot cover the other
    two keys `eval.grid.assert_banks_comparable` compares. They cannot see
    `backbone`, because a score frame records none. And they cannot see
    `manifest_sha256`, because `score_grid` fills `image_idx` from
    `bank.meta["image_idx"]` -- the POSITIONAL manifest index -- so two banks
    built over two DIFFERENT manifests of equal length yield byte-identical
    `image_idx` sets. A rung pair straddling a manifest re-split therefore
    passes every frame-level check with one rung's labels misaligned, and the
    §6.4 headline is then chosen from it. Defaulting `banks` to "skip the
    check" made that failure the path of least resistance, so there is no
    default; a caller with no banks passes `BANKS_NOT_VERIFIED`, which records
    the gap in a `banks_verified` column and banners every rendering.
    """
    if not per_rung:
        raise ValueError("per_rung is empty; there is nothing to compare")
    if banks is _REQUIRED:
        raise ValueError(
            "robustness_table requires `banks`: pass {rung: FeatureBank} so the "
            "rungs go through eval.grid.assert_banks_comparable, or pass "
            "banks=BANKS_NOT_VERIFIED to build the table anyway. The frame-level "
            "checks cannot see the backbone, and cannot see the manifest either "
            "(image_idx is a positional index, so two different manifests of "
            "equal length look identical). BANKS_NOT_VERIFIED is not a way to "
            "silence that -- it stamps the table and every rendering of it as "
            "unverified.")
    if isinstance(banks, str):
        if banks != BANKS_NOT_VERIFIED:
            raise ValueError(
                f"banks must be a {{rung: FeatureBank}} mapping or the exact "
                f"sentinel BANKS_NOT_VERIFIED, got {banks!r}")
        banks_verified = False
    elif isinstance(banks, Mapping):
        banks_verified = True
    else:
        raise ValueError(
            f"banks must be a {{rung: FeatureBank}} mapping or the exact "
            f"sentinel BANKS_NOT_VERIFIED, got {type(banks).__name__}")
    _check_tier(tier)
    conditions = _check_rungs_comparable(per_rung)
    _check_tier_coverage(tier, conditions)
    colliding = [c for c in conditions if _is_summary_column(str(c))]
    if colliding:
        raise ValueError(
            f"condition name(s) {colliding} collide with the table's reserved "
            f"summary columns (names {list(_SUMMARY_NAMES)}, prefixes "
            f"{list(_SUMMARY_PREFIXES)}); rename the condition")
    if metric not in METRIC_COLUMNS:
        # Checked BEFORE the first rung is scored: validating inside the loop
        # spends a full 20 x n_boot resampling pass before rejecting the call,
        # and `heldout_severity` -- a bool flag, not a metric -- passed it.
        raise ValueError(f"unknown metric {metric!r}; it must be one of "
                         f"{list(METRIC_COLUMNS)}")
    if banks_verified:
        _check_banks(banks, per_rung, conditions)

    heldout = [c for c in conditions if c in HELDOUT_SEVERITY_CONDITIONS]
    degraded = [c for c in conditions if c != "clean"]
    seen = [c for c in degraded if c not in HELDOUT_SEVERITY_CONDITIONS]

    rows = {}
    for rung, scores in per_rung.items():
        m = condition_metrics(scores, seed=seed, n_boot=n_boot).set_index("condition")
        row = {c: float(m.loc[c, metric]) for c in conditions}
        for label, subset in (("robust", degraded), ("heldout", heldout),
                              ("seen", seen)):
            row[f"{label}_{metric}"] = (float(m.loc[subset, metric].mean())
                                        if subset else float("nan"))
        row["tier"] = tier
        row["n_images"] = int(scores["image_idx"].nunique()
                              if "image_idx" in scores.columns
                              else int(m["n"].max()))
        row["boot_seed"] = seed
        row["boot_n"] = n_boot
        row["banks_verified"] = banks_verified
        rows[rung] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.attrs["tier"] = tier
    table.attrs["metric"] = metric
    table.attrs["heldout_severity_conditions"] = heldout
    return table


# --- rendering -------------------------------------------------------------

def _tier_of(table: pd.DataFrame) -> str:
    """The tier a table carries, from its column, falling back to its attrs.

    Reports the label; does not validate it. `_check_renderable` runs
    `_check_tier` on every rendering path, so validating here as well was
    duplication that no mutant could reach -- the mutation run is what
    established that, and dead defensive code is worse than none because it
    reads as a guarantee.
    """
    if "tier" in table.columns:
        values = sorted(set(map(str, table["tier"].tolist())))
        if len(values) != 1:
            raise ValueError(
                f"the table mixes evaluation tiers {values}; rungs from "
                "different tiers were never evaluated on the same thing and "
                "must not share a table")
        return values[0]
    attr = table.attrs.get("tier")
    if attr is None:
        raise ValueError(
            "the table states no evaluation tier: it has neither a 'tier' "
            "column nor a 'tier' entry in attrs. Build it with "
            "robustness_table(..., tier=...) rather than labelling it by hand.")
    return str(attr)


def _metric_of(table: pd.DataFrame) -> str:
    """The metric a robustness table holds, from its `robust_<metric>` column.

    Read from the column names rather than `attrs` for the same reason the
    tier is: attrs do not survive a round trip, and the metric decides the
    heatmap's colour scale, so losing it silently rescales the figure.
    """
    for col in table.columns:
        name = str(col)
        if name.startswith("robust_"):
            return name[len("robust_"):]
    return str(table.attrs.get("metric", "auc"))


def heatmap_limits(metric: str) -> tuple[float, float]:
    """Colour limits for the heatmap, per metric.

    AUC is floored at 0.5 because chance is a meaningful bottom of scale and
    anchoring every figure there makes them comparable. Every other metric --
    notably `tpr_at_1pct`, the §6.4 model-selection metric, which routinely
    sits well below 0.5 -- gets the full 0..1 range: clamping those at 0.5
    would render every struggling rung the identical colour and hide exactly
    the differences the figure is drawn to show.
    """
    return (0.5, 1.0) if metric == "auc" else (0.0, 1.0)


def _check_heldout_marking(table: pd.DataFrame, tier: str) -> list[str]:
    """The tier's unseen-severity conditions must still be in the table.

    A reshape that drops columns is easy to do and impossible to see in the
    rendered output, and the unseen-severity columns are the ones a reader
    would most notice the absence of only if told.
    """
    allowed = TIER_CONDITIONS[tier]
    expected = ([] if allowed is None
                else sorted(set(allowed) & set(HELDOUT_SEVERITY_CONDITIONS)))
    present = [c for c in _condition_columns(table)
               if c in HELDOUT_SEVERITY_CONDITIONS]
    dropped = sorted(set(expected) - set(present))
    if dropped:
        raise ValueError(
            f"the unseen-severity condition(s) {dropped} are missing from a "
            f"{tier!r} table that is defined over them. These columns are the "
            "distinction between robustness to trained-on degradations and to "
            "unseen ones; rendering the table without them, unmarked, would "
            "misrepresent it.")
    return [c for c in _condition_columns(table) if c in HELDOUT_SEVERITY_CONDITIONS]


def _banks_verified(table: pd.DataFrame) -> bool:
    """Whether this table went through the bank-level comparability check.

    A table with no `banks_verified` column predates the column or was built by
    hand, and is treated as unverified: the banner is the safe default, since
    the cost of an unnecessary one is a line of text and the cost of a missing
    one is an unverified comparison that reads as a verified one.
    """
    if "banks_verified" not in table.columns:
        return False
    return bool(table["banks_verified"].astype(bool).all())


def _check_renderable(table: pd.DataFrame, tier: str) -> tuple[list[str], list[str]]:
    """Every check a rendering must pass, for EVERY renderer.

    Factored out because `save_heatmap` used to perform none of them, which
    made the PNG a complete bypass of the guarantees `to_markdown` enforces --
    and `docs/robustness_table.png` is a shipped deliverable a reader looks at
    before the markdown. A relabelled tier or a dropped unseen-severity column
    was invisible in the figure and caught in the table, which is the wrong way
    round: the figure is the thing that gets screenshotted.

    Returns the marked unseen-severity conditions and the banner lines the
    rendering must carry.
    """
    _check_tier(tier)
    carried = _tier_of(table)
    if carried != tier:
        raise ValueError(
            f"asked to render this table as tier {tier!r} but it was built as "
            f"{carried!r}. The tier is a property of the evaluation that was "
            "run, not a label chosen at render time; rebuild the table at the "
            "tier you mean.")
    marked = _check_heldout_marking(table, tier)
    _check_tier_coverage(tier, _condition_columns(table))
    banners = []
    if tier in UNPUBLISHABLE_TIERS:
        banners.append(NOT_FOR_PUBLICATION)
    if not _banks_verified(table):
        banners.append(UNVERIFIED_BANNER)
    return marked, banners


def _display_label(name: str) -> str:
    return f"{name}{HELDOUT_MARK}" if name in HELDOUT_SEVERITY_CONDITIONS else str(name)


def _escape(text: str) -> str:
    """Escape the cell separator.

    Rung names are caller-supplied dict keys. An unescaped `|` in one produces
    a row with more cells than the header has columns, which shifts every value
    one column left of its heading -- a silent, total misattribution of the
    numbers.
    """
    return str(text).replace("|", r"\|")


def _format(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{float(value):.4f}"
    return str(value)


def _markdown_table(table: pd.DataFrame) -> str:
    """A GitHub-flavoured markdown table.

    Hand-rolled rather than `DataFrame.to_markdown`, which requires the
    optional `tabulate` package -- not a project dependency and not installed
    here, so the pandas route raises ImportError at write time. Writing it out
    also lets the unseen-severity conditions be marked in the header.
    """
    header = ["rung"] + [_escape(_display_label(str(c))) for c in table.columns]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for rung in table.index:
        cells = [_escape(rung)] + [_format(table.loc[rung, c]) for c in table.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


#: One line per tier, written into every report so the row/condition budget the
#: number was produced under travels with it.
TIER_DESCRIPTIONS: dict[str, str] = {
    "ablation": ("Selection tier: 5k internal validation plus a 5k stratified "
                 "benchmark subsample, over the full 20-condition grid. Used "
                 "for ablations and model selection only."),
    "final_report": ("Final-report tier: the complete benchmark over the 15 "
                     "core conditions, run once. Numbers quoted in the report "
                     "come from here."),
    "smoke": ("Smoke tier: an arbitrary subset of conditions on an arbitrary "
              "number of rows, run to prove the pipeline executes. The numbers "
              "are not an evaluation of anything and must not be quoted."),
}


def to_markdown(table: pd.DataFrame, tier: str, path: str) -> None:
    """Write the table to `path` with its tier stated and unseen severities marked.

    `tier` is cross-checked against the tier the table itself carries. Passing
    a tier the table does not have is an error, not an override: relabelling a
    selection-tier table as a final-report one is precisely the mistake the
    label exists to prevent, and it is invisible once written.
    """
    marked, banners = _check_renderable(table, tier)

    seeds = sorted(set(table["boot_seed"].tolist())) if "boot_seed" in table else []
    boots = sorted(set(table["boot_n"].tolist())) if "boot_n" in table else []
    provenance = (
        f"**Bootstrap 95% CIs:** {boots[0]} resamples, seed {seeds[0]}.\n\n"
        if len(seeds) == 1 and len(boots) == 1 else
        "**Bootstrap 95% CIs:** provenance not recorded in this table.\n\n")

    # `encoding="utf-8"` is not decoration: the body below contains the section
    # sign, and a bare `open(path, "w")` encodes through the locale codec. Under
    # LC_ALL=C -- the default in many container and CI images, and Kaggle is in
    # this project's critical path -- that is ANSI_X3.4-1968 and the write dies
    # with UnicodeEncodeError. Same crash-at-write-time class as `tabulate`.
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Robustness table\n\n")
        for banner in banners:
            f.write(f"> **{banner.upper()}**\n\n")
        f.write(f"**Evaluation tier:** {tier}\n\n")
        f.write(f"{TIER_DESCRIPTIONS[tier]}\n\n")
        f.write(provenance)
        f.write(f"Conditions marked `{HELDOUT_MARK.strip()}` use a severity the "
                "training sampler never drew (spec §4.6, held-out severity "
                f"bands): {', '.join(marked)}.\n\n")
        f.write(_markdown_table(table))
        f.write("\n")


def save_heatmap(table: pd.DataFrame, path: str) -> None:
    """Render the rungs x conditions matrix as a PNG at `path`.

    Only condition columns are plotted: the summary and provenance columns are
    on other scales (or are strings) and would either wash out the colour range
    or fail to cast.

    Runs `_check_renderable`, the same gate `to_markdown` runs, and carries the
    same banners. The figure is a shipped deliverable that a reader looks at
    before the table, so it must not be the one rendering path where a
    relabelled tier or a dropped unseen-severity column goes unnoticed.
    """
    import matplotlib
    matplotlib.use("Agg")  # never touch a display; this runs headless in CI
    import matplotlib.pyplot as plt

    cols = _condition_columns(table)
    if not cols:
        raise ValueError("the table has no condition columns to plot")
    tier = _tier_of(table)
    _, banners = _check_renderable(table, tier)
    metric = _metric_of(table)
    vmin, vmax = heatmap_limits(metric)

    fig, ax = plt.subplots(figsize=(2 + 0.45 * len(cols), 1.5 + 0.45 * len(table)))
    im = ax.imshow(table[cols].to_numpy(dtype=float), aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_display_label(str(c)) for c in cols], rotation=90, fontsize=7)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=8)
    title = f"{metric} by condition (tier: {tier})"
    if banners:
        title += "\n" + " | ".join(b.upper() for b in banners)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- degradation-head validation (spec §3.4) -------------------------------

def validate_degradation_head(pred_severity: np.ndarray, proxies: np.ndarray,
                              families: Sequence[str] = ("jpeg", "blur", "noise"),
                              ) -> pd.DataFrame:
    """Spearman correlation between the learned severity and its model-free proxy.

    Run on day 4 (spec §3.4): a weak correlation means the dashboard's
    degradation readout is not trustworthy, and it is much better to find that
    out before the readout is in a report.

    The correlation is reported SIGNED, with the sign a correct head should
    show (`expected_sign`) and their product (`spearman_aligned`, higher is
    better, negative means the head learned the relationship backwards).
    `jpeg_quality` and `laplacian_var` both fall as their family's severity
    rises, so a healthy head shows a strongly negative Spearman for jpeg and
    blur and a strongly positive one for noise. Reporting `abs(rho)` would
    destroy exactly the signal that distinguishes a working head from an
    inverted one.

    Families with no model-free proxy (`resize`, `jitter`, `crop`) are rejected
    rather than skipped: dropping them silently yields a table that looks like
    a complete validation and is not.
    """
    pred_severity = np.asarray(pred_severity, dtype=float)
    proxies = np.asarray(proxies, dtype=float)
    if pred_severity.ndim != 2 or pred_severity.shape[1] != N_FAMILIES:
        raise ValueError(
            f"pred_severity must be (n, {N_FAMILIES}) with columns in FAMILIES "
            f"order {list(FAMILIES)}, got shape {pred_severity.shape}")
    if proxies.ndim != 2 or proxies.shape[1] != len(PROXY_NAMES):
        raise ValueError(
            f"proxies must be (n, {len(PROXY_NAMES)}) with columns "
            f"{list(PROXY_NAMES)}, got shape {proxies.shape}")
    if pred_severity.shape[0] != proxies.shape[0]:
        raise ValueError(
            f"pred_severity has {pred_severity.shape[0]} rows but proxies has "
            f"{proxies.shape[0]}; they must describe the same views")
    if not families:
        raise ValueError("families is empty; there is nothing to validate")
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown degradation family/families {unknown}; "
                         f"FAMILIES is {list(FAMILIES)}")
    unproxied = [f for f in families if f not in _PROXY_FOR_FAMILY]
    if unproxied:
        raise ValueError(
            f"no model-free proxy exists for family/families {unproxied}; only "
            f"{sorted(_PROXY_FOR_FAMILY)} can be validated this way. Skipping "
            "them silently would produce a table that reads as a complete "
            "validation of the head while saying nothing about them.")

    rows = []
    for fam in families:
        proxy_name, expected_sign = _PROXY_FOR_FAMILY[fam]
        x = pred_severity[:, FAMILIES.index(fam)]
        z = proxies[:, PROXY_NAMES.index(proxy_name)]
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(z) == 0:
            # spearmanr returns nan and warns on a constant input; say so in
            # the table instead of emitting a warning nobody reads.
            rho, p = float("nan"), float("nan")
        else:
            result = spearmanr(x, z)
            rho, p = float(result.statistic), float(result.pvalue)
        rows.append({
            "family": fam,
            "proxy": proxy_name,
            "expected_sign": expected_sign,
            "spearman": rho,
            "spearman_aligned": rho * expected_sign,
            "p_value": p,
            "n": int(len(x)),
            "sign_ok": bool(rho * expected_sign > 0) if not np.isnan(rho) else False,
        })
    return pd.DataFrame(rows)
