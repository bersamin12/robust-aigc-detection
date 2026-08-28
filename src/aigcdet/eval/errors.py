"""Model selection and error analysis (spec §6.4, §6.6).

This module owns the single most consequential decision in the plan: which
rung the submission is built around. Spec §6.4 fixes the rule BEFORE any
result exists -- the headline model is the rung in A3-A6 with the highest
robust TPR @ 1% FPR on internal validation against held-out generators --
precisely so the choice cannot be fitted to the numbers once they are on the
table. Three ways that goes wrong, each guarded here rather than left to a
caller's discipline:

1. **The candidate set is A3-A6, not "the best rung".** A0-A2 are ablation
   CONTROLS. If A1 tops the table on the day it is still not the headline
   model, because the ladder's claim is about what A3+ adds. Ineligible rungs
   are therefore named in `selection_report`, and an ineligible rung that
   OUTSCORES the winner raises `IneligibleRungWarning` rather than being
   dropped quietly: a silent filter is indistinguishable from a bug that
   dropped a rung by accident.

2. **The metric is robust TPR@1%FPR, not `val_auc`.** `train_rung` returns
   `val_auc` -- the CLEAN VIEW ONLY, view 0 -- with `val_auc_mean_views`
   beside it. Neither is the selection metric. `val_auc` measures the one
   condition robustness training helps least with, so the robustness grid will
   disagree with it; selecting on it would pick a model by a rule the write-up
   says was not used, and every downstream number would then describe the
   wrong model. `select_headline` reads `SELECTION_METRIC` and nothing else,
   and refuses a result dict that does not carry it instead of falling back to
   whatever AUC-shaped key happens to be present.

3. **The population is internal validation, held-out generators.** Two distinct
   things go wrong if it is not.

   Selecting on the EXTERNAL BENCHMARK spends the organisers' demo set on model
   selection: the headline rung is then chosen to fit the very images the
   report claims it generalises to, and no untouched set is left to say so.

   Separately -- and this is the coupling the day-2 note (C-C) is about -- §6.4
   selects on `val_internal`, and `aigcdet.calibrate` may fit the temperature,
   the EQI gate and the decision policy ONLY on `val_internal`. The two share
   rows by design. That is tolerable exactly as long as selection uses the
   agreed metric over the agreed population, because then the shared use is a
   stated one; it stops being tolerable the moment selection quietly uses a
   different rule, since the calibration set has then been spent on a choice
   nobody wrote down.

   `heldout_robust_tpr` builds the population itself, from the bank's own split
   column, so a caller cannot pass benchmark rows in by mistake; and
   `select_headline` refuses any result that DECLARES a different population,
   split set or target FPR. A caller that declares nothing cannot be checked --
   the docstrings state the requirement, and `run_ablation.py` always declares.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from aigcdet.eval.metrics import tpr_at_fpr
from aigcdet.operating_point import TARGET_FPR, fpr_label, tpr_column_name

#: The rungs eligible to be the headline model (spec §6.4). A0-A2 are ablation
#: controls: they exist to show what A3+ buys, so a control winning is a
#: finding to report, never a model to ship.
ELIGIBLE_RUNGS: tuple[str, ...] = ("a3", "a4", "a5", "a6")

#: The operating point the rule is specified at (spec §6.1, §6.4).
#:
#: NOT a literal, and no longer this module's own: it is the project-wide
#: `operating_point.TARGET_FPR`, which `eval.report.condition_metrics`,
#: `calibrate.policy.fit_policy` and `scripts/make_error_sheet.py` default from
#: as well. As four separate literals, moving this one moved the headline rule
#: and left the reported `tpr_at_1pct` column, the deployed decision policy and
#: the error sheet's diagnostic threshold behind at 1%.
#:
#: A module constant rather than a call-site literal because `SELECTION_RULE`
#: below and `SELECTION_METRIC`'s NAME are both rendered FROM it: a rule string
#: that says "1% FPR" while the call site passes 0.05 would make
#: `selection.json` -- the artefact whose entire purpose is to record the rule
#: -- state a rule that was not used.
SELECTION_TARGET_FPR: float = TARGET_FPR

#: The one key `select_headline` reads. Named for the whole rule -- robust,
#: held-out-generator, TPR at the operating point -- so that a result dict
#: carrying only `val_auc` fails loudly rather than being selected on the wrong
#: criterion. The operating point in the name is DERIVED from
#: `SELECTION_TARGET_FPR`: at the project default this is exactly
#: `heldout_robust_tpr_at_1pct`, and at any other it says so rather than
#: leaving a key that names an operating point nobody selected at.
SELECTION_METRIC: str = f"heldout_robust_{tpr_column_name(SELECTION_TARGET_FPR)}"

#: Keys that are NOT the selection metric, listed in the error message so the
#: reader is told why the AUC sitting right there is not a substitute.
NON_SELECTION_KEYS: tuple[str, ...] = (
    "val_auc", "val_auc_mean_views", "clean_auc", "auc", "robust_auc")

#: Result-dict keys through which a caller may declare the provenance of its
#: numbers -- the population, the splits and the operating point -- so the
#: declaration can be checked rather than taken on trust.
POPULATION_KEY: str = "population"
SPLITS_KEY: str = "splits"
TARGET_FPR_KEY: str = "target_fpr"

#: The splits the selection metric is computed over: authentic images from the
#: internal validation split, generated images from generators withheld from
#: training entirely. `benchmark` and `train` rows are excluded by
#: construction.
SELECTION_SPLITS: tuple[str, str] = ("val_internal", "heldout_generator")
SELECTION_POPULATION: str = (
    "val_internal authentic images vs heldout_generator generated images")

SELECTION_RULE: str = (
    "the rung among " + "/".join(r.upper() for r in ELIGIBLE_RUNGS) + " with "
    f"the highest mean TPR @ {fpr_label(SELECTION_TARGET_FPR)} FPR over the degraded "
    "conditions, computed on " + SELECTION_POPULATION + " (spec §6.4). Fixed "
    "before any result existed; not clean AUC, not val_auc, not the external "
    "benchmark.")


class IneligibleRungWarning(UserWarning):
    """An ineligible rung outscored the selected headline model.

    Not an error -- §6.4 says the headline comes from A3-A6 and that stands --
    but it is a result worth reporting, and the exclusion must be visible at
    the moment it happens rather than inferred later from a table.
    """


# --- the §6.4 selection metric ---------------------------------------------

def check_selection_population(splits: Sequence[str]) -> None:
    """Refuse a bank that cannot supply the §6.4 population, before any work.

    Split coverage is a property of the eval bank alone, knowable in the first
    millisecond. `heldout_robust_tpr` cannot check it any earlier than the
    scores it is handed, which in an ablation run is after a rung has TRAINED;
    an operator who points the ladder at a benchmark-only bank should not burn
    a rung to find that out, so `run_ablation.py` calls this up front.

    This is a NECESSARY condition, not a sufficient one: it says the splits are
    present, not that every condition covers both classes once the population
    filter is applied. `heldout_robust_tpr` still checks the rest.
    """
    values = np.asarray(splits).astype(str)
    missing = [s for s in SELECTION_SPLITS if not (values == s).any()]
    if missing:
        present = {str(s): int(n) for s, n in
                   zip(*np.unique(values, return_counts=True))} if len(values) \
            else {}
        raise ValueError(
            f"this bank has no {missing} rows, so the §6.4 selection population "
            f"cannot be built from it. It contains splits {present}; selection "
            f"requires {list(SELECTION_SPLITS)}. A bank holding only benchmark "
            "rows is the organisers' demo set and must never decide the "
            "headline model.")


def heldout_robust_tpr(scores_df: pd.DataFrame, splits: Sequence[str],
                       target_fpr: float = SELECTION_TARGET_FPR) -> float:
    """Mean TPR @ `target_fpr` FPR over the DEGRADED conditions, §6.4 population.

    `scores_df` is an `eval.grid.score_grid` frame; `splits` is the eval bank's
    per-row split column (`bank.meta["split"]`), indexed by `image_idx` exactly
    as `score_grid` emits it.

    The population is built HERE, from `splits`, rather than being trusted from
    the caller:

    - authentic rows come from `val_internal` only. Benchmark authentic images
      (COCO val2017) are the organisers' demo set; selecting against them
      contaminates the headline choice and, worse, spends the demo set before
      it is reported on.
    - generated rows come from `heldout_generator` only. `val_internal` fakes
      are from generators the head trained on, so including them measures
      memorisation rather than the generalisation §6.4 selects for.
    - `clean` is excluded, per §6.1's robust-metric definition: the mean is
      over the degraded conditions, which is what "robust" means here.

    NOTE FOR TASK 8 (fusion). `splits` is ONE bank's split column, indexed by
    `image_idx`. A fused score frame (rung A5, two independently-trained banks
    combined by `eval.fusion.fuse_scores`) has no single owning bank, so before
    this function is called on one, fusion must define which bank's split
    column applies -- and the honest answer is that it is only defined when the
    two banks were extracted from the same frozen manifest, which is what
    `assert_banks_comparable` already requires of anything sharing a table.
    Passing the first bank's splits without checking that would silently label
    A5's rows from the wrong manifest.

    Every refusal below exists because the obvious lenient alternative is
    silent. Skipping a condition that has only one class after the population
    filter, and averaging what is left, yields a number that is a mean over an
    unstated subset of the grid; returning 0.0 when NO condition survives makes
    every rung score 0.0 and hands the headline to whichever rung sorts first.
    """
    required = {"condition", "image_idx", "label", "score"}
    missing = required - set(scores_df.columns)
    if missing:
        raise ValueError(
            f"scores_df is missing column(s) {sorted(missing)}; it must be a "
            "frame as produced by eval.grid.score_grid")

    splits = np.asarray(splits).astype(str)
    idx = scores_df["image_idx"].to_numpy()
    if len(idx) and (idx.min() < 0 or idx.max() >= len(splits)):
        raise ValueError(
            f"scores_df has image_idx in [{idx.min()}, {idx.max()}] but `splits` "
            f"has {len(splits)} entries; `splits` must be the eval bank's own "
            "split column, positionally indexed by image_idx")
    row_split = splits[idx]
    label = scores_df["label"].to_numpy()

    held = row_split == "heldout_generator"
    if held.any() and (label[held] == 0).any():
        raise ValueError(
            f"{int((label[held] == 0).sum())} row(s) in the heldout_generator "
            "split carry label 0. That split is defined by generator family, so "
            "every row in it is generated; an authentic row there means the "
            "manifest and the bank disagree, and the selection population "
            "cannot be built from it.")

    authentic = (row_split == "val_internal") & (label == 0)
    generated = held & (label == 1)
    if not authentic.any() or not generated.any():
        present = {str(s): int(n) for s, n in
                   zip(*np.unique(row_split, return_counts=True))}
        raise ValueError(
            f"the §6.4 selection population is empty: {int(authentic.sum())} "
            f"val_internal authentic and {int(generated.sum())} "
            "heldout_generator generated rows. The scored bank contains splits "
            f"{present}. Selection requires {list(SELECTION_SPLITS)}; a bank "
            "holding only benchmark rows is the external demo set and must "
            "never decide the headline model.")

    sub = scores_df[authentic | generated]
    conditions = [c for c in dict.fromkeys(sub["condition"].tolist()) if c != "clean"]
    if not conditions:
        raise ValueError(
            "scores_df has no degraded condition; a robust metric is a mean "
            "over the degraded conditions and there are none to average "
            f"(conditions seen: {sorted(set(sub['condition']))})")

    values = []
    for cond in conditions:
        g = sub[sub["condition"] == cond]
        y = g["label"].to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"condition {cond!r} has only class {sorted(set(y.tolist()))} "
                "once restricted to the selection population, so its TPR@FPR is "
                "undefined. Averaging the conditions that did survive would "
                "report a mean over an unstated subset of the grid; re-extract "
                "the eval bank so every condition covers both classes.")
        values.append(tpr_at_fpr(y, g["score"].to_numpy(), target_fpr))
    return float(np.mean(values))


# --- the §6.4 selection rule -----------------------------------------------

def _normalise(rung: str) -> str:
    return str(rung).strip().lower()


def _check_provenance(rung: str, result: Mapping) -> None:
    """Refuse a result that DECLARES a population, split set or operating point
    other than §6.4's.

    This is the only contamination the function can detect: a dict of floats
    does not say where it came from unless the producer said so.

    The `target_fpr` arm exists because the operating point is the one part of
    the rule that a caller can change without changing anything visible -- the
    metric key stays `heldout_robust_tpr_at_1pct`, the number stays in [0, 1],
    and `selection.json` keeps quoting SELECTION_RULE. Refusing a declared
    mismatch means a producer that moved the operating point either says so and
    is rejected, or does not say so and is not this project's code.
    """
    declared_fpr = result.get(TARGET_FPR_KEY)
    if declared_fpr is not None and float(declared_fpr) != SELECTION_TARGET_FPR:
        raise ValueError(
            f"rung {rung!r} declares target_fpr {declared_fpr!r}, but §6.4 "
            f"selects at {SELECTION_TARGET_FPR}. The metric key, the value "
            "range and the recorded rule all look identical at another "
            "operating point, so a mismatch here is refused rather than "
            "recorded.")
    declared = result.get(POPULATION_KEY)
    if declared is not None and str(declared) != SELECTION_POPULATION:
        raise ValueError(
            f"rung {rung!r} declares population {declared!r}, but §6.4 selects "
            f"on {SELECTION_POPULATION!r}. Selecting on any other population -- "
            "the external benchmark above all -- contaminates the choice and "
            "leaves val_internal unusable for calibration afterwards.")
    splits = result.get(SPLITS_KEY)
    if splits is not None and tuple(splits) != SELECTION_SPLITS:
        raise ValueError(
            f"rung {rung!r} declares splits {tuple(splits)!r}, but §6.4 selects "
            f"over {SELECTION_SPLITS!r}.")


def _metric_of(rung: str, result: Mapping) -> float:
    if SELECTION_METRIC not in result:
        raise ValueError(
            f"rung {rung!r} carries no {SELECTION_METRIC!r} (it has "
            f"{sorted(result)}). The §6.4 rule selects on robust TPR @ "
            f"{fpr_label(SELECTION_TARGET_FPR)} FPR "
            f"and on nothing else: {list(NON_SELECTION_KEYS)} are clean-view or "
            "whole-grid AUCs and are NOT substitutes -- the robustness grid "
            "routinely disagrees with them. Compute the metric with "
            "heldout_robust_tpr() and store it under that key.")
    value = float(result[SELECTION_METRIC])
    if not np.isfinite(value):
        raise ValueError(
            f"rung {rung!r} has a non-finite {SELECTION_METRIC} ({value!r}); a "
            "rung whose selection metric could not be computed cannot be "
            "compared, and NaN silently loses every comparison it takes part in")
    return value


def _partition(results: Mapping[str, Mapping]) -> tuple[dict, dict]:
    eligible, ineligible = {}, {}
    for rung, result in results.items():
        target = eligible if _normalise(rung) in ELIGIBLE_RUNGS else ineligible
        target[rung] = result
    return eligible, ineligible


def _warn_if_outscored(winner: str, best: float, ineligible: Mapping) -> None:
    beaten = {}
    for rung, result in ineligible.items():
        value = result.get(SELECTION_METRIC)
        if value is not None and np.isfinite(float(value)) and float(value) > best:
            beaten[rung] = float(value)
    if beaten:
        warnings.warn(
            f"rung(s) {sorted(beaten)} scored higher on {SELECTION_METRIC} "
            f"({beaten}) than the selected headline {winner!r} ({best}), but "
            f"they are not eligible: §6.4 chooses only among "
            f"{list(ELIGIBLE_RUNGS)} and the rest are ablation controls. The "
            "headline stands; report the control's win as a finding.",
            IneligibleRungWarning, stacklevel=3)


def select_headline(results: Mapping[str, Mapping]) -> str:
    """The headline rung under spec §6.4. See this module's docstring.

    `results` maps rung name -> that rung's result dict, each of which must
    carry `SELECTION_METRIC` computed by `heldout_robust_tpr` over the §6.4
    population. Rung names are matched case-insensitively, so a rung recorded
    as `"A3"` is a candidate rather than being silently filtered out.

    Ties are broken by rung name ascending, so the same results always select
    the same model.
    """
    if not results:
        raise ValueError(
            "results is empty; there is no rung to select from. Expected one "
            f"entry per trained rung, of which {list(ELIGIBLE_RUNGS)} are "
            "eligible for the headline.")
    eligible, ineligible = _partition(results)
    if not eligible:
        raise ValueError(
            f"no eligible rung among {sorted(results)}: §6.4 chooses the "
            f"headline model only from {list(ELIGIBLE_RUNGS)}, and "
            f"{sorted(ineligible)} are ablation controls. A control winning the "
            "table is a result to report, not a model to ship.")
    for rung, result in eligible.items():
        _check_provenance(rung, result)
    scored = {rung: _metric_of(rung, result) for rung, result in eligible.items()}
    winner = min(scored, key=lambda rung: (-scored[rung], str(rung)))
    _warn_if_outscored(winner, scored[winner], ineligible)
    return winner


def selection_report(results: Mapping[str, Mapping]) -> dict:
    """The record written to `selection.json`: the rule, the inputs, the choice.

    The rule and the population travel WITH the numbers, because the whole
    point of fixing the rule in advance is lost if the artefact that records
    the choice does not also record what the choice was made on.

    A failure to select (no eligible rung, a missing metric) is recorded as
    `headline: null` plus `headline_error`, not raised: the table and the
    summary are still worth writing when the selection cannot be made.
    """
    eligible, ineligible = _partition(results)
    headline, error = None, None
    try:
        headline = select_headline(results)
    except ValueError as exc:
        error = str(exc)
    return {
        "rule": SELECTION_RULE,
        "metric": SELECTION_METRIC,
        "target_fpr": SELECTION_TARGET_FPR,
        "population": SELECTION_POPULATION,
        "splits": list(SELECTION_SPLITS),
        "eligible_rungs": list(ELIGIBLE_RUNGS),
        "headline": headline,
        "headline_error": error,
        "candidates": {r: v.get(SELECTION_METRIC) for r, v in eligible.items()},
        "excluded_as_ineligible": {r: v.get(SELECTION_METRIC)
                                   for r, v in ineligible.items()},
        "summary": {r: dict(v) for r, v in results.items()},
    }


# --- error analysis (spec §6.6) --------------------------------------------

#: Columns used, in this order, to break score ties in `top_errors`. Without
#: them the sheet is only as stable as the row order of the frame it was given,
#: and two runs over identical data can render different images.
TIE_BREAK_COLUMNS: tuple[str, ...] = ("condition", "image_idx", "path")


def top_errors(scores_df: pd.DataFrame, k: int = 24, kind: str = "fp") -> pd.DataFrame:
    """The `k` most confidently mis-scored rows of one class.

    "Top" means most extreme score in the wrong direction, and the function
    takes no threshold on purpose: these are the rows that are wrong at the
    largest number of operating points, so they are the mistakes for any
    threshold strict enough to include them.

    - `kind="fp"`: authentic rows (label 0) by DESCENDING score -- the images
      the detector was most confident were generated.
    - `kind="fn"`: generated rows (label 1) by ASCENDING score.

    The order is fully deterministic: a stable sort on the score, with ties
    broken on `TIE_BREAK_COLUMNS` ascending. Ties are common in practice (a
    saturated head, or two views of one image), and `nlargest`'s default
    tie-breaking follows the input row order, so an upstream `groupby` or a
    re-read of the frame would otherwise change which images the sheet shows.
    """
    if kind not in ("fp", "fn"):
        raise ValueError(f"kind must be 'fp' or 'fn', got {kind!r}")
    missing = {"label", "score"} - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing column(s) {sorted(missing)}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    ascending = kind == "fn"
    pool = scores_df[scores_df["label"] == (1 if ascending else 0)]
    tie = [c for c in TIE_BREAK_COLUMNS if c in pool.columns]
    ordered = pool.sort_values(["score", *tie],
                              ascending=[ascending] + [True] * len(tie),
                              kind="mergesort")
    return ordered.head(k).reset_index(drop=True)


def fp_rate_by_source(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """False-positive rate per source dataset (spec §6.6).

    False positives concentrated in one dataset indicate a confound in that
    dataset, not a weakness of the detector, so the per-source split is what
    makes the overall rate readable.

    EVERY source in the frame gets a row, including sources that contributed no
    authentic image -- theirs is `fp_rate = NaN` with `n_authentic = 0`, never
    0.0. Dropping them makes the table look like complete coverage when it is
    not, and a 0.0 against an empty denominator reads as "this source never
    false-positives", which is the opposite of what the data says.

    A row counts as a false positive when `score >= threshold`, matching the
    `>=` convention in `calibrate.policy.decide`.
    """
    required = {"label", "score", "source"}
    missing = required - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing column(s) {sorted(missing)}")

    df = scores_df.assign(
        _authentic=(scores_df["label"] == 0),
        _fp=((scores_df["label"] == 0) & (scores_df["score"] >= threshold)))
    grouped = df.groupby("source", as_index=False, sort=True).agg(
        n_images=("_fp", "size"),
        n_authentic=("_authentic", "sum"),
        n_fp=("_fp", "sum"))
    grouped["n_authentic"] = grouped["n_authentic"].astype(int)
    grouped["n_fp"] = grouped["n_fp"].astype(int)
    n_authentic = grouped["n_authentic"].to_numpy()
    n_fp = grouped["n_fp"].to_numpy()
    grouped["fp_rate"] = np.where(n_authentic > 0,
                                  n_fp / np.maximum(n_authentic, 1), np.nan)
    return grouped


def contact_sheet(rows: pd.DataFrame, out_path: str,
                  annotations: Sequence[str] | None = None,
                  cols: int = 6, thumb: int = 180) -> None:
    """Render a grid of the worst errors to `out_path`, each one annotated.

    `rows` is a frame as returned by `top_errors`, carrying a `path` column.
    `annotations` must be one string per row when given; the brief's version
    indexed it positionally without checking, which turns a short list into an
    IndexError halfway through writing a partial sheet.

    Matplotlib is used with the `Agg` backend, set explicitly, exactly as
    `eval.report.save_heatmap` does: this runs headless and must never try to
    reach a display.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("nothing to render: `rows` is empty")
    if "path" not in rows.columns:
        raise ValueError("`rows` has no `path` column; merge the eval bank's "
                         "meta onto the scores before building a sheet")
    if annotations is None and "score" not in rows.columns:
        raise ValueError("`rows` has no `score` column and no `annotations` "
                         "were given, so the tiles cannot be labelled")
    if annotations is not None and len(annotations) != n:
        raise ValueError(f"annotations has {len(annotations)} entries for "
                         f"{n} rows; there must be exactly one per row")

    import matplotlib
    matplotlib.use("Agg")  # never touch a display; this runs headless in CI
    import matplotlib.pyplot as plt
    from PIL import Image

    ncols = min(int(cols), n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * thumb / 100.0,
                                      nrows * (thumb + 26) / 100.0))
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, row in enumerate(rows.itertuples()):
        ax = axes[i // ncols][i % ncols]
        with Image.open(row.path) as im:
            ax.imshow(im.convert("RGB").resize((thumb, thumb), Image.BILINEAR))
        ax.set_title(annotations[i] if annotations is not None
                     else f"score={row.score:+.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
