"""Rung A5: paradigm-diverse ensemble fusion (spec §6.4).

Two backbones trained independently produce logits on different scales, so a
raw average would let whichever has the larger spread dominate. Standardising
within each condition first makes the average a genuine vote.

Fusing per condition rather than globally is deliberate: score distributions
shift under degradation, and we want the fusion to be fair at every operating
point, not just on clean data. Standardising within (condition, LABEL) would
be a different thing again, and a broken one -- it centres both classes on
zero, which removes exactly the signal being fused while leaving every
"is it standardised?" check satisfied. The grouping is the condition alone.

**Which rows set the z-score parameters.** Standardising is a fit: each
condition's mean and sigma are estimated from a population of rows and then
applied. The ratio of the two parents' sigmas is the ratio of their
contributions to the fused score, so whichever rows are in that population
decide how much each backbone votes. The ablation tier's bank is 5k internal
validation plus a 5k stratified subsample of the organisers' benchmark (spec
§4.4a), so fitting over the WHOLE frame lets the benchmark set half of it --
and A5's §6.4 selection number becomes a function of how widely the demo set's
scores happen to spread under each backbone. That is exactly what
`errors.heldout_robust_tpr` works to keep out of every other rung's number
("selecting on the external benchmark spends the organisers' demo set on model
selection"), and it does not cancel: two backbones do not separate the
benchmark equally, and the spec expects the benchmark to separate unusually
well.

`fuse_scores` therefore takes the population as an argument -- `splits` (the
bank's own split column, positionally indexed by `image_idx`, i.e. exactly
what `fused_splits` returns) and `fit_splits` (the split names to fit on) --
estimates mean and sigma from those rows only, and APPLIES them to every row.
The fused frame still covers the whole bank; only the fit is restricted.

Because the fused score is not otherwise a fixed function of the two heads,
the population is RECORDED on the output: a `zscore_population` column, not a
`DataFrame.attrs` entry, since pandas drops attrs through most reshapes and a
provenance marking that vanishes on a reshape is no marking.

`fit_splits` is REQUIRED. A caller who genuinely wants the whole-frame fit
passes `fit_splits=ALL_ROWS` and gets a column that says so. Defaulting to it
would have made the contaminated fit the path of least resistance, and a
default is what ships: this branch's history is guards that could be forgotten
being forgotten, and `report.BANKS_NOT_VERIFIED` exists for the same reason
after `banks=None` let a bank skip verification. The whole-frame fit stays
expressible; it stops being inheritable by silence.
`FIT_SPLITS_FOR_SELECTION` is the population `scripts/run_ablation.py` uses --
the §6.4 selection population itself, so the rows that set the weights are the
rows the selection metric is read on, and no benchmark row is in either.

**Whose splits apply to a fused frame.** `errors.heldout_robust_tpr` takes one
bank's `split` column, positionally indexed by `image_idx`. A fused frame has
no single owning bank, so this module answers the question rather than leaving
the caller to pick a parent: `fused_splits` returns the shared column and only
exists when the parents genuinely share one. "Share" is enforced, by
`assert_fusion_parents`, as all four of

1. the same `manifest_sha256`, present on both -- the identity of the frozen
   manifest the banks index positionally;
2. the same condition axis (`conditions` and `n_views`), because view j must
   mean the same thing in both;
3. the same `image_idx` sequence, so the rows line up positionally;
4. an element-for-element identical `split` (and `label`) column. The
   fingerprint alone is not enough: it covers the manifest's path column, so a
   re-split that kept the paths fingerprints identically while moving rows
   between splits.

**A composite row is a declared one, never a borrowed one.** The backbone is
deliberately NOT part of that check. A5 is defined as "+ second backbone"
(spec §6.4), so a differing backbone is the treatment under test. Whether the
resulting row may sit in the same robustness table as a single-backbone rung
is a separate question, and `FusedEvalBank` answers it honestly rather than
quietly: its `backbone` is the composite of its parents', so
`eval.grid.assert_banks_comparable` sees a two-backbone row for what it is.
That guard exempts a DECLARED composite -- one carrying `fused_from`,
`fused_backbones`, and a `backbone` that is exactly the composite of those --
from the backbone key, because the spec's A5 (DINOv3 + SigLIP2) would
otherwise be refused a place in the results table at all (R43). The exemption
attaches to the declaration, not to the row: borrowing one parent's name fails
the consistency check in `eval.grid`, because that is the R24 confound
laundered through a label.

`FusedEvalBank` also carries `n_images`, the row count its parents share.
`report._check_banks` compares that against the number of distinct
`image_idx` in the scores, which is how a 5k selection-tier bank is stopped
from sitting beside a 13.8k final-report score frame; a config without it was
skipped by that check, so every A5 row would have been exempt from it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from aigcdet.eval.errors import SELECTION_SPLITS, SELECTION_TARGET_FPR
from aigcdet.eval.metrics import tpr_at_fpr

#: The columns that identify a row of a `score_grid` frame.
_KEYS = ["condition", "image_idx"]

#: Bank config keys a fused rung's parents must agree on. `backbone` is absent
#: on purpose: see the module docstring.
_PARENT_KEYS = ("n_views", "conditions", "manifest_sha256")

#: The column `fuse_scores` records its z-score population in.
POPULATION_COLUMN = "zscore_population"

#: The explicit opt-out for `fuse_scores`'s required `fit_splits` argument, and
#: the value recorded in `POPULATION_COLUMN` when it is used: mean and sigma
#: were fitted over every row of each frame, benchmark rows included.
#:
#: It does not silence the concern, it records it. A whole-frame fit is a real
#: choice -- there are frames with no split column to speak of -- but it is the
#: choice that lets the organisers' demo set set A5's fusion weights, so it has
#: to appear in a diff and in `selection.json` rather than being inherited by
#: silence. This is the same shape as `report.BANKS_NOT_VERIFIED`.
ALL_ROWS = "all_rows"

#: Sentinel default marking `fit_splits` as required while still allowing an
#: error that names both ways out, rather than a bare TypeError that never
#: mentions ALL_ROWS.
_REQUIRED = object()

#: The population to fit A5's z-score parameters on, for the §6.4 selection
#: run. It is the selection population itself (`errors.SELECTION_SPLITS`), so
#: the rows that decide how much each backbone contributes are the rows the
#: selection metric is computed on -- and the organisers' benchmark subsample
#: is in neither. Aliased rather than re-spelled so the two cannot drift.
FIT_SPLITS_FOR_SELECTION: tuple[str, ...] = tuple(SELECTION_SPLITS)

#: The z-score population to use when the fusion WEIGHT is being fitted too.
#:
#: `FIT_SPLITS_FOR_SELECTION` above is defensible for an A5 whose weights are a
#: fixed constant: the standardisation sees the held-out rows, but nothing is
#: CHOSEN from what it sees. The moment a weight is selected, that stops being
#: true -- the sweep's objective is computed on scores whose scale was set
#: partly by held-out rows, so the held-out number is no longer read exactly
#: once. `scripts/family_experts.py` made this argument first, for the same
#: reason, and fixed it the same way; the constant lives here so the two
#: cannot drift into disagreeing about what a clean fused number is.
FIT_SPLITS_WHEN_FITTING_WEIGHT: tuple[str, ...] = ("val_internal",)


def _population_label(fit_splits: Sequence[str]) -> str:
    """The value recorded in `POPULATION_COLUMN` for a declared population."""
    return "split=" + "+".join(sorted({str(s) for s in fit_splits}))


def zscore_by_condition(df: pd.DataFrame,
                        fit_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Standardise `score` within each condition, leaving every other column.

    Population standard deviation (`ddof=0`), so a condition's scores come out
    with exactly unit variance rather than `sqrt((n-1)/n)` of it. A condition
    whose scores are constant has nothing to standardise and is centred on zero
    rather than divided by it.

    `fit_mask`, when given, is a boolean over the frame's rows selecting the
    population the mean and sigma are ESTIMATED from. Every row is still
    standardised and returned -- restricting the fit is not the same as
    dropping rows, and the fused frame has to cover the whole bank for the
    robustness table. See the module docstring for why the fit population is a
    decision rather than "all of it".
    """
    if "score" not in df.columns or "condition" not in df.columns:
        raise ValueError(
            "zscore_by_condition needs 'condition' and 'score' columns; it "
            f"was given {sorted(df.columns)}")
    out = df.copy()
    if fit_mask is None:
        g = out.groupby("condition", observed=True)["score"]
        std = g.transform("std", ddof=0).replace(0.0, 1.0).fillna(1.0)
        out["score"] = (out["score"] - g.transform("mean")) / std
        return out

    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != (len(out),):
        raise ValueError(
            f"fit_mask must be one boolean per row: the frame has {len(out)} "
            f"rows and the mask has shape {mask.shape}")
    fit = out.loc[mask]
    if fit.empty:
        raise ValueError(
            "the declared z-score population selects no row of this frame, so "
            "there is nothing to fit the standardisation on")
    grouped = fit.groupby("condition", observed=True)["score"]
    mean = grouped.mean()
    sigma = grouped.std(ddof=0)
    conditions = out["condition"]
    absent = sorted(set(conditions.unique().tolist()) - set(mean.index.tolist()))
    if absent:
        # Fitting a condition on another condition's rows, or on none, is not
        # a standardisation; refuse rather than emit NaN scores that every
        # downstream metric would quietly propagate.
        raise ValueError(
            f"condition(s) {absent} have no row in the declared z-score "
            "population, so their mean and sigma are undefined. The fit "
            "population must cover every condition the frame scores.")
    mu = conditions.map(mean).to_numpy(dtype=float)
    sd = conditions.map(sigma).to_numpy(dtype=float)
    sd = np.where(np.isfinite(sd) & (sd != 0.0), sd, 1.0)
    out["score"] = (out["score"].to_numpy(dtype=float) - mu) / sd
    return out


def _fit_mask(df: pd.DataFrame, splits: np.ndarray,
              fit_splits: Sequence[str], position: int) -> np.ndarray:
    """The rows of `df` whose split is one of `fit_splits`.

    Derived from the frame's own `image_idx`, not from row order, so it is
    correct for a frame listed in any order -- the same reason `_aligned`
    matches on keys rather than position.
    """
    if "image_idx" not in df.columns:
        raise ValueError(
            f"frame {position} has no 'image_idx' column, so its rows cannot "
            "be matched to the bank's split column; a declared z-score "
            "population needs one")
    idx = df["image_idx"].to_numpy()
    if len(idx) and (idx.min() < 0 or idx.max() >= len(splits)):
        raise ValueError(
            f"frame {position} has image_idx in [{idx.min()}, {idx.max()}] but "
            f"`splits` has {len(splits)} entries; `splits` must be the eval "
            "bank's own split column, positionally indexed by image_idx (it is "
            "what `fused_splits` returns)")
    return np.isin(splits[idx], np.asarray([str(s) for s in fit_splits]))


def _resolve_population(splits, fit_splits) -> tuple[np.ndarray | None, tuple[str, ...], str]:
    """Validate the declared population and name it for the output column."""
    if fit_splits is _REQUIRED:
        raise ValueError(
            "fuse_scores requires `fit_splits`, because the ratio of the two "
            "parents' sigmas IS the ratio of their contributions to the fused "
            "score, and which rows set those sigmas is therefore a decision. "
            "Pass the bank's `splits` column (what `fused_splits` returns) "
            "together with the split names to fit on -- for the §6.4 selection "
            f"run that is fit_splits=FIT_SPLITS_FOR_SELECTION, i.e. "
            f"{list(FIT_SPLITS_FOR_SELECTION)} -- or fit_splits=ALL_ROWS to fit "
            "over every row. There is no default: on the ablation tier a "
            "whole-frame fit lets the organisers' benchmark subsample set half "
            "of each parent's sigma, and A5's selection number then moves with "
            "the demo set. ALL_ROWS is not a way to silence that; it is "
            "recorded in the output's zscore_population column.")
    if isinstance(fit_splits, str):
        if fit_splits != ALL_ROWS:
            raise ValueError(
                f"fit_splits={fit_splits!r} is a single string, which would be "
                "read one character at a time. Pass a sequence of split names "
                f"(e.g. {list(FIT_SPLITS_FOR_SELECTION)}) or the exact sentinel "
                "ALL_ROWS.")
        if splits is not None:
            raise ValueError(
                "fit_splits=ALL_ROWS fits over every row, so the `splits` "
                "column would not be used; passing both does not say which was "
                "meant. Pass ALL_ROWS alone, or `splits` with the split names "
                "to fit on.")
        return None, (), ALL_ROWS
    if splits is None:
        raise ValueError(
            "`splits` and `fit_splits` go together: `splits` is the bank's "
            "split column (what `fused_splits` returns) and `fit_splits` names "
            "the splits to fit the standardisation on. One without the other "
            "does not describe a population.")
    fit = tuple(str(s) for s in fit_splits)
    if not fit:
        raise ValueError(
            "fit_splits is empty, so the declared z-score population is empty; "
            f"pass the splits to fit on (e.g. {list(FIT_SPLITS_FOR_SELECTION)}) "
            "or pass neither argument for the whole-frame fit")
    values = np.asarray(splits).astype(str)
    present = sorted(set(values.tolist()))
    missing = [s for s in dict.fromkeys(fit) if s not in present]
    if missing:
        raise ValueError(
            f"the declared z-score population names split(s) {missing}, which "
            f"the bank's split column does not contain (it holds {present}). "
            "The recorded population would then describe rows that are not "
            "there, which overstates what the fused score was fitted on.")
    return values, fit, _population_label(fit)


def _aligned(df: pd.DataFrame, base: pd.DataFrame, position: int,
             fit_mask: np.ndarray | None = None) -> np.ndarray:
    """One frame's z-scored column, reordered onto `base`'s rows.

    Reordering rather than sorting matters: `report._check_rungs_comparable`
    compares rungs on condition ORDER as well as membership, so a fused frame
    sorted into alphabetical condition order could not share a table with the
    rungs it is meant to be compared against.
    """
    keyed = zscore_by_condition(df, fit_mask).set_index(_KEYS)
    if keyed.index.has_duplicates:
        raise ValueError(
            f"frame {position} has duplicate (condition, image_idx) rows, so "
            "its scores cannot be matched to the other frames' one for one")
    base_index = pd.MultiIndex.from_frame(base[_KEYS])
    if len(keyed) != len(base_index) or not keyed.index.sort_values().equals(
            base_index.sort_values()):
        raise ValueError(
            f"all frames must cover the same rows (condition, image_idx): "
            f"frame 0 has {len(base_index)} rows and frame {position} has "
            f"{len(keyed)}, and their keys are not the same set")
    keyed = keyed.reindex(base_index)
    if "label" in keyed.columns and "label" in base.columns:
        mismatched = int((keyed["label"].to_numpy()
                          != base["label"].to_numpy()).sum())
        if mismatched:
            raise ValueError(
                f"frames 0 and {position} disagree on the label of "
                f"{mismatched} row(s). The fused frame is built from frame 0, "
                "so the disagreement would be resolved silently in favour of "
                "whichever bank was passed first and every metric downstream "
                "would be scored against one parent's labels.")
    return keyed["score"].to_numpy()


def fuse_scores(dfs: Sequence[pd.DataFrame],
                weights: Sequence[float] | None = None, *,
                splits: Sequence[str] | np.ndarray | None = None,
                fit_splits: Sequence[str] | str = _REQUIRED) -> pd.DataFrame:
    """Weighted mean of z-scored `score_grid` frames, on frame 0's rows.

    Rows are matched on `(condition, image_idx)`, not on position, and the
    output keeps frame 0's row order so the fused rung stays comparable with
    the rungs it is tabulated beside.

    `fit_splits` is REQUIRED, and declares which rows the per-condition mean
    and sigma are fitted on; they are applied to every row either way. Pass it
    with the bank's `splits` column, or pass `fit_splits=ALL_ROWS` for the
    whole-frame fit. There is no default, because on the ablation tier a
    whole-frame fit lets the organisers' benchmark subsample set half of each
    parent's spread and therefore half of how much each backbone votes -- see
    the module docstring -- and a default is what actually ships. Whichever is
    chosen is recorded on the returned frame, in the `zscore_population`
    column, because the fused score is not a fixed function of the two parents'
    scores without it.
    """
    if len(dfs) == 0:
        raise ValueError("nothing to fuse: fuse_scores was given no frames")
    split_values, fit, population = _resolve_population(splits, fit_splits)

    base = dfs[0].reset_index(drop=True)
    masks = [None if split_values is None else _fit_mask(d, split_values, fit, i)
             for i, d in enumerate(dfs)]
    stacked = np.stack([_aligned(d, base, i, m)
                        for i, (d, m) in enumerate(zip(dfs, masks))])

    w = np.asarray(weights if weights is not None else [1.0] * len(dfs),
                   dtype=float)
    if w.shape != (len(dfs),):
        raise ValueError(
            f"weights must match the number of frames: {len(dfs)} frame(s), "
            f"{w.size} weight(s)")
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"weights must be finite and sum to more than 0, got "
                         f"{list(w)}")
    w = w / total

    fused = base.copy()
    fused["score"] = (stacked * w[:, None]).sum(axis=0)
    fused[POPULATION_COLUMN] = population
    fused.attrs[POPULATION_COLUMN] = population
    return fused


# --- fitting the weight, on val_internal alone ------------------------------
#
# Equal weighting is `fuse_scores`' default, not its only option, and the
# default has a known failure mode: a weaker parent dilutes a stronger one in
# exact proportion to how much weight it is handed for free. Fitting the weight
# lets a parent that is better on SOME subset contribute there without paying
# for it everywhere -- and lets a parent that is simply worse fall to a small
# weight instead of half the vote.
#
# The whole difficulty is that fitting is a SELECTION, and a selection that has
# seen the held-out rows has spent them. `heldout_robust_tpr` is read on
# val_internal authentic against heldout_generator generated; a weight swept
# against that objective is a weight fitted on the rows being reported. So the
# sweep maximises `val_robust_tpr` below -- the same metric shape, both classes
# drawn from val_internal -- and the held-out number is read exactly once,
# after w is fixed. This discipline was written for `scripts/family_experts.py`
# and lives here so A5 and the family-expert probe cannot drift apart on it.

#: Weight on parent 0; parent 1 gets 1 - w. 21 points is a 0.05 grid, and finer
#: buys nothing when the objective is a mean of TPRs over a few thousand
#: validation images and therefore moves in steps of 1/n.
WEIGHT_GRID: np.ndarray = np.linspace(0.0, 1.0, 21)


def val_robust_tpr(scores_df: pd.DataFrame, splits,
                   target_fpr: float = SELECTION_TARGET_FPR) -> float:
    """Mean TPR @ `target_fpr` over the DEGRADED conditions, val_internal only.

    The same shape as `errors.heldout_robust_tpr` -- mean over the degraded
    grid, same operating point -- with both classes drawn from `val_internal`
    instead of authentic-from-val against generated-from-heldout. It exists so
    a fusion weight can be chosen without the held-out rows entering the choice.

    It is NOT a substitute for the selection metric and must never be reported
    as one: its positives come from families the heads trained on, so it
    measures fit, not generalisation. It is a knob-setter.
    """
    row_split = np.asarray(splits).astype(str)[scores_df["image_idx"].to_numpy()]
    sub = scores_df[row_split == "val_internal"]
    sub = sub[sub["condition"] != "clean"]
    if sub.empty:
        raise ValueError(
            "no val_internal rows in a degraded condition, so the weight "
            "objective is empty; the eval bank must carry val_internal rows "
            "over the degraded grid for a weight to be fitted off the "
            "held-out families at all")
    values = []
    for cond, g in sub.groupby("condition", sort=False):
        y = g["label"].to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"condition {cond!r} has only class {sorted(set(y.tolist()))} "
                "among val_internal rows, so its TPR@FPR is undefined. "
                "Averaging what survived would be a mean over an unstated "
                "subset of the grid -- the same refusal heldout_robust_tpr "
                "makes for the same reason.")
        values.append(tpr_at_fpr(y, g["score"].to_numpy(), target_fpr))
    return float(np.mean(values))


def fit_fusion_weight(dfs: Sequence[pd.DataFrame], splits, *,
                      fit_splits: Sequence[str] | str = _REQUIRED,
                      grid: np.ndarray = WEIGHT_GRID,
                      target_fpr: float = SELECTION_TARGET_FPR
                      ) -> tuple[tuple[float, float], list[dict]]:
    """Choose `(w, 1 - w)` for two parents by maximising `val_robust_tpr`.

    Returns the weights and the full sweep, so `selection.json` can record the
    objective at every grid point rather than only the argmax -- a flat sweep
    and a sharply peaked one justify very different confidence in the same w.

    `fit_splits` is passed through to `fuse_scores` unchanged and is still
    REQUIRED: the z-score population and the weight are two separate fits, and
    silently defaulting either is how A5's number becomes a function of the
    organisers' demo set (see this module's docstring).

    **Ties go to 0.5, not to the lowest w.** Equal weighting is the null this
    sweep exists to test; moving off it on a tie would report a fitted weight
    that bought nothing. `max` would otherwise return the first grid point,
    making w=0.0 -- "drop parent 0 entirely" -- the silent winner of a flat
    objective.
    """
    if len(dfs) != 2:
        raise ValueError(
            f"fit_fusion_weight sweeps a single scalar w over TWO parents, got "
            f"{len(dfs)}. An n-parent fit is a different problem with a "
            "different number of degrees of freedom to justify.")
    sweep = []
    for w in grid:
        fused = fuse_scores(dfs, weights=[float(w), 1.0 - float(w)],
                            splits=splits, fit_splits=fit_splits)
        sweep.append({"w0": float(w),
                      "val_robust_tpr": val_robust_tpr(fused, splits, target_fpr)})
    best = max(sweep, key=lambda r: (r["val_robust_tpr"], -abs(r["w0"] - 0.5)))
    return (best["w0"], 1.0 - best["w0"]), sweep


# --- which bank a fused frame belongs to -----------------------------------

def _config(bank, key):
    return getattr(bank, "config", {}).get(key)


def assert_fusion_parents(banks: Sequence) -> None:
    """Refuse banks whose fusion would have no defined row set.

    See the module docstring for what "the same rows" is taken to require and
    why the backbone is not part of it.
    """
    if len(banks) < 2:
        raise ValueError(
            f"a fusion needs at least two banks, got {len(banks)}; rung A5 is "
            "an ensemble of independently-trained banks")
    for i, bank in enumerate(banks):
        if _config(bank, "manifest_sha256") is None:
            raise ValueError(
                f"the fusion parent at {getattr(bank, 'path', '?')} records no "
                "manifest_sha256, so it cannot be shown to index the same "
                "frozen manifest as the bank it is fused with. Banks index the "
                "manifest positionally; two unfingerprinted banks agree only in "
                f"the sense that neither will say (parent {i}).")
    ref = banks[0]
    for other in banks[1:]:
        differing = {k: (_config(ref, k), _config(other, k))
                     for k in _PARENT_KEYS if _config(ref, k) != _config(other, k)}
        if "manifest_sha256" in differing:
            raise ValueError(
                f"the banks at {getattr(ref, 'path', '?')} and "
                f"{getattr(other, 'path', '?')} were built from different "
                f"frozen manifests ({differing['manifest_sha256']}), so a fused "
                "frame has no defined split column and its rows describe two "
                "different sets of images.")
        if differing:
            raise ValueError(
                f"the banks at {getattr(ref, 'path', '?')} and "
                f"{getattr(other, 'path', '?')} do not share a condition axis: "
                f"they disagree on {differing} (first, this one). View j must "
                "mean the same thing in both before their scores may be "
                "averaged.")

    ref_meta = ref.meta
    for other in banks[1:]:
        meta = other.meta
        if len(meta) != len(ref_meta) or not np.array_equal(
                meta["image_idx"].to_numpy(), ref_meta["image_idx"].to_numpy()):
            raise ValueError(
                f"the banks at {getattr(ref, 'path', '?')} and "
                f"{getattr(other, 'path', '?')} hold different rows "
                f"({len(ref_meta)} vs {len(meta)} images); a fused frame is "
                "matched row for row and there is no correspondence to use.")
        for column in ("split", "label"):
            a = ref_meta[column].to_numpy().astype(str)
            b = meta[column].to_numpy().astype(str)
            if not np.array_equal(a, b):
                raise ValueError(
                    f"the banks at {getattr(ref, 'path', '?')} and "
                    f"{getattr(other, 'path', '?')} disagree on the {column} of "
                    f"{int((a != b).sum())} row(s), despite recording the same "
                    "manifest fingerprint (which covers the path column, so a "
                    "re-split that kept the paths fingerprints identically). "
                    f"Which parent's {column} applies to the fused row is then "
                    "undefined, and scoring against either would mislabel the "
                    "rows the two disagree about.")


def fused_splits(banks: Sequence) -> np.ndarray:
    """The split column that applies to a frame fused from `banks`.

    `errors.heldout_robust_tpr` needs one split column per scored frame. This
    is the only circumstance under which a fused frame has one -- parents that
    agree on every count in `assert_fusion_parents` -- and asking for it any
    other way raises rather than silently adopting the first parent's.

    It is also the `splits` argument `fuse_scores` wants: the population the
    z-score parameters are fitted on is named in the same vocabulary as the
    population the selection metric is read on.
    """
    assert_fusion_parents(banks)
    return banks[0].meta["split"].to_numpy()


class FusedEvalBank:
    """The evaluation identity of a fused rung, as `robustness_table` reads it.

    Duck-types the part of `FeatureBank` that `eval.grid.assert_banks_comparable`
    and `eval.report._check_banks` use -- `path` and `config` -- so the A5 row
    registers the evaluation that actually produced it instead of borrowing one
    parent's bank. Borrowing would make the R24 comparability check pass on a
    row it never covered, which is the confound it exists to prevent.

    `config["backbone"]` is the composite of the parents' backbones, collapsing
    to the single name when they agree, and `fused_from`/`fused_backbones`
    declare what it is a composite OF -- which is what earns the row its
    exemption from the backbone key in `assert_banks_comparable` (R43).

    `config["n_images"]` is the row count the parents share. Without it,
    `report._check_banks` skips its bank-size check entirely, so a fused row
    would be the one row in the table that could pair a selection-tier bank
    with a final-report score frame unremarked (C3).
    """

    def __init__(self, parents: Sequence):
        assert_fusion_parents(parents)
        self.parents = tuple(parents)
        paths = [getattr(p, "path", "?") for p in self.parents]
        backbones = [str(_config(p, "backbone")) for p in self.parents]
        unique = list(dict.fromkeys(backbones))
        # `assert_fusion_parents` has already proved the parents' metas match
        # row for row, so this count describes all of them. It is taken from
        # the meta rather than a parent's `n_images` because the meta is what
        # `score_grid` actually scores; a parent whose config disagrees with
        # its own rows is a mis-declared bank, and passing that through as the
        # fused row's size would put the wrong number in front of the check
        # that exists to catch a tier mismatch.
        n_images = int(len(self.parents[0].meta))
        for path, parent in zip(paths, self.parents):
            declared = _config(parent, "n_images")
            if declared is not None and int(declared) != n_images:
                raise ValueError(
                    f"the fusion parent at {path} records n_images={int(declared)} "
                    f"but holds {n_images} metadata row(s). The fused row's size "
                    "is what `report._check_banks` compares against the scores, "
                    "and it cannot be stated from a bank that disagrees with "
                    "itself; re-extract or repair that bank.")
        self.path = "fused(" + ", ".join(paths) + ")"
        self.config = {
            "n_views": _config(self.parents[0], "n_views"),
            "conditions": list(_config(self.parents[0], "conditions") or []),
            "manifest_sha256": _config(self.parents[0], "manifest_sha256"),
            "backbone": "+".join(unique),
            "n_images": n_images,
            "fused_from": paths,
            "fused_backbones": backbones,
        }

    def __repr__(self) -> str:
        return f"FusedEvalBank({self.path!r})"
