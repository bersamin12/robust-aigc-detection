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

The backbone is deliberately NOT part of that check. A5 is defined as "+ second
backbone" (spec §6.4), so a differing backbone is the treatment under test.
Whether the resulting row may sit in the same robustness table as a
single-backbone rung is a separate question, and `FusedEvalBank` answers it
honestly rather than quietly: its `backbone` is the composite of its parents',
so `eval.grid.assert_banks_comparable` sees a two-backbone row for what it is.
Under the current `_COMPARABLE_KEYS` that means a cross-backbone A5 row is
REFUSED a place in a single-backbone table -- a real limitation, recorded here
because the alternative (borrowing one parent's name) is the R24 confound
laundered through a label.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: The columns that identify a row of a `score_grid` frame.
_KEYS = ["condition", "image_idx"]

#: Bank config keys a fused rung's parents must agree on. `backbone` is absent
#: on purpose: see the module docstring.
_PARENT_KEYS = ("n_views", "conditions", "manifest_sha256")


def zscore_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise `score` within each condition, leaving every other column.

    Population standard deviation (`ddof=0`), so a condition's scores come out
    with exactly unit variance rather than `sqrt((n-1)/n)` of it. A condition
    whose scores are constant has nothing to standardise and is centred on zero
    rather than divided by it.
    """
    if "score" not in df.columns or "condition" not in df.columns:
        raise ValueError(
            "zscore_by_condition needs 'condition' and 'score' columns; it "
            f"was given {sorted(df.columns)}")
    out = df.copy()
    g = out.groupby("condition", observed=True)["score"]
    std = g.transform("std", ddof=0).replace(0.0, 1.0).fillna(1.0)
    out["score"] = (out["score"] - g.transform("mean")) / std
    return out


def _aligned(df: pd.DataFrame, base: pd.DataFrame, position: int) -> np.ndarray:
    """One frame's z-scored column, reordered onto `base`'s rows.

    Reordering rather than sorting matters: `report._check_rungs_comparable`
    compares rungs on condition ORDER as well as membership, so a fused frame
    sorted into alphabetical condition order could not share a table with the
    rungs it is meant to be compared against.
    """
    keyed = zscore_by_condition(df).set_index(_KEYS)
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
                weights: Sequence[float] | None = None) -> pd.DataFrame:
    """Weighted mean of z-scored `score_grid` frames, on frame 0's rows.

    Rows are matched on `(condition, image_idx)`, not on position, and the
    output keeps frame 0's row order so the fused rung stays comparable with
    the rungs it is tabulated beside.
    """
    if len(dfs) == 0:
        raise ValueError("nothing to fuse: fuse_scores was given no frames")
    base = dfs[0].reset_index(drop=True)
    stacked = np.stack([_aligned(d, base, i) for i, d in enumerate(dfs)])

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
    return fused


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
    to the single name when they agree.
    """

    def __init__(self, parents: Sequence):
        assert_fusion_parents(parents)
        self.parents = tuple(parents)
        paths = [getattr(p, "path", "?") for p in self.parents]
        backbones = [str(_config(p, "backbone")) for p in self.parents]
        unique = list(dict.fromkeys(backbones))
        self.path = "fused(" + ", ".join(paths) + ")"
        self.config = {
            "n_views": _config(self.parents[0], "n_views"),
            "conditions": list(_config(self.parents[0], "conditions") or []),
            "manifest_sha256": _config(self.parents[0], "manifest_sha256"),
            "backbone": "+".join(unique),
            "fused_from": paths,
            "fused_backbones": backbones,
        }

    def __repr__(self) -> str:
        return f"FusedEvalBank({self.path!r})"
