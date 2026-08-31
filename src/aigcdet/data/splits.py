"""Splits, frozen before any training (spec §4.6).

The held-out-transform-family split (A3-LOTO) is NOT here: it is a property of
the training recipe sampler, not of the image set, and Plan 2 configures it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aigcdet.data.sources import is_heldout_eligible

DEFAULT_SEED = 20260827
MIN_HELDOUT_IMAGES = 200


def choose_heldout_generators(df: pd.DataFrame, n: int = 2, seed: int = DEFAULT_SEED) -> list[str]:
    """Pick n generator families to exclude from training entirely.

    Restricted to families that are

    1. genuine generator families -- a dataset-level pseudo-generator (see
       `aigcdet.data.sources.PSEUDO_GENERATORS`) names a *source*, and
       holding it out would measure dataset shift rather than the
       unseen-generator generalisation spec §4.6 defines; and
    2. large enough (>= MIN_HELDOUT_IMAGES) for the held-out evaluation to
       have usable confidence intervals.

    A human who wants a specific pair anyway passes them to
    `build_dataset(heldout_generators=...)` rather than reseeding this.
    """
    counts = df[df["label"] == 1]["generator"].value_counts()
    big_enough = sorted(counts[counts >= MIN_HELDOUT_IMAGES].index.tolist())
    eligible = [g for g in big_enough if is_heldout_eligible(g)]
    if len(eligible) < n:
        ineligible = [g for g in big_enough if not is_heldout_eligible(g)]
        raise ValueError(
            f"need {n} generators with >={MIN_HELDOUT_IMAGES} images, have "
            f"{len(eligible)}"
            + (f" (excluding dataset-level pseudo-generators {ineligible}, "
               "which name a source rather than a generator family)"
               if ineligible else ""))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(eligible), size=n, replace=False).tolist())


def assign_splits(
    df: pd.DataFrame,
    heldout_generators: list[str],
    val_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
    group_key: pd.Series | None = None,
) -> pd.DataFrame:
    """Assign train / val_internal / heldout_generator.

    `group_key` exists for paired corpora such as AI-OV7, where a fake is
    generated FROM a specific real and both rows carry the same ImageID. Drawn
    per row -- the only behaviour before this argument -- a real and its own
    fake land on opposite sides of the val boundary about 18% of the time, and
    the model then trains on a scene it is validated against under the other
    label. The validation number that comes back is not measuring what it says
    it measures.

    With a group key the draw is taken once per group and applied to the whole
    group, and held-out membership PROPAGATES: if any row of a group is held
    out, all of it is. Otherwise the real paired with a held-out fake stays in
    training, and the held-out rung -- the one that is supposed to measure
    generalisation to an unseen decoder -- evaluates on scenes the model has
    already memorised under the other label.

    Passing `group_key=None` keeps the original per-row draw and the original
    RNG stream, so manifests frozen before this argument existed rebuild
    byte-identically. Every feature bank on disk fingerprints its manifest, so
    that is a hard requirement, not a courtesy.
    """
    present = set(df["generator"].unique())
    missing = [g for g in heldout_generators if g not in present]
    if missing:
        raise ValueError(f"held-out generators not present in manifest: {missing}")

    out = df.copy()
    out["split"] = ""
    held = out["generator"].isin(heldout_generators)

    if group_key is None:
        out.loc[held, "split"] = "heldout_generator"
        rest = ~held
        rng = np.random.default_rng(seed)
        draws = rng.random(int(rest.sum()))
        out.loc[rest, "split"] = np.where(draws < val_fraction, "val_internal", "train")
        return out

    groups = pd.Series(group_key, index=out.index).astype(str)
    if groups.isna().any() or (groups == "").any():
        raise ValueError("group_key has blank entries; a row with no group "
                         "cannot be kept with its pair")
    held_groups = set(groups[held])
    out.loc[groups.isin(held_groups), "split"] = "heldout_generator"

    rest_groups = sorted(set(groups) - held_groups)
    rng = np.random.default_rng(seed)
    draws = rng.random(len(rest_groups))
    verdict = {g: ("val_internal" if d < val_fraction else "train")
               for g, d in zip(rest_groups, draws)}
    rest = out["split"] == ""
    out.loc[rest, "split"] = groups[rest].map(verdict)
    return out


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["split", "label"], as_index=False)
              .size().rename(columns={"size": "n"}))
