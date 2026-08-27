"""Splits, frozen before any training (spec §4.6).

The held-out-transform-family split (A3-LOTO) is NOT here: it is a property of
the training recipe sampler, not of the image set, and Plan 2 configures it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEED = 20260827
MIN_HELDOUT_IMAGES = 200


def choose_heldout_generators(df: pd.DataFrame, n: int = 2, seed: int = DEFAULT_SEED) -> list[str]:
    """Pick n generator families to exclude from training entirely.

    Restricted to families with at least MIN_HELDOUT_IMAGES images so the
    held-out evaluation has enough support for a usable confidence interval.
    """
    counts = df[df["label"] == 1]["generator"].value_counts()
    eligible = sorted(counts[counts >= MIN_HELDOUT_IMAGES].index.tolist())
    if len(eligible) < n:
        raise ValueError(
            f"need {n} generators with >={MIN_HELDOUT_IMAGES} images, have {len(eligible)}")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(eligible), size=n, replace=False).tolist())


def assign_splits(
    df: pd.DataFrame,
    heldout_generators: list[str],
    val_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    present = set(df["generator"].unique())
    missing = [g for g in heldout_generators if g not in present]
    if missing:
        raise ValueError(f"held-out generators not present in manifest: {missing}")

    out = df.copy()
    out["split"] = ""
    held = out["generator"].isin(heldout_generators)
    out.loc[held, "split"] = "heldout_generator"

    rest = ~held
    rng = np.random.default_rng(seed)
    draws = rng.random(int(rest.sum()))
    out.loc[rest, "split"] = np.where(draws < val_fraction, "val_internal", "train")
    return out


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["split", "label"], as_index=False)
              .size().rename(columns={"size": "n"}))
