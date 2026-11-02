#!/usr/bin/env python3
"""Amendment 7, re-analysis R3 of Test 9: squish - combiner per size bucket.

    python scripts/amend5_test9_supp_a7.py

A supplement. It changes no verdict, and it does not touch Test 9's json or
md. Nothing is refitted: it reads the per-image rows Test 9 saved
(`outputs/test9/test9_rows_{A,B}.parquet`: the combiner logit, the frozen
squish logit and the grid9 mean logit of every `ntire_val_hard` image).

Statistic, per branch and per size bucket of the image's short side
(80-336, 352-512, 528-736, 752+ px, as in Test 2):

    robust AUC(squish) - robust AUC(combiner)

the quantity Test 9's verdict reads, pooled over size. Robust AUC is the AUC
on the distorted rows. The clean-row difference and grid9 - combiner are
reported alongside, from the same draws.

The size is min(width, height) of the image as stored in `ntire_val_hard`,
which for a distorted row is its size after its recorded distortions (the
size the model was shown).

Bootstrap: the index scheme of paper_analyses.paired_diff (seed 20260827,
2,000 draws, strata label x is_distorted, cells in np.unique order), which
is also the scheme of paired_bootstrap.paired_bootstrap that Test 9 used.
Each bucket gets its own draws on its own rows. The script asserts that
(i) its pooled all-row squish - combiner interval equals a fresh paired_diff
call, and (ii) its pooled robust squish - combiner equals Test 9's reported
number and interval.

Writes docs/ntire/amend5/test9_supp_a7.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

ROWS_DIR = "/scratch/dion_r13/dion_extensions/outputs/test9"
TEST9_JSON = "docs/ntire/amend5/test9_mil-combiner.json"
OUT = "docs/ntire/amend5/test9_supp_a7.json"
BRANCHES = ("A", "B")
ARMS = ("squish", "combiner", "grid9")
BUCKETS = {"80-336": (80, 336), "352-512": (352, 512), "528-736": (528, 736), "752+": (752, 100000)}
N_BOOT = 2000


def load_rows(branch: str) -> pd.DataFrame:
    """Test 9's saved per-image rows on ntire_val_hard, with the short side."""
    rows = pd.read_parquet(f"{ROWS_DIR}/test9_rows_{branch}.parquet")
    assert not rows["path"].str.contains("ntire_test|test_images").any(), "R3 never reads ntire_test"
    assert rows["path"].str.contains("val_images_hard").all(), "Test 9 evaluates ntire_val_hard only"
    rows["short"] = rows[["width", "height"]].min(axis=1)
    return rows


def strata_of(rows: pd.DataFrame) -> np.ndarray:
    """label x is_distorted, coded as paired_bootstrap codes it (label * 2 + is_distorted)."""
    return rows["label"].to_numpy().astype(int) * 2 + rows["is_distorted"].to_numpy().astype(int)


def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def statistics(rows: pd.DataFrame, index: np.ndarray | None = None) -> dict[str, float]:
    """AUC of each arm on all / distorted / clean rows, and the paired differences."""
    drawn = rows if index is None else rows.iloc[index]
    y = drawn["label"].to_numpy().astype(int)
    distorted = drawn["is_distorted"].to_numpy().astype(int) == 1
    out = {}
    for arm in ARMS:
        scores = drawn[arm].to_numpy().astype(np.float64)
        out[f"auc_{arm}"] = auc(y, scores)
        out[f"robust_auc_{arm}"] = auc(y[distorted], scores[distorted])
        out[f"clean_auc_{arm}"] = auc(y[~distorted], scores[~distorted])
    for prefix in ("auc", "robust_auc", "clean_auc"):
        out[f"{prefix}_squish_minus_combiner"] = out[f"{prefix}_squish"] - out[f"{prefix}_combiner"]
        out[f"{prefix}_grid9_minus_combiner"] = out[f"{prefix}_grid9"] - out[f"{prefix}_combiner"]
    return out


def with_intervals(rows: pd.DataFrame) -> dict:
    """Point estimate and 95% percentile interval of every statistic, from one draw set."""
    point = statistics(rows)
    draws = [statistics(rows, k) for k in draw_indices(strata_of(rows))]
    distorted = rows["is_distorted"] == 1
    result = {"n": int(len(rows)), "n_robust": int(distorted.sum()),
              "n_robust_fake": int(rows.loc[distorted, "label"].sum()),
              "n_clean": int((~distorted).sum()), "n_clean_fake": int(rows.loc[~distorted, "label"].sum())}
    for name, value in point.items():
        values = np.array([draw[name] for draw in draws])
        lo, hi = np.nanquantile(values, [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


def reading(cell: dict) -> str:
    """Test 9's rule applied to one bucket, as a description (not a verdict)."""
    lo, hi = cell["robust_auc_squish_minus_combiner"]["ci"]
    if lo > 0:
        return "squish above combiner"
    if hi < 0:
        return "combiner above squish"
    return "interval includes zero"


def main() -> int:
    with open(TEST9_JSON) as handle:
        test9 = json.load(handle)
    report = {"supplement": "amendment 7, R3 (Test 9)", "changes_verdict": False, "commit": commit(),
              "seed": SEED, "n_boot": N_BOOT, "rows_dir": ROWS_DIR, "refitted": False, "branches": {}}

    for branch in BRANCHES:
        rows = load_rows(branch)
        pooled = with_intervals(rows)

        # (i) Same index scheme as paired_diff: same all-row interval.
        y = rows["label"].to_numpy().astype(int)
        reference = paired_diff(y, rows["combiner"].to_numpy(), rows["squish"].to_numpy(), strata_of(rows))
        assert np.allclose(pooled["auc_squish_minus_combiner"]["ci"], reference["ci"]), (pooled, reference)
        # (ii) Same robust number as Test 9 reported.
        reported = test9["branches"][branch]["robust_squish_minus_combiner"]
        assert np.isclose(pooled["robust_auc_squish_minus_combiner"]["point"], reported["diff"]), reported
        assert np.allclose(pooled["robust_auc_squish_minus_combiner"]["ci"], reported["ci"]), reported
        pooled["reading"] = reading(pooled)

        buckets = {}
        in_any_bucket = np.zeros(len(rows), dtype=bool)
        for name, (lo, hi) in BUCKETS.items():
            mask = ((rows["short"] >= lo) & (rows["short"] <= hi)).to_numpy()
            in_any_bucket |= mask
            cell = with_intervals(rows[mask].reset_index(drop=True))
            cell["reading"] = reading(cell)
            buckets[name] = cell
            d = cell["robust_auc_squish_minus_combiner"]
            print(f"{branch} {name}: n_robust {cell['n_robust']}, squish - combiner {d['point']:+.4f} "
                  f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]", flush=True)
        report["branches"][branch] = {"pooled": pooled, "buckets": buckets,
                                      "n_outside_buckets": int((~in_any_bucket).sum())}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"R3 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
