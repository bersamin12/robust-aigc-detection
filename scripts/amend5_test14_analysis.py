#!/usr/bin/env python3
"""Amendment 7, Test 14, analysis: coverage (f) vs shrink (k) main effects.

    python scripts/amend5_test14_analysis.py

Reads the parquets of amend5_test14.py. Loads no model.

Statistics (registered), robust AUC = AUC on distorted rows:
  dF = mean over k in {1, 2} of [AUC(k, 1) - AUC(k, 0.25)]
  dK = mean over f in {0.25, 0.5, 1} of [AUC(2, f) - AUC(1, f)]
  C  = dF - dK
Conditions (95% intervals, both branches, primary rows = both splits pooled):
  COVERAGE  dF above zero and C above zero
  CLEANING  dK above zero and C below zero
  BOTH      dF and dK above zero, C includes zero
  NEITHER   otherwise (the three are reported)
Secondary, reported, not a condition:
  R_g = (AUC(gridall) - AUC(center)) / (AUC(squish) - AUC(center)),
  gridall = tiles(1, 1). Only an interval above 0.60 is read (as support
  for coverage without resampling); a low R_g is not read as cleaning.
Every AUC is also reported on clean rows, and each split alone.

Bootstrap: paper_analyses.paired_diff's index scheme (seed 20260827),
strata label x is_distorted, every statistic on the same draw. The script
asserts that its all-row squish - center interval equals paired_diff's.

Writes docs/ntire/amend5/test14_tile-factorial.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test14 import CELLS, cell_name  # noqa: E402
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

SCORES = "/scratch/dion_r13/dion_extensions/outputs/test14/scores"
OUT = "docs/ntire/amend5/test14_tile-factorial.json"
BRANCHES = ("A2", "B2swa")
ARMS = ["center", "squish"] + [cell_name(k, f) for k, f in CELLS]
N_BOOT = 2000
GRIDALL_SUPPORT_ABOVE = 0.60


def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def statistics(y: np.ndarray, dist: np.ndarray, logits: dict[str, np.ndarray]) -> dict[str, float]:
    """Every registered statistic on one set of rows."""
    out = {}
    for arm in ARMS:
        out[f"robust_{arm}"] = auc(y[dist == 1], logits[arm][dist == 1])
        out[f"clean_{arm}"] = auc(y[dist == 0], logits[arm][dist == 0])
    for kind in ("robust", "clean"):
        a = {(k, f): out[f"{kind}_{cell_name(k, f)}"] for k, f in CELLS}
        out[f"{kind}_dF"] = float(np.mean([a[(k, 1.0)] - a[(k, 0.25)] for k in (1, 2)]))
        out[f"{kind}_dK"] = float(np.mean([a[(2, f)] - a[(1, f)] for f in (0.25, 0.5, 1.0)]))
        out[f"{kind}_C"] = out[f"{kind}_dF"] - out[f"{kind}_dK"]
        out[f"{kind}_dK_full_cover_4v1"] = a[(4, 1.0)] - a[(1, 1.0)]
        squish_gap = out[f"{kind}_squish"] - out[f"{kind}_center"]
        out[f"{kind}_R_g"] = (a[(1, 1.0)] - out[f"{kind}_center"]) / squish_gap
    return out


def with_intervals(frame: pd.DataFrame) -> dict:
    y = frame["label"].to_numpy().astype(int)
    dist = frame["is_distorted"].to_numpy().astype(int)
    logits = {arm: frame[f"logit_{arm}"].to_numpy().astype(np.float64) for arm in ARMS}
    point = statistics(y, dist, logits)
    strata = y * 2 + dist
    draws = [statistics(y[k], dist[k], {arm: z[k] for arm, z in logits.items()}) for k in draw_indices(strata)]
    result = {"n": int(len(frame)), "n_distorted": int(dist.sum()), "n_fake": int(y.sum()),
              "mean_windows": {arm: float(frame[f"n_windows_{arm}"].mean()) for arm in ARMS}}
    for name, value in point.items():
        lo, hi = np.nanquantile(np.array([draw[name] for draw in draws]), [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


def above(cell: dict) -> bool:
    return cell["ci"][0] > 0


def below(cell: dict) -> bool:
    return cell["ci"][1] < 0


def verdict(stats: dict) -> str:
    dF, dK, C = stats["robust_dF"], stats["robust_dK"], stats["robust_C"]
    if above(dF) and above(C):
        return "COVERAGE"
    if above(dK) and below(C):
        return "CLEANING"
    if above(dF) and above(dK):
        return "BOTH"
    return "NEITHER"


def main() -> int:
    report = {"test": 14, "commit": commit(), "n_boot": N_BOOT, "seed": SEED, "branches": {}}
    for branch in BRANCHES:
        frame = pd.read_parquet(f"{SCORES}/{branch}_tiles.parquet")
        assert not frame["split"].eq("ntire_test").any(), "Test 14 never reads ntire_test"

        # The index scheme must be paired_diff's: same interval for one difference.
        y = frame["label"].to_numpy().astype(int)
        strata = y * 2 + frame["is_distorted"].to_numpy().astype(int)
        reference = paired_diff(y, frame["logit_center"].to_numpy(), frame["logit_squish"].to_numpy(), strata)
        draws = []
        for k in draw_indices(strata):
            draws.append(auc(y[k], frame["logit_squish"].to_numpy()[k]) - auc(y[k], frame["logit_center"].to_numpy()[k]))
        assert np.allclose(np.nanquantile(draws, [0.025, 0.975]), reference["ci"])

        pooled = with_intervals(frame)
        per_split = {split: with_intervals(frame[frame["split"] == split].reset_index(drop=True))
                     for split in ("ntire_val", "ntire_val_hard")}
        r_g = pooled["robust_R_g"]
        report["branches"][branch] = {
            "pooled": pooled, "verdict": verdict(pooled),
            "gridall_supports_coverage": bool(r_g["ci"][0] > GRIDALL_SUPPORT_ABOVE),
            "splits": {s: dict(stats, verdict=verdict(stats)) for s, stats in per_split.items()},
        }
        print(f"{branch}: dF {pooled['robust_dF']['point']:+.4f} {pooled['robust_dF']['ci']}, "
              f"dK {pooled['robust_dK']['point']:+.4f} {pooled['robust_dK']['ci']}, "
              f"C {pooled['robust_C']['point']:+.4f} {pooled['robust_C']['ci']}, "
              f"R_g {r_g['point']:.2f} {r_g['ci']} -> {verdict(pooled)}", flush=True)

    verdicts = {report["branches"][b]["verdict"] for b in BRANCHES}
    report["verdict"] = verdicts.pop() if len(verdicts) == 1 else "BRANCHES DISAGREE"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 14: {report['verdict']} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
