#!/usr/bin/env python3
"""Amendment 9, Test 17, analysis: split Test 14/16's shrink effect dK in two.

    python scripts/amend5_test17_analysis.py

Reads the parquets of amend5_test17.py and the tiles(1, f), tiles(2, f)
logits of Tests 14 and 16 (joined on path; the rows must match exactly).
Loads no model.

Statistics (registered), robust AUC = AUC on distorted rows, f in {0.25, 0.5, 1}:
  dA = mean over f of [AUC(rt(2, f)) - AUC(tiles(1, f))]   same scene per window, averaged pixels
  dS = mean over f of [AUC(tiles(2, f)) - AUC(rt(2, f))]   same averaged pixels, 4x the scene per window
  dA + dS = dK, Test 14/16's shrink effect (asserted).
Conditions (95% intervals, per branch, per row set):
  ATTENUATION  dA above zero, dS not above zero
  CONTENT      dS above zero, dA not above zero
  BOTH         dA and dS above zero
  NEITHER      otherwise
Clean-row dA is reported: with no added damage, it measures what the round
trip itself costs (lost detail, the enlargement's softness).

Bootstrap: amend5_test14_analysis.draw_indices (paired_diff's index scheme,
seed 20260827, strata label x is_distorted), every statistic on the same draw.

Writes docs/ntire/amend5/test17_roundtrip.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test14 import cell_name as tiles_name  # noqa: E402
from amend5_test14_analysis import N_BOOT, above, draw_indices  # noqa: E402
from amend5_test17 import cell_name as rt_name  # noqa: E402
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

ROOT = "/scratch/dion_r13/dion_extensions/outputs"
EARLIER = {"ntire": f"{ROOT}/test14/scores", "socialrf": f"{ROOT}/test16/scores"}
SCORES = f"{ROOT}/test17/scores"
OUT = "docs/ntire/amend5/test17_roundtrip.json"
BRANCHES = ("A2", "B2swa")
FS = (0.25, 0.5, 1.0)
ARMS = [tiles_name(1, f) for f in FS] + [tiles_name(2, f) for f in FS] + [rt_name(2, f) for f in FS]


def load(branch: str, row_set: str) -> pd.DataFrame:
    """Test 17's logits beside the earlier test's tiles(1, f) and tiles(2, f), same rows."""
    earlier = pd.read_parquet(f"{EARLIER[row_set]}/{branch}_tiles.parquet")
    new = pd.read_parquet(f"{SCORES}/{branch}_{row_set}_roundtrip.parquet")
    assert earlier["path"].tolist() == new["path"].tolist(), f"{branch} {row_set}: rows differ"
    assert not new["split"].eq("ntire_test").any(), "Test 17 never reads ntire_test"
    # Same scene per window: rt(2, f) must have exactly tiles(1, f)'s window count.
    for f in FS:
        assert (new[f"n_windows_{rt_name(2, f)}"] == earlier[f"n_windows_{tiles_name(1, f)}"]).all()
    frame = earlier.copy()
    for f in FS:
        frame[f"logit_{rt_name(2, f)}"] = new[f"logit_{rt_name(2, f)}"].to_numpy()
    return frame


def statistics(y: np.ndarray, dist: np.ndarray, logits: dict[str, np.ndarray]) -> dict[str, float]:
    """Every registered statistic on one set of rows."""
    out = {}
    for kind, rows in (("robust", dist == 1), ("clean", dist == 0)):
        a = {arm: auc(y[rows], logits[arm][rows]) for arm in ARMS}
        out.update({f"{kind}_{arm}": value for arm, value in a.items()})
        out[f"{kind}_dA"] = float(np.mean([a[rt_name(2, f)] - a[tiles_name(1, f)] for f in FS]))
        out[f"{kind}_dS"] = float(np.mean([a[tiles_name(2, f)] - a[rt_name(2, f)] for f in FS]))
        out[f"{kind}_dK"] = float(np.mean([a[tiles_name(2, f)] - a[tiles_name(1, f)] for f in FS]))
        assert np.isclose(out[f"{kind}_dA"] + out[f"{kind}_dS"], out[f"{kind}_dK"])
    return out


def with_intervals(frame: pd.DataFrame) -> dict:
    y = frame["label"].to_numpy().astype(int)
    dist = frame["is_distorted"].to_numpy().astype(int)
    logits = {arm: frame[f"logit_{arm}"].to_numpy().astype(np.float64) for arm in ARMS}
    point = statistics(y, dist, logits)
    draws = [statistics(y[k], dist[k], {arm: z[k] for arm, z in logits.items()})
             for k in draw_indices(y * 2 + dist)]
    result = {"n": int(len(frame)), "n_distorted": int(dist.sum()), "n_fake": int(y.sum())}
    for name, value in point.items():
        lo, hi = np.nanquantile(np.array([draw[name] for draw in draws]), [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


def verdict(stats: dict) -> str:
    attenuation, content = above(stats["robust_dA"]), above(stats["robust_dS"])
    if attenuation and content:
        return "BOTH"
    if attenuation:
        return "ATTENUATION"
    if content:
        return "CONTENT"
    return "NEITHER"


def main() -> int:
    report = {"test": 17, "commit": commit(), "n_boot": N_BOOT, "seed": SEED, "row_sets": {}}
    for row_set in EARLIER:
        report["row_sets"][row_set] = {}
        for branch in BRANCHES:
            frame = load(branch, row_set)
            stats = with_intervals(frame)
            entry = {"pooled": stats, "verdict": verdict(stats)}
            if row_set == "ntire":
                entry["splits"] = {}
                for split in ("ntire_val", "ntire_val_hard"):
                    split_stats = with_intervals(frame[frame["split"] == split].reset_index(drop=True))
                    entry["splits"][split] = dict(split_stats, verdict=verdict(split_stats))
            report["row_sets"][row_set][branch] = entry
            print(f"{row_set} {branch}: dA {stats['robust_dA']['point']:+.4f} {stats['robust_dA']['ci']}, "
                  f"dS {stats['robust_dS']['point']:+.4f} {stats['robust_dS']['ci']}, "
                  f"clean dA {stats['clean_dA']['point']:+.4f} {stats['clean_dA']['ci']} "
                  f"-> {entry['verdict']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 17 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
