#!/usr/bin/env python3
"""Amendment 7, Test 13, analysis: retained share Q of squish's damage rise.

    python scripts/amend5_test13_analysis.py

Reads Test 2's per-image parquets (center and native-damage squish logits)
and Test 13's (squish_post). Loads no model.

Statistics (registered), per branch, pooled over size (primary) and per
size bucket (reported):
  D_nat(c)  = AUC(squish, c)      - AUC(center, c)    native damage, Test 2
  D_post(c) = AUC(squish_post, c) - AUC(center, c)    damage on the n x n view
  Rise_nat  = mean over op types of D_nat(heavy)  - D(none)
  Rise_post = mean over op types of D_post(heavy) - D(none)
  Q         = Rise_post / Rise_nat                 (also per op type)
D(none) is one number: with no damage, squish_post is squish.

Conditions (decided on the 95% interval, both branches):
  NOT DECIDABLE  Rise_nat's interval includes zero
  CLEANING       Q's interval lies below 0.30
  COVERAGE       Q's interval lies above 0.60
  UNRESOLVED     otherwise

Bootstrap: the index scheme of paper_analyses.paired_diff (seed 20260827,
resampling within label; every row in a condition has the same damage
status), with every AUC computed on the same draw. The script asserts that
its D_nat(jpeg_heavy) interval equals paired_diff's.

Writes docs/ntire/amend5/test13_postsquish.json.
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

TEST2_SCORES = "/scratch/dion_r13/dion_extensions/outputs/test2/scores"
TEST13_SCORES = "/scratch/dion_r13/dion_extensions/outputs/test13/scores"
OUT = "docs/ntire/amend5/test13_postsquish.json"
BRANCHES = ("A2", "B2swa")
OPS = ("jpeg", "blur", "noise")
LEVELS = ("light", "medium", "heavy")
BUCKETS = {"352-512": (352, 512), "528-736": (528, 736), "752+": (752, 100000)}
N_BOOT = 2000
CLEANING_BELOW, COVERAGE_ABOVE = 0.30, 0.60


def load_branch(branch: str) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """One frame with every logit column, and the column name of each (arm, condition)."""
    post = pd.read_parquet(f"{TEST13_SCORES}/{branch}_ntire_clean_squish_post.parquet")
    assert not post["split"].eq("ntire_test").any(), "Test 13 never reads ntire_test"
    frame = post[["path", "label", "width", "height"]].copy()
    columns = {"center": {}, "squish": {}, "squish_post": {}}
    conditions = ["none"] + [f"{op}_{level}" for op in OPS for level in LEVELS]
    for condition in conditions:
        native = pd.read_parquet(f"{TEST2_SCORES}/{branch}_ntire_clean_{condition}.parquet")
        assert (native["path"].to_numpy() == frame["path"].to_numpy()).all(), condition
        for arm in ("center", "squish"):
            frame[f"{arm}_{condition}"] = native[f"logit_{arm}"].to_numpy()
            columns[arm][condition] = f"{arm}_{condition}"
        if condition == "none":
            columns["squish_post"][condition] = "squish_none"
        else:
            columns["squish_post"][condition] = f"squish_post_{condition}"
            frame[f"squish_post_{condition}"] = post[f"logit_squish_post_{condition}"].to_numpy()
    frame["short"] = frame[["width", "height"]].min(axis=1)
    return frame, columns


def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, nan when the denominator is 0 (a draw with no rise)."""
    return numerator / denominator if denominator != 0 else float("nan")


def statistics(frame: pd.DataFrame, columns: dict, index: np.ndarray | None = None) -> dict[str, float]:
    """Every registered statistic on one set of rows (the full rows, or one draw)."""
    rows = frame if index is None else frame.iloc[index]
    y = rows["label"].to_numpy().astype(int)

    def d(arm: str, condition: str) -> float:
        return (auc(y, rows[columns[arm][condition]].to_numpy())
                - auc(y, rows[columns["center"][condition]].to_numpy()))

    out = {"D_none": d("squish", "none")}
    for op in OPS:
        for level in LEVELS:
            out[f"D_nat_{op}_{level}"] = d("squish", f"{op}_{level}")
            out[f"D_post_{op}_{level}"] = d("squish_post", f"{op}_{level}")
        out[f"rise_nat_{op}"] = out[f"D_nat_{op}_heavy"] - out["D_none"]
        out[f"rise_post_{op}"] = out[f"D_post_{op}_heavy"] - out["D_none"]
        out[f"Q_{op}"] = ratio(out[f"rise_post_{op}"], out[f"rise_nat_{op}"])
    out["rise_nat"] = float(np.mean([out[f"rise_nat_{op}"] for op in OPS]))
    out["rise_post"] = float(np.mean([out[f"rise_post_{op}"] for op in OPS]))
    out["Q"] = ratio(out["rise_post"], out["rise_nat"])
    return out


def with_intervals(frame: pd.DataFrame, columns: dict) -> dict:
    point = statistics(frame, columns)
    strata = frame["label"].to_numpy().astype(int)
    draws = [statistics(frame, columns, k) for k in draw_indices(strata)]
    result = {"n": int(len(frame)), "n_fake": int(strata.sum())}
    for name, value in point.items():
        values = np.array([draw[name] for draw in draws])
        lo, hi = np.nanquantile(values, [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


def verdict(cell: dict) -> str:
    rise_lo, rise_hi = cell["rise_nat"]["ci"]
    q_lo, q_hi = cell["Q"]["ci"]
    if rise_lo <= 0 <= rise_hi:
        return "NOT DECIDABLE"
    if q_hi < CLEANING_BELOW:
        return "CLEANING"
    if q_lo > COVERAGE_ABOVE:
        return "COVERAGE"
    return "UNRESOLVED"


def main() -> int:
    report = {"test": 13, "commit": commit(), "n_boot": N_BOOT, "seed": SEED, "branches": {}}
    for branch in BRANCHES:
        frame, columns = load_branch(branch)

        # The index scheme must be paired_diff's: same interval for one difference.
        y = frame["label"].to_numpy().astype(int)
        reference = paired_diff(y, frame[columns["center"]["jpeg_heavy"]].to_numpy(),
                                frame[columns["squish"]["jpeg_heavy"]].to_numpy(), y)
        pooled = with_intervals(frame, columns)
        assert np.allclose(pooled["D_nat_jpeg_heavy"]["ci"], reference["ci"]), (pooled["D_nat_jpeg_heavy"], reference)

        buckets = {}
        for name, (lo, hi) in BUCKETS.items():
            in_bucket = frame[(frame["short"] >= lo) & (frame["short"] <= hi)].reset_index(drop=True)
            buckets[name] = with_intervals(in_bucket, columns)
            buckets[name]["verdict"] = verdict(buckets[name])
        report["branches"][branch] = {"pooled": pooled, "verdict": verdict(pooled), "buckets": buckets,
                                      "n_below_352": int((frame["short"] < 352).sum())}
        print(f"{branch}: Rise_nat {pooled['rise_nat']['point']:+.4f} {pooled['rise_nat']['ci']}, "
              f"Q {pooled['Q']['point']:.2f} {pooled['Q']['ci']} -> {report['branches'][branch]['verdict']}",
              flush=True)

    verdicts = {report["branches"][b]["verdict"] for b in BRANCHES}
    report["verdict"] = verdicts.pop() if len(verdicts) == 1 else "BRANCHES DISAGREE"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 13: {report['verdict']} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
