#!/usr/bin/env python3
"""Amendment 7, Test 15, analysis: native 512 window vs 200 px window at input 512.

    python scripts/amend5_test15_analysis.py

Registered condition (paper_prereg.md, amendment 7, Test 15; single seed):

  Enlargement penalty present: robust AUC (native512 - crop200@512) on
  ntire_val_hard has an interval above zero. Its size is reported as a share
  of A2's squish - center robust gap, as a description.
  Absent: the interval includes zero or lies below it.

Robust AUC = AUC on the distorted rows. The interval is paired_diff's
(seed 20260827, 2,000 draws), run on the distorted rows with label strata,
which is label x is_distorted restricted to is_distorted = 1. Also reported:
clean AUC, ntire_val, clamp counts, and robust AUC per size bucket.

Writes docs/ntire/amend5/test15_native512-proxy.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

SCORES = "/scratch/dion_r13/dion_extensions/outputs/test15"
OUT = "docs/ntire/amend5/test15_native512-proxy.json"
ARMS = {"crop200": 200, "native512": 512}  # arm -> crop side
SPLITS = ("ntire_val_hard", "ntire_val")
BUCKETS = ((80, 336), (352, 512), (528, 736), (752, 4016))  # ntire_crop_probe.BUCKETS


def load(split: str) -> pd.DataFrame:
    """Both arms' logits on one split, joined on path, with the native size."""
    frames = [pd.read_parquet(f"{SCORES}/abl512_{arm}_s1_{split}.parquet")
              .set_index("path")[["label", "is_distorted", "logit"]]
              .rename(columns={"logit": f"logit_{arm}"}) for arm in ARMS]
    joined = frames[0].join(frames[1].drop(columns=["label", "is_distorted"]), how="inner")
    assert len(joined) == len(frames[0]) == len(frames[1]), "the two arms scored different rows"
    # The native size comes from the probe parquets, which carry width and height.
    sizes = pd.read_parquet(f"outputs/scores/probe_A2_center_{split}.parquet"
                            if split == "ntire_val_hard" else f"outputs/scores/A2_{split}.parquet")
    if "width" in sizes:
        joined = joined.join(sizes.set_index("path")[["width", "height"]])
    return joined.reset_index()


def compare(rows: pd.DataFrame) -> dict:
    """native512 - crop200 AUC on these rows, paired_diff interval, label strata."""
    y = rows["label"].to_numpy().astype(int)
    r = paired_diff(y, rows["logit_crop200"].to_numpy(), rows["logit_native512"].to_numpy(), y)
    return {"n": len(rows), "crop200": r["a"], "native512": r["b"], "diff": r["diff"], "ci": r["ci"]}


def a2_gap() -> dict:
    """A2's squish - center robust AUC on ntire_val_hard, the registered yardstick."""
    center = pd.read_parquet("outputs/scores/probe_A2_center_ntire_val_hard.parquet").set_index("path")
    squish = pd.read_parquet("outputs/scores/probe_A2_squish_ntire_val_hard.parquet").set_index("path")
    rows = center[["label", "is_distorted", "logit"]].join(squish[["logit"]], rsuffix="_squish")
    rows = rows[rows["is_distorted"] == 1]
    y = rows["label"].to_numpy().astype(int)
    r = paired_diff(y, rows["logit"].to_numpy(), rows["logit_squish"].to_numpy(), y)
    return {"center": r["a"], "squish": r["b"], "gap": r["diff"], "ci": r["ci"]}


def main() -> int:
    report = {"test": 15, "commit": commit(), "seed": SEED, "n_boot": 2000, "splits": {}}
    for split in SPLITS:
        assert split != "ntire_test", "Test 15 never reads ntire_test"
        rows = load(split)
        short = rows[["width", "height"]].min(axis=1) if "width" in rows else None
        entry = {"robust": compare(rows[rows["is_distorted"] == 1]),
                 "clean": compare(rows[rows["is_distorted"] == 0])}
        if short is not None:
            entry["clamped"] = {arm: int((short < side).sum()) for arm, side in ARMS.items()}
            entry["robust_by_bucket"] = {
                f"{lo}-{hi}": compare(rows[(rows["is_distorted"] == 1) & short.between(lo, hi)])
                for lo, hi in BUCKETS if ((rows["is_distorted"] == 1) & short.between(lo, hi)).sum() > 0}
        report["splits"][split] = entry
        r = entry["robust"]
        print(f"{split}: robust crop200 {r['crop200']:.4f} native512 {r['native512']:.4f} "
              f"diff {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]", flush=True)

    primary = report["splits"]["ntire_val_hard"]["robust"]
    report["verdict"] = "PENALTY PRESENT" if primary["ci"][0] > 0 else "ABSENT"
    gap = a2_gap()
    report["a2_squish_minus_center"] = gap
    report["share_of_a2_gap"] = primary["diff"] / gap["gap"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"A2 squish - center robust gap {gap['gap']:+.4f}; share {report['share_of_a2_gap']:.2f}")
    print(f"Test 15: {report['verdict']} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
