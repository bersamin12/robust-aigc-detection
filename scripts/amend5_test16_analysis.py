#!/usr/bin/env python3
"""Amendment 8, Test 16, analysis: Test 14's statistics and verdict on SocialRF.

    python scripts/amend5_test16_analysis.py

The statistics (dF, dK, C, R_g), the bootstrap and the verdict rule are
amend5_test14_analysis's functions, imported unchanged; only the input and
output paths differ. One pooled set of rows: SocialRF clean + `_deg`.

Writes docs/ntire/amend5/test16_tile-factorial-socialrf.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test14_analysis import GRIDALL_SUPPORT_ABOVE, N_BOOT, draw_indices, verdict, with_intervals  # noqa: E402
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

SCORES = "/scratch/dion_r13/dion_extensions/outputs/test16/scores"
OUT = "docs/ntire/amend5/test16_tile-factorial-socialrf.json"
BRANCHES = ("A2", "B2swa")


def main() -> int:
    report = {"test": 16, "commit": commit(), "n_boot": N_BOOT, "seed": SEED, "branches": {}}
    for branch in BRANCHES:
        frame = pd.read_parquet(f"{SCORES}/{branch}_tiles.parquet")
        assert not frame["split"].eq("ntire_test").any(), "Test 16 never reads ntire_test"

        # The index scheme must be paired_diff's: same interval for one difference.
        y = frame["label"].to_numpy().astype(int)
        strata = y * 2 + frame["is_distorted"].to_numpy().astype(int)
        center, squish = frame["logit_center"].to_numpy(), frame["logit_squish"].to_numpy()
        reference = paired_diff(y, center, squish, strata)
        draws = [auc(y[k], squish[k]) - auc(y[k], center[k]) for k in draw_indices(strata)]
        assert np.allclose(np.nanquantile(draws, [0.025, 0.975]), reference["ci"])

        pooled = with_intervals(frame)
        r_g = pooled["robust_R_g"]
        report["branches"][branch] = {"pooled": pooled, "verdict": verdict(pooled),
                                      "gridall_supports_coverage": bool(r_g["ci"][0] > GRIDALL_SUPPORT_ABOVE)}
        print(f"{branch}: dF {pooled['robust_dF']['point']:+.4f} {pooled['robust_dF']['ci']}, "
              f"dK {pooled['robust_dK']['point']:+.4f} {pooled['robust_dK']['ci']}, "
              f"C {pooled['robust_C']['point']:+.4f} {pooled['robust_C']['ci']}, "
              f"R_g {r_g['point']:.2f} {r_g['ci']} -> {verdict(pooled)}", flush=True)

    verdicts = {report["branches"][b]["verdict"] for b in BRANCHES}
    report["verdict"] = verdicts.pop() if len(verdicts) == 1 else "BRANCHES DISAGREE"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 16: {report['verdict']} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
