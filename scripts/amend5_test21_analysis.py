#!/usr/bin/env python3
"""Amendment 12, Test 21, analysis: Test 14's rule on B-Free's 504 px tiles.

    python scripts/amend5_test21_analysis.py [--scores FILE] [--out FILE]

Reads the parquet of amend5_test21.py. Loads no model. Test 14's code
unchanged, with `native` (the authors' five-crop) in `center`'s role; it
enters only R_g, which is reported and not read. Robust AUC = AUC on the
`_deg` rows. Prediction (registered): dK not above zero.

Writes docs/ntire/amend5/test21_bfree-tiles.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amend5_test14_analysis as t14  # noqa: E402
from amend5_test21 import OUT_ROOT  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT = "docs/ntire/amend5/test21_bfree-tiles.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=f"{OUT_ROOT}/scores/bfree_tiles504.parquet")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    frame = pd.read_parquet(a.scores)
    assert not frame["split"].eq("ntire_test").any(), "Test 21 never reads ntire_test"
    frame = frame.assign(logit_center=frame["logit_native"], n_windows_center=frame["n_windows_native"])
    stats = t14.with_intervals(frame)
    dK = stats["robust_dK"]
    report = {"test": 21, "commit": commit(), "n_boot": t14.N_BOOT, "seed": SEED,
              "note": "center = the authors' native five-crop; windows are 504 px, fed unresized",
              "stats": stats, "verdict": t14.verdict(stats),
              "downscale_effect_carries_to_bfree": t14.above(dK)}
    print(f"B-Free: native {stats['robust_center']['point']:.4f}, squish {stats['robust_squish']['point']:.4f} | "
          f"dF {stats['robust_dF']['point']:+.4f} {stats['robust_dF']['ci']}, "
          f"dK {dK['point']:+.4f} {dK['ci']}, C {stats['robust_C']['point']:+.4f} {stats['robust_C']['ci']} "
          f"-> {report['verdict']}; dK above zero: {report['downscale_effect_carries_to_bfree']}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 21 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
