#!/usr/bin/env python3
"""Amendment 12, Test 20, analysis: Tests 16 and 17's verdicts on the CLIP proxies.

    python scripts/amend5_test20_analysis.py [--scores DIR] [--out FILE]

Reads the parquets of amend5_test20.py. Loads no model.

Per seed, on Test 16's rows (robust AUC = AUC on the `_deg` rows):
  Test 16's rule, with Test 14's code: dF, dK, C -> COVERAGE / CLEANING / BOTH / NEITHER
  Test 17's rule, with Test 17's code: dA, dS    -> ATTENUATION / CONTENT / BOTH / NEITHER
Verdict per rule: the one at least two of three seeds share, else SEEDS DISAGREE.
Floor: if the seed-mean robust AUC of `center` is below 0.60, no verdict is read.
The paper's claims carry to CLIP if the rules give CLEANING and CONTENT.

Writes docs/ntire/amend5/test20_clip-decomp.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amend5_test14_analysis as t14  # noqa: E402
import amend5_test17_analysis as t17  # noqa: E402
from amend5_test20 import OUT_ROOT, SEEDS  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT = "docs/ntire/amend5/test20_clip-decomp.json"
FLOOR = 0.60
CLAIMS_HOLD = {"t16_rule": "CLEANING", "t17_rule": "CONTENT"}


def majority(verdicts: list[str]) -> str:
    """The verdict at least two seeds share (with one seed, that seed's)."""
    if len(verdicts) == 1:
        return verdicts[0]
    label, count = Counter(verdicts).most_common(1)[0]
    return label if count >= 2 else "SEEDS DISAGREE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=f"{OUT_ROOT}/scores")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    report = {"test": 20, "commit": commit(), "n_boot": t14.N_BOOT, "seed": SEED, "seeds": {}}
    paths = None
    for seed in SEEDS:
        file = f"{a.scores}/{seed}_views.parquet"
        if not os.path.exists(file):
            print(f"{seed}: no scores (budget fallback or not run)", flush=True)
            continue
        frame = pd.read_parquet(file)
        assert not frame["split"].eq("ntire_test").any(), "Test 20 never reads ntire_test"
        assert paths is None or frame["path"].tolist() == paths, f"{seed}: rows differ between seeds"
        paths = frame["path"].tolist()
        # rt(2, f) holds tiles(1, f)'s scene: same window count (Test 17's check).
        for f in t17.FS:
            assert (frame[f"n_windows_{t17.rt_name(2, f)}"] == frame[f"n_windows_{t17.tiles_name(1, f)}"]).all()
        s14, s17 = t14.with_intervals(frame), t17.with_intervals(frame)
        report["seeds"][seed] = {"t16_rule": {"stats": s14, "verdict": t14.verdict(s14)},
                                 "t17_rule": {"stats": s17, "verdict": t17.verdict(s17)}}
        print(f"{seed}: center {s14['robust_center']['point']:.4f}, squish {s14['robust_squish']['point']:.4f} | "
              f"dF {s14['robust_dF']['point']:+.4f} {s14['robust_dF']['ci']}, "
              f"dK {s14['robust_dK']['point']:+.4f} {s14['robust_dK']['ci']}, "
              f"C {s14['robust_C']['point']:+.4f} -> {t14.verdict(s14)} | "
              f"dA {s17['robust_dA']['point']:+.4f} {s17['robust_dA']['ci']}, "
              f"dS {s17['robust_dS']['point']:+.4f} {s17['robust_dS']['ci']} -> {t17.verdict(s17)}", flush=True)

    seeds = report["seeds"]
    assert seeds, "no seed was scored"
    report["n_seeds"] = len(seeds)
    mean_center = float(np.mean([s["t16_rule"]["stats"]["robust_center"]["point"] for s in seeds.values()]))
    report["seed_mean_robust_center"] = mean_center
    # Descriptive: every point estimate averaged over seeds.
    report["seed_mean"] = {
        rule: {name: float(np.mean([s[rule]["stats"][name]["point"] for s in seeds.values()]))
               for name in next(iter(seeds.values()))[rule]["stats"] if name.startswith(("robust_", "clean_"))}
        for rule in CLAIMS_HOLD}
    if mean_center < FLOOR:
        report["verdict"] = {rule: f"NOT READ (center robust AUC {mean_center:.3f} < {FLOOR})" for rule in CLAIMS_HOLD}
        report["claims_carry_to_clip"] = None
    else:
        report["verdict"] = {rule: majority([s[rule]["verdict"] for s in seeds.values()]) for rule in CLAIMS_HOLD}
        report["claims_carry_to_clip"] = all(report["verdict"][r] == v for r, v in CLAIMS_HOLD.items())
    print(f"Test 20 ({len(seeds)} seed(s)): {report['verdict']}, claims carry: {report['claims_carry_to_clip']}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 20 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
