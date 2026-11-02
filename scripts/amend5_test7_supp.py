#!/usr/bin/env python3
"""Amendment 5, Test 7: supplementary numbers, not part of the registered verdict.

    python scripts/amend5_test7_supp.py

Two descriptive additions to docs/ntire/amend5/test7_multiscale-proxy.md:

1. The same three-arm paired bootstrap on `ntire_val`. The squish proxy was
   chosen as the paper's arm with `ntire_val_hard` in view, so `ntire_val` is
   the split on which neither squish nor the multi-scale arm was selected
   against the other. (The multi-scale presentation was chosen on
   `ntire_val`, so this split is not fully held out for it either.)
2. How much of the crop-to-squish robust-AUC gap the multi-scale proxy
   recovers, per seed: (multi-scale - crop) / (squish - crop).

Reuses `compare_seed` from scripts/amend5_test7_analysis.py, so the
statistics are the registered ones (paired_bootstrap, 2,000 resamples,
seed 20260827). Writes docs/ntire/amend5/test7_multiscale-proxy_supp.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test7_analysis import SEED_INDICES, STEM, compare_seed  # noqa: E402
from paper_analyses import commit  # noqa: E402


def recovered_fraction(seed_result: dict) -> dict:
    """Share of the crop -> squish robust gap that the multi-scale arm closes.

    `vs_baseline` diffs are (arm - multi-scale), so
    squish - crop = squish_diff - crop_diff and multi-scale - crop = -crop_diff.
    """
    squish_minus_ms = seed_result["vs_baseline"]["squish"]["robust_auc"]["diff"]
    crop_minus_ms = seed_result["vs_baseline"]["crop"]["robust_auc"]["diff"]
    squish_minus_crop = squish_minus_ms - crop_minus_ms
    ms_minus_crop = -crop_minus_ms
    return {"squish_minus_crop": squish_minus_crop,
            "multiscale_minus_crop": ms_minus_crop,
            "fraction_recovered": ms_minus_crop / squish_minus_crop}


def main() -> int:
    with open(f"{STEM}.json") as f:
        registered = json.load(f)
    gap_on_eval_split = {i: recovered_fraction(registered["ntire_val_hard"][str(i)])
                         for i in SEED_INDICES}
    ntire_val = {i: compare_seed(i, "ntire_val") for i in SEED_INDICES}
    gap_on_ntire_val = {i: recovered_fraction(ntire_val[i]) for i in SEED_INDICES}
    out = {"commit": commit(), "registered": False,
           "gap_recovered_ntire_val_hard": gap_on_eval_split,
           "ntire_val": ntire_val, "gap_recovered_ntire_val": gap_on_ntire_val}
    with open(f"{STEM}_supp.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    for i in SEED_INDICES:
        r = ntire_val[i]["vs_baseline"]["squish"]["robust_auc"]
        print(f"seed {i}: ntire_val squish - multi-scale robust {r['diff']:+.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]; gap recovered "
              f"hard {gap_on_eval_split[i]['fraction_recovered']:.2f}, "
              f"val {gap_on_ntire_val[i]['fraction_recovered']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
