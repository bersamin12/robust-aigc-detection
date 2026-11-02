#!/usr/bin/env python3
"""Amendment 5, Test 8: fair native crop baseline, analysis.

    python scripts/amend5_test8_analysis.py

Registered condition (docs/ntire/paper_prereg.md, "Amendment 5", Test 8):

  Pass: mean over seeds of robust AUC (squish proxy - native-224 proxy) on
        `ntire_val_hard` > 0, with a CI above zero in at least 2 of 3 seeds.
        The 200 px crop proxies are reported alongside.

The registration names no fail branch; a mean <= 0 is reported as FAIL, and
a positive mean with fewer than 2 seeds' CI above zero as IN BETWEEN.

Arms per seed index i: abl_nat224_s<i> (this test), abl_squish_s<i> and
abl_crop_s<i> (the existing G1 proxies, same seed, same scorer). Statistics:
scripts/paired_bootstrap.py, 2,000 resamples stratified by label x
is_distorted, seed 20260827, the same indices for all three arms of a seed.
`ntire_val` is reported as well, not judged.

Writes docs/ntire/amend5/test8_native-crop-proxy.json and _fig.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import join, load, paired_bootstrap  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT = "/scratch/dion_r13/dion_extensions/outputs/test8"
EXISTING_PROXY_SCORES = "outputs/scores/paper"
DOCS = "docs/ntire/amend5"
STEM = f"{DOCS}/test8_native-crop-proxy"
SEED_INDICES = (1, 2, 3)
EVAL_SPLIT = "ntire_val_hard"
ALSO_REPORTED_SPLIT = "ntire_val"
N_BOOT = 2000


def compare_seed(seed_index: int, split: str) -> dict:
    """Paired bootstrap of the three arms of one seed; baseline native-224,
    so each `vs_baseline` diff reads (arm - native-224)."""
    assert split != "ntire_test", "Test 8 never reads ntire_test"
    paths = {
        "native224": f"{OUT}/abl_nat224_s{seed_index}_{split}.parquet",
        "squish": f"{EXISTING_PROXY_SCORES}/abl_squish_s{seed_index}_{split}.parquet",
        "crop200": f"{EXISTING_PROXY_SCORES}/abl_crop_s{seed_index}_{split}.parquet",
    }
    joined = join([load(name, path, "logit") for name, path in paths.items()])
    result = paired_bootstrap(joined, list(paths), baseline="native224",
                              n_boot=N_BOOT, seed=SEED)
    result["paths"] = paths
    return result


def verdict(per_seed: dict) -> dict:
    diffs = [per_seed[i]["vs_baseline"]["squish"]["robust_auc"]["diff"] for i in SEED_INDICES]
    ci_lows = [per_seed[i]["vs_baseline"]["squish"]["robust_auc"]["ci"][0] for i in SEED_INDICES]
    mean_diff = float(np.mean(diffs))
    seeds_ci_above_zero = int(sum(low > 0 for low in ci_lows))
    if mean_diff > 0 and seeds_ci_above_zero >= 2:
        reading = "PASS"
    elif mean_diff <= 0:
        reading = "FAIL (not a registered branch: mean difference <= 0)"
    else:
        reading = "IN BETWEEN (mean > 0, CI above zero in fewer than 2 seeds)"
    return {"squish_minus_native224_robust": diffs, "ci_low": ci_lows,
            "mean_diff": mean_diff, "seeds_ci_above_zero": seeds_ci_above_zero,
            "reading": reading}


def figure(per_seed: dict, pdf: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    for row, (arm, colour) in enumerate((("squish", "#2166ac"), ("crop200", "#b2182b"))):
        for k, i in enumerate(SEED_INDICES):
            r = per_seed[i]["vs_baseline"][arm]["robust_auc"]
            ax.errorbar(r["diff"], row + (k - 1) * 0.15,
                        xerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]],
                        fmt="o", ms=3, color=colour, lw=0.9, capsize=1.5)
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["squish - native 224", "crop 200 - native 224"])
    ax.invert_yaxis()
    ax.set_xlabel(f"robust AUC difference on {EVAL_SPLIT} (95% CI), seeds 1-3")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf)


def main() -> int:
    os.makedirs(DOCS, exist_ok=True)
    per_seed = {i: compare_seed(i, EVAL_SPLIT) for i in SEED_INDICES}
    also_reported = {i: compare_seed(i, ALSO_REPORTED_SPLIT) for i in SEED_INDICES}
    result = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT, "eval_split": EVAL_SPLIT,
              "note": "ntire_val_hard was used to choose squish: fit-split numbers",
              "verdict": verdict(per_seed), EVAL_SPLIT: per_seed,
              ALSO_REPORTED_SPLIT: also_reported}
    with open(f"{STEM}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    figure(per_seed, f"{STEM}_fig.pdf")
    v = result["verdict"]
    print(f"Test 8: {v['reading']}; squish - native224 robust per seed "
          f"{[round(d, 4) for d in v['squish_minus_native224_robust']]}, mean {v['mean_diff']:+.4f}, "
          f"CI above zero in {v['seeds_ci_above_zero']}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
