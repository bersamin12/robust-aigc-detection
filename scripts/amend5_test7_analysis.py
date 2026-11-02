#!/usr/bin/env python3
"""Amendment 5, Test 7: multi-scale training baseline, analysis.

    python scripts/amend5_test7_analysis.py choose --seed-index 1
    python scripts/amend5_test7_analysis.py report

`choose` picks one seed's test-time presentation (center, short_center or
squish) on `ntire_val` by mean(robust AUC, clean AUC), from the probe json
that scripts/amend5_test7.sh writes, and records it in
outputs/test7/abl_rrc_s<i>_choice.json. It runs before that seed is scored on
`ntire_val_hard` or any external set, as registered. Ties go to the earlier
presentation in the registered order.

`report` runs the registered comparison and writes
docs/ntire/amend5/test7_multiscale-proxy.json and _fig.pdf:

  Pass: mean over seeds of robust AUC (squish proxy - multi-scale proxy) on
        `ntire_val_hard` > 0, with a CI above zero in at least 2 of 3 seeds.
  Fail: multi-scale matches or beats squish (mean difference <= 0).

Comparators are the existing G1 proxies, abl_squish_s<i> and abl_crop_s<i>,
same seed index. Statistics: scripts/paired_bootstrap.py, 2,000 resamples
stratified by label x is_distorted, seed 20260827, the same indices for all
three arms of a seed. The external sets as distributed are reported, not
judged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import join, load, paired_bootstrap  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT = "/scratch/dion_r13/dion_extensions/outputs/test7"
EXISTING_PROXY_SCORES = "outputs/scores/paper"
DOCS = "docs/ntire/amend5"
STEM = f"{DOCS}/test7_multiscale-proxy"
PRESENTATIONS = ("center", "short_center", "squish")   # registered order
SEED_INDICES = (1, 2, 3)
EVAL_SPLIT = "ntire_val_hard"
EXTERNAL_SPLITS = ("ext_aigi_socialrf", "ext_aigi_community", "ext_synthbuster")
N_BOOT = 2000


def mean_robust_clean(arm_metrics: dict) -> float:
    return float(np.mean([arm_metrics["robust_auc"], arm_metrics["clean_auc"]]))


def cmd_choose(seed_index: int) -> int:
    """Pick the presentation for one seed on ntire_val only."""
    prefix = f"{OUT}/abl_rrc_s{seed_index}"
    with open(f"{prefix}_probe_ntire_val.json") as f:
        probe = json.load(f)
    assert probe["split"] == "ntire_val", probe["split"]
    objective = {arm: mean_robust_clean(probe["arms"][arm]) for arm in PRESENTATIONS}
    best_value = max(objective.values())
    chosen = next(arm for arm in PRESENTATIONS if objective[arm] == best_value)
    record = {"seed_index": seed_index, "split": "ntire_val",
              "objective": "mean(robust_auc, clean_auc)",
              "per_presentation": objective, "chosen": chosen,
              "tie_rule": "earlier in (center, short_center, squish)",
              "commit": commit()}
    with open(f"{prefix}_choice.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"abl_rrc_s{seed_index}: {objective} -> {chosen}")
    return 0


def arm_paths(seed_index: int, split: str) -> dict:
    """Per-image parquet of each arm of one seed on one split."""
    with open(f"{OUT}/abl_rrc_s{seed_index}_choice.json") as f:
        chosen = json.load(f)["chosen"]
    on_ntire = split.startswith("ntire_")
    comparator_dir = EXISTING_PROXY_SCORES if on_ntire else OUT
    return {
        "multiscale": f"{OUT}/abl_rrc_s{seed_index}_{chosen}_{split}.parquet",
        "squish": f"{comparator_dir}/abl_squish_s{seed_index}_{split}.parquet",
        "crop": f"{comparator_dir}/abl_crop_s{seed_index}_{split}.parquet",
    }


def compare_seed(seed_index: int, split: str) -> dict:
    """Paired bootstrap of the three arms of one seed; baseline multi-scale,
    so each `vs_baseline` diff reads (arm - multi-scale)."""
    assert split != "ntire_test", "Test 7 never reads ntire_test"
    paths = arm_paths(seed_index, split)
    frames = [load(name, path, "logit") for name, path in paths.items()]
    joined = join(frames)
    result = paired_bootstrap(joined, list(paths), baseline="multiscale",
                              n_boot=N_BOOT, seed=SEED)
    result["paths"] = paths
    return result


def verdict(per_seed: dict) -> dict:
    """The registered pass / fail reading on ntire_val_hard."""
    diffs = [per_seed[i]["vs_baseline"]["squish"]["robust_auc"]["diff"] for i in SEED_INDICES]
    ci_lows = [per_seed[i]["vs_baseline"]["squish"]["robust_auc"]["ci"][0] for i in SEED_INDICES]
    mean_diff = float(np.mean(diffs))
    seeds_ci_above_zero = int(sum(low > 0 for low in ci_lows))
    if mean_diff > 0 and seeds_ci_above_zero >= 2:
        reading = "PASS"
    elif mean_diff <= 0:
        reading = "FAIL"
    else:
        reading = "IN BETWEEN (mean > 0, CI above zero in fewer than 2 seeds)"
    return {"squish_minus_multiscale_robust": diffs, "ci_low": ci_lows,
            "mean_diff": mean_diff, "seeds_ci_above_zero": seeds_ci_above_zero,
            "reading": reading}


def figure(per_seed: dict, pdf: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    for row, (arm, colour) in enumerate((("squish", "#2166ac"), ("crop", "#b2182b"))):
        for k, i in enumerate(SEED_INDICES):
            r = per_seed[i]["vs_baseline"][arm]["robust_auc"]
            ax.errorbar(r["diff"], row + (k - 1) * 0.15,
                        xerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]],
                        fmt="o", ms=3, color=colour, lw=0.9, capsize=1.5)
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["squish - multi-scale", "crop - multi-scale"])
    ax.invert_yaxis()
    ax.set_xlabel(f"robust AUC difference on {EVAL_SPLIT} (95% CI), seeds 1-3")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf)


def cmd_report() -> int:
    os.makedirs(DOCS, exist_ok=True)
    choices = {}
    for i in SEED_INDICES:
        with open(f"{OUT}/abl_rrc_s{i}_choice.json") as f:
            choices[i] = json.load(f)
    per_seed = {i: compare_seed(i, EVAL_SPLIT) for i in SEED_INDICES}
    external = {}
    for split in EXTERNAL_SPLITS:
        external[split] = {}
        for i in SEED_INDICES:
            try:
                external[split][i] = compare_seed(i, split)
            except (FileNotFoundError, SystemExit) as err:
                external[split][i] = {"missing": str(err)}
    result = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT,
              "presentation_choice": choices, "eval_split": EVAL_SPLIT,
              "note": "ntire_val_hard was used to choose squish: fit-split numbers",
              "verdict": verdict(per_seed), "ntire_val_hard": per_seed,
              "external_as_distributed": external}
    with open(f"{STEM}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    figure(per_seed, f"{STEM}_fig.pdf")
    v = result["verdict"]
    print(f"Test 7: {v['reading']}; squish - multi-scale robust per seed "
          f"{[round(d, 4) for d in v['squish_minus_multiscale_robust']]}, mean {v['mean_diff']:+.4f}, "
          f"CI above zero in {v['seeds_ci_above_zero']}/3")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("choose")
    choose.add_argument("--seed-index", type=int, required=True, choices=SEED_INDICES)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.command == "choose":
        return cmd_choose(a.seed_index)
    return cmd_report()


if __name__ == "__main__":
    raise SystemExit(main())
