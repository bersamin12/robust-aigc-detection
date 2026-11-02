#!/usr/bin/env python3
"""Amendment 7, re-analysis R4 (supplement to Test 11). Changes no verdict.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/amend5_test11_supp_a7.py

Registered (docs/ntire/paper_prereg.md, Amendment 7, "Re-analyses", R4):
"logits z-scored against each model's own reals before any cross-model logit
comparison; the raw-logit 5.28 -> 3.94 figure is withdrawn."

Why: a crop-trained and a squish-trained model put their logits on different
scales, so "the squish-trained model's mean fake logit is lower" can mean the
fakes look less generated, or only that the model's whole scale moved. Here
each model's fake logits are expressed in units of that model's own reals:

    z = (logit - mean(real logits of that model)) / sd(real logits of that model)

and the fakes-only comparison of Test 11, analysis 2 (crop-trained vs
squish-trained, per branch, fake short side <= 1100 / > 1100 on clean
Synthbuster) is redone on z, with 95% intervals.

Statistics on every bootstrap draw (one draw set for all of them):
  - mean fake z per model and bucket; squish - crop per bucket;
  - the difference of those differences, (> 1100) - (<= 1100): does the large
    bucket lose more than the small one, on a common scale;
  - TPR@1%FPR per model and bucket (score_plan_splits.tpr_at_fpr, threshold
    re-derived on each draw's reals); squish - crop per bucket. TPR is
    invariant to the z transform, so it needs no rescaling; it is added
    because Test 11 gave it without an interval;
  - pooled clean AUC difference (squish - crop) per branch, used only to
    assert that the draws are exactly paired_diff's.

Bootstrap: the index scheme of paper_analyses.paired_diff (seed 20260827,
default_rng, cells in np.unique(strata) order, rng.integers within each
cell, 2000 draws, nanquantile 2.5 / 97.5). Strata = label (every row is
clean, so label x is_distorted reduces to label). The reals' mean and sd are
re-estimated on each draw's reals.

Also written down here (no new computation beyond reading Test 11's JSON and
two fusion configs): whether Synthbuster has power for the per-generator AUC
comparison, and which checkpoints the paper's 0.974 -> 0.880 comparison uses.

Writes docs/ntire/amend5/test11_supp_a7.json. The .md is written by hand
from it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test11 import BIG, MODELS, load_synthbuster  # noqa: E402
from paper_analyses import SEED, auc, commit, paired_diff  # noqa: E402
from score_plan_splits import tpr_at_fpr  # noqa: E402

OUT_JSON = "docs/ntire/amend5/test11_supp_a7.json"
TEST11_JSON = "docs/ntire/amend5/test11_clean-cost.json"
N_BOOT = 2000
#: Fusion configs of the two `ntire_test` reads (which checkpoints each used).
READ1_CONFIG = "docs/ntire/fusion_ntire2_final.json"
READ2_CONFIG = "docs/ntire/paper/fusion_squish_final.json"
BUCKETS = {"<= 1100": "small", "> 1100": "large"}


def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def z_against_own_reals(logits: np.ndarray, is_real: np.ndarray) -> np.ndarray:
    """Logits in units of the model's own reals: (logit - real mean) / real sd (ddof=1)."""
    real_logits = logits[is_real]
    real_mean = real_logits.mean()
    real_sd = real_logits.std(ddof=1)
    return (logits - real_mean) / real_sd


def statistics(frame: pd.DataFrame, index: np.ndarray | None = None) -> dict[str, float]:
    """Every R4 statistic on one set of rows (the full rows, or one draw)."""
    rows = frame if index is None else frame.iloc[index]
    labels = rows["label"].to_numpy().astype(int)
    is_real = labels == 0
    is_large = rows["short"].to_numpy() > BIG
    bucket_masks = {"small": (labels == 1) & ~is_large, "large": (labels == 1) & is_large}

    out = {}
    for branch, (crop_model, squish_model) in MODELS.items():
        for policy, model in (("crop", crop_model), ("squish", squish_model)):
            logits = rows[model].to_numpy().astype(np.float64)
            z = z_against_own_reals(logits, is_real)
            for bucket, is_fake_in_bucket in bucket_masks.items():
                out[f"{branch}_{policy}_{bucket}_mean_z"] = float(z[is_fake_in_bucket].mean())
                bucket_labels = labels[is_real | is_fake_in_bucket]
                bucket_logits = logits[is_real | is_fake_in_bucket]
                out[f"{branch}_{policy}_{bucket}_tpr"] = tpr_at_fpr(bucket_labels, bucket_logits)

        for bucket in bucket_masks:
            out[f"{branch}_{bucket}_mean_z_diff"] = (out[f"{branch}_squish_{bucket}_mean_z"]
                                                     - out[f"{branch}_crop_{bucket}_mean_z"])
            out[f"{branch}_{bucket}_tpr_diff"] = (out[f"{branch}_squish_{bucket}_tpr"]
                                                  - out[f"{branch}_crop_{bucket}_tpr"])
        out[f"{branch}_large_minus_small_mean_z_diff"] = (out[f"{branch}_large_mean_z_diff"]
                                                          - out[f"{branch}_small_mean_z_diff"])
        out[f"{branch}_pooled_auc_diff"] = (auc(labels, rows[squish_model].to_numpy())
                                            - auc(labels, rows[crop_model].to_numpy()))
    return out


def with_intervals(frame: pd.DataFrame) -> dict:
    """Point value and 95% percentile interval of every statistic."""
    point = statistics(frame)
    strata = frame["label"].to_numpy().astype(int)
    draws = [statistics(frame, index) for index in draw_indices(strata)]
    result = {}
    for name, value in point.items():
        values = np.array([draw[name] for draw in draws])
        lo, hi = np.nanquantile(values, [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


def assert_matches_paired_diff(frame: pd.DataFrame, stats: dict) -> dict:
    """One AUC difference per branch must reproduce paired_diff's interval exactly."""
    labels = frame["label"].to_numpy().astype(int)
    checked = {}
    for branch, (crop_model, squish_model) in MODELS.items():
        reference = paired_diff(labels, frame[crop_model].to_numpy(), frame[squish_model].to_numpy(),
                                labels, n_boot=N_BOOT, seed=SEED)
        mine = stats[f"{branch}_pooled_auc_diff"]
        assert np.allclose(mine["ci"], reference["ci"], rtol=0, atol=1e-12), (branch, mine, reference)
        assert abs(mine["point"] - reference["diff"]) < 1e-12, (branch, mine, reference)
        checked[branch] = reference
    return checked


def power_statement(frame: pd.DataFrame) -> dict:
    """What Synthbuster can and cannot show for the per-generator AUC comparison.

    Read from Test 11's JSON: the crop-trained AUC per generator and the width
    of each paired interval. Added: how far each generator's fakes sit above
    the reals in the crop-trained model's own real-sd units (median fake z),
    a measure of how much room the ranking has before AUC moves at all.
    """
    with open(TEST11_JSON) as f:
        test11 = json.load(f)
    labels = frame["label"].to_numpy().astype(int)
    is_real = labels == 0
    out = {}
    for branch, (crop_model, _) in MODELS.items():
        crop_z = z_against_own_reals(frame[crop_model].to_numpy().astype(np.float64), is_real)
        per_generator = {}
        for generator, entry in test11["per_generator"].items():
            is_this_generator = (frame["generator"].to_numpy() == generator) & (labels == 1)
            ci_low, ci_high = entry[branch]["ci"]
            per_generator[generator] = {"crop_auc": entry[branch]["crop_auc"],
                                        "diff": entry[branch]["diff"],
                                        "ci_width": ci_high - ci_low,
                                        "crop_median_fake_z": float(np.median(crop_z[is_this_generator]))}
        crop_aucs = [g["crop_auc"] for g in per_generator.values()]
        widths = [g["ci_width"] for g in per_generator.values()]
        out[branch] = {"per_generator": per_generator,
                       "min_crop_auc": float(min(crop_aucs)),
                       "max_ci_width": float(max(widths))}
    return out


def read_checkpoints() -> dict:
    """Which branch-A checkpoint each `ntire_test` read used (the 0.974 -> 0.880 figure)."""
    out = {}
    for name, path in (("read1_crop", READ1_CONFIG), ("read2_squish", READ2_CONFIG)):
        with open(path) as f:
            config = json.load(f)
        out[name] = {"config": path, "branch_a": config["branch_a"], "branch_b": config["branch_b"]}
    return out


def main() -> int:
    synthbuster = load_synthbuster()
    assert not synthbuster["path"].str.contains("ntire_test").any(), "R4 never reads ntire_test"
    assert set(synthbuster["label"].unique()) == {0, 1}

    stats = with_intervals(synthbuster)
    reference = assert_matches_paired_diff(synthbuster, stats)

    large = synthbuster[(synthbuster["label"] == 1) & (synthbuster["short"] > BIG)]
    result = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT, "strata": "label (all rows clean)",
              "z": "(logit - mean of model's reals) / sd (ddof=1) of model's reals, re-estimated per draw",
              "tpr": "score_plan_splits.tpr_at_fpr, 1% FPR, threshold re-derived per draw",
              "models": MODELS, "size_cut_px": BIG,
              "n_real": int((synthbuster["label"] == 0).sum()),
              "n_fake_small": int(((synthbuster["label"] == 1) & (synthbuster["short"] <= BIG)).sum()),
              "n_fake_large": int(len(large)),
              "large_generators": sorted(large["generator"].unique().tolist()),
              "statistics": stats,
              "paired_diff_check": reference,
              "power": power_statement(synthbuster),
              "test_read_checkpoints": read_checkpoints()}
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    for name, value in stats.items():
        print(f"{name:40s} {value['point']:+.4f} [{value['ci'][0]:+.4f}, {value['ci'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
