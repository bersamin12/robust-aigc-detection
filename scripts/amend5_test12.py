#!/usr/bin/env python3
"""Amendment 5, Test 12: numbers behind the proposed writing fixes.

    PYTHONPATH=$PWD/src python scripts/amend5_test12.py

Test 12 scores nothing. It reads per-image scores that already exist and
computes the three things the registration names (docs/ntire/paper_prereg.md,
"Amendment 5", Test 12):

1. TPR@1%FPR for every table of docs/paper/main.tex that has per-image scores
   behind it. Each row's AUC is recomputed next to the TPR and compared with
   the value printed in main.tex, so a row that does not reproduce is visible.
2. The 416-496 px bin of Figure 1 (distorted ntire_val_hard rows), per frozen
   branch instead of fused.
3. Crop's final-vs-SWA choice redone on ntire_val alone (A2 vs A2swa, B2 vs
   B2swa), with the rule the squish models used: mean of robust and clean AUC
   on ntire_val, a tie goes to final (docs/ntire/paper/picks.json).

ntire_test is never read. Table "tab:test" already prints TPR@1%.

Writes docs/ntire/amend5/test12_writing.json and the per-row TPR table as
/scratch/dion_r13/dion_extensions/outputs/test12/test12_tpr_rows.parquet.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_bootstrap as pb  # noqa: E402
from paper_analyses import (SEED, _tokens, commit, distortion_group,  # noqa: E402
                            paired_diff, with_size)

from aigcdet.eval.metrics import tpr_at_fpr  # noqa: E402

OUT = "docs/ntire/amend5"
STEM = f"{OUT}/test12_writing"
ROWS_OUT = "/scratch/dion_r13/dion_extensions/outputs/test12"
SCORES = "outputs/scores"
PAPER = "outputs/scores/paper"
#: The fused per-image parquets are git-ignored, so they exist only in
#: Justin's checkout. Read-only.
FUSED = "/mnt/r13justinssd/techjam/aigcdet/docs/ntire/paper"
N_BOOT = 2000
#: Below this many reals a 1% FPR cannot be resolved: the threshold is then
#: "above every real", i.e. the TPR is at zero false positives.
MIN_REALS_FOR_1PCT = 100


def fused(tag: str, split: str) -> str:
    return f"{FUSED}/fusion_{tag}_{split}_fused.parquet"


# ---------------------------------------------------------------------------
# 1. TPR@1%FPR per table row
# ---------------------------------------------------------------------------

def table_rows() -> list[dict]:
    """Every table row with per-image scores behind it.

    Each entry names the parquet, the score column, the split, and the AUC
    values main.tex prints for that row (`paper`), keyed by the metric the
    table shows. Rows that main.tex reports as a mean over three seeds list
    three parquets.
    """
    rows = []

    # Table tab:probe: frozen crop-trained branches under four test-time
    # presentations, ntire_val_hard.
    probe_paper = {
        ("A2", "center"): (0.8561, 0.9912), ("A2", "grid9"): (0.8495, 0.9978),
        ("A2", "fov3"): (0.8703, 0.9977), ("A2", "squish"): (0.9482, 0.9906),
        ("B2swa", "center"): (0.8872, 0.9801), ("B2swa", "grid9"): (0.9049, 0.9931),
        ("B2swa", "fov3"): (0.9215, 0.9942), ("B2swa", "squish"): (0.9599, 0.9886),
    }
    for (branch, arm), (robust, clean) in probe_paper.items():
        rows.append({"table": "tab:probe", "row": f"{branch} {arm}",
                     "paths": [f"{SCORES}/probe_{branch}_{arm}_ntire_val_hard.parquet"],
                     "col": "logit", "split": "ntire_val_hard",
                     "paper": {"robust_auc": robust, "clean_auc": clean}})

    # Table tab:aspect: aspect-preserving presentations. main.tex prints
    # only differences, so there is no absolute AUC to check here.
    for branch in ("A2", "B2swa"):
        for arm in ("short_center", "letterbox"):
            rows.append({"table": "tab:aspect", "row": f"{branch} {arm}",
                         "paths": [f"{PAPER}/aspect_{branch}_{arm}_ntire_val_hard.parquet"],
                         "col": "logit", "split": "ntire_val_hard", "paper": {}})

    # Table tab:branches: crop-trained as trained, the same model under
    # squish at test time, and the squish-trained pick.
    branches_paper = [
        ("A crop-trained", f"{SCORES}/A2_ntire_val_hard.parquet", 0.8561, 0.9913),
        ("A crop-trained, squish at test", f"{SCORES}/probe_A2_squish_ntire_val_hard.parquet", 0.9482, 0.9906),
        ("A squish-trained", f"{PAPER}/A3sqswa_ntire_val_hard.parquet", 0.9425, 0.9934),
        ("B crop-trained", f"{SCORES}/B2swa_ntire_val_hard.parquet", 0.8873, 0.9801),
        ("B crop-trained, squish at test", f"{SCORES}/probe_B2swa_squish_ntire_val_hard.parquet", 0.9599, 0.9886),
        ("B squish-trained", f"{PAPER}/B3sqswa_ntire_val_hard.parquet", 0.9589, 0.9898),
    ]
    for label, path, robust, clean in branches_paper:
        rows.append({"table": "tab:branches", "row": label, "paths": [path],
                     "col": "logit", "split": "ntire_val_hard",
                     "paper": {"robust_auc": robust, "clean_auc": clean}})

    # Table tab:clip: proxies, mean of three seeds.
    clip_paper = {
        ("abl", "crop-trained"): (0.7994, 0.9255),
        ("abl", "crop-trained, squish at test"): (0.8310, 0.9087),
        ("abl", "squish-trained"): (0.8841, 0.9357),
        ("clip", "crop-trained"): (0.7482, 0.9655),
        ("clip", "crop-trained, squish at test"): (0.8485, 0.9324),
        ("clip", "squish-trained"): (0.8924, 0.9645),
    }
    for (prefix, label), (robust, clean) in clip_paper.items():
        if label == "crop-trained":
            pattern = f"{PAPER}/{prefix}_probe_s{{seed}}_center_ntire_val_hard.parquet"
        elif label == "crop-trained, squish at test":
            pattern = f"{PAPER}/{prefix}_probe_s{{seed}}_squish_ntire_val_hard.parquet"
        else:
            pattern = f"{PAPER}/{prefix}_squish_s{{seed}}_ntire_val_hard.parquet"
        backbone = "DINOv2" if prefix == "abl" else "CLIP"
        rows.append({"table": "tab:clip", "row": f"{backbone} {label}",
                     "paths": [pattern.format(seed=s) for s in (1, 2, 3)],
                     "col": "logit", "split": "ntire_val_hard",
                     "paper": {"robust_auc": robust, "clean_auc": clean}})

    # Table tab:external: fused crop vs squish. The external sets are all
    # clean, so only pooled AUC / TPR exist. main.tex prints AUC only for the
    # files as distributed.
    external_paper = {"ext_aigi_socialrf": (0.9633, 0.9776),
                      "ext_aigi_community": (0.9809, 0.9920),
                      "ext_synthbuster": (0.9995, 0.9984)}
    for base, (crop_auc, squish_auc) in external_paper.items():
        for variant in ("", "_jpeg95", "_deg"):
            split = base + variant
            for tag, paper_auc in (("crop", crop_auc), ("squish", squish_auc)):
                paper = {"auc": paper_auc} if variant == "" else {}
                rows.append({"table": "tab:external", "row": f"{tag} fusion",
                             "paths": [fused(tag, split)], "col": "fused_z",
                             "split": split, "paper": paper})

    # Table tab:public: B-Free under three presentations.
    public_paper = {
        "ntire_val_hard": {"robust_auc": (0.636, 0.631, 0.618), "clean_auc": (0.909, 0.753, 0.722)},
        "ntire_val": {"robust_auc": (0.735, 0.754, 0.744), "clean_auc": (0.920, 0.848, 0.825)},
        "ext_aigi_socialrf": {"auc": (0.922, 0.889, 0.871)},
        "ext_aigi_socialrf_deg": {"auc": (0.757, 0.829, 0.820)},
        "ext_aigi_community": {"auc": (0.891, 0.742, 0.703)},
        "ext_aigi_community_deg": {"auc": (0.683, 0.568, 0.539)},
        "ext_synthbuster": {"auc": (0.998, 0.987, 0.976)},
        "ext_synthbuster_deg": {"auc": (0.896, 0.893, 0.892)},
    }
    for split, metrics in public_paper.items():
        for i, arm in enumerate(("native", "train_fov", "squish")):
            paper = {metric: values[i] for metric, values in metrics.items()}
            rows.append({"table": "tab:public", "row": f"B-Free {arm}",
                         "paths": [f"{PAPER}/public/bfree_{split}.parquet"],
                         "col": f"logit_{arm}", "split": split, "paper": paper})
    return rows


def metrics_one(y: np.ndarray, s: np.ndarray, dist: np.ndarray | None) -> dict:
    """AUC and TPR@1%FPR pooled, and on the distorted / clean halves.

    The robust TPR uses a threshold set on the distorted reals only, and the
    clean TPR one set on the clean reals only, so each matches the AUC it
    sits next to. `n_real_*` records how many reals set each threshold.
    """
    out = {"auc": pb.auc(y, s), "tpr": tpr_at_fpr(y, s), "n_real": int((y == 0).sum())}
    if dist is None:
        return out
    for half, mask in (("robust", dist == 1), ("clean", dist == 0)):
        y_half, s_half = y[mask], s[mask]
        if len(set(y_half.tolist())) < 2:
            # The external sets are all clean: their distorted half is empty.
            continue
        out[f"{half}_auc"] = pb.auc(y_half, s_half)
        out[f"{half}_tpr"] = tpr_at_fpr(y_half, s_half)
        out[f"n_real_{half}"] = int((y_half == 0).sum())
    return out


def score_row(entry: dict) -> dict:
    """Metrics for one table row, averaged over its parquets (seeds)."""
    assert entry["split"] != "ntire_test", "Test 12 never reads ntire_test"
    per_parquet = []
    for path in entry["paths"]:
        frame = pd.read_parquet(path)
        y = frame["label"].to_numpy().astype(int)
        s = frame[entry["col"]].to_numpy().astype(np.float64)
        dist = frame["is_distorted"].to_numpy().astype(int) if "is_distorted" in frame else None
        per_parquet.append(metrics_one(y, s, dist))
    mean = {k: float(np.mean([m[k] for m in per_parquet])) for k in per_parquet[0]}

    # Compare each AUC main.tex prints with the recomputed one, at the
    # number of decimals main.tex uses.
    checks = {}
    for metric, printed in entry["paper"].items():
        decimals = len(str(printed).split(".")[1])
        recomputed = round(mean[metric], decimals)
        checks[metric] = {"paper": printed, "recomputed": recomputed,
                          "matches": bool(abs(recomputed - printed) <= 10 ** -decimals + 1e-12)}
    return {**{k: entry[k] for k in ("table", "row", "split", "col", "paths")},
            "metrics": mean, "per_seed": per_parquet if len(per_parquet) > 1 else None,
            "paper_check": checks}


def groups_tpr() -> dict:
    """Robust TPR@1%FPR, centre vs squish, within the distortion groups of
    tab:groups (paper_analyses.groups, same grouping function)."""
    out = {}
    for branch in ("A2", "B2swa"):
        centre = pd.read_parquet(f"{SCORES}/probe_{branch}_center_ntire_val_hard.parquet")
        squish = pd.read_parquet(f"{SCORES}/probe_{branch}_squish_ntire_val_hard.parquet")[["path", "logit"]]
        joined = centre.merge(squish, on="path", suffixes=("_c", "_s"), validate="1:1")
        distorted = joined[joined["is_distorted"] == 1].copy()
        distorted["group"] = distorted["distortions"].apply(_tokens).apply(distortion_group)
        for group, rows in distorted.groupby("group"):
            y = rows["label"].to_numpy().astype(int)
            entry = out.setdefault(group, {"n": int(len(rows)), "n_real": int((y == 0).sum())})
            entry[branch] = {"centre_tpr": tpr_at_fpr(y, rows["logit_c"].to_numpy()),
                             "squish_tpr": tpr_at_fpr(y, rows["logit_s"].to_numpy())}
    return out


# ---------------------------------------------------------------------------
# 2. The 416-496 px bin, per frozen branch
# ---------------------------------------------------------------------------

def size_bin_416_496() -> dict:
    """Robust AUC in Figure 1's fourth bin, per branch.

    The bin edges are recomputed exactly as paper_analyses.figsize does
    (8 equal-count bins of native short side over the distorted
    ntire_val_hard rows of the fused parquets), and the bin must come out as
    416-496 px with 171 rows. Within the bin, each branch's squish arm and
    squish-trained model are compared with its centre window by paired_diff
    stratified by label, as in figsize.
    """
    crop_fused = pd.read_parquet(fused("crop", "ntire_val_hard"))[["path", "label", "is_distorted", "fused_z"]]
    squish_fused = pd.read_parquet(fused("squish", "ntire_val_hard"))[["path", "fused_z"]]
    both = with_size(crop_fused.merge(squish_fused, on="path", suffixes=("_c", "_s")))
    distorted = both[both["is_distorted"] == 1].reset_index(drop=True)
    edges = np.quantile(distorted["short"], np.linspace(0, 1, 9))
    bin_index = np.clip(np.searchsorted(edges, distorted["short"], side="right") - 1, 0, 7)
    in_bin = distorted[bin_index == 3]
    lo, hi = float(in_bin["short"].min()), float(in_bin["short"].max())
    assert (lo, hi, len(in_bin)) == (416.0, 496.0, 171), (lo, hi, len(in_bin))
    bin_paths = set(in_bin["path"])

    y_fused = in_bin["label"].to_numpy()
    fused_result = paired_diff(y_fused, in_bin["fused_z_c"].to_numpy(), in_bin["fused_z_s"].to_numpy(), y_fused)

    result = {"lo": lo, "hi": hi, "n": len(in_bin), "n_fake": int(y_fused.sum()),
              "window_fov_range": [200 / hi, 200 / lo],
              "fused": fused_result, "branches": {}}
    arms = {"A": {"centre": f"{SCORES}/probe_A2_center_ntire_val_hard.parquet",
                  "squish at test": f"{SCORES}/probe_A2_squish_ntire_val_hard.parquet",
                  "squish-trained": f"{PAPER}/A3sqswa_ntire_val_hard.parquet"},
            "B": {"centre": f"{SCORES}/probe_B2swa_center_ntire_val_hard.parquet",
                  "squish at test": f"{SCORES}/probe_B2swa_squish_ntire_val_hard.parquet",
                  "squish-trained": f"{PAPER}/B3sqswa_ntire_val_hard.parquet"}}
    for branch, paths in arms.items():
        frames = [pb.load(name, path, "logit") for name, path in paths.items()]
        joined = pb.join(frames)
        joined = joined[joined["path"].isin(bin_paths)].reset_index(drop=True)
        assert len(joined) == len(in_bin), f"{branch}: bin rows missing"
        y = joined["label"].to_numpy()
        centre = joined["centre"].to_numpy()
        result["branches"][branch] = {
            "centre": pb.auc(y, centre),
            "squish at test": paired_diff(y, centre, joined["squish at test"].to_numpy(), y),
            "squish-trained": paired_diff(y, centre, joined["squish-trained"].to_numpy(), y),
        }
    return result


# ---------------------------------------------------------------------------
# 3. Crop final vs SWA, reselected on ntire_val alone
# ---------------------------------------------------------------------------

def final_vs_swa() -> dict:
    """The squish models' selection rule applied to the crop models.

    Rule (picks.json): mean of robust and clean AUC on ntire_val; a tie goes
    to final. The paired bootstrap (label x is_distorted strata) gives the
    interval on SWA minus final for each metric. For reference the same
    criterion is also computed on ntire_val_hard, and on both splits pooled,
    since main.tex says the crop choice was "made earlier on both validation
    splits" without saying how the two were combined.
    """
    out = {}
    for branch, final_name, swa_name, kept in (("A", "A2", "A2swa", "A2"), ("B", "B2", "B2swa", "B2swa")):
        entry = {"kept_in_paper": kept, "splits": {}}
        pooled_parts = []
        for split in ("ntire_val", "ntire_val_hard"):
            frames = [pb.load("final", f"{SCORES}/{final_name}_{split}.parquet", "logit"),
                      pb.load("swa", f"{SCORES}/{swa_name}_{split}.parquet", "logit")]
            joined = pb.join(frames)
            pooled_parts.append(joined)
            boot = pb.paired_bootstrap(joined, ["final", "swa"], "final", n_boot=N_BOOT, seed=SEED)
            criterion = {m: (boot["models"][m]["robust_auc"]["value"] + boot["models"][m]["clean_auc"]["value"]) / 2
                         for m in ("final", "swa")}
            entry["splits"][split] = {"criterion": criterion,
                                      "pick": "swa" if criterion["swa"] > criterion["final"] else "final",
                                      "models": boot["models"], "swa_minus_final": boot["vs_baseline"]["swa"]}
        pooled = pd.concat(pooled_parts, ignore_index=True)
        pooled_metrics = {m: pb.all_metrics(pooled["label"].to_numpy(), pooled["is_distorted"].to_numpy(),
                                            pooled[m].to_numpy()) for m in ("final", "swa")}
        pooled_criterion = {m: (v["robust_auc"] + v["clean_auc"]) / 2 for m, v in pooled_metrics.items()}
        mean_of_splits = {m: np.mean([entry["splits"][s]["criterion"][m] for s in entry["splits"]])
                          for m in ("final", "swa")}
        entry["both_splits_pooled_rows"] = {"criterion": pooled_criterion,
                                            "pick": "swa" if pooled_criterion["swa"] > pooled_criterion["final"] else "final"}
        entry["both_splits_mean"] = {"criterion": {k: float(v) for k, v in mean_of_splits.items()},
                                     "pick": "swa" if mean_of_splits["swa"] > mean_of_splits["final"] else "final"}
        pick_on_val = entry["splits"]["ntire_val"]["pick"]
        entry["reselected_on_ntire_val"] = swa_name if pick_on_val == "swa" else final_name
        entry["changes_the_paper"] = entry["reselected_on_ntire_val"] != kept
        out[branch] = entry
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ROWS_OUT, exist_ok=True)
    scored = [score_row(entry) for entry in table_rows()]
    result = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT,
              "min_reals_for_1pct": MIN_REALS_FOR_1PCT,
              "tables": scored, "groups_tpr": groups_tpr(),
              "bin_416_496": size_bin_416_496(), "final_vs_swa": final_vs_swa()}

    flat = pd.DataFrame([{"table": r["table"], "row": r["row"], "split": r["split"], **r["metrics"]}
                         for r in scored])
    flat.to_parquet(f"{ROWS_OUT}/test12_tpr_rows.parquet", index=False)
    with open(f"{STEM}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    mismatches = [(r["table"], r["row"], r["split"], m, c) for r in scored
                  for m, c in r["paper_check"].items() if not c["matches"]]
    print(f"{len(scored)} table rows, {len(mismatches)} AUC mismatches against main.tex")
    for mismatch in mismatches:
        print("  MISMATCH", mismatch)
    for branch, entry in result["final_vs_swa"].items():
        print(branch, "kept", entry["kept_in_paper"], "reselected on ntire_val", entry["reselected_on_ntire_val"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
