#!/usr/bin/env python3
"""Amendment 5, Test 1: coverage vs cleaning (`freqmatch`).

    python scripts/amend5_test1.py

Question. Squish shows the whole image AND shrinks it. Is squish's robust-AUC
gain over the 200 px centre window explained by the wider field of view
(coverage), or by the shrink averaging away noise and JPEG blocking
(cleaning)?

Arm. `freqmatch` (scripts/ntire_crop_probe.py, `views_freqmatch`): squish to
n x n, resize back to h x w with bicubic, then the branch's own centre
window. It keeps the centre window's field of view but carries squish's
frequency content.

Registered statistic, per branch, on ntire_val_hard (a fit split):

    R = (robust AUC freqmatch - centre) / (robust AUC squish - centre)

    coverage holds        R < 0.40 on both branches
    cleaning explains it  R > 0.70 on both branches
    otherwise             both contribute; report R per branch

Inputs (per-image logits, no model is loaded here):
  - centre and freqmatch: scored for this test by ntire_crop_probe.py into
    /scratch/dion_r13/dion_extensions/outputs/test1/ (centre is re-scored so
    the reproduction of the frozen number can be checked);
  - squish: the existing probe parquets in outputs/scores/ (same script, same
    checkpoints).

Writes docs/ntire/amend5/test1_freqmatch.json and test1_freqmatch_fig.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc, paired_bootstrap  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT_DIR = "docs/ntire/amend5"
STEM = f"{OUT_DIR}/test1_freqmatch"
SCORES_DIR = "/scratch/dion_r13/dion_extensions/outputs/test1"
BRANCHES = {"A": "A2", "B": "B2swa"}
SPLITS = ("ntire_val_hard", "ntire_val")
#: Native-short-side buckets of amendment 5 (ntire_crop_probe.BUCKETS).
BUCKETS = ((80, 336), (352, 512), (528, 736), (752, 4016))
N_BOOT = 2000

#: Frozen reference for the centre arm, to check that re-scoring reproduces it.
#: ntire_val_hard: the probe's own centre parquet. ntire_val: no probe centre
#: parquet exists (plan D14), so the as-trained score parquet is the reference.
FROZEN_CENTRE = {
    "ntire_val_hard": "outputs/scores/probe_{tag}_center_ntire_val_hard.parquet",
    "ntire_val": "outputs/scores/{tag}_ntire_val.parquet",
}


def read_logits(path: str, name: str) -> pd.DataFrame:
    """One arm's per-image logits, renamed to `name`, keyed by path."""
    frame = pd.read_parquet(path)
    keep = frame[["path", "label", "is_distorted", "width", "height", "logit"]]
    return keep.rename(columns={"logit": name})


def load_branch(split: str, tag: str) -> pd.DataFrame:
    """Join centre, freqmatch and squish logits for one branch on one split.

    Every arm must cover exactly the same rows with the same labels; the
    paired bootstrap needs one row per image with all three scores.
    """
    assert split != "ntire_test", "Test 1 never reads ntire_test"
    centre = read_logits(f"{SCORES_DIR}/probe_{tag}_center_{split}.parquet", "center")
    freqmatch = read_logits(f"{SCORES_DIR}/probe_{tag}_freqmatch_{split}.parquet", "freqmatch")
    squish = read_logits(f"outputs/scores/probe_{tag}_squish_{split}.parquet", "squish")

    joined = centre.merge(freqmatch[["path", "label", "freqmatch"]], on="path",
                          suffixes=("", "_f"), validate="1:1")
    joined = joined.merge(squish[["path", "label", "squish"]], on="path",
                          suffixes=("", "_s"), validate="1:1")
    assert len(joined) == len(centre) == len(freqmatch) == len(squish), "arms cover different rows"
    assert (joined["label"] == joined["label_f"]).all() and (joined["label"] == joined["label_s"]).all()
    joined = joined.drop(columns=["label_f", "label_s"])
    joined["short"] = joined[["width", "height"]].min(axis=1)
    return joined


def reproduction_check(split: str, tag: str, rescored: pd.DataFrame) -> dict:
    """Compare the re-scored centre arm with the frozen centre scores."""
    frozen = pd.read_parquet(FROZEN_CENTRE[split].format(tag=tag))[["path", "logit"]]
    both = rescored[["path", "label", "is_distorted", "center"]].merge(frozen, on="path", validate="1:1")
    distorted = both["is_distorted"] == 1
    y = both["label"].to_numpy()
    return {
        "reference": FROZEN_CENTRE[split].format(tag=tag),
        "n_matched": int(len(both)),
        "max_abs_logit_diff": float((both["center"] - both["logit"]).abs().max()),
        "robust_auc_rescored": auc(y[distorted], both["center"].to_numpy()[distorted]),
        "robust_auc_frozen": auc(y[distorted], both["logit"].to_numpy()[distorted]),
    }


def recovered_share(boot_vs_centre: dict) -> dict:
    """R = (freqmatch - centre) / (squish - centre), on robust AUC.

    Point estimate only: the registered rule is on the point value. The two
    differences it is built from carry their own paired intervals.
    """
    gain_freqmatch = boot_vs_centre["freqmatch"]["robust_auc"]["diff"]
    gain_squish = boot_vs_centre["squish"]["robust_auc"]["diff"]
    return {"gain_freqmatch": gain_freqmatch, "gain_squish": gain_squish,
            "R": gain_freqmatch / gain_squish if gain_squish != 0 else float("nan")}


def by_bucket(joined: pd.DataFrame) -> dict:
    """Descriptive robust AUC per arm and native-size bucket (not registered)."""
    table = {}
    for lo, hi in BUCKETS:
        in_bucket = (joined["short"] >= lo) & (joined["short"] <= hi) & (joined["is_distorted"] == 1)
        rows = joined[in_bucket]
        y = rows["label"].to_numpy()
        entry = {"n_distorted": int(len(rows))}
        for arm in ("center", "freqmatch", "squish"):
            entry[arm] = auc(y, rows[arm].to_numpy()) if len(rows) else None
        if entry["n_distorted"] and entry["squish"] != entry["center"]:
            entry["R"] = (entry["freqmatch"] - entry["center"]) / (entry["squish"] - entry["center"])
        table[f"{lo}-{hi}"] = entry
    return table


def verdict(r_by_branch: dict) -> str:
    """Apply the registered rule to the per-branch R on ntire_val_hard."""
    values = list(r_by_branch.values())
    if all(r < 0.40 for r in values):
        return "COVERAGE HOLDS"
    if all(r > 0.70 for r in values):
        return "CLEANING EXPLAINS IT"
    return "BOTH CONTRIBUTE"


def figure(res: dict, pdf_path: str) -> None:
    """Robust-AUC gain over centre, freqmatch vs squish, per branch and split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), sharey=True)
    for ax, split in zip(axes, SPLITS):
        labels, positions = [], []
        for i, branch in enumerate(BRANCHES):
            vs_centre = res["splits"][split][branch]["vs_center"]["vs_baseline"]
            for j, (arm, colour) in enumerate((("freqmatch", "#b2182b"), ("squish", "#2166ac"))):
                d = vs_centre[arm]["robust_auc"]
                x = i * 3 + j
                ax.bar(x, d["diff"], color=colour, width=0.8)
                ax.errorbar(x, d["diff"], yerr=[[d["diff"] - d["ci"][0]], [d["ci"][1] - d["diff"]]],
                            color="0.2", lw=0.8, capsize=2)
                labels.append(f"{branch}\n{arm}")
                positions.append(x)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(split + (" (fit split)" if split == "ntire_val_hard" else ""), fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("robust AUC gain over centre")
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf_path)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT, "splits": {}}
    for split in SPLITS:
        res["splits"][split] = {}
        for branch, tag in BRANCHES.items():
            joined = load_branch(split, tag)
            entry = {
                "n": int(len(joined)),
                "reproduction": reproduction_check(split, tag, joined),
                # Same resample indices for all three arms (stratified label x is_distorted).
                "vs_center": paired_bootstrap(joined, ["center", "freqmatch", "squish"], "center",
                                              n_boot=N_BOOT, seed=SEED),
                "vs_squish": paired_bootstrap(joined, ["squish", "freqmatch"], "squish",
                                              n_boot=N_BOOT, seed=SEED),
                "by_bucket_robust": by_bucket(joined),
            }
            entry["recovered_share"] = recovered_share(entry["vs_center"]["vs_baseline"])
            res["splits"][split][branch] = entry
            joined.assign(split=split, branch=branch).to_parquet(
                f"{SCORES_DIR}/test1_rows_{branch}_{split}.parquet", index=False)

    r_primary = {b: res["splits"]["ntire_val_hard"][b]["recovered_share"]["R"] for b in BRANCHES}
    res["verdict"] = {"R_ntire_val_hard": r_primary, "reading": verdict(r_primary),
                      "R_ntire_val_secondary": {b: res["splits"]["ntire_val"][b]["recovered_share"]["R"]
                                                for b in BRANCHES}}
    figure(res, f"{STEM}_fig.pdf")
    with open(f"{STEM}.json", "w") as f:
        json.dump(res, f, indent=2)

    for split in SPLITS:
        for branch in BRANCHES:
            e = res["splits"][split][branch]
            m = e["vs_center"]["models"]
            print(split, branch, "repro", round(e["reproduction"]["max_abs_logit_diff"], 6),
                  "robust c/f/s", *(round(m[a]["robust_auc"]["value"], 4) for a in ("center", "freqmatch", "squish")),
                  "R", round(e["recovered_share"]["R"], 3))
    print("verdict", res["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
