#!/usr/bin/env python3
"""Amendment 5, Test 10: masking test.

    python scripts/amend5_test10.py

Question. Under `squish`, where does the evidence the model uses sit? If the
squish gain comes from the wider field of view, the periphery (what the
200 px centre window never shows) should matter more than the centre.

Arms, scored by scripts/ntire_crop_probe.py on ntire_val_hard (a fit split):
  mask_centre     squish, centred square of side n/2 filled with the
                  ImageNet mean (25% of the area);
  mask_periphery  squish, everything outside the centred square of side
                  round(n * sqrt(0.75)) filled (also ~25%).
  squish          unmasked reference, re-scored with the same code.

Registered condition: on distorted rows, robust AUC (mask_centre -
mask_periphery) has a CI above zero on both branches. The same difference on
clean rows is reported, not used.

Writes docs/ntire/amend5/test10_masking.json and test10_masking_fig.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import paired_bootstrap  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT_DIR = "docs/ntire/amend5"
STEM = f"{OUT_DIR}/test10_masking"
SCORES_DIR = "/scratch/dion_r13/dion_extensions/outputs/test10"
#: Test 1's re-scored centre-window arm, used only for a supplementary comparison.
TEST1_SCORES = "/scratch/dion_r13/dion_extensions/outputs/test1"
SPLIT = "ntire_val_hard"
BRANCHES = {"A": "A2", "B": "B2swa"}
ARMS = ("squish", "mask_centre", "mask_periphery")
N_BOOT = 2000


def load_branch(tag: str) -> pd.DataFrame:
    """One row per image with the logits of all three arms."""
    assert SPLIT != "ntire_test", "Test 10 never reads ntire_test"
    joined = None
    for arm in ARMS:
        frame = pd.read_parquet(f"{SCORES_DIR}/probe_{tag}_{arm}_{SPLIT}.parquet")
        frame = frame[["path", "label", "is_distorted", "logit"]].rename(columns={"logit": arm})
        if joined is None:
            joined = frame
        else:
            before = len(joined)
            joined = joined.merge(frame[["path", arm]], on="path", validate="1:1")
            assert len(joined) == before == len(frame), f"{tag} {arm}: arms cover different rows"
    return joined


def squish_reproduction(tag: str, joined: pd.DataFrame) -> float:
    """Max |logit| difference between this squish re-score and the frozen probe parquet."""
    frozen = pd.read_parquet(f"outputs/scores/probe_{tag}_squish_{SPLIT}.parquet")[["path", "logit"]]
    both = joined[["path", "squish"]].merge(frozen, on="path", validate="1:1")
    assert len(both) == len(joined)
    return float((both["squish"] - both["logit"]).abs().max())


def vs_centre_window(tag: str, joined: pd.DataFrame) -> dict:
    """Supplementary, not registered: squish with its centre masked against the
    200 px centre window (Test 1's re-scored `center` arm, same rows).

    For images with a short side above ~400 px the centre window shows a part
    of the scene that lies inside the masked centre square, so `mask_centre`
    sees none of what `center` sees. If it still beats `center`, the evidence
    outside the window is enough on its own.
    """
    centre = pd.read_parquet(f"{TEST1_SCORES}/probe_{tag}_center_{SPLIT}.parquet")[["path", "logit"]]
    both = joined.merge(centre.rename(columns={"logit": "center"}), on="path", validate="1:1")
    assert len(both) == len(joined)
    boot = paired_bootstrap(both, ["center", "mask_centre"], "center", n_boot=N_BOOT, seed=SEED)
    return {"mask_centre_minus_center": boot["vs_baseline"]["mask_centre"],
            "center": boot["models"]["center"]}


def figure(res: dict, pdf_path: str) -> None:
    """Robust and clean AUC per arm and branch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
    colours = {"squish": "0.6", "mask_centre": "#2166ac", "mask_periphery": "#b2182b"}
    for ax, metric in zip(axes, ("robust_auc", "clean_auc")):
        for i, branch in enumerate(BRANCHES):
            models = res["branches"][branch]["bootstrap"]["models"]
            for j, arm in enumerate(ARMS):
                m = models[arm][metric]
                x = i * 4 + j
                ax.bar(x, m["value"], color=colours[arm], width=0.8, label=arm if i == 0 else None)
                ax.errorbar(x, m["value"], yerr=[[m["value"] - m["ci"][0]], [m["ci"][1] - m["value"]]],
                            color="0.2", lw=0.8, capsize=2)
        ax.set_xticks([1, 5])
        ax.set_xticklabels(list(BRANCHES))
        ax.set_title(metric.replace("_", " "), fontsize=8)
        low = min(res["branches"][b]["bootstrap"]["models"][a][metric]["ci"][0] for b in BRANCHES for a in ARMS)
        ax.set_ylim(max(0.5, low - 0.02), 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf_path)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT, "split": SPLIT, "branches": {}}
    passes = {}
    for branch, tag in BRANCHES.items():
        joined = load_branch(tag)
        # Baseline mask_periphery: vs_baseline["mask_centre"] is (centre-masked - periphery-masked).
        boot = paired_bootstrap(joined, list(ARMS), "mask_periphery", n_boot=N_BOOT, seed=SEED)
        centre_minus_periphery = boot["vs_baseline"]["mask_centre"]
        passes[branch] = centre_minus_periphery["robust_auc"]["ci"][0] > 0
        res["branches"][branch] = {
            "n": int(len(joined)),
            "squish_max_abs_logit_diff_vs_frozen": squish_reproduction(tag, joined),
            "bootstrap": boot,
            "robust_centre_minus_periphery": centre_minus_periphery["robust_auc"],
            "clean_centre_minus_periphery_reported_only": centre_minus_periphery["clean_auc"],
            "ci_above_zero": bool(passes[branch]),
        }
        res["branches"][branch]["supplementary_vs_centre_window"] = vs_centre_window(tag, joined)
        joined.to_parquet(f"{SCORES_DIR}/test10_rows_{branch}.parquet", index=False)
    res["verdict"] = "PASS" if all(passes.values()) else "FAIL"
    figure(res, f"{STEM}_fig.pdf")
    with open(f"{STEM}.json", "w") as f:
        json.dump(res, f, indent=2)
    for branch, e in res["branches"].items():
        r, c = e["robust_centre_minus_periphery"], e["clean_centre_minus_periphery_reported_only"]
        print(branch, "squish repro", round(e["squish_max_abs_logit_diff_vs_frozen"], 6),
              f"robust {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]",
              f"clean {c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
    print("verdict", res["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
