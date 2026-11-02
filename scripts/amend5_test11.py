#!/usr/bin/env python3
"""Amendment 5, Test 11: where does the squish-trained branch lose clean AUC?

    python scripts/amend5_test11.py

The paper (Sec. "Where whole-image input costs") reports that on clean
`ntire_test` images with a short side above 1100 px, branch A loses clean
AUC when it is trained and run under squish instead of crop. `ntire_test`
carries no generator labels, so the paper cannot say which generated images
cause the loss. Synthbuster has generator labels, so this test looks for the
same loss there, split by generator and by the fakes' native size.

Compared models (the frozen checkpoints of amendment 5):
    crop-trained:    A2 (branch A), B2swa (branch B)
    squish-trained:  A3sqswa, B3sqswa (docs/ntire/paper/picks.json)
Each model is scored as trained (its own input policy). Only existing
per-image scores are read; no model and no image is loaded.

Three analyses, as registered in docs/ntire/paper_prereg.md,
"Amendment 5", Test 11:
    1. per generator: clean AUC of that generator's fakes against ALL RAISE
       reals, squish-trained minus crop-trained, paired 95% CI;
    2. fakes only, by fake short side (<= 1100, > 1100): mean logit and TPR
       at the threshold giving 1% FPR on the reals; the AUC of the bucket's
       fakes against all reals is added so the pass condition can be judged
       on buckets too;
    3. clean `ntire_val` / `ntire_val_hard` rows above 1100 px, as a check
       of the numbers already in the paper (871 and 295 images).

Writes docs/ntire/amend5/test11_clean-cost.json and _fig.pdf, and the
per-image join to /scratch/dion_r13/dion_extensions/outputs/test11/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analyses import BIG, SEED, commit, paired_diff, with_size  # noqa: E402

OUT_DIR = "docs/ntire/amend5"
STEM = f"{OUT_DIR}/test11_clean-cost"
ROWS_DIR = "/scratch/dion_r13/dion_extensions/outputs/test11"
SCORES = "outputs/scores/paper"
EXTERNAL_MANIFEST = "data/manifest_external.parquet"

#: branch -> (crop-trained model, squish-trained model)
MODELS = {"A": ("A2", "A3sqswa"), "B": ("B2swa", "B3sqswa")}

#: The registered size of a clean-cost drop: squish below crop by more than this.
DROP = 0.020
#: Target false positive rate on the reals for the TPR column.
FPR = 0.01
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Loading

def load_synthbuster() -> pd.DataFrame:
    """One row per clean Synthbuster image, with all four models' logits and
    the image's native short side (from the external manifest).

    Only the clean split `ext_synthbuster` is used: the registration asks
    for clean AUC. The `_deg` and `_jpeg95` copies are not read.
    """
    split = "ext_synthbuster"
    joined = None
    for crop_model, squish_model in MODELS.values():
        for model in (crop_model, squish_model):
            scores = pd.read_parquet(f"{SCORES}/{model}_{split}.parquet")
            assert (scores["split"] == split).all()
            if joined is None:
                joined = scores[["path", "generator", "label", "is_distorted"]].copy()
            per_model = scores[["path", "logit"]].rename(columns={"logit": model})
            joined = joined.merge(per_model, on="path", how="inner", validate="1:1")
            assert len(joined) == len(scores), f"{model}: rows differ between models"

    manifest = pd.read_parquet(EXTERNAL_MANIFEST, columns=["path", "split", "width", "height"])
    manifest = manifest[manifest["split"] == split].drop(columns="split")
    joined = joined.merge(manifest, on="path", how="left", validate="1:1")
    assert joined["width"].notna().all(), "some Synthbuster rows are not in the manifest"
    joined["short"] = joined[["width", "height"]].min(axis=1)

    assert (joined["is_distorted"] == 0).all(), "the clean split should have no distorted rows"
    return joined


def load_ntire_large(split: str, branch: str) -> pd.DataFrame:
    """Clean rows of an NTIRE fit split with a short side above 1100 px, with
    the crop-trained and the squish-trained logit of one branch."""
    assert split != "ntire_test", "Test 11 never reads ntire_test"
    crop_model, squish_model = MODELS[branch]
    crop = pd.read_parquet(f"{SCORES}/{crop_model}_{split}.parquet")
    squish = pd.read_parquet(f"{SCORES}/{squish_model}_{split}.parquet")
    crop = crop[["path", "label", "is_distorted", "logit"]].rename(columns={"logit": "crop"})
    squish = squish[["path", "logit"]].rename(columns={"logit": "squish"})
    both = with_size(crop.merge(squish, on="path", validate="1:1"))
    is_clean = both["is_distorted"] == 0
    is_large = both["short"] > BIG
    return both[is_clean & is_large].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Measures

def tpr_at_fpr(real_logits: np.ndarray, fake_logits: np.ndarray, fpr: float = FPR) -> dict:
    """True positive rate at the threshold where `fpr` of the reals score above it.

    The threshold is the (1 - fpr) quantile of the real logits; a fake counts
    as detected when its logit is strictly above it. With 994 reals, 1% FPR
    is about 10 reals, so the threshold rests on the top ~10 real scores.
    """
    threshold = float(np.quantile(real_logits, 1.0 - fpr))
    realised_fpr = float(np.mean(real_logits > threshold))
    tpr = float(np.mean(fake_logits > threshold))
    return {"threshold": threshold, "realised_fpr": realised_fpr, "tpr": tpr}


def compare_on_subset(rows: pd.DataFrame, branch: str) -> dict:
    """Clean AUC of crop-trained vs squish-trained on `rows` (fakes + reals).

    `diff` = AUC(squish-trained) - AUC(crop-trained), so a clean cost of
    squish is a negative diff. Resampling is stratified by label (every row
    here is clean, so label x is_distorted reduces to label), with the same
    indices for both models: `paired_diff`.
    """
    crop_model, squish_model = MODELS[branch]
    labels = rows["label"].to_numpy()
    crop_logits = rows[crop_model].to_numpy()
    squish_logits = rows[squish_model].to_numpy()
    result = paired_diff(labels, crop_logits, squish_logits, labels, n_boot=N_BOOT, seed=SEED)
    result["crop_auc"] = result.pop("a")
    result["squish_auc"] = result.pop("b")
    ci_low, ci_high = result["ci"]
    result["is_drop"] = bool(result["diff"] < -DROP and ci_high < 0)
    return result


def describe_fakes(fakes: pd.DataFrame, reals: pd.DataFrame, branch: str) -> dict:
    """Fakes-only numbers for one size bucket: mean logit and TPR@1%FPR,
    for the crop-trained and the squish-trained model of `branch`."""
    out = {}
    for policy, model in zip(("crop", "squish"), MODELS[branch]):
        fake_logits = fakes[model].to_numpy()
        real_logits = reals[model].to_numpy()
        out[policy] = {"mean_logit": float(fake_logits.mean()),
                       **tpr_at_fpr(real_logits, fake_logits)}
    return out


# ---------------------------------------------------------------------------
# Analyses

def per_generator(sb: pd.DataFrame) -> dict:
    """Analysis 1: each generator's fakes against all reals."""
    reals = sb[sb["label"] == 0]
    table = {}
    for generator in sorted(sb.loc[sb["label"] == 1, "generator"].unique()):
        fakes = sb[(sb["label"] == 1) & (sb["generator"] == generator)]
        subset = pd.concat([fakes, reals])
        entry = {"n_fake": int(len(fakes)), "n_real": int(len(reals)),
                 "short_min": int(fakes["short"].min()), "short_max": int(fakes["short"].max())}
        for branch in MODELS:
            entry[branch] = compare_on_subset(subset, branch)
        table[generator] = entry
    return table


def by_fake_size(sb: pd.DataFrame) -> dict:
    """Analysis 2: fakes split by native short side at 1100 px.

    The reals (all 833-835 px) are never split by size: every bucket is
    compared against the same full set of reals.
    """
    reals = sb[sb["label"] == 0]
    fakes = sb[sb["label"] == 1]
    buckets = {f"<= {BIG}": fakes[fakes["short"] <= BIG],
               f"> {BIG}": fakes[fakes["short"] > BIG]}
    table = {}
    for name, bucket_fakes in buckets.items():
        generators = bucket_fakes["generator"].value_counts().to_dict()
        entry = {"n_fake": int(len(bucket_fakes)), "n_real": int(len(reals)),
                 "generators": {g: int(n) for g, n in generators.items()}}
        subset = pd.concat([bucket_fakes, reals])
        for branch in MODELS:
            entry[branch] = {"auc": compare_on_subset(subset, branch),
                             "fakes": describe_fakes(bucket_fakes, reals, branch)}
        table[name] = entry
    return table


def ntire_check() -> dict:
    """Analysis 3: clean NTIRE fit-split rows above 1100 px (recomputes
    docs/ntire/paper/analysis_valsize.md as a check)."""
    table = {}
    for split in ("ntire_val", "ntire_val_hard"):
        for branch in MODELS:
            rows = load_ntire_large(split, branch)
            labels = rows["label"].to_numpy()
            result = paired_diff(labels, rows["crop"].to_numpy(), rows["squish"].to_numpy(),
                                 labels, n_boot=N_BOOT, seed=SEED)
            result.update(n=int(len(rows)), n_fake=int(labels.sum()))
            table.setdefault(split, {})[branch] = result
    return table


def verdict(generators: dict, sizes: dict) -> dict:
    """Apply the registered condition, per branch.

    Registered: "at least one generator or size bucket where squish's clean
    AUC is below crop's by more than 0.020 with a CI below zero, and no such
    drop outside it."

    Reading used here: list every generator and every size bucket with such
    a drop. The pass needs at least one, and all of them inside one location,
    meaning a single generator, or a single size bucket that contains every
    flagged generator's fakes. The flagged lists are reported in full so
    another reading can be checked against them.
    """
    result = {}
    for branch in MODELS:
        flagged_generators = [g for g, e in generators.items() if e[branch]["is_drop"]]
        flagged_buckets = [b for b, e in sizes.items() if e[branch]["auc"]["is_drop"]]

        if not flagged_generators and not flagged_buckets:
            location = None
        elif len(flagged_buckets) == 0 and len(flagged_generators) == 1:
            location = f"generator {flagged_generators[0]}"
        elif len(flagged_buckets) == 1:
            bucket_generators = set(sizes[flagged_buckets[0]]["generators"])
            all_inside = set(flagged_generators) <= bucket_generators
            location = f"size bucket {flagged_buckets[0]}" if all_inside else None
        else:
            location = None

        passes = location is not None
        result[branch] = {"flagged_generators": flagged_generators,
                          "flagged_size_buckets": flagged_buckets,
                          "location": location,
                          "pass": passes,
                          "reading": f"cost located: {location}" if passes else "cost unexplained"}
    return result


# ---------------------------------------------------------------------------
# Figure

def figure(generators: dict, pdf: str) -> None:
    """Per-generator clean AUC change (squish-trained - crop-trained), both branches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})

    names = list(generators)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for offset, branch, colour, marker in ((-0.15, "A", "#b2182b", "o"), (0.15, "B", "#2166ac", "s")):
        diffs = np.array([generators[g][branch]["diff"] for g in names])
        lower = diffs - np.array([generators[g][branch]["ci"][0] for g in names])
        upper = np.array([generators[g][branch]["ci"][1] for g in names]) - diffs
        ax.errorbar(diffs, positions + offset, xerr=[lower, upper], fmt=marker, ms=3,
                    color=colour, lw=0.9, capsize=1.5, label=f"branch {branch}")
    ax.axvline(0, color="0.5", lw=0.6)
    ax.axvline(-DROP, color="0.5", lw=0.6, ls=":")
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{g} ({generators[g]['short_min']}-{generators[g]['short_max']} px)"
                        for g in names])
    ax.invert_yaxis()
    ax.set_xlabel("clean AUC change, squish-trained - crop-trained (95% CI)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf)


# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ROWS_DIR, exist_ok=True)

    synthbuster = load_synthbuster()
    synthbuster.to_parquet(f"{ROWS_DIR}/test11_synthbuster_rows.parquet", index=False)

    generators = per_generator(synthbuster)
    sizes = by_fake_size(synthbuster)
    ntire = ntire_check()
    decision = verdict(generators, sizes)

    result = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT, "drop_threshold": DROP,
              "fpr": FPR, "size_cut_px": BIG, "models": MODELS,
              "synthbuster": {"n": int(len(synthbuster)),
                              "n_real": int((synthbuster["label"] == 0).sum()),
                              "n_fake": int((synthbuster["label"] == 1).sum()),
                              "real_short_range": [int(synthbuster.loc[synthbuster["label"] == 0, "short"].min()),
                                                   int(synthbuster.loc[synthbuster["label"] == 0, "short"].max())]},
              "per_generator": generators, "by_fake_size": sizes, "ntire_above_1100": ntire,
              "verdict": decision}

    figure(generators, f"{STEM}_fig.pdf")
    with open(f"{STEM}.json", "w") as f:
        json.dump(result, f, indent=2)

    for generator, entry in generators.items():
        print(f"{generator:22s} n={entry['n_fake']:4d}",
              *(f"{b} {entry[b]['diff']:+.4f} [{entry[b]['ci'][0]:+.4f},{entry[b]['ci'][1]:+.4f}]"
                for b in MODELS))
    for bucket, entry in sizes.items():
        print(f"fakes {bucket}: n={entry['n_fake']}",
              *(f"{b} auc {entry[b]['auc']['diff']:+.4f} tpr {entry[b]['fakes']['crop']['tpr']:.3f}"
                f"->{entry[b]['fakes']['squish']['tpr']:.3f}" for b in MODELS))
    for branch, v in decision.items():
        print(branch, v["reading"], "| generators", v["flagged_generators"], "| buckets", v["flagged_size_buckets"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
