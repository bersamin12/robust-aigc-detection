#!/usr/bin/env python3
"""Amendment 5, Test 2, analysis: D by model x condition x size bucket.

    python scripts/amend5_test2_analysis.py [--scores-dir DIR]

Reads the per-image parquets written by amend5_test2.py. Loads no model.

Statistic (registered). Within each (model x condition x size bucket) cell
of the primary source:

    D = AUC(squish) - AUC(native)

with a paired CI (`paper_analyses.paired_diff`: 2,000 resamples, seed
20260827, resampling within label; every row in a cell has the same
is_distorted, so label x is_distorted strata reduce to label strata).
"native" is `center` for our branches and the authors' five-crop for B-Free.
After damage every row is distorted, so this is the robust AUC; at "none"
it is the AUC over all rows, as registered.

"Damage level" averages D over the three op types (JPEG, blur, noise) at
each level. Pass rule, for both of our branches:
  (i)  the level-averaged D does not decrease none -> light -> medium ->
       heavy in at least 3 of the 4 size buckets, and
  (ii) D changes sign along that axis in at least one bucket.
Secondary: B-Free's level-averaged D rises with level.

Size buckets: native short side 80-336, 352-512, 528-736, 752+ px. Rows
outside them (short side < 80) are counted and left out of the bucketed
statistic. The secondary sources (SocialRF, Synthbuster) are not bucketed.

Writes docs/ntire/amend5/test2_regime-grid.json and _fig.pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

OUT_DIR = "docs/ntire/amend5"
STEM = f"{OUT_DIR}/test2_regime-grid"
MODELS = {"A2": "center", "B2swa": "center", "bfree": "native"}  # model -> its "native" presentation
OUR_BRANCHES = ("A2", "B2swa")
OPS = ("jpeg", "blur", "noise")
LEVELS = ("none", "light", "medium", "heavy")
BUCKETS = ((80, 336), (352, 512), (528, 736), (752, 100000))
BUCKET_NAMES = ("80-336", "352-512", "528-736", "752+")
SECONDARY = ("socialrf", "synthbuster")


def condition_name(op: str, level: str) -> str:
    return "none" if level == "none" else f"{op}_{level}"


def read_cell(scores_dir: str, model: str, source: str, condition: str) -> pd.DataFrame:
    frame = pd.read_parquet(f"{scores_dir}/{model}_{source}_{condition}.parquet")
    assert not frame["split"].eq("ntire_test").any(), "Test 2 never reads ntire_test"
    frame["short"] = frame[["width", "height"]].min(axis=1)
    return frame


def d_statistic(rows: pd.DataFrame, native: str) -> dict:
    """D = AUC(squish) - AUC(native) on these rows, with a paired CI.

    A cell with no rows, or only one class, has no AUC: D is None.
    """
    y = rows["label"].to_numpy().astype(int)
    if len(rows) == 0 or y.min() == y.max():
        return {"n": int(len(rows)), "n_fake": int(y.sum()), "D": None}
    result = paired_diff(y, rows[f"logit_{native}"].to_numpy(), rows["logit_squish"].to_numpy(), y)
    return {"n": int(len(rows)), "n_fake": int(y.sum()), "auc_native": result["a"],
            "auc_squish": result["b"], "D": result["diff"], "ci": result["ci"]}


def bucket_of(short: pd.Series) -> pd.Series:
    names = pd.Series(pd.NA, index=short.index, dtype="object")
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        names[(short >= lo) & (short <= hi)] = name
    return names


def primary_cells(scores_dir: str, model: str) -> dict:
    """D for every (condition, bucket) of the primary source, plus pooled over size."""
    native = MODELS[model]
    cells = {}
    for op in OPS:
        for level in LEVELS:
            condition = condition_name(op, level)
            if condition in cells:
                continue
            frame = read_cell(scores_dir, model, "ntire_clean", condition)
            frame["bucket"] = bucket_of(frame["short"])
            entry = {"pooled": d_statistic(frame, native),
                     "n_outside_buckets": int(frame["bucket"].isna().sum()), "buckets": {}}
            for name in BUCKET_NAMES:
                entry["buckets"][name] = d_statistic(frame[frame["bucket"] == name], native)
            cells[condition] = entry
    return cells


def level_curve(cells: dict, where: str) -> dict | None:
    """Level-averaged D (mean over op types) along none -> heavy, for one bucket or 'pooled'.

    None when the bucket has no AUC (e.g. no primary row has a short side
    of 80-336 px: clean NTIRE images start at 352 px).
    """
    def d_of(condition):
        e = cells[condition]
        return e["pooled"]["D"] if where == "pooled" else e["buckets"][where]["D"]
    if d_of("none") is None:
        return None
    curve = {}
    for level in LEVELS:
        if level == "none":
            curve[level] = d_of("none")
        else:
            curve[level] = float(np.mean([d_of(condition_name(op, level)) for op in OPS]))
    return curve


def non_decreasing(curve: dict) -> bool:
    values = [curve[level] for level in LEVELS]
    return all(b >= a for a, b in zip(values, values[1:]))


def changes_sign(curve: dict) -> bool:
    values = [curve[level] for level in LEVELS]
    return min(values) < 0 < max(values)


def judge_branch(cells: dict) -> dict:
    per_bucket = {}
    for name in BUCKET_NAMES:
        curve = level_curve(cells, name)
        if curve is None:
            # An empty bucket cannot show "D does not decrease"; it counts against the 3-of-4 rule.
            per_bucket[name] = {"curve": None, "non_decreasing": False, "changes_sign": False, "empty": True}
            continue
        per_bucket[name] = {"curve": curve, "non_decreasing": non_decreasing(curve),
                            "changes_sign": changes_sign(curve), "empty": False}
    n_monotone = sum(v["non_decreasing"] for v in per_bucket.values())
    any_sign_change = any(v["changes_sign"] for v in per_bucket.values())
    return {"per_bucket": per_bucket, "n_buckets_non_decreasing": n_monotone,
            "any_bucket_changes_sign": any_sign_change,
            "passes": bool(n_monotone >= 3 and any_sign_change),
            "pooled_curve": level_curve(cells, "pooled")}


def per_op_curves(cells: dict) -> dict:
    """D per op type along the levels, pooled over size (reported alongside)."""
    return {op: {level: cells[condition_name(op, level)]["pooled"]["D"] for level in LEVELS} for op in OPS}


def secondary(scores_dir: str, model: str) -> dict:
    out = {}
    native = MODELS[model]
    for source in SECONDARY:
        out[source] = {}
        for op in OPS:
            for level in LEVELS:
                condition = condition_name(op, level)
                if condition not in out[source]:
                    out[source][condition] = d_statistic(read_cell(scores_dir, model, source, condition), native)
    return out


def figure(res: dict, pdf_path: str) -> None:
    """Level-averaged D per size bucket, one panel per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    colours = ("#fdae61", "#f46d43", "#d73027", "#a50026")
    for ax, model in zip(axes, MODELS):
        judged = res["models"][model]["judged"]
        for name, colour in zip(BUCKET_NAMES, colours):
            curve = judged["per_bucket"][name]["curve"]
            if curve is None:
                continue
            ax.plot(range(len(LEVELS)), [curve[l] for l in LEVELS], marker="o", ms=3, color=colour, lw=1, label=name)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(LEVELS)
        ax.set_title({"A2": "Branch A", "B2swa": "Branch B", "bfree": "B-Free"}[model], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("D = AUC(squish) - AUC(native)")
    axes[0].legend(frameon=False, fontsize=6, title="short side", title_fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", default="/scratch/dion_r13/dion_extensions/outputs/test2/scores")
    ap.add_argument("--stem", default=STEM)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.stem), exist_ok=True)

    res = {"commit": commit(), "seed": SEED, "scores_dir": a.scores_dir, "models": {}}
    for model in MODELS:
        cells = primary_cells(a.scores_dir, model)
        res["models"][model] = {"primary": cells, "judged": judge_branch(cells),
                                "per_op_pooled": per_op_curves(cells), "secondary": secondary(a.scores_dir, model)}

    branch_results = {b: res["models"][b]["judged"]["passes"] for b in OUR_BRANCHES}
    bfree_pooled = res["models"]["bfree"]["judged"]["pooled_curve"]
    res["verdict"] = {
        "regime_rule_holds": bool(all(branch_results.values())),
        "per_branch": branch_results,
        "bfree_D_rises_with_level_pooled": non_decreasing(bfree_pooled),
        "bfree_buckets_non_decreasing": res["models"]["bfree"]["judged"]["n_buckets_non_decreasing"],
    }
    # L5 prediction: JPEG's D rises more steeply (heavy - none) than noise's, pooled, on both branches.
    l5 = {}
    for b in OUR_BRANCHES:
        per_op = res["models"][b]["per_op_pooled"]
        rise_jpeg = per_op["jpeg"]["heavy"] - per_op["jpeg"]["none"]
        rise_noise = per_op["noise"]["heavy"] - per_op["noise"]["none"]
        l5[b] = {"rise_jpeg": rise_jpeg, "rise_noise": rise_noise, "holds": bool(rise_jpeg > rise_noise)}
    res["verdict"]["L5_prediction"] = l5

    figure(res, f"{a.stem}_fig.pdf")
    with open(f"{a.stem}.json", "w") as f:
        json.dump(res, f, indent=2)
    for model in MODELS:
        judged = res["models"][model]["judged"]
        print(model, "monotone buckets", judged["n_buckets_non_decreasing"], "sign change", judged["any_bucket_changes_sign"],
              "pooled", {k: round(v, 4) for k, v in judged["pooled_curve"].items()})
        for name in BUCKET_NAMES:
            c = judged["per_bucket"][name]["curve"]
            print("   ", name, "empty" if c is None else {k: round(v, 4) for k, v in c.items()})
    print("verdict", res["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
