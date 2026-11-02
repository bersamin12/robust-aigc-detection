#!/usr/bin/env python3
"""Amendment 5, Test 6, val part: fit and freeze the two-view routed fusion.

    python scripts/amend5_test6_fit.py
    python scripts/amend5_test6_fit.py --rows-dir DIR --docs-dir DIR   # smoke test

Registered rule (docs/ntire/paper_prereg.md, "Amendment 5", Test 6). Per
image, per branch:

    s = w * z_squish + (1 - w) * z_grid9,   w = sigmoid(a + b * d)

d is the branch's predicted overall damage severity on the squish view, and
each z is a z-score fitted on `ntire_val`. The two branches are then fused as
in the paper (z-score each branch's s, sweep the branch weight; the
`eval.late_fusion` code the paper's fusions use). (a, b) are chosen by grid
search on `ntire_val` only, a in [-4, 4], b in [0, 8], step 0.25, by
mean(robust AUC, clean AUC).

Choices the registration leaves open, fixed here before any routed number
exists, and disclosed in the md:
  * "overall damage severity" d = the MAX over the degradation head's six
    per-family severities. The head's per-family target is the harshest op
    of that family in the chain (augment/recipes.py), so the harshest
    family is the same reading one level up.
  * (a, b) are fitted PER BRANCH, on that branch's own routed score s,
    because each branch has its own degradation head and its own d scale.
  * Grid ties: the smallest b (least routing), then the a closest to 0.

Inputs (all fit-split; this script refuses ntire_test by name):
  outputs/test6/test6_squish_<tag>_<split>.parquet   squish logit + severity
                                                     (amend5_test6_score.py)
  outputs/test9/grid9_features_<tag>_<split>.npz     every grid9 window logit
                                                     (Test 9); mean = grid9
  docs/ntire/fusion_squish_valfit.json               "squish alone" fusion
  docs/ntire/fusion_ntire2_final.json + outputs/scores/{A2,B2swa}_<split>
                                                     read 1's crop fusion

Writes:
  docs/ntire/amend5/test6_frozen.json   everything read 3 needs, frozen.
                                        Dion commits it before read 3.
  docs/ntire/amend5/test6_val.json      fit-split results: condition 1 on
                                        ntire_val_hard and the "also
                                        reported" clean > 1100 px rows.
  outputs/test6/test6_rows_<split>.parquet   per-image scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc, paired_bootstrap  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402
from aigcdet.augment.recipes import FAMILIES  # noqa: E402
from aigcdet.eval.late_fusion import (  # noqa: E402
    Fusion, branch_metrics, fit_fusion, fusion_objective, zscore_fit,
)

#: Output locations; main() can point both elsewhere for a smoke test.
OUT_DOCS = "docs/ntire/amend5"
OUT_ROWS = "/scratch/dion_r13/dion_extensions/outputs/test6"
GRID9_DIR = "/scratch/dion_r13/dion_extensions/outputs/test9"
FIT_SPLIT = "ntire_val"
EVAL_SPLIT = "ntire_val_hard"
BRANCHES = {"A": "A2", "B": "B2swa"}
#: Registered grid, both ends included.
A_GRID = np.round(np.arange(-4.0, 4.0 + 1e-9, 0.25), 4)
B_GRID = np.round(np.arange(0.0, 8.0 + 1e-9, 0.25), 4)
#: Clean-cost rows of condition 2 and of the "also reported" comparisons.
LARGE_SHORT_SIDE = 1100
SEVERITY_COLUMNS = [f"sev_{family}" for family in FAMILIES]
SQUISH_ALONE_RECORD = "docs/ntire/fusion_squish_valfit.json"
CROP_FUSION_RECORD = "docs/ntire/fusion_ntire2_final.json"


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def grid9_logit_per_image(tag: str, split: str) -> pd.DataFrame:
    """Mean of each image's grid9 window logits: exactly the probe's grid9
    score (Test 9 checked this against the frozen grid9 parquet)."""
    path = f"{GRID9_DIR}/grid9_features_{tag}_{split}.npz"
    saved = np.load(path, allow_pickle=True)
    window_logit, offsets = saved["window_logit"], saved["offsets"]
    image_logit = np.array([window_logit[offsets[i]:offsets[i + 1]].mean()
                            for i in range(len(offsets) - 1)])
    return pd.DataFrame({"path": saved["path"], "grid9_logit": image_logit})


def load_branch(tag: str, split: str) -> pd.DataFrame:
    """One branch, one split: squish logit, overall severity d, grid9 logit."""
    assert split != "ntire_test", "Test 6's val part never reads ntire_test"
    squish = pd.read_parquet(f"{OUT_ROWS}/test6_squish_{tag}_{split}.parquet")
    squish = squish.rename(columns={"logit": "squish_logit"})
    squish["d"] = squish[SEVERITY_COLUMNS].max(axis=1)
    grid9 = grid9_logit_per_image(tag, split)
    both = squish.merge(grid9, on="path", validate="1:1")
    assert len(both) == len(squish) == len(grid9), f"{tag} {split}: views cover different rows"
    return both


def routed_score(rows: pd.DataFrame, z_squish, z_grid9, a: float, b: float) -> np.ndarray:
    """s = w * z_squish + (1 - w) * z_grid9 with w = sigmoid(a + b * d)."""
    w = sigmoid(a + b * rows["d"].to_numpy())
    return (w * z_squish.apply(rows["squish_logit"])
            + (1.0 - w) * z_grid9.apply(rows["grid9_logit"]))


def fit_routing(fit_rows: pd.DataFrame) -> dict:
    """Z-scores and (a, b) for one branch, on the fit split only."""
    z_squish = zscore_fit(fit_rows["squish_logit"])
    z_grid9 = zscore_fit(fit_rows["grid9_logit"])
    y = fit_rows["label"].to_numpy()
    dist = fit_rows["is_distorted"].to_numpy()

    objective = np.empty((len(A_GRID), len(B_GRID)))
    for i, a in enumerate(A_GRID):
        for j, b in enumerate(B_GRID):
            s = routed_score(fit_rows, z_squish, z_grid9, float(a), float(b))
            objective[i, j] = fusion_objective(branch_metrics(y, s, dist))

    best_value = objective.max()
    tied = [(float(A_GRID[i]), float(B_GRID[j]))
            for i, j in zip(*np.nonzero(objective == best_value))]
    # Ties: least routing first (smallest b), then a closest to zero.
    a_best, b_best = min(tied, key=lambda ab: (ab[1], abs(ab[0])))
    return {"z_squish": z_squish, "z_grid9": z_grid9, "a": a_best, "b": b_best,
            "objective": float(best_value), "n_tied": len(tied),
            "objective_grid": objective}


def frozen_fusion_score(record_path: str, score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
    with open(record_path) as f:
        record = json.load(f)
    return Fusion.from_record(record["fusion"]).apply(score_a, score_b)


def centre_logits(tag: str, split: str) -> pd.DataFrame:
    """The as-trained centre-window score: read 1's crop presentation."""
    rows = pd.read_parquet(f"outputs/scores/{tag}_{split}.parquet")[["path", "logit"]]
    return rows.rename(columns={"logit": f"centre_{tag}"})


def build_split(split: str, routing: dict, branch_fusion: Fusion | None) -> pd.DataFrame:
    """Every score Test 6 compares, for one split, one row per image."""
    joined = None
    for branch, tag in BRANCHES.items():
        rows = load_branch(tag, split)
        r = routing[branch]
        rows[f"routed_{branch}"] = routed_score(rows, r["z_squish"], r["z_grid9"], r["a"], r["b"])
        rows = rows.rename(columns={"squish_logit": f"squish_{branch}",
                                    "grid9_logit": f"grid9_{branch}", "d": f"d_{branch}"})
        keep = ["path", f"squish_{branch}", f"grid9_{branch}", f"d_{branch}", f"routed_{branch}"]
        if joined is None:
            keep = ["label", "is_distorted", "width", "height"] + keep
            joined = rows[keep]
        else:
            joined = joined.merge(rows[keep], on="path", validate="1:1")
    for tag in BRANCHES.values():
        joined = joined.merge(centre_logits(tag, split), on="path", validate="1:1")
    joined["short"] = joined[["width", "height"]].min(axis=1)
    joined["squish_alone"] = frozen_fusion_score(SQUISH_ALONE_RECORD,
                                                 joined["squish_A"], joined["squish_B"])
    joined["crop_fusion"] = frozen_fusion_score(CROP_FUSION_RECORD,
                                                joined["centre_A2"], joined["centre_B2swa"])
    if branch_fusion is not None:
        joined["routed"] = branch_fusion.apply(joined["routed_A"], joined["routed_B"])
    return joined


def clean_large_comparison(rows: pd.DataFrame) -> dict:
    """Clean AUC on clean rows with native short side > 1100 px: routed
    fusion against read 1's crop fusion and against squish alone."""
    large_clean = rows[(rows["is_distorted"] == 0) & (rows["short"] > LARGE_SHORT_SIDE)]
    y = large_clean["label"].to_numpy()
    out = {"n": int(len(large_clean)), "n_fake": int(y.sum())}
    if len(set(y.tolist())) < 2:
        out["note"] = "one class only; no AUC"
        return out
    for column in ("routed", "crop_fusion", "squish_alone"):
        out[f"{column}_clean_auc"] = auc(y, large_clean[column].to_numpy())
    out["routed_minus_crop"] = out["routed_clean_auc"] - out["crop_fusion_clean_auc"]
    out["within_0.010_of_crop"] = bool(abs(out["routed_minus_crop"]) <= 0.010)
    return out


def main() -> int:
    global OUT_DOCS, OUT_ROWS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-dir", default=OUT_ROWS,
                    help="where the squish+severity parquets are read and row parquets written")
    ap.add_argument("--docs-dir", default=OUT_DOCS, help="where the two json files are written")
    args = ap.parse_args()
    OUT_ROWS, OUT_DOCS = args.rows_dir, args.docs_dir
    os.makedirs(OUT_DOCS, exist_ok=True)

    # 1. Routing per branch, fitted on ntire_val only.
    routing = {}
    for branch, tag in BRANCHES.items():
        fit_rows = load_branch(tag, FIT_SPLIT)
        routing[branch] = fit_routing(fit_rows)
        r = routing[branch]
        print(f"branch {branch}: a={r['a']:+.2f} b={r['b']:.2f} objective {r['objective']:.4f} "
              f"(ties {r['n_tied']})")

    # 2. Branch fusion of the routed scores, as in the paper, on ntire_val.
    fit_split_rows = build_split(FIT_SPLIT, routing, None)
    fusion_input = pd.DataFrame({"label": fit_split_rows["label"],
                                 "is_distorted": fit_split_rows["is_distorted"],
                                 "score_a": fit_split_rows["routed_A"],
                                 "score_b": fit_split_rows["routed_B"]})
    branch_fusion, weight_sweep = fit_fusion(fusion_input, f"split={FIT_SPLIT}")
    fit_split_rows["routed"] = branch_fusion.apply(fit_split_rows["routed_A"],
                                                   fit_split_rows["routed_B"])
    print(f"branch fusion weight {branch_fusion.weight}")

    # 3. Freeze. Everything read 3 needs, nothing it could tune.
    inputs = [SQUISH_ALONE_RECORD, CROP_FUSION_RECORD]
    for tag in BRANCHES.values():
        inputs.append(f"{OUT_ROWS}/test6_squish_{tag}_{FIT_SPLIT}.parquet")
        inputs.append(f"{GRID9_DIR}/grid9_features_{tag}_{FIT_SPLIT}.npz")
    frozen = {
        "test": "amendment 5, Test 6, routed fusion",
        "commit": commit(),
        "fit_split": FIT_SPLIT,
        "rule": "s = w*z_squish + (1-w)*z_grid9, w = sigmoid(a + b*d), per branch; "
                "branches fused by eval.late_fusion",
        "d_definition": "max over the degradation head's per-family severities "
                        f"{list(FAMILIES)}, on the squish view",
        "grid": {"a": [float(A_GRID[0]), float(A_GRID[-1]), 0.25],
                 "b": [float(B_GRID[0]), float(B_GRID[-1]), 0.25],
                 "objective": "mean(robust AUC, clean AUC) on ntire_val",
                 "ties": "smallest b, then a closest to 0", "fitted": "per branch"},
        "branches": {branch: {"checkpoint_tag": BRANCHES[branch],
                              "z_squish": {"mu": r["z_squish"].mu, "sigma": r["z_squish"].sigma},
                              "z_grid9": {"mu": r["z_grid9"].mu, "sigma": r["z_grid9"].sigma},
                              "a": r["a"], "b": r["b"],
                              "fit_objective": r["objective"], "n_grid_ties": r["n_tied"]}
                     for branch, r in routing.items()},
        "branch_fusion": branch_fusion.as_record(),
        "checkpoints": {
            "A2": "/scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt (final)",
            "B2swa": "/scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt (--swa)"},
        "views": {"squish": "ntire_crop_probe.views_squish",
                  "grid9": "ntire_crop_probe.views_grid9, mean of window logits"},
        "read3_condition": "clean ntire_test rows with native short side > 1100 px: routed "
                           "clean AUC within 0.010 of read 1's crop fusion "
                           f"({CROP_FUSION_RECORD}) on the same rows",
        "input_md5": {p: md5(p) for p in inputs},
    }
    with open(f"{OUT_DOCS}/test6_frozen.json", "w") as f:
        json.dump(frozen, f, indent=2)

    # 4. Fit-split results with the frozen parameters.
    eval_rows = build_split(EVAL_SPLIT, routing, branch_fusion)
    boot_frame = eval_rows[["path", "label", "is_distorted", "routed", "squish_alone",
                            "crop_fusion"]]
    condition_1 = paired_bootstrap(boot_frame, ["routed", "squish_alone", "crop_fusion"],
                                   baseline="squish_alone", n_boot=2000, seed=SEED)
    robust_vs_squish = condition_1["vs_baseline"]["routed"]["robust_auc"]
    condition_1_pass = robust_vs_squish["ci"][1] >= 0.0

    results = {
        "commit": commit(), "seed": SEED, "n_boot": 2000,
        "frozen": f"{OUT_DOCS}/test6_frozen.json",
        "fit_split_objective": {b: r["objective"] for b, r in routing.items()},
        "branch_weight_sweep": weight_sweep,
        "condition_1": {
            "split": EVAL_SPLIT,
            "note": "ntire_val_hard was used to choose squish: fit-split number",
            "routed_minus_squish_robust": robust_vs_squish,
            "pass": bool(condition_1_pass),
            "bootstrap": condition_1},
        "mean_w": {split: {b: float(sigmoid(routing[b]["a"] + routing[b]["b"]
                                            * rows[f"d_{b}"]).mean()) for b in BRANCHES}
                   for split, rows in ((FIT_SPLIT, fit_split_rows), (EVAL_SPLIT, eval_rows))},
        "also_reported_clean_large": {FIT_SPLIT: clean_large_comparison(fit_split_rows),
                                      EVAL_SPLIT: clean_large_comparison(eval_rows)},
        "not_yet_run": "external sets (as distributed, _jpeg95, _deg) need grid9 and squish "
                       "severity scored on them; Synthbuster's clean > 1100 comparison is "
                       "among them",
    }
    with open(f"{OUT_DOCS}/test6_val.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    for split, rows in ((FIT_SPLIT, fit_split_rows), (EVAL_SPLIT, eval_rows)):
        rows.to_parquet(f"{OUT_ROWS}/test6_rows_{split}.parquet", index=False)

    print(f"condition 1 ({EVAL_SPLIT}): routed - squish robust "
          f"{robust_vs_squish['diff']:+.4f} {robust_vs_squish['ci']} -> "
          f"{'PASS' if condition_1_pass else 'FAIL'}")
    for split, r in results["also_reported_clean_large"].items():
        print(f"clean > {LARGE_SHORT_SIDE} px on {split}: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
