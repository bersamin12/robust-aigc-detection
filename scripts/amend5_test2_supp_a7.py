#!/usr/bin/env python3
"""Amendment 7, re-analysis R1 of Test 2: intervals on the level-averaged D.

    python scripts/amend5_test2_supp_a7.py

A supplement. It changes no verdict, and it does not touch Test 2's json or
md. It reads Test 2's per-image parquets and loads no model.

What Test 2 reported without an interval. Test 2's statistic is

    D(c) = AUC(squish, c) - AUC(native, c)

per condition c and size bucket ("native" is `center` for our branches and
B-Free's five-crop for B-Free). Its pass rule reads the level average

    L(level) = mean over op types (JPEG, blur, noise) of D(op, level),
    L(none)  = D(none),

and asks whether L does not decrease none -> light -> medium -> heavy and
whether L changes sign. Test 2 gave a paired interval for each single
condition, but not for L, for the steps of the curve, or for the sign
change. This script computes them, as registered in amendment 7 (R1):

  - L(level) with an interval, per bucket and pooled over size;
  - the steps L(light) - L(none), L(medium) - L(light), L(heavy) - L(medium),
    and the total rise L(heavy) - L(none), each with an interval;
  - the sign-change cell: the most negative and most positive level of L
    in each bucket, each with its interval. The crossing is "resolved" only
    if the negative level's interval lies below zero and the positive
    level's lies above zero.

All of these come from ONE set of bootstrap draws per (model, bucket): every
condition's AUCs are computed on the same resampled rows, so the average and
the differences between levels get honest intervals. Rows are aligned across
the ten condition parquets by path (asserted).

The L5 prediction (JPEG's D rises more steeply than noise's) is reported as
untestable, as amendment 7 registers: the three op types' "heavy" levels are
not matched in damage strength, so a steeper rise for one op says nothing
about the op type itself. To show the mismatch, the script measures the
pixel RMSE (0-255 scale, native resolution) that each condition causes, on
a seeded sample of the primary images, with Test 2's own `damage` function.
The per-op rises are still listed, with intervals, as a description.

Bootstrap: the index scheme of paper_analyses.paired_diff (seed 20260827,
2,000 draws, resampling within label; every row of a condition has the same
damage status, so label x is_distorted reduces to label). The script asserts
that every single-condition interval it computes equals the one in Test 2's
json (which came from paired_diff), and that the pooled D(jpeg_heavy)
interval equals a fresh paired_diff call.

Writes docs/ntire/amend5/test2_supp_a7.json.
"""
from __future__ import annotations

import json
import os
import sys

# The RMSE pass imports amend5_test2, which imports torch. This script must
# never touch the GPU (a training job shares the box).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test2_analysis import (BUCKET_NAMES, LEVELS, MODELS, OPS, bucket_of,  # noqa: E402
                                   condition_name, read_cell)
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

SCORES_DIR = "/scratch/dion_r13/dion_extensions/outputs/test2/scores"
TEST2_JSON = "docs/ntire/amend5/test2_regime-grid.json"
OUT = "docs/ntire/amend5/test2_supp_a7.json"
N_BOOT = 2000
CONDITIONS = ["none"] + [condition_name(op, level) for op in OPS for level in LEVELS if level != "none"]
DAMAGED_LEVELS = ("light", "medium", "heavy")
RMSE_SAMPLE = 200
RMSE_SEED = 20261019


def load_model(model: str) -> pd.DataFrame:
    """One frame per model: path, label, size bucket, and native/squish logits per condition."""
    native = MODELS[model]
    base = read_cell(SCORES_DIR, model, "ntire_clean", "none")
    frame = base[["path", "label", "short"]].copy()
    frame["bucket"] = bucket_of(frame["short"])
    for condition in CONDITIONS:
        cell = read_cell(SCORES_DIR, model, "ntire_clean", condition)
        assert (cell["path"].to_numpy() == frame["path"].to_numpy()).all(), (model, condition)
        assert (cell["label"].to_numpy() == frame["label"].to_numpy()).all(), (model, condition)
        frame[f"native_{condition}"] = cell[f"logit_{native}"].to_numpy()
        frame[f"squish_{condition}"] = cell["logit_squish"].to_numpy()
    return frame


def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def statistics(frame: pd.DataFrame, index: np.ndarray | None = None) -> dict[str, float]:
    """D per condition, the level average L, its steps, and per-op rises, on one set of rows."""
    rows = frame if index is None else frame.iloc[index]
    y = rows["label"].to_numpy().astype(int)
    out = {}
    for condition in CONDITIONS:
        auc_squish = auc(y, rows[f"squish_{condition}"].to_numpy())
        auc_native = auc(y, rows[f"native_{condition}"].to_numpy())
        out[f"D_{condition}"] = auc_squish - auc_native

    # Level average, as Test 2 defines it: D(none) alone, else the mean over op types.
    out["L_none"] = out["D_none"]
    for level in DAMAGED_LEVELS:
        per_op = [out[f"D_{condition_name(op, level)}"] for op in OPS]
        out[f"L_{level}"] = float(np.mean(per_op))

    # Steps along the curve, and the total rise.
    for previous, level in zip(LEVELS, LEVELS[1:]):
        out[f"step_{previous}_to_{level}"] = out[f"L_{level}"] - out[f"L_{previous}"]
    out["rise_none_to_heavy"] = out["L_heavy"] - out["L_none"]

    # Per-op rise (descriptive; the L5 comparison is untestable, see the docstring).
    for op in OPS:
        out[f"rise_{op}"] = out[f"D_{op}_heavy"] - out["D_none"]
    out["rise_noise_minus_jpeg"] = out["rise_noise"] - out["rise_jpeg"]
    return out


def with_intervals(frame: pd.DataFrame) -> dict:
    """Point estimate and 95% percentile interval of every statistic, from one draw set."""
    point = statistics(frame)
    strata = frame["label"].to_numpy().astype(int)
    draws = [statistics(frame, k) for k in draw_indices(strata)]
    result = {"n": int(len(frame)), "n_fake": int(strata.sum())}
    for name, value in point.items():
        values = np.array([draw[name] for draw in draws])
        lo, hi = np.nanquantile(values, [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


def above_zero(stat: dict) -> bool:
    return stat["ci"][0] > 0


def below_zero(stat: dict) -> bool:
    return stat["ci"][1] < 0


def curve_reading(cell: dict) -> dict:
    """What the intervals say about Test 2's two curve properties in one bucket."""
    steps = [f"step_{previous}_to_{level}" for previous, level in zip(LEVELS, LEVELS[1:])]
    levels_by_point = sorted(LEVELS, key=lambda level: cell[f"L_{level}"]["point"])
    lowest, highest = levels_by_point[0], levels_by_point[-1]
    sign_change_point = cell[f"L_{lowest}"]["point"] < 0 < cell[f"L_{highest}"]["point"]
    return {
        "steps_above_zero": [s for s in steps if above_zero(cell[s])],
        "steps_below_zero": [s for s in steps if below_zero(cell[s])],
        "rise_above_zero": above_zero(cell["rise_none_to_heavy"]),
        "levels_below_zero": [level for level in LEVELS if below_zero(cell[f"L_{level}"])],
        "levels_above_zero": [level for level in LEVELS if above_zero(cell[f"L_{level}"])],
        "lowest_level": lowest,
        "highest_level": highest,
        "sign_change_point_estimate": bool(sign_change_point),
        "sign_change_resolved": bool(below_zero(cell[f"L_{lowest}"]) and above_zero(cell[f"L_{highest}"])),
    }


def check_against_test2(model: str, where: str, cell: dict, test2: dict) -> None:
    """Every single-condition D and interval must equal Test 2's (paired_diff's)."""
    for condition in CONDITIONS:
        entry = test2["models"][model]["primary"][condition]
        reported = entry["pooled"] if where == "pooled" else entry["buckets"][where]
        assert np.isclose(cell[f"D_{condition}"]["point"], reported["D"]), (model, where, condition)
        assert np.allclose(cell[f"D_{condition}"]["ci"], reported["ci"]), (model, where, condition)


def damage_rmse() -> dict:
    """Pixel RMSE (0-255) between each damaged image and its clean original, native scale.

    A seeded sample of the primary images, damaged with Test 2's own
    `damage` function (same op parameters, same per-image noise seed).
    """
    from amend5_test2 import damage, load_sources
    from public_detector_probe import decode

    primary = load_sources()["ntire_clean"]
    assert not primary["split"].eq("ntire_test").any(), "R1 never reads ntire_test"
    sample = primary.sample(n=RMSE_SAMPLE, random_state=RMSE_SEED)
    per_condition = {condition: [] for condition in CONDITIONS if condition != "none"}
    for path, sha in zip(sample["path"], sample["content_sha256"]):
        clean_pixels = decode(path)
        clean = clean_pixels.astype(np.float64)
        for condition in per_condition:
            damaged = damage(clean_pixels.copy(), condition, sha).astype(np.float64)
            per_condition[condition].append(float(np.sqrt(np.mean((damaged - clean) ** 2))))
    summary = {}
    for condition, values in per_condition.items():
        values = np.array(values)
        summary[condition] = {"mean": float(values.mean()), "median": float(np.median(values)),
                              "p10": float(np.quantile(values, 0.10)), "p90": float(np.quantile(values, 0.90))}
    return {"n_images": RMSE_SAMPLE, "sample_seed": RMSE_SEED, "scale": "0-255, native resolution",
            "per_condition": summary}


def main() -> int:
    with open(TEST2_JSON) as handle:
        test2 = json.load(handle)
    report = {"supplement": "amendment 7, R1 (Test 2)", "changes_verdict": False, "commit": commit(),
              "seed": SEED, "n_boot": N_BOOT, "scores_dir": SCORES_DIR, "models": {}}

    for model in MODELS:
        frame = load_model(model)

        # The index scheme must be paired_diff's: same interval for one difference.
        y = frame["label"].to_numpy().astype(int)
        reference = paired_diff(y, frame["native_jpeg_heavy"].to_numpy(), frame["squish_jpeg_heavy"].to_numpy(), y)
        pooled = with_intervals(frame)
        assert np.allclose(pooled["D_jpeg_heavy"]["ci"], reference["ci"]), (pooled["D_jpeg_heavy"], reference)
        check_against_test2(model, "pooled", pooled, test2)
        pooled["reading"] = curve_reading(pooled)

        buckets = {}
        for name in BUCKET_NAMES:
            in_bucket = frame[frame["bucket"] == name].reset_index(drop=True)
            if len(in_bucket) == 0:
                buckets[name] = {"n": 0, "empty": True}
                continue
            cell = with_intervals(in_bucket)
            check_against_test2(model, name, cell, test2)
            cell["reading"] = curve_reading(cell)
            buckets[name] = cell
        report["models"][model] = {"pooled": pooled, "buckets": buckets}
        for name, cell in [("pooled", pooled)] + list(buckets.items()):
            if cell.get("empty"):
                continue
            print(model, name, {level: round(cell[f"L_{level}"]["point"], 4) for level in LEVELS},
                  "rise", [round(v, 4) for v in cell["rise_none_to_heavy"]["ci"]],
                  "sign change resolved", cell["reading"]["sign_change_resolved"], flush=True)

    report["L5"] = {"status": "untestable",
                    "reason": "damage levels are not matched across op types (see damage_rmse)",
                    "damage_rmse": damage_rmse()}
    for condition in ("jpeg_heavy", "blur_heavy", "noise_heavy"):
        print(condition, "RMSE mean", round(report["L5"]["damage_rmse"]["per_condition"][condition]["mean"], 2))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"R1 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
