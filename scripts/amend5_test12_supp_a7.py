#!/usr/bin/env python3
"""Amendment 7, re-analysis R5 (supplement to Test 12). Changes no verdict.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/amend5_test12_supp_a7.py

Registered (docs/ntire/paper_prereg.md, Amendment 7, "Re-analyses", R5):
"the recomputed-AUC count is corrected; E2 and E3 get intervals; E10 is
withdrawn."

1. Count. test12_writing.md says "All 66 recomputed AUCs match". The count is
   re-derived from test12_writing.json: how many table rows were scored, how
   many of them have an AUC printed in main.tex to compare against, how many
   AUC values were compared, and how many matched.
2. E2 interval. Robust TPR@1%FPR (distorted ntire_val_hard rows, threshold
   set on the distorted reals), squish at test - centre window, frozen
   crop-trained branches A2 and B2swa (the probe parquets Test 12 used).
3. E3 interval. Pooled TPR@1%FPR on SocialRF, squish fusion - crop fusion
   (`fused_z`), as distributed and after JPEG 95 (the two variants E3
   quotes).
4. E10: withdrawn; nothing is computed for it (see the .md).

TPR is score_plan_splits.tpr_at_fpr, threshold re-derived on each draw's
reals. Test 12 used aigcdet.eval.metrics.tpr_at_fpr (lowest ROC threshold
with FPR <= 1%); its point values are recomputed here next to the new ones
so the two definitions can be compared.

Bootstrap: the index scheme of paper_analyses.paired_diff (seed 20260827,
default_rng, cells in np.unique(strata) order, rng.integers within each
cell, 2000 draws, nanquantile 2.5 / 97.5), strata label x is_distorted,
every statistic on the same draw. Each data set asserts that its pooled AUC
difference reproduces paired_diff's interval exactly.

Writes docs/ntire/amend5/test12_supp_a7.json. The .md is written by hand
from it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analyses import SEED, auc, commit, paired_diff  # noqa: E402
from score_plan_splits import tpr_at_fpr  # noqa: E402

from aigcdet.eval.metrics import tpr_at_fpr as test12_tpr_at_fpr  # noqa: E402

OUT_JSON = "docs/ntire/amend5/test12_supp_a7.json"
TEST12_JSON = "docs/ntire/amend5/test12_writing.json"
SCORES = "outputs/scores"
#: Fused per-image parquets, git-ignored, only in Justin's checkout (as in Test 12). Read-only.
FUSED = "/mnt/r13justinssd/techjam/aigcdet/docs/ntire/paper"
N_BOOT = 2000


# ---------------------------------------------------------------------------
# 1. The recomputed-AUC count

def recount() -> dict:
    """Rows scored, rows with a printed AUC, AUC values compared, matches."""
    with open(TEST12_JSON) as f:
        test12 = json.load(f)
    rows = test12["tables"]
    rows_with_printed_auc = [r for r in rows if r["paper_check"]]
    rows_without_printed_auc = [f"{r['table']} {r['row']} {r['split']}" for r in rows if not r["paper_check"]]
    compared = [check for r in rows for check in r["paper_check"].values()]
    matched = [check for check in compared if check["matches"]]
    per_table = {}
    for r in rows:
        entry = per_table.setdefault(r["table"], {"rows": 0, "rows_with_printed_auc": 0, "auc_values": 0})
        entry["rows"] += 1
        entry["rows_with_printed_auc"] += int(bool(r["paper_check"]))
        entry["auc_values"] += len(r["paper_check"])
    return {"rows_scored": len(rows), "rows_with_printed_auc": len(rows_with_printed_auc),
            "auc_values_compared": len(compared), "auc_values_matched": len(matched),
            "per_table": per_table, "rows_without_printed_auc": rows_without_printed_auc}


# ---------------------------------------------------------------------------
# Bootstrap

def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def label_by_distortion(frame: pd.DataFrame) -> np.ndarray:
    """Stratum code: label x is_distorted (0..3)."""
    return frame["label"].to_numpy().astype(int) * 2 + frame["is_distorted"].to_numpy().astype(int)


def statistics(frame: pd.DataFrame, base: str, other: str, rows_for_tpr: str,
               index: np.ndarray | None = None) -> dict[str, float]:
    """TPR of `base` and `other` on the rows named by `rows_for_tpr`, their
    difference (other - base), and the pooled AUC difference on all rows."""
    rows = frame if index is None else frame.iloc[index]
    labels = rows["label"].to_numpy().astype(int)
    if rows_for_tpr == "distorted":
        keep = rows["is_distorted"].to_numpy() == 1
    else:
        keep = np.ones(len(rows), dtype=bool)
    base_tpr = tpr_at_fpr(labels[keep], rows[base].to_numpy()[keep])
    other_tpr = tpr_at_fpr(labels[keep], rows[other].to_numpy()[keep])
    pooled_auc_diff = auc(labels, rows[other].to_numpy()) - auc(labels, rows[base].to_numpy())
    return {"base_tpr": base_tpr, "other_tpr": other_tpr, "tpr_diff": other_tpr - base_tpr,
            "pooled_auc_diff": pooled_auc_diff}


def compare(frame: pd.DataFrame, base: str, other: str, rows_for_tpr: str) -> dict:
    """Point values, intervals, the paired_diff check, and Test 12's TPR definition."""
    assert not frame["path"].astype(str).str.contains("ntire_test").any(), "R5 never reads ntire_test"
    strata = label_by_distortion(frame)
    point = statistics(frame, base, other, rows_for_tpr)
    draws = [statistics(frame, base, other, rows_for_tpr, index) for index in draw_indices(strata)]
    result = {"n": int(len(frame)), "n_fake": int(frame["label"].sum()), "rows_for_tpr": rows_for_tpr}
    for name, value in point.items():
        values = np.array([draw[name] for draw in draws])
        lo, hi = np.nanquantile(values, [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}

    labels = frame["label"].to_numpy().astype(int)
    reference = paired_diff(labels, frame[base].to_numpy(), frame[other].to_numpy(), strata,
                            n_boot=N_BOOT, seed=SEED)
    assert np.allclose(result["pooled_auc_diff"]["ci"], reference["ci"], rtol=0, atol=1e-12), (result, reference)
    result["paired_diff_check"] = reference

    if rows_for_tpr == "distorted":
        keep = frame["is_distorted"].to_numpy() == 1
    else:
        keep = np.ones(len(frame), dtype=bool)
    result["n_real_for_threshold"] = int((labels[keep] == 0).sum())
    result["test12_definition_point"] = {
        "base_tpr": test12_tpr_at_fpr(labels[keep], frame[base].to_numpy()[keep]),
        "other_tpr": test12_tpr_at_fpr(labels[keep], frame[other].to_numpy()[keep])}
    lo, hi = result["tpr_diff"]["ci"]
    result["interval_excludes_zero"] = bool(lo > 0 or hi < 0)
    return result


# ---------------------------------------------------------------------------
# 2. E2 and 3. E3

def e2() -> dict:
    """Robust TPR, squish at test vs centre window, per frozen crop-trained branch."""
    out = {}
    for branch in ("A2", "B2swa"):
        centre = pd.read_parquet(f"{SCORES}/probe_{branch}_center_ntire_val_hard.parquet")
        squish = pd.read_parquet(f"{SCORES}/probe_{branch}_squish_ntire_val_hard.parquet")
        centre = centre[["path", "label", "is_distorted", "logit"]].rename(columns={"logit": "centre"})
        squish = squish[["path", "logit"]].rename(columns={"logit": "squish"})
        joined = centre.merge(squish, on="path", how="inner", validate="1:1")
        assert len(joined) == len(centre) == 2500, branch
        out[branch] = compare(joined, base="centre", other="squish", rows_for_tpr="distorted")
    return out


def e3() -> dict:
    """Pooled TPR on SocialRF, squish fusion vs crop fusion, as distributed and JPEG 95."""
    out = {}
    for split in ("ext_aigi_socialrf", "ext_aigi_socialrf_jpeg95"):
        crop = pd.read_parquet(f"{FUSED}/fusion_crop_{split}_fused.parquet")
        squish = pd.read_parquet(f"{FUSED}/fusion_squish_{split}_fused.parquet")
        assert (crop["split"] == split).all() and (squish["split"] == split).all()
        crop = crop[["path", "label", "is_distorted", "fused_z"]].rename(columns={"fused_z": "crop"})
        squish = squish[["path", "fused_z"]].rename(columns={"fused_z": "squish"})
        joined = crop.merge(squish, on="path", how="inner", validate="1:1")
        assert len(joined) == len(crop), split
        out[split] = compare(joined, base="crop", other="squish", rows_for_tpr="all")
    return out


def main() -> int:
    result = {"commit": commit(), "seed": SEED, "n_boot": N_BOOT,
              "strata": "label x is_distorted",
              "tpr": "score_plan_splits.tpr_at_fpr, 1% FPR, threshold re-derived per draw",
              "count": recount(), "E2": e2(), "E3": e3(),
              "E10": "withdrawn by amendment 7; nothing computed"}
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(json.dumps({k: v for k, v in result["count"].items() if k != "rows_without_printed_auc"}, indent=1))
    for edit in ("E2", "E3"):
        for name, entry in result[edit].items():
            print(edit, name, *(f"{k} {entry[k]['point']:+.4f} [{entry[k]['ci'][0]:+.4f}, {entry[k]['ci'][1]:+.4f}]"
                                for k in ("base_tpr", "other_tpr", "tpr_diff", "pooled_auc_diff")),
                  "test12-def", entry["test12_definition_point"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
