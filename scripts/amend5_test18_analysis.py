#!/usr/bin/env python3
"""Amendment 10, Test 18, analysis: does the damage reduction from the shrink depend on k?

    python scripts/amend5_test18_analysis.py                     # full run
    python scripts/amend5_test18_analysis.py --smoke --scores .../smoke/scores --out .../smoke/t18.json

Reads the parquets of amend5_test18.py, plus Test 2's squish logits (clean
and native heavy damage) and Test 13's squish_post logits (heavy damage on
the squished view), joined on path, for the descriptive squish rows. Loads
no model.

Statistics (registered), per branch. Every row is a clean NTIRE image; in a
damaged view every image, real and fake, carries the same op, so the
"robust AUC" of a view is its AUC over all rows.

  R_op(k) = AUC(before(op, k)) - AUC(after(op, k))
            damage reduction from the shrink: positive when damaging first
            and shrinking afterwards helps, i.e. the shrink removed part of
            the damage (or rescaled it, see the amendment's confounds).
  R(k)    = mean over the 3 ops of R_op(k)                (the registered S(k))
  R(k_max) - R(2)
  clean AUC(k), and drop(k) = clean AUC(k) - damaged AUC, for each order.
At k = 1 the two orders are the same pixels, so R(1) = 0 exactly.

Conditions (95% intervals), per branch; pooled R is primary, the same rule on
each op's R_op is secondary:
  STRONG-ONLY  R(2)'s CI includes zero AND lies within [-0.01, +0.01]
               AND [R(k_max) - R(2)]'s CI is above zero
  EVERYWHERE   R(2)'s CI is above zero
  NONE         every R(k)'s CI includes zero or is below it
               AND [R(k_max) - R(2)]'s CI includes zero
  UNRESOLVED   otherwise
The output flags per-op verdicts that disagree with each other or with the
pooled one.

Bootstrap: amend5_test14_analysis.draw_indices (paired_diff's index scheme,
seed 20260827, 2,000 draws), stratified by label (every row is clean, so
label x is_distorted is label). Every statistic is computed on the same
draw. The script asserts that its R_jpeg(2) interval equals paired_diff's.

Joins asserted: both branches score the same rows in the same order; the full
run has the registered row counts; window counts are equal across clean,
before and after at each k; every row has exactly one Test 2 and one Test 13
squish logit.

Writes docs/ntire/amend5/test18_shrinkorder.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test14_analysis import N_BOOT, draw_indices  # noqa: E402
from amend5_test18 import OPS, OUT_ROOT, REGISTERED_OUTCOME  # noqa: E402
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

ROOT = "/scratch/dion_r13/dion_extensions/outputs"
TEST2_SCORES = f"{ROOT}/test2/scores"
TEST13_SCORES = f"{ROOT}/test13/scores"
OUT = "docs/ntire/amend5/test18_shrinkorder.json"
BRANCHES = ("B2swa", "A2")
KS = REGISTERED_OUTCOME["ks"]
K_MAX = max(KS)
EQUIVALENCE_MARGIN = 0.01
STATISTIC_LABEL = "damage reduction from the shrink, R_op(k) = AUC(before) - AUC(after)"


# ---------------------------------------------------------------- loading

def load_branch(branch: str, scores_dir: str, smoke: bool) -> dict[str, np.ndarray]:
    """Every view's image logit for one branch, the squish logits joined on path, and the labels."""
    frame = pd.read_parquet(f"{scores_dir}/{branch}_shrinkorder.parquet")
    assert not frame["split"].eq("ntire_test").any(), "Test 18 never reads ntire_test"
    assert frame["path"].is_unique
    if not smoke:
        counts = {"n": len(frame), "n_real": int((frame["label"] == 0).sum()),
                  "n_fake": int((frame["label"] == 1).sum())}
        assert counts == {key: REGISTERED_OUTCOME[key] for key in counts}, counts

    logits = {"path": frame["path"].to_numpy(), "label": frame["label"].to_numpy().astype(int)}
    for k in KS:
        logits[f"clean_k{k}"] = frame[f"logit_clean_k{k}"].to_numpy().astype(np.float64)
        clean_windows = frame[f"n_windows_clean_k{k}"].to_numpy()
        for op in OPS:
            before = f"before_{op}_k{k}"
            # k = 1: the orders are the same pixels. The full run scores them once; the
            # smoke run scores both, and they must agree exactly.
            after = f"after_{op}_k{k}" if f"logit_after_{op}_k{k}" in frame else before
            if k == 1 and after != before:
                assert np.array_equal(frame[f"logit_{before}"], frame[f"logit_{after}"]), (branch, op)
            assert (frame[f"n_windows_{before}"].to_numpy() == clean_windows).all(), (branch, before)
            assert (frame[f"n_windows_{after}"].to_numpy() == clean_windows).all(), (branch, after)
            logits[f"before_{op}_k{k}"] = frame[f"logit_{before}"].to_numpy().astype(np.float64)
            logits[f"after_{op}_k{k}"] = frame[f"logit_{after}"].to_numpy().astype(np.float64)
    logits.update(squish_logits(branch, frame["path"]))
    return logits


def squish_logits(branch: str, paths: pd.Series) -> dict[str, np.ndarray]:
    """Descriptive squish views on the same rows: Test 2 (clean, damage before) and Test 13 (damage after)."""
    def joined(parquet: str, column: str) -> np.ndarray:
        scored = pd.read_parquet(parquet, columns=["path", column])
        assert scored["path"].is_unique, parquet
        matched = paths.to_frame().merge(scored, on="path", how="left", validate="one_to_one")
        assert matched["path"].tolist() == paths.tolist() and matched[column].notna().all(), parquet
        return matched[column].to_numpy().astype(np.float64)

    out = {"squish_clean": joined(f"{TEST2_SCORES}/{branch}_ntire_clean_none.parquet", "logit_squish")}
    for op in OPS:
        out[f"squish_before_{op}"] = joined(f"{TEST2_SCORES}/{branch}_ntire_clean_{op}_heavy.parquet", "logit_squish")
        out[f"squish_after_{op}"] = joined(f"{TEST13_SCORES}/{branch}_ntire_clean_squish_post.parquet",
                                           f"logit_squish_post_{op}_heavy")
    return out


# ---------------------------------------------------------------- statistics

def statistics(y: np.ndarray, logits: dict[str, np.ndarray]) -> dict[str, float]:
    """Every reported statistic on one set of rows (the full rows, or one bootstrap draw)."""
    a = {name: auc(y, z) for name, z in logits.items() if name not in ("path", "label")}
    out = {f"auc_{name}": value for name, value in a.items()}
    for k in KS:
        for op in OPS:
            out[f"reduction_{op}_k{k}"] = a[f"before_{op}_k{k}"] - a[f"after_{op}_k{k}"]
            out[f"drop_before_{op}_k{k}"] = a[f"clean_k{k}"] - a[f"before_{op}_k{k}"]
            out[f"drop_after_{op}_k{k}"] = a[f"clean_k{k}"] - a[f"after_{op}_k{k}"]
        for prefix in ("reduction", "drop_before", "drop_after"):
            out[f"{prefix}_k{k}"] = float(np.mean([out[f"{prefix}_{op}_k{k}"] for op in OPS]))
    for scope in ("",) + tuple(f"{op}_" for op in OPS):
        out[f"reduction_{scope}kmax_minus_k2"] = out[f"reduction_{scope}k{K_MAX}"] - out[f"reduction_{scope}k2"]
    # Descriptive: the same before - after difference for squish's single view.
    for op in OPS:
        out[f"squish_reduction_{op}"] = a[f"squish_before_{op}"] - a[f"squish_after_{op}"]
    out["squish_reduction"] = float(np.mean([out[f"squish_reduction_{op}"] for op in OPS]))
    return out


def with_intervals(logits: dict[str, np.ndarray]) -> dict:
    y = logits["label"]
    point = statistics(y, logits)
    draws = [statistics(y[index], {name: z[index] for name, z in logits.items()}) for index in draw_indices(y)]
    result = {"n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((y == 0).sum())}
    for name, value in point.items():
        lo, hi = np.nanquantile(np.array([draw[name] for draw in draws]), [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}
    return result


# ---------------------------------------------------------------- verdicts

def includes_zero(ci: list[float]) -> bool:
    return ci[0] <= 0 <= ci[1]


def verdict(stats: dict, scope: str) -> str:
    """The registered rule on one scope: "" (pooled over ops, primary) or "<op>_" (secondary)."""
    r2 = stats[f"reduction_{scope}k2"]["ci"]
    rise = stats[f"reduction_{scope}kmax_minus_k2"]["ci"]
    r2_equivalent_to_zero = includes_zero(r2) and -EQUIVALENCE_MARGIN <= r2[0] and r2[1] <= EQUIVALENCE_MARGIN
    if r2_equivalent_to_zero and rise[0] > 0:
        return "STRONG-ONLY"
    if r2[0] > 0:
        return "EVERYWHERE"
    no_k_above_zero = all(stats[f"reduction_{scope}k{k}"]["ci"][0] <= 0 for k in KS)
    if no_k_above_zero and includes_zero(rise):
        return "NONE"
    return "UNRESOLVED"


def check_verdict_rule() -> None:
    """The rule on hand-made intervals, one case per outcome (runs before every analysis)."""
    def stats(r2, r4, rise):
        return {"reduction_k1": {"ci": [0.0, 0.0]}, "reduction_k2": {"ci": r2},
                f"reduction_k{K_MAX}": {"ci": r4}, "reduction_kmax_minus_k2": {"ci": rise}}
    assert verdict(stats([-0.004, 0.006], [0.01, 0.03], [0.005, 0.02]), "") == "STRONG-ONLY"
    assert verdict(stats([-0.004, 0.012], [0.01, 0.03], [0.005, 0.02]), "") == "UNRESOLVED"  # past the margin
    assert verdict(stats([0.002, 0.008], [0.01, 0.03], [0.005, 0.02]), "") == "EVERYWHERE"
    assert verdict(stats([-0.004, 0.006], [-0.003, 0.004], [-0.002, 0.003]), "") == "NONE"
    assert verdict(stats([-0.004, 0.006], [0.001, 0.009], [-0.002, 0.003]), "") == "UNRESOLVED"


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=f"{OUT_ROOT}/scores")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true", help="smoke rows: skip the registered row-count assert")
    a = ap.parse_args()
    check_verdict_rule()

    report = {"test": 18, "commit": commit(), "n_boot": N_BOOT, "seed": SEED, "ks": list(KS), "k_max": K_MAX,
              "min_short_side": REGISTERED_OUTCOME["min_short"], "equivalence_margin": EQUIVALENCE_MARGIN,
              "statistic": STATISTIC_LABEL, "smoke": a.smoke, "branches": {}}
    paths = None
    for branch in BRANCHES:
        logits = load_branch(branch, a.scores, a.smoke)
        if paths is None:
            paths = logits["path"]
        assert (logits["path"] == paths).all(), f"{branch}: rows differ from {BRANCHES[0]}"

        stats = with_intervals(logits)
        # The index scheme must be paired_diff's: same interval for one difference.
        y = logits["label"]
        reference = paired_diff(y, logits["after_jpeg_k2"], logits["before_jpeg_k2"], y)
        assert np.allclose(stats["reduction_jpeg_k2"]["ci"], reference["ci"]), (stats["reduction_jpeg_k2"], reference)

        pooled_verdict = verdict(stats, "")
        per_op = {op: verdict(stats, f"{op}_") for op in OPS}
        report["branches"][branch] = {
            "stats": stats,
            "verdict": pooled_verdict,
            "verdict_per_op": per_op,
            "per_op_disagree_with_each_other": len(set(per_op.values())) > 1,
            "per_op_disagree_with_pooled": any(v != pooled_verdict for v in per_op.values()),
        }
        cells = ", ".join(f"R({k}) {stats[f'reduction_k{k}']['point']:+.4f} "
                          f"[{stats[f'reduction_k{k}']['ci'][0]:+.4f}, {stats[f'reduction_k{k}']['ci'][1]:+.4f}]"
                          for k in KS)
        rise = stats["reduction_kmax_minus_k2"]
        print(f"{branch}: {cells}; R({K_MAX}) - R(2) {rise['point']:+.4f} "
              f"[{rise['ci'][0]:+.4f}, {rise['ci'][1]:+.4f}] -> {pooled_verdict}; per op {per_op}"
              + ("  [PER-OP VERDICTS DISAGREE]" if report["branches"][branch]["per_op_disagree_with_pooled"]
                 or report["branches"][branch]["per_op_disagree_with_each_other"] else ""), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"Test 18{' (SMOKE, not a result)' if a.smoke else ''} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
