"""N-parent score fusion on the simplex (off-ladder).

    python scripts/fuse_simplex.py \
        --parent band=data/banks/eval_probe_band_dinov2regl:outputs/.../checkpoint.pt \
        --parent crop=data/banks/eval_probe_crop_dinov2regl:outputs/.../checkpoint.pt \
        --parent siglip=data/banks/eval_probe_band_siglipso400m:outputs/.../checkpoint.pt \
        --out docs/fuse_simplex.json

WHY THIS IS A SEPARATE SCRIPT. `fusion.fit_fusion_weight` refuses more than two
parents on purpose, and its refusal is the specification for this file:

    "fit_fusion_weight sweeps a single scalar w over TWO parents, got {n}. An
     n-parent fit is a different problem with a different number of degrees of
     freedom to justify."

So the guard is not weakened and no signature is widened. The extra degrees of
freedom are justified here instead, where a reader can see the cost:

    parents   free params   lattice   grid points
       2           1           1/20            21
       3           2           1/21           253
       4           3           1/20         1,771

(the lattice denominator is snapped up to a multiple of the parent count so
that equal weighting is representable -- see `simplex_grid`)

Two parents fit one number on `val_internal`; three fit two. That is strictly
more room to fit noise in the selection knob, and it is why this script reports
the whole sweep and its flatness rather than only the argmax -- a simplex whose
objective is flat has not learned a weighting, it has picked one.

THE MEASUREMENT. Wave 2 produced two fusions that each beat their parents:

    dinov2regl band + crop            0.8714   (parents 0.7242 / 0.7858)
    dinov2regl band + siglipso400m    0.8534   (parents 0.7242 / 0.7111)

Both gains came from FUSION, not from the weight: band+crop's fitted weight
bought +0.0012 over equal weighting. The open question is whether the two
fusions are finding the SAME complementary signal or different ones. If
three-way beats band+crop, the SigLIP tower carries something neither DINO view
has; if it does not, band+crop already spans what these three towers know, and
the third tower is cost without coverage.

DISCIPLINE, INHERITED NOT RESTATED. The weights are swept to maximise
`fusion.val_robust_tpr` -- both classes from `val_internal` -- and the held-out
number is read once, after the weights are fixed. `fit_splits` for the z-score
population is `FIT_SPLITS_WHEN_FITTING_WEIGHT` (`val_internal` alone), because
the z-score population and the weights are two separate fits and defaulting
either is how A5's number becomes a function of the organisers' demo set.

TIES GO TO THE CENTROID, mirroring `fit_fusion_weight`'s "ties go to 0.5, not
to the lowest w". Equal weighting is the null this sweep exists to test, so on
a flat objective the sweep must return it rather than whichever vertex the grid
happened to visit first -- otherwise "drop parent 2 entirely" silently wins a
tie and gets reported as a fitted three-way weighting.

OFF-LADDER. A5 is score-level fusion of TWO banks (`FUSION_PARTNER_NAME` trains
one partner from a3's own config). Three parents is not that rung and is not a
one-flag step from it, so this writes its own JSON and is never a §6.4
candidate. It is evidence about whether a third tower is worth extracting at
full scale, which is a cost decision, not a headline.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (
    FIT_SPLITS_WHEN_FITTING_WEIGHT, assert_fusion_parents, fuse_scores,
    val_robust_tpr,
)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector


def simplex_grid(n: int, step: float) -> tuple[list[tuple[float, ...]], int]:
    """All `n`-vectors on the unit simplex with coordinates on `step`'s lattice.

    Built from integer compositions so the coordinates sum to exactly 1 in
    integer arithmetic before division -- a float `arange` accumulates error and
    would put points slightly off the simplex, which `fuse_scores` then
    renormalises silently, making the recorded weight differ from the one used.

    THE LATTICE IS SNAPPED SO THE CENTROID LIES ON IT. Equal weighting is the
    null this whole sweep exists to test, and for three parents 1/3 is not a
    multiple of 0.05 -- so the requested step would build a grid that cannot
    represent the null, cannot report its objective, and cannot break a tie
    toward it. `k` is therefore rounded UP to the next multiple of `n`, which
    is the smallest change that keeps `(1/n, ..., 1/n)` a lattice point.
    Returns the grid and the `k` actually used, so the JSON records the step
    that ran rather than the one that was asked for.
    """
    k = int(round(1.0 / step))
    if abs(k * step - 1.0) > 1e-9:
        raise SystemExit(f"--step {step} does not divide 1 evenly")
    if k % n:
        k += n - (k % n)
    pts = []
    for cut in itertools.combinations(range(1, k + n), n - 1):
        prev, coords = 0, []
        for c in cut:
            coords.append(c - prev - 1)
            prev = c
        coords.append(k + n - 1 - prev)
        pts.append(tuple(c / k for c in coords))
    return pts, k


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", action="append", required=True,
                    metavar="NAME=EVAL_BANK:CHECKPOINT", help="repeatable")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    parents = []
    for spec in a.parent:
        name, _, rest = spec.partition("=")
        eval_dir, _, ckpt = rest.partition(":")
        if not (name and eval_dir and ckpt):
            raise SystemExit(f"--parent wants NAME=EVAL_BANK:CHECKPOINT, got {spec!r}")
        parents.append((name, eval_dir, ckpt))
    if len(parents) < 2:
        raise SystemExit("fusion needs at least two parents")

    banks = [FeatureBank(d) for _, d, _ in parents]
    # The row set of a fused frame is only defined when the parents came from
    # the same frozen manifest with the same view axis; `canon_policy` is
    # deliberately NOT among the checked keys, which is what makes band+crop
    # fusion legal in the first place.
    assert_fusion_parents(banks)

    splits = banks[0].meta.set_index("image_idx")["split"]
    dfs, singles = [], []
    for (name, eval_dir, ckpt), bank in zip(parents, banks):
        model, ck = load_detector(ckpt, device=a.device)
        use_recon = bool(ck["config"].get("use_recon", False))
        df = score_grid(model, bank, use_recon=use_recon, device=a.device)
        dfs.append(df)
        solo = heldout_robust_tpr(df, splits, SELECTION_TARGET_FPR)
        singles.append({"parent": name, "eval_bank": eval_dir,
                        "checkpoint": ckpt,
                        "heldout_robust_tpr_at_1pct": float(solo)})
        print(f"parent {name:>8s}: {solo:.4f}", flush=True)

    n = len(dfs)
    grid, k_used = simplex_grid(n, a.step)
    step_used = 1.0 / k_used
    if abs(step_used - a.step) > 1e-12:
        print(f"step {a.step} snapped to {step_used:.6f} (1/{k_used}) so the "
              f"equal-weight null is a lattice point", flush=True)
    print(f"\nsweeping {len(grid)} simplex points over {n} parents "
          f"({n - 1} free params)", flush=True)

    centroid = tuple([1.0 / n] * n)
    sweep = []
    for w in grid:
        fused = fuse_scores(dfs, weights=list(w), splits=splits,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        sweep.append({"w": [float(x) for x in w],
                      "val_robust_tpr": float(
                          val_robust_tpr(fused, splits, SELECTION_TARGET_FPR))}
                     )

    def _dist_to_centroid(r):
        return float(np.abs(np.asarray(r["w"]) - np.asarray(centroid)).sum())

    best = max(sweep, key=lambda r: (r["val_robust_tpr"], -_dist_to_centroid(r)))
    w_best = best["w"]

    objs = np.asarray([r["val_robust_tpr"] for r in sweep])
    equal_obj = next(r["val_robust_tpr"] for r in sweep
                     if np.allclose(r["w"], centroid))
    flatness = {
        "objective_max": float(objs.max()),
        "objective_min": float(objs.min()),
        "objective_at_equal_weights": float(equal_obj),
        "gain_over_equal_on_FIT_split": float(best["val_robust_tpr"] - equal_obj),
        "n_points_within_1e-4_of_max": int((objs >= objs.max() - 1e-4).sum()),
        "n_points": len(sweep),
    }

    # Held out is read exactly once per reported weighting, after the sweep.
    equal_fused = fuse_scores(dfs, weights=list(centroid), splits=splits,
                              fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
    best_fused = fuse_scores(dfs, weights=w_best, splits=splits,
                             fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
    equal_held = heldout_robust_tpr(equal_fused, splits, SELECTION_TARGET_FPR)
    best_held = heldout_robust_tpr(best_fused, splits, SELECTION_TARGET_FPR)

    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        pf = fuse_scores([dfs[i], dfs[j]], splits=splits,
                         fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        pairs.append({
            "parents": [parents[i][0], parents[j][0]], "weights": [0.5, 0.5],
            "heldout_robust_tpr_at_1pct": float(
                heldout_robust_tpr(pf, splits, SELECTION_TARGET_FPR))})

    best_parent = max(s["heldout_robust_tpr_at_1pct"] for s in singles)
    best_pair = max(p["heldout_robust_tpr_at_1pct"] for p in pairs)
    print(f"\n{n}-way equal   : {equal_held:.4f}")
    print(f"{n}-way fitted  : {best_held:.4f}  w={[round(x, 3) for x in w_best]}")
    print(f"best single    : {best_parent:.4f}")
    print(f"best equal pair: {best_pair:.4f}")
    print(f"\nfit-split objective is flat within 1e-4 at "
          f"{flatness['n_points_within_1e-4_of_max']}/{flatness['n_points']} "
          f"points; fitting bought {flatness['gain_over_equal_on_FIT_split']:+.4f} "
          f"on the FIT split")
    verdict = ("the third parent adds coverage"
               if best_held > best_pair + 1e-9 else
               "no gain over the best PAIR: the extra tower is cost without coverage")
    print(f"VERDICT: {verdict}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "fuse_simplex", "off_ladder": True,
            "not_eligible_reason": "A5 is fusion of TWO banks; three parents "
                                   "is not that rung and not a one-flag step "
                                   "from it",
            "metric": "heldout_robust_tpr_at_1pct",
            "n_parents": n, "free_params": n - 1,
            "step_requested": a.step, "step_used": float(step_used),
            "n_simplex_points": len(grid),
            "zscore_fit_splits": list(FIT_SPLITS_WHEN_FITTING_WEIGHT),
            "weight_fit_objective": "val_robust_tpr (val_internal both classes)",
            "tie_break": "toward the centroid (equal weights), so a flat "
                         "objective returns the null rather than a vertex",
            "parents": singles,
            "equal_weight": {"w": list(centroid),
                             "heldout_robust_tpr_at_1pct": float(equal_held)},
            "fitted_weight": {"w": [float(x) for x in w_best],
                              "heldout_robust_tpr_at_1pct": float(best_held)},
            "equal_weight_pairs": pairs,
            "flatness": flatness,
            "verdict": verdict,
            "sweep": sweep,
        }, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
