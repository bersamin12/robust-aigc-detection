"""Do the THREE- and FOUR-way fusions survive a different held-out family?

    python scripts/second_holdout_lattice.py \
        --arm band_dinov2regl=dinov2regl:BANK:EVAL_BANK ... \
        --holdout-generators sid_set --out docs/second_holdout_lattice_sid.json

WHY A SIBLING AND NOT A FLAG ON `second_holdout.py`. That script fuses only
when exactly two arms are supplied, because it delegates to `fit_fusion_weight`
and that function refuses a third parent on purpose: "an n-parent fit is a
different problem with a different number of degrees of freedom to justify."
Its two-arm numbers are already on record (band/crop, and the three siglip
pairs), so it is left byte-identical and the arity generalisation lives here.
`relabelled_splits` is IMPORTED from it rather than re-typed -- that function
has already been wrong once, in a way the repo's own empty-class guard caught,
and one copy is the whole point.

WHAT THIS ANSWERS. `fusion_lattice` ranked every arity on the PRIMARY split and
found the four-way top of the legal table at 0.9247. But the primary split is
two wildfake generators, and the 2026-08-31 second-holdout runs showed the
single-arm ordering INVERTS across families: crop beat band by +0.062 on the
primary pair and LOST by 0.129 on sid_set. A ranking that inverts for one arm
can invert for four. This re-runs the whole legal lattice under a different
held-out family, so the arity question is asked where it actually matters.

WHAT IS COMPARABLE AND WHAT IS NOT. Levels are NOT comparable with the primary
split's -- a different family at a different sample size is a different
population. The ORDER is what transfers, and the specific things to read are:
does the four-way still beat the three-ways, do they still beat the pairs, and
does fusion still beat every parent? The last of those was the one property
that held on all three splits last time.

EACH ARM IS RETRAINED ONCE, NOT ONCE PER COMBINATION. The excluded-family head
for `crop_dinov2regl` is the same head whether it is fused with one arm or
three, so the retraining loop is over ARMS and the combination loop is numpy
over cached frames. That is what makes a 36-combination lattice affordable on
CPU beside a running extraction.

EQUAL WEIGHTS ARE THE DEFAULT REPORT HERE. On a split whose whole purpose is to
test whether a ranking is real, a fitted weight adds a second thing that could
fail to transfer. Equal weighting is the null; the simplex fit is run for the
top few combinations only, and is reported beside the null rather than instead.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (FIT_SPLITS_WHEN_FITTING_WEIGHT, fuse_scores,
                                 val_robust_tpr)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector, train_rung
from run_ablation import load_rung_config  # noqa: E402

#: OUR OWN spec line, not the track's. The design doc's line 74 reads "Final
#: model uses at most two backbones, to hold total parameters and inference
#: latency at defensible levels" -- a self-imposed rule with a stated rationale.
#: The TRACK's only hard constraint is the 2B parameter cap (design.md:5, and
#: the out-of-scope list at :358). Half that rationale is now measured: all five
#: cached backbones sum to 1,998,494,848, inside the cap. The other half,
#: latency, is not measured, so the rule stays a DEFAULT rather than becoming a
#: constant -- relax it with --max-backbones and say why.
DEFAULT_MAX_BACKBONES = 2


def _sibling(name: str, attr: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=BACKBONE:BANK:EVAL_BANK")
    ap.add_argument("--holdout-generators", required=True)
    ap.add_argument("--config", default="configs/rungs/a3.yaml")
    ap.add_argument("--out-dir", default="outputs/second_holdout_lattice")
    ap.add_argument("--max-arity", type=int, default=4)
    ap.add_argument("--max-backbones", type=int, default=DEFAULT_MAX_BACKBONES,
                    help="distinct backbones a combination may use; the track "
                         "caps parameters, not tower count")
    ap.add_argument("--simplex-top", type=int, default=2)
    ap.add_argument("--simplex-step", type=float, default=0.05)
    ap.add_argument("--simplex-max-arity", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    families = [g.strip() for g in a.holdout_generators.split(",") if g.strip()]
    tag = "_".join(families)[:40]
    relabelled_splits = _sibling("second_holdout", "relabelled_splits")

    arms = {}
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        backbone, _, paths = rest.partition(":")
        bank_dir, _, eval_dir = paths.partition(":")
        if not (name and backbone and bank_dir and eval_dir):
            raise SystemExit(f"--arm wants NAME=BACKBONE:BANK:EVAL_BANK, got {spec!r}")
        arms[name] = (backbone, bank_dir, eval_dir)

    frames, singles, splits_by_arm = {}, {}, {}
    for name, (backbone, bank_dir, eval_dir) in arms.items():
        print(f"\n=== {name}: retraining a3 without {families} ===", flush=True)
        eval_bank = FeatureBank(eval_dir)
        splits_by_arm[name] = relabelled_splits(eval_bank, families)
        cfg = load_rung_config(a.config, bank_dir, f"{a.out_dir}/{tag}/{name}",
                               a.device, train_exclude_generators=families)
        cfg.name = f"a3_no_{tag}"
        result = train_rung(cfg)
        model, _ = load_detector(result["checkpoint"], device=a.device)
        frames[name] = score_grid(model, eval_bank, use_recon=cfg.use_recon,
                                  device=a.device)
        singles[name] = float(heldout_robust_tpr(frames[name], splits_by_arm[name],
                                                 SELECTION_TARGET_FPR))
        print(f"{name}: second-holdout robust TPR@1%FPR = {singles[name]:.4f}",
              flush=True)

    # Every arm shares the frozen manifest, so every relabelled split column
    # must name the same rows. A fused frame has no single owning bank, so this
    # is asserted rather than assumed (errors.heldout_robust_tpr, "NOTE FOR TASK 8").
    names = list(arms)
    s0 = splits_by_arm[names[0]]
    for n in names[1:]:
        if not splits_by_arm[n].equals(s0):
            raise SystemExit(f"arm {n!r} disagrees with {names[0]!r} on the "
                             "relabelled split column; the fused frame would "
                             "have no defined population.")
    n_pos = int((s0 == "heldout_generator").sum())

    rows: dict[str, dict] = {}
    for k in range(2, min(a.max_arity, len(names)) + 1):
        legal = [sub for sub in itertools.combinations(names, k)
                 if len({arms[x][0] for x in sub}) <= a.max_backbones]
        print(f"\n=== arity {k}: {len(legal)} combinations "
              f"(<= {a.max_backbones} backbones) ===", flush=True)
        for sub in legal:
            fused = fuse_scores([frames[x] for x in sub], splits=s0,
                                fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            rows["+".join(sub)] = {
                "arity": k, "arms": list(sub),
                "backbones": sorted({arms[x][0] for x in sub}),
                "equal": float(heldout_robust_tpr(fused, s0, SELECTION_TARGET_FPR)),
                "fitted": None, "w_fitted": None,
                "min_parent": min(singles[x] for x in sub),
                "max_parent": max(singles[x] for x in sub),
            }
        for kk, v in sorted([(kk, v) for kk, v in rows.items() if v["arity"] == k],
                            key=lambda kv: -kv[1]["equal"]):
            print(f"  {kk:<62s} {v['equal']:.4f}  "
                  f"(best parent {v['max_parent']:.4f})", flush=True)

    simplex_grid = _sibling("fuse_simplex", "simplex_grid")
    for k in range(2, min(a.simplex_max_arity, a.max_arity, len(names)) + 1):
        sel = sorted([(kk, v) for kk, v in rows.items() if v["arity"] == k],
                     key=lambda kv: -kv[1]["equal"])[:a.simplex_top]
        if not sel:
            continue
        grid, k_used = simplex_grid(k, a.simplex_step)
        centroid = tuple([1.0 / k] * k)
        print(f"\n=== simplex fit, arity {k}: {len(grid)} points ===", flush=True)
        for kk, v in sel:
            dfs = [frames[x] for x in v["arms"]]
            sweep = [(w, val_robust_tpr(
                fuse_scores(dfs, weights=list(w), splits=s0,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT),
                s0, SELECTION_TARGET_FPR)) for w in grid]
            w_best, _ = max(sweep, key=lambda t: (
                t[1], -float(np.abs(np.asarray(t[0]) - np.asarray(centroid)).sum())))
            fitted = fuse_scores(dfs, weights=list(w_best), splits=s0,
                                 fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            v["fitted"] = float(heldout_robust_tpr(fitted, s0, SELECTION_TARGET_FPR))
            v["w_fitted"] = [float(x) for x in w_best]
            print(f"  {kk:<62s} equal={v['equal']:.4f} fitted={v['fitted']:.4f} "
                  f"w={[round(x, 3) for x in w_best]}", flush=True)

    best = max(rows, key=lambda kk: rows[kk]["equal"])
    beats_parents = sum(1 for v in rows.values() if v["equal"] > v["max_parent"])
    by_arity = {k: max((v["equal"] for v in rows.values() if v["arity"] == k),
                       default=None)
                for k in range(2, min(a.max_arity, len(names)) + 1)}
    print(f"\n{'arm':>26s} {'2nd holdout':>12s}")
    for n in sorted(singles, key=singles.get, reverse=True):
        print(f"{n:>26s} {singles[n]:12.4f}")
    print(f"\nbest combination : {best} = {rows[best]['equal']:.4f}")
    print("best per arity   : " + "  ".join(
        f"{k}-way {v:.4f}" for k, v in by_arity.items() if v is not None))
    print(f"fusion beats its best parent in {beats_parents}/{len(rows)} combinations")
    print(f"\nheld-out positives: {n_pos} images x conditions "
          "-- read the ORDER, not the level")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "second_holdout_lattice", "off_ladder": True,
            "holdout_generators": families, "n_heldout_positive_images": n_pos,
            "metric": "heldout_robust_tpr_at_1pct",
            "max_backbones": a.max_backbones,
            "singles": singles, "combinations": rows,
            "best": best, "best_per_arity": by_arity,
            "fusion_beats_best_parent": [beats_parents, len(rows)],
            "caveat": ("levels are NOT comparable with the primary split; a "
                       "different family at a different sample size is a "
                       "different population. Read the order."),
        }, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
