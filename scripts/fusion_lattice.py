"""Every fusion of every arity the cached arms can express -- legal and barred.

    python scripts/fusion_lattice.py --arm NAME=BACKBONE:EVAL_BANK:CKPT ... \
        --max-arity 8 --simplex-top 3 --out docs/fusion_lattice.json

WHAT THIS ADDS OVER `pair_lattice`. That script stopped at three arms and only
ever weighted them equally. Two questions were left open by it and both are
cheap to close once the eight score frames are in memory:

  1. Does a FOURTH arm buy anything? Under the two-backbone rule a four-way is
     forced to be 2+2 -- both canon policies of two towers -- so there are
     exactly THREE legal four-ways, not seventy. Enumerating the other
     sixty-seven anyway is not wasted: they measure what the rule costs.
  2. Do FITTED weights change the ordering at arity 3 and 4? `pair_lattice`
     fitted only pairs, where a single scalar w has an analytic-feeling sweep;
     at higher arity the weights live on a simplex and the null (equal
     weighting) is a specific lattice point that the grid must contain.

LEGAL AND BARRED ARE BOTH REPORTED, SEPARATELY AND ALWAYS. `design.md:74` caps
the shipped bundle at two backbones. A subset over that cap is not a candidate
and is never allowed to win a table here -- but suppressing it entirely hides
the only measurement that says whether the cap is expensive or free. The same
convention the repo already uses for the gated `dinov3l` reference: measured,
reported, and marked unshippable. Every row carries `n_backbones` and `legal`,
and the selection line at the end reads from the legal rows only.

THE PARAMETER CAP IS NOT CHECKED HERE. `n_backbones <= 2` is a different
constraint from the 2B parameter budget, which `tests/features/test_backbones.py`
enforces against the actual registry. This script reports which towers a
combination needs; it does not re-derive their sizes, because a param count
typed into a second place is a param count that will disagree with the first.

STILL THE PRIMARY SPLIT ONLY. The 2026-08-31 second-holdout runs showed the
single-arm ordering INVERTS across held-out families, and nothing here revisits
that. Arity is a shortlist axis. The selector is `scripts/second_holdout.py`.
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

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (FIT_SPLITS_WHEN_FITTING_WEIGHT, assert_fusion_parents,
                                 fuse_scores, val_robust_tpr)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector

MAX_BACKBONES = 2   # design.md:74



def aux_flags(ck) -> dict:
    """The auxiliary-block flags a checkpoint was TRAINED with.

    `score_grid` takes three, not one. Passing only `use_recon` was silently
    wrong the moment a rung above a4 existed: an `aF` head is built with a
    frequency block on its input, and scoring it with `use_freq=False` feeds
    the head a different vector from the one it was fitted on. Derive all
    three from the checkpoint rather than from the caller, because the
    checkpoint is the only thing that knows.
    """
    cfg = ck["config"]
    return {"use_recon": bool(cfg.get("use_recon", False)),
            "use_recon_vq": bool(cfg.get("use_recon_vq", False)),
            "use_freq": bool(cfg.get("use_freq", False))}

def _sibling(name: str, attr: str):
    """Import one symbol from a sibling SCRIPT by path.

    `fuse_simplex.simplex_grid` and `family_experts.bootstrap_panel` both live
    in `__main__`-guarded scripts rather than the package. Loading them by path
    keeps one implementation of each; re-typing the simplex snap in particular
    would reintroduce the bug it was written to fix (for n=3, 1/3 is not a
    multiple of 0.05, so an unsnapped lattice cannot represent the null).
    """
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def parse_arm(spec: str) -> tuple[str, str, str, str]:
    name, _, rest = spec.partition("=")
    backbone, _, rest = rest.partition(":")
    eval_dir, _, ckpt = rest.partition(":")
    if not (name and backbone and eval_dir and ckpt):
        raise SystemExit(f"--arm wants NAME=BACKBONE:EVAL_BANK:CKPT, got {spec!r}")
    return name, backbone, eval_dir, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=BACKBONE:EVAL_BANK:CKPT")
    ap.add_argument("--max-arity", type=int, default=4)
    ap.add_argument("--simplex-top", type=int, default=3,
                    help="fit simplex weights for the top-N LEGAL combos of each arity")
    ap.add_argument("--simplex-step", type=float, default=0.05)
    ap.add_argument("--simplex-max-arity", type=int, default=4,
                    help="above this the lattice explodes; equal weights only")
    ap.add_argument("--boot-n", type=int, default=1000)
    ap.add_argument("--boot-top", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = [parse_arm(s) for s in a.arm]
    names = [n for n, _, _, _ in arms]
    backbone_of = {n: b for n, b, _, _ in arms}
    banks = {n: FeatureBank(d) for n, _, d, _ in arms}
    assert_fusion_parents(list(banks.values()))
    splits = banks[names[0]].meta.set_index("image_idx")["split"]

    frames, singles = {}, {}
    for name, backbone, eval_dir, ckpt in arms:
        model, ck = load_detector(ckpt, device=a.device)
        frames[name] = score_grid(model, banks[name],
                                  device=a.device, **aux_flags(ck))
        singles[name] = float(heldout_robust_tpr(frames[name], splits,
                                                 SELECTION_TARGET_FPR))
        print(f"single {name:>26s}  {singles[name]:.4f}", flush=True)

    max_arity = min(a.max_arity, len(names))
    rows: dict[str, dict] = {}
    for k in range(2, max_arity + 1):
        subsets = list(itertools.combinations(names, k))
        n_legal = sum(1 for s in subsets
                      if len({backbone_of[x] for x in s}) <= MAX_BACKBONES)
        print(f"\n=== arity {k}: {len(subsets)} subsets "
              f"({n_legal} legal, {len(subsets) - n_legal} barred) ===", flush=True)
        for sub in subsets:
            bbs = sorted({backbone_of[x] for x in sub})
            fused = fuse_scores([frames[x] for x in sub], splits=splits,
                                fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            rows["+".join(sub)] = {
                "arity": k, "arms": list(sub), "backbones": bbs,
                "n_backbones": len(bbs), "legal": len(bbs) <= MAX_BACKBONES,
                "equal": float(heldout_robust_tpr(fused, splits, SELECTION_TARGET_FPR)),
                "fitted": None, "w_fitted": None,
            }
        for tag, want in (("LEGAL", True), ("barred", False)):
            sel = [(kk, v) for kk, v in rows.items()
                   if v["arity"] == k and v["legal"] is want]
            if not sel:
                continue
            for kk, v in sorted(sel, key=lambda kv: -kv[1]["equal"])[:3]:
                print(f"  {tag:>6s} {kk:<62s} {v['equal']:.4f} "
                      f"({v['n_backbones']} backbone(s))", flush=True)

    # --- fitted weights on the simplex, for the legal shortlist only ---------
    simplex_grid = _sibling("fuse_simplex", "simplex_grid")
    for k in range(2, min(a.simplex_max_arity, max_arity) + 1):
        sel = sorted([(kk, v) for kk, v in rows.items()
                      if v["arity"] == k and v["legal"]],
                     key=lambda kv: -kv[1]["equal"])[:a.simplex_top]
        if not sel:
            continue
        grid, k_used = simplex_grid(k, a.simplex_step)
        centroid = tuple([1.0 / k] * k)
        print(f"\n=== simplex fit, arity {k}: {len(grid)} points "
              f"(step snapped to 1/{k_used}) ===", flush=True)
        for kk, v in sel:
            dfs = [frames[x] for x in v["arms"]]
            # Swept on val_internal ONLY; held out is read once, afterwards.
            sweep = [(w, val_robust_tpr(
                fuse_scores(dfs, weights=list(w), splits=splits,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT),
                splits, SELECTION_TARGET_FPR)) for w in grid]
            objs = np.asarray([o for _, o in sweep])
            equal_obj = next(o for w, o in sweep if np.allclose(w, centroid))
            # Ties break TOWARD the null: equal weighting is what this tests.
            w_best, _ = max(sweep, key=lambda t: (
                t[1], -float(np.abs(np.asarray(t[0]) - np.asarray(centroid)).sum())))
            fitted = fuse_scores(dfs, weights=list(w_best), splits=splits,
                                 fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            v["fitted"] = float(heldout_robust_tpr(fitted, splits, SELECTION_TARGET_FPR))
            v["w_fitted"] = [float(x) for x in w_best]
            v["fit_gain_on_FIT_split"] = float(objs.max() - equal_obj)
            v["fit_flat_within_1e-4"] = int((objs >= objs.max() - 1e-4).sum())
            v["fit_grid_points"] = len(sweep)
            print(f"  {kk:<62s} equal={v['equal']:.4f} fitted={v['fitted']:.4f} "
                  f"w={[round(x, 3) for x in w_best]} "
                  f"(flat at {v['fit_flat_within_1e-4']}/{len(sweep)})", flush=True)

    legal = {kk: v for kk, v in rows.items() if v["legal"]}
    best_legal = max(legal, key=lambda kk: legal[kk]["equal"])
    barred = {kk: v for kk, v in rows.items() if not v["legal"]}
    best_barred = max(barred, key=lambda kk: barred[kk]["equal"]) if barred else None

    # --- paired bootstrap over the legal shortlist --------------------------
    shortlist = sorted(legal, key=lambda kk: -legal[kk]["equal"])[:a.boot_top]
    panel_frames = {kk: fuse_scores([frames[x] for x in legal[kk]["arms"]],
                                    splits=splits,
                                    fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
                    for kk in shortlist}
    best_single = max(singles, key=singles.get)
    panel_frames[best_single] = frames[best_single]
    if best_barred is not None:
        panel_frames[f"[BARRED] {best_barred}"] = fuse_scores(
            [frames[x] for x in barred[best_barred]["arms"]], splits=splits,
            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
    print(f"\n=== paired bootstrap: {len(panel_frames)} heads, {a.boot_n} "
          f"resamples over IMAGES, baseline={best_legal} ===", flush=True)
    panel = _sibling("family_experts", "bootstrap_panel")(
        panel_frames, splits, baseline=best_legal, n_boot=a.boot_n)
    for kk in sorted(panel, key=lambda x: -panel[x]["point"]):
        p, vb = panel[kk], panel[kk]["vs_baseline"]
        lo, hi = vb["paired_ci95"]
        print(f"  {kk:<62s} {p['point']:.4f} "
              f"[{p['ci95'][0]:.4f},{p['ci95'][1]:.4f}] "
              f"d={vb['delta']:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{'separated' if (lo > 0 or hi < 0) else 'TIE'}", flush=True)

    print(f"\nbest LEGAL  : {best_legal} = {legal[best_legal]['equal']:.4f}")
    if best_barred is not None:
        cost = barred[best_barred]["equal"] - legal[best_legal]["equal"]
        print(f"best barred : {best_barred} = {barred[best_barred]['equal']:.4f} "
              f"({barred[best_barred]['n_backbones']} backbones)")
        print(f"the two-backbone rule costs {cost:+.4f} on the primary split "
              "-- read it against the bootstrap intervals above, not on its own")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "fusion_lattice", "metric": "heldout_robust_tpr_at_1pct",
            "target_fpr": SELECTION_TARGET_FPR,
            "max_backbones_for_legal": MAX_BACKBONES,
            "fit_splits_for_weight": list(FIT_SPLITS_WHEN_FITTING_WEIGHT),
            "caveat": ("primary split only; single-arm ordering is known to invert "
                       "across held-out families, so this is a shortlist, not a "
                       "selection"),
            "singles": singles, "combinations": rows,
            "best_legal": best_legal, "best_barred": best_barred,
            "bootstrap": panel,
        }, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
