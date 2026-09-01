"""Every fusion the cached probe banks can express, with paired bootstrap CIs.

    python scripts/pair_lattice.py --arm NAME=BACKBONE:EVAL_BANK:CKPT ... \
        --baseline crop_dinov2regl+band_siglipso400m --out docs/pair_lattice.json

WHY THIS EXISTS. The current headline -- `dinov2regl:crop + siglipso400m:band`
at 0.9105 -- was crowned after trying THREE of the twenty-eight pairs the
cached banks can express. The three were not chosen by a search; they were the
pairs an earlier queue happened to schedule. "Best of three arbitrary pairs"
and "best pair" are different claims and only one of them is worth shipping.

WHY IT IS FREE. Every arm here already has a trained a3 checkpoint and a scored
eval bank on disk, and A5's partner is defined as a3's own config trained on
the partner bank (`FUSION_PARTNER_NAME`), so the partner head for bank B *is*
B's a3. Nothing is retrained. The cost is eight `score_grid` passes over cached
float16 features; the 54 combinations after that are numpy over frames already
in memory, which is why this runs on CPU beside a GPU extraction.

THE TWO-BACKBONE RULE IS ENFORCED HERE, NOT ASSUMED. `design.md:74` caps the
shipped bundle at two backbones. A pair can never break it, but a three-way
can, so three-ways are enumerated as "two policies of one tower plus one arm of
another" rather than over all triples -- and every emitted row carries its
distinct-backbone count so the constraint is auditable from the JSON.

WHAT THE BOOTSTRAP IS FOR. The pair table will have near-ties, and reading a
winner off a fourth decimal place is how a ranking gets shipped that a second
sample would reverse. `family_experts.bootstrap_panel` resamples IMAGES (each
image contributes a row to all 19 degraded conditions, so rows are not
independent) and reports the PAIRED interval on each candidate minus the
baseline, which is the interval the comparison actually lives in. It is
imported rather than re-typed, per this repo's rule for the section 6.4
machinery.

READ THE ORDER, NOT THE LEVEL, AND THEN ONLY ON THE PRIMARY SPLIT. The
2026-08-31 second-holdout runs showed the single-arm ordering INVERTS across
held-out families. This script measures the primary split only. It is therefore
a shortlist generator, not a selector: whatever wins here still has to survive
`scripts/second_holdout.py` before it can be believed.
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
                                 fit_fusion_weight, fuse_scores)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector


def _load_bootstrap_panel():
    """Import `bootstrap_panel` out of the family-experts script.

    `family_experts.py` is a script with a `__main__` guard, not a package
    module, so it is loaded by path. The alternative -- copying the resampler
    -- is the thing this repo's fusion rule exists to prevent: two bootstraps
    that drift apart give two different answers to one question.
    """
    path = Path(__file__).resolve().parent / "family_experts.py"
    spec = importlib.util.spec_from_file_location("_family_experts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bootstrap_panel


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
                    metavar="NAME=BACKBONE:EVAL_BANK:CKPT", help="repeatable")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boot-top", type=int, default=6,
                    help="bootstrap the top-N pairs plus the baseline and its parents")
    ap.add_argument("--boot-n", type=int, default=1000)
    ap.add_argument("--baseline", default=None,
                    help="name of the combination the paired CIs are read against")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = [parse_arm(s) for s in a.arm]
    names = [n for n, _, _, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit("--arm names must be unique; they key every table below")

    banks = {n: FeatureBank(d) for n, _, d, _ in arms}
    # Same frozen manifest, same view axis. `canon_policy` is deliberately NOT
    # among the checked keys -- that omission is what makes band+crop legal.
    assert_fusion_parents(list(banks.values()))
    splits = banks[names[0]].meta.set_index("image_idx")["split"]

    backbone_of = {n: b for n, b, _, _ in arms}
    frames, singles = {}, []
    for name, backbone, eval_dir, ckpt in arms:
        model, ck = load_detector(ckpt, device=a.device)
        use_recon = bool(ck["config"].get("use_recon", False))
        frames[name] = score_grid(model, banks[name], use_recon=use_recon,
                                  device=a.device)
        solo = float(heldout_robust_tpr(frames[name], splits, SELECTION_TARGET_FPR))
        singles.append({"arm": name, "backbone": backbone, "eval_bank": eval_dir,
                        "checkpoint": ckpt, "heldout_robust_tpr_at_1pct": solo})
        print(f"single {name:>26s}  {solo:.4f}", flush=True)

    combos: dict[str, dict] = {}
    print(f"\n=== pairs ({len(names) * (len(names) - 1) // 2}) ===", flush=True)
    for i, j in itertools.combinations(names, 2):
        dfs = [frames[i], frames[j]]
        equal = fuse_scores(dfs, splits=splits,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        # The weight is swept on val_internal alone; held out is read ONCE,
        # after the sweep, for each of the two reported weightings.
        (w0, w1), sweep = fit_fusion_weight(
            dfs, splits, fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        fitted = fuse_scores(dfs, weights=[w0, w1], splits=splits,
                             fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        key = f"{i}+{j}"
        combos[key] = {
            "kind": "pair", "arms": [i, j],
            "backbones": sorted({backbone_of[i], backbone_of[j]}),
            "n_backbones": len({backbone_of[i], backbone_of[j]}),
            "equal": float(heldout_robust_tpr(equal, splits, SELECTION_TARGET_FPR)),
            "fitted": float(heldout_robust_tpr(fitted, splits, SELECTION_TARGET_FPR)),
            "w_fitted": [float(w0), float(w1)],
            "fit_objective_flat_within_1e-4": int(sum(
                r["val_robust_tpr"] >= max(s["val_robust_tpr"] for s in sweep) - 1e-4
                for r in sweep)),
            "fit_grid_points": len(sweep),
        }
        c = combos[key]
        print(f"  {key:<50s} equal={c['equal']:.4f} fitted={c['fitted']:.4f} "
              f"w={c['w_fitted'][0]:.2f}", flush=True)

    # Three-ways: TWO policies of one backbone plus ONE arm of another. Any
    # other triple would need three towers and is barred by design.md:74.
    by_backbone: dict[str, list[str]] = {}
    for n in names:
        by_backbone.setdefault(backbone_of[n], []).append(n)
    triples = []
    for bb, arms_of in by_backbone.items():
        if len(arms_of) < 2:
            continue
        for pair in itertools.combinations(sorted(arms_of), 2):
            for third in names:
                if backbone_of[third] != bb:
                    triples.append((*pair, third))
    print(f"\n=== three-ways ({len(triples)}, equal weights only) ===", flush=True)
    for tri in triples:
        dfs = [frames[t] for t in tri]
        equal = fuse_scores(dfs, splits=splits,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        key = "+".join(tri)
        combos[key] = {
            "kind": "three_way", "arms": list(tri),
            "backbones": sorted({backbone_of[t] for t in tri}),
            "n_backbones": len({backbone_of[t] for t in tri}),
            "equal": float(heldout_robust_tpr(equal, splits, SELECTION_TARGET_FPR)),
            "fitted": None, "w_fitted": None,
        }
        print(f"  {key:<50s} equal={combos[key]['equal']:.4f}", flush=True)

    illegal = [k for k, v in combos.items() if v["n_backbones"] > 2]
    if illegal:
        raise SystemExit(f"BUG: {len(illegal)} combination(s) exceed two backbones: "
                         f"{illegal[:3]}")

    ranked = sorted(combos.items(), key=lambda kv: -kv[1]["equal"])
    baseline = a.baseline or ranked[0][0]
    if baseline not in combos:
        raise SystemExit(f"--baseline {baseline!r} is not among the "
                         f"{len(combos)} combinations measured")

    # Bootstrap a SHORTLIST, not the lattice: 54 paired intervals invite
    # exactly the "some interval excluded zero" reading that a multiple
    # comparison makes meaningless.
    shortlist = [k for k, _ in ranked[:a.boot_top]]
    for extra in [baseline, *combos[baseline]["arms"]]:
        if extra not in shortlist:
            shortlist.append(extra)
    panel_frames = {}
    for key in shortlist:
        if key in frames:
            panel_frames[key] = frames[key]
            continue
        panel_frames[key] = fuse_scores([frames[x] for x in combos[key]["arms"]],
                                        splits=splits,
                                        fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
    print(f"\n=== paired bootstrap: {len(panel_frames)} heads, "
          f"{a.boot_n} resamples over IMAGES, baseline={baseline} ===", flush=True)
    panel = _load_bootstrap_panel()(panel_frames, splits, baseline=baseline,
                                    n_boot=a.boot_n)
    for key in sorted(panel, key=lambda k: -panel[k]["point"]):
        p = panel[key]
        vb = p["vs_baseline"]
        lo, hi = vb["paired_ci95"]
        verdict = "separated" if (lo > 0 or hi < 0) else "TIE with baseline"
        print(f"  {key:<50s} {p['point']:.4f} "
              f"[{p['ci95'][0]:.4f},{p['ci95'][1]:.4f}]  "
              f"d={vb['delta']:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "pair_lattice", "off_ladder": False,
            "metric": "heldout_robust_tpr_at_1pct",
            "target_fpr": SELECTION_TARGET_FPR,
            "fit_splits_for_weight": list(FIT_SPLITS_WHEN_FITTING_WEIGHT),
            "caveat": ("primary split only; the single-arm ordering is known to "
                       "invert across held-out families, so this is a shortlist "
                       "generator and not a selector"),
            "singles": singles,
            "combinations": combos,
            "baseline": baseline,
            "bootstrap": panel,
        }, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
