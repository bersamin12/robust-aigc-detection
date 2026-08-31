"""Fourteen ways to combine arms, at every arity, with the parameter budget attached.

    python scripts/fusion_rules.py --arm NAME=BACKBONE:EVAL_BANK:CKPT ... \
        --max-arity 8 --out docs/fusion_rules.json

TWO QUESTIONS, ONE SWEEP.

1. IS THE WEIGHTED MEAN THE RIGHT COMBINER? Everything measured so far fuses by
   averaging z-scored logits. That is one choice out of many and it was never
   compared against another. A mean is the right combiner when arms make
   independent, similarly-scaled errors; it is the WRONG one when arms disagree
   informatively (a max or a disagreement penalty reads that), when one arm is
   miscalibrated relative to the others (a rank mean is immune), or when a
   single confident arm should be able to carry a decision (most-confident).

2. HOW MUCH OF THE 2B PARAMETER BUDGET IS ACTUALLY SPENT? Two constraints are
   in play and they are NOT the same constraint. `design.md:74` caps the bundle
   at TWO BACKBONES; the bundle also has a 2B PARAMETER cap. Today the count
   rule binds and the parameter cap does not come close: the shipping candidate
   is dinov2regl + siglipso400m = 732,598,336, about 37% of budget. Every row
   below therefore carries both `n_backbones` and the summed parameter count,
   so "what would maxing out the budget actually buy" is answerable from the
   JSON instead of being argued from intuition.

   a3 sets `use_recon: false`, so these arms carry NO VAE and NO LPIPS. The
   budget is the towers alone. That is what makes the eight-arm combination
   interesting rather than absurd -- see `--recon-overhead`, which adds them
   back for anyone pricing an a4 variant.

THE MULTIPLE-COMPARISON WARNING, UP FRONT AND NOT IN A FOOTNOTE. Fourteen rules
across 247 subsets is 3,458 numbers read off a metric whose paired bootstrap
interval is about +/-0.025 wide. Something will look like a winner by chance;
that is arithmetic, not evidence. So: the sweep RANKS, the bootstrap DECIDES,
and a rule is only worth believing if it beats the mean by more than the
interval AND does so at more than one arity. The expected result is that almost
everything ties, and that is a real finding about the shape of the problem, not
a failed experiment.

HOW THE Z-SCORING IS SHARED. `fuse_scores` z-scores each frame against a fit
population and aligns it onto frame 0's rows. Those per-frame columns do not
depend on which subset is being fused, so they are computed ONCE against a
single canonical base and every subset is then a row-slice of one matrix. This
is a speed change and must not be a semantics change, so `--self-check` fuses
one subset both ways and refuses to continue unless the metrics agree exactly.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (FIT_SPLITS_WHEN_FITTING_WEIGHT, _aligned, _fit_mask,
                                 _resolve_population, assert_fusion_parents,
                                 fuse_scores)
from aigcdet.eval.grid import score_grid
from aigcdet.features.backbones import BACKBONES
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector

MAX_BACKBONES = 2          # design.md:74
PARAM_CAP = 2_000_000_000  # the bundle cap the backbone registry test enforces
VAE_PARAMS = 84_000_000    # SD 1.5 KL VAE, only present when use_recon is true
LPIPS_PARAMS = 2_500_000



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

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def build_rules(Z: np.ndarray, R: np.ndarray) -> dict:
    """Every combiner, as a dict of name -> fused score vector.

    `Z` is (k, n_rows) of z-scored logits; `R` is the same arms' within-
    condition normalised RANKS, precomputed because a rank is not a function of
    the subset either.

    Monotone transforms are applied in log space where the natural form
    saturates: `noisy_or` as -sum(log(1-p)) rather than 1-prod(1-p), because
    with eight arms the latter rounds to 1.0 for most positives and a metric
    read at 1% FPR is destroyed by ties, not by scale.
    """
    k = Z.shape[0]
    mean, std = Z.mean(0), Z.std(0)
    p = np.clip(_sigmoid(Z), 1e-9, 1 - 1e-9)
    # Confidence = 1 - binary entropy of each arm's own probability. An arm that
    # is 50/50 on a row contributes nothing to that row; the weights are
    # per-ROW, which is the thing a fitted global weight cannot express.
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    conf = np.clip(1.0 - H, 1e-6, None)
    rules = {
        "mean": mean,
        "median": np.median(Z, axis=0),
        "max": Z.max(0),
        "min": Z.min(0),
        "rank_mean": R.mean(0),
        "rank_median": np.median(R, axis=0),
        "disagreement_only": std,          # control: must be near chance
        "mean_minus_std": mean - std,
        "mean_minus_half_std": mean - 0.5 * std,
        "mean_plus_std": mean + std,
        "most_confident": Z[np.abs(Z).argmax(0), np.arange(Z.shape[1])],
        "entropy_weighted": (conf * Z).sum(0) / conf.sum(0),
        "noisy_or": -np.log1p(-p).sum(0),
        "noisy_and": np.log(p).sum(0),
    }
    if k >= 4:
        # Drop one arm from each end: needs at least four to leave two behind.
        srt = np.sort(Z, axis=0)
        rules["trimmed_mean"] = srt[1:-1].mean(0)
    return rules


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=BACKBONE:EVAL_BANK:CKPT")
    ap.add_argument("--max-arity", type=int, default=8)
    ap.add_argument("--recon-overhead", action="store_true",
                    help="add the SD 1.5 VAE + LPIPS to every parameter total "
                         "(price an a4 variant; a3 carries neither)")
    ap.add_argument("--boot-n", type=int, default=1000)
    ap.add_argument("--boot-top", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        backbone, _, rest = rest.partition(":")
        eval_dir, _, ckpt = rest.partition(":")
        if not (name and backbone and eval_dir and ckpt):
            raise SystemExit(f"--arm wants NAME=BACKBONE:EVAL_BANK:CKPT, got {spec!r}")
        if backbone not in BACKBONES:
            raise SystemExit(f"{backbone!r} is not in the backbone registry, so "
                             "its parameter count cannot be reported")
        arms.append((name, backbone, eval_dir, ckpt))
    names = [n for n, _, _, _ in arms]
    backbone_of = {n: b for n, b, _, _ in arms}
    overhead = (VAE_PARAMS + LPIPS_PARAMS) if a.recon_overhead else 0

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
        print(f"single {name:>26s} {singles[name]:.4f}  "
              f"{BACKBONES[backbone].params:>13,} params", flush=True)

    # --- one alignment, reused by every subset -----------------------------
    split_values, fit, population = _resolve_population(
        splits, FIT_SPLITS_WHEN_FITTING_WEIGHT)
    base = frames[names[0]].reset_index(drop=True)
    Z_all = np.stack([
        _aligned(frames[n], base, i,
                 None if split_values is None
                 else _fit_mask(frames[n], split_values, fit, i))
        for i, n in enumerate(names)])
    cond = base["condition"].to_numpy()
    R_all = np.stack([
        pd.Series(z).groupby(cond).rank(pct=True).to_numpy() for z in Z_all])
    template = base.copy()

    def metric(score: np.ndarray) -> float:
        f = template.copy()
        f["score"] = score
        return float(heldout_robust_tpr(f, splits, SELECTION_TARGET_FPR))

    # The shared-alignment shortcut must not change any number it speeds up.
    probe = (names[0], names[1])
    ref = float(heldout_robust_tpr(
        fuse_scores([frames[x] for x in probe], splits=splits,
                    fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT),
        splits, SELECTION_TARGET_FPR))
    got = metric(Z_all[[names.index(x) for x in probe]].mean(0))
    if abs(ref - got) > 1e-12:
        raise SystemExit(f"SELF-CHECK FAILED: shared alignment gives {got:.6f} "
                         f"where fuse_scores gives {ref:.6f}; the speed-up "
                         "changed the semantics and no result below is valid.")
    print(f"\nself-check ok: {'+'.join(probe)} mean = {got:.4f} both ways", flush=True)

    rows: dict[str, dict] = {}
    for k in range(2, min(a.max_arity, len(names)) + 1):
        subsets = list(itertools.combinations(range(len(names)), k))
        print(f"\n=== arity {k}: {len(subsets)} subsets x rules ===", flush=True)
        for sub in subsets:
            sub_names = [names[i] for i in sub]
            bbs = sorted({backbone_of[n] for n in sub_names})
            params = sum(BACKBONES[b].params for b in bbs) + overhead
            res = {name: metric(v)
                   for name, v in build_rules(Z_all[list(sub)], R_all[list(sub)]).items()}
            best_rule = max(res, key=res.get)
            rows["+".join(sub_names)] = {
                "arity": k, "arms": sub_names, "backbones": bbs,
                "n_backbones": len(bbs), "params": int(params),
                "fits_param_cap": bool(params < PARAM_CAP),
                "legal_two_backbone": len(bbs) <= MAX_BACKBONES,
                "rules": {r: float(v) for r, v in res.items()},
                "best_rule": best_rule, "best": float(res[best_rule]),
                "mean": float(res["mean"]),
            }
        top = sorted([kv for kv in rows.items() if kv[1]["arity"] == k],
                     key=lambda kv: -kv[1]["best"])[:3]
        for kk, v in top:
            flag = "LEGAL " if v["legal_two_backbone"] else "barred"
            print(f"  {flag} {kk:<58s} {v['best']:.4f} via {v['best_rule']:<20s} "
                  f"(mean {v['mean']:.4f}) {v['params']:>13,}p", flush=True)

    # --- which rule wins, across everything, not on one lucky subset --------
    rule_names = sorted({r for v in rows.values() for r in v["rules"]})
    print(f"\n{'rule':>22s} {'best any':>9s} {'best legal':>11s} "
          f"{'mean d':>9s} {'wins':>6s}")
    rule_summary = {}
    for r in rule_names:
        have = [v for v in rows.values() if r in v["rules"]]
        deltas = [v["rules"][r] - v["mean"] for v in have]
        legal = [v["rules"][r] for v in have if v["legal_two_backbone"]]
        rule_summary[r] = {
            "best_any": max(v["rules"][r] for v in have),
            "best_legal": max(legal) if legal else None,
            "mean_delta_vs_mean_rule": float(np.mean(deltas)),
            "subsets_where_it_beats_mean": int(sum(d > 0 for d in deltas)),
            "n_subsets": len(have),
        }
        s = rule_summary[r]
        bl = f"{s['best_legal']:.4f}" if s["best_legal"] is not None else "     -"
        print(f"{r:>22s} {s['best_any']:9.4f} {bl:>11s} "
              f"{s['mean_delta_vs_mean_rule']:+9.4f} "
              f"{s['subsets_where_it_beats_mean']:>3d}/{s['n_subsets']:<3d}")

    legal_rows = {k: v for k, v in rows.items() if v["legal_two_backbone"]}
    best_legal = max(legal_rows, key=lambda k: legal_rows[k]["best"])
    capped = {k: v for k, v in rows.items() if v["fits_param_cap"]}
    best_capped = max(capped, key=lambda k: capped[k]["best"])
    biggest = max(capped, key=lambda k: capped[k]["params"])
    print(f"\nbest LEGAL (<= {MAX_BACKBONES} backbones): {best_legal} "
          f"{legal_rows[best_legal]['best']:.4f} via "
          f"{legal_rows[best_legal]['best_rule']}, "
          f"{legal_rows[best_legal]['params']:,} params "
          f"({100 * legal_rows[best_legal]['params'] / PARAM_CAP:.0f}% of cap)")
    print(f"best under the PARAMETER cap alone: {best_capped} "
          f"{capped[best_capped]['best']:.4f} via {capped[best_capped]['best_rule']}, "
          f"{capped[best_capped]['params']:,} params "
          f"({capped[best_capped]['n_backbones']} backbones)")
    print(f"largest combination that still fits: {biggest} "
          f"{capped[biggest]['params']:,} params, "
          f"{PARAM_CAP - capped[biggest]['params']:,} to spare")
    if not all(v["fits_param_cap"] for v in rows.values()):
        over = [k for k, v in rows.items() if not v["fits_param_cap"]]
        print(f"{len(over)} combination(s) exceed the parameter cap outright")

    # --- bootstrap the shortlist: rules vs the mean, on the same images -----
    path = Path(__file__).resolve().parent / "family_experts.py"
    spec = importlib.util.spec_from_file_location("_fe", path)
    fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)

    shortlist = sorted(legal_rows, key=lambda k: -legal_rows[k]["best"])[:3]
    panel_frames = {}
    for kk in shortlist:
        idx = [names.index(x) for x in legal_rows[kk]["arms"]]
        for r, v in build_rules(Z_all[idx], R_all[idx]).items():
            f = template.copy(); f["score"] = v
            panel_frames[f"{kk} | {r}"] = f
    baseline = f"{shortlist[0]} | mean"
    keep = sorted(panel_frames, key=lambda k: -metric(panel_frames[k]["score"].to_numpy()))
    panel_frames = {k: panel_frames[k] for k in keep[:a.boot_top]} | {baseline: panel_frames[baseline]}
    print(f"\n=== paired bootstrap: {len(panel_frames)} combiners, {a.boot_n} "
          f"resamples over IMAGES, baseline = {baseline} ===", flush=True)
    panel = fe.bootstrap_panel(panel_frames, splits, baseline=baseline, n_boot=a.boot_n)
    for kk in sorted(panel, key=lambda x: -panel[x]["point"]):
        p, vb = panel[kk], panel[kk]["vs_baseline"]
        lo, hi = vb["paired_ci95"]
        print(f"  {kk:<74s} {p['point']:.4f} d={vb['delta']:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {'SEPARATED' if (lo > 0 or hi < 0) else 'tie'}",
              flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "fusion_rules", "metric": "heldout_robust_tpr_at_1pct",
            "param_cap": PARAM_CAP, "max_backbones_for_legal": MAX_BACKBONES,
            "recon_overhead_included": bool(a.recon_overhead),
            "zscore_population": population,
            "singles": singles,
            "backbone_params": {b: BACKBONES[b].params
                                for b in sorted(set(backbone_of.values()))},
            "combinations": rows, "rule_summary": rule_summary,
            "best_legal": best_legal, "best_under_param_cap": best_capped,
            "largest_fitting": biggest, "bootstrap": panel,
            "caveat": ("primary split only, and 14 rules x 247 subsets is a "
                       "multiple-comparison surface: rank here, decide on the "
                       "bootstrap and on second_holdout_lattice"),
        }, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
