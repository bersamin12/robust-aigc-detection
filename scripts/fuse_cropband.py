"""Two-parent fusion of the SAME tower trained under two canonicalisation
policies (crop and band), plus the complementarity diagnostic that decides
whether fusing them can help at all.

    python scripts/fuse_cropband.py \
        --parent crop=data/banks/eval_ov7_d24:outputs/unfreeze/d24/checkpoint.pt \
        --parent band=data/banks/eval_ov7_d24band:outputs/unfreeze/d24band/checkpoint.pt \
        --out docs/fuse_cropband_ov7.json

WHY NOT `scripts/fuse_simplex.py`. That script fits its weight by maximising
`fusion.val_robust_tpr`, which needs BOTH classes inside `val_internal`. The
OV7 eval bank does not have them -- its `val_internal` rows are authentic and
its generated rows are all `heldout_generator` -- so the weight objective is
empty there and the sweep raises. Rather than weaken that guard (it is the only
thing keeping a fitted weight off the held-out rows), this script reports what
IS defined on such a bank: equal weighting, which is a fixed constant and not a
selection, plus a weight TRANSFERRED from a tier where the objective exists.
Where `val_internal` does carry both classes the sweep runs exactly as
`fuse_simplex` runs it, over the same grid and the same objective.

WHAT ACTUALLY DECIDES A FUSION. A parent that is simply worse everywhere makes
a fusion worse; a parent that is worse ON AVERAGE but wrong about DIFFERENT
images can still lift one. The standalone gap does not distinguish those two
cases, so the numbers below that matter most are not the fused TPRs but the
complementarity block: at each parent's own 1%-FPR threshold, how many of the
held-out generated images does the weaker parent catch that the stronger one
misses, and what would a perfect oracle over the two reach. If the oracle
ceiling is barely above the better parent, no weighting of these two scores can
help and the fused number is noise around the stronger parent.

DISCIPLINE, INHERITED NOT RESTATED. The z-score population is
`FIT_SPLITS_WHEN_FITTING_WEIGHT` (`val_internal` alone) whenever a weight is
fitted or transferred, because the standardisation and the weight are two
separate fits and the held-out rows must set neither. Each reported weighting
reads the held-out number exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (
    FIT_SPLITS_WHEN_FITTING_WEIGHT, WEIGHT_GRID, assert_fusion_parents,
    fuse_scores, val_robust_tpr, zscore_by_condition,
)
from aigcdet.eval.grid import score_grid
from aigcdet.eval.metrics import threshold_at_fpr
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector
from score_unfreeze_ladder import _load_head as _load_unfreeze_head

_KEYS = ["condition", "image_idx"]


def load_parent(ckpt: str, device: str):
    """The head this checkpoint holds, whichever of the two shapes it is.

    `train.finetune` writes a `config` with no `use_recon` key at all -- the
    fine-tune path has no aux branch to configure -- so `load_detector`, which
    reads `cfg["use_recon"]` directly, raises KeyError on an unfreeze rung. The
    ladder's own loader understands that shape, and it is IMPORTED rather than
    re-spelled here so a fused number and the ladder number it is compared
    against come out of an identically-built head. (Re-spelling one is how
    `score_d8_aux.py` acquired a duplicate loader that then drifted.)
    """
    try:
        model, ck = load_detector(ckpt, device=device)
        return model, bool(ck["config"].get("use_recon", False))
    except KeyError:
        return _load_unfreeze_head(ckpt, device)[0], False


def selection_rows(df: pd.DataFrame, splits: np.ndarray) -> pd.DataFrame:
    """The rows `heldout_robust_tpr` actually reads, marked with their role.

    Same population, built the same way: authentic from `val_internal`,
    generated from `heldout_generator`, `clean` dropped. Rebuilt here rather
    than imported because `heldout_robust_tpr` returns a scalar and the
    complementarity question is about which individual images survive it.
    """
    row_split = splits[df["image_idx"].to_numpy()]
    keep = (((row_split == "val_internal") & (df["label"].to_numpy() == 0))
            | ((row_split == "heldout_generator") & (df["label"].to_numpy() == 1)))
    out = df[keep & (df["condition"] != "clean")].copy()
    return out


def per_condition_hits(df: pd.DataFrame) -> pd.DataFrame:
    """`(condition, image_idx) -> caught`, at this parent's own 1% FPR point.

    The threshold is refitted per condition from that condition's authentic
    rows, which is what `tpr_at_fpr` does internally; using one global
    threshold would compare the parents at operating points neither was
    reported at.
    """
    parts = []
    for cond, g in df.groupby("condition", sort=False):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        thr = threshold_at_fpr(y, s, SELECTION_TARGET_FPR)
        parts.append(pd.DataFrame({"condition": cond,
                                   "image_idx": g["image_idx"].to_numpy(),
                                   "label": y, "caught": s >= thr}))
    return pd.concat(parts, ignore_index=True)


def complementarity(dfs, names, splits) -> dict:
    """Do the parents miss the SAME held-out images, or different ones?"""
    sel = [selection_rows(d, splits) for d in dfs]
    hits = [per_condition_hits(s).set_index(_KEYS) for s in sel]
    a, b = hits[0], hits[1].reindex(hits[0].index)
    pos = a["label"].to_numpy() == 1
    ca, cb = a["caught"].to_numpy(), b["caught"].to_numpy()

    # Score correlation is computed on the z-scored columns, because a raw
    # correlation between two heads with different logit spreads mostly
    # measures the spreads.
    zs = [zscore_by_condition(s).set_index(_KEYS)["score"] for s in sel]
    z0, z1 = zs[0], zs[1].reindex(zs[0].index)
    return {
        "n_heldout_positive_rows": int(pos.sum()),
        "pearson_r_zscored": float(np.corrcoef(z0.to_numpy(), z1.to_numpy())[0, 1]),
        "spearman_r_zscored": float(pd.Series(z0.to_numpy()).corr(
            pd.Series(z1.to_numpy()), method="spearman")),
        f"tpr_{names[0]}": float(ca[pos].mean()),
        f"tpr_{names[1]}": float(cb[pos].mean()),
        "tpr_either_ORACLE_CEILING": float((ca | cb)[pos].mean()),
        "tpr_both": float((ca & cb)[pos].mean()),
        f"{names[1]}_catch_rate_on_{names[0]}_misses": float(
            cb[pos & ~ca].mean()) if (pos & ~ca).any() else None,
        f"{names[0]}_catch_rate_on_{names[1]}_misses": float(
            ca[pos & ~cb].mean()) if (pos & ~cb).any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", action="append", required=True,
                    metavar="NAME=EVAL_BANK:CHECKPOINT", help="exactly two")
    ap.add_argument("--transfer-weight", type=float, default=None,
                    metavar="W0", help="weight on parent 0, fitted elsewhere")
    ap.add_argument("--heldout-sweep", action="store_true",
                    help="DIAGNOSTIC ONLY: read the held-out metric at every "
                         "grid point. Its argmax is not a selectable weight -- "
                         "it is chosen ON the rows it is read on -- and is "
                         "recorded only to answer whether ANY weighting of "
                         "these two parents could beat the better one, which "
                         "is a question about the parents, not a result.")
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
    if len(parents) != 2:
        raise SystemExit("this script fuses exactly two parents")
    names = [p[0] for p in parents]

    banks = [FeatureBank(d) for _, d, _ in parents]
    assert_fusion_parents(banks)          # canon_policy is deliberately not checked
    splits = banks[0].meta["split"].to_numpy().astype(str)
    split_series = banks[0].meta.set_index("image_idx")["split"]

    dfs, singles = [], []
    for (name, eval_dir, ckpt), bank in zip(parents, banks):
        model, use_recon = load_parent(ckpt, a.device)
        df = score_grid(model, bank, use_recon=use_recon, device=a.device)
        dfs.append(df)
        solo = float(heldout_robust_tpr(df, split_series, SELECTION_TARGET_FPR))
        singles.append({"parent": name, "eval_bank": eval_dir, "checkpoint": ckpt,
                        "canon_policy": bank.config.get("canon_policy"),
                        "heldout_robust_tpr_at_1pct": solo})
        print(f"parent {name:>6s}: {solo:.4f}   "
              f"({bank.config.get('canon_policy', {}).get('mode')})", flush=True)

    comp = complementarity(dfs, names, splits)
    print(f"\n--- complementarity on the {comp['n_heldout_positive_rows']} held-out "
          "generated rows, each parent at its own 1% FPR point")
    print(f"  score correlation (z-scored): pearson {comp['pearson_r_zscored']:+.4f}"
          f"   spearman {comp['spearman_r_zscored']:+.4f}")
    print(f"  TPR {names[0]:>6s}          : {comp['tpr_' + names[0]]:.4f}")
    print(f"  TPR {names[1]:>6s}          : {comp['tpr_' + names[1]]:.4f}")
    print(f"  TPR both agree      : {comp['tpr_both']:.4f}")
    print(f"  TPR either (ORACLE) : {comp['tpr_either_ORACLE_CEILING']:.4f}"
          "   <- no score fusion of these two can beat this")
    for k in (f"{names[1]}_catch_rate_on_{names[0]}_misses",
              f"{names[0]}_catch_rate_on_{names[1]}_misses"):
        print(f"  {k}: {comp[k]}")

    weightings, sweep, fit_note = {}, None, None
    def read_held(w0: float) -> float:
        fused = fuse_scores(dfs, weights=[w0, 1.0 - w0], splits=splits,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        return float(heldout_robust_tpr(fused, split_series, SELECTION_TARGET_FPR))

    weightings["equal"] = {"w0": 0.5, "heldout_robust_tpr_at_1pct": read_held(0.5)}
    print(f"\nequal weight (0.5/0.5): {weightings['equal']['heldout_robust_tpr_at_1pct']:.4f}")

    try:
        sweep = []
        for w in WEIGHT_GRID:
            fused = fuse_scores(dfs, weights=[float(w), 1.0 - float(w)],
                                splits=splits, fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            sweep.append({"w0": float(w),
                          "val_robust_tpr": float(val_robust_tpr(
                              fused, splits, SELECTION_TARGET_FPR))})
    except ValueError as exc:
        sweep, fit_note = None, str(exc).split("\n")[0]
        print(f"\nno weight fitted here: {fit_note}", flush=True)
    else:
        best = max(sweep, key=lambda r: (r["val_robust_tpr"], -abs(r["w0"] - 0.5)))
        flat = sum(1 for r in sweep
                   if r["val_robust_tpr"] >= best["val_robust_tpr"] - 1e-4)
        weightings["fitted"] = {
            "w0": best["w0"],
            "heldout_robust_tpr_at_1pct": read_held(best["w0"]),
            "val_robust_tpr_at_best": best["val_robust_tpr"],
            "n_grid_points_within_1e-4_of_max": flat,
            "n_grid_points": len(sweep)}
        print(f"fitted weight w0={best['w0']:.2f} : "
              f"{weightings['fitted']['heldout_robust_tpr_at_1pct']:.4f}"
              f"   (objective flat at {flat}/{len(sweep)} grid points)")

    if a.transfer_weight is not None:
        w0 = float(a.transfer_weight)
        weightings["transferred"] = {"w0": w0, "heldout_robust_tpr_at_1pct": read_held(w0)}
        print(f"transferred w0={w0:.2f}   : "
              f"{weightings['transferred']['heldout_robust_tpr_at_1pct']:.4f}")

    diagnostic = None
    if a.heldout_sweep:
        curve = [{"w0": float(w), "heldout_robust_tpr_at_1pct": read_held(float(w))}
                 for w in WEIGHT_GRID]
        top = max(curve, key=lambda r: r["heldout_robust_tpr_at_1pct"])
        diagnostic = {"NOT_A_SELECTABLE_WEIGHT": True, "curve": curve,
                      "argmax_w0": top["w0"],
                      "ceiling_heldout_robust_tpr_at_1pct":
                          top["heldout_robust_tpr_at_1pct"]}
        print("\nDIAGNOSTIC held-out sweep (argmax is fitted ON the reported "
              "rows -- never selectable):")
        print("  " + "  ".join(f"{r['w0']:.2f}:{r['heldout_robust_tpr_at_1pct']:.4f}"
                               for r in curve))
        print(f"  best any weighting: {top['heldout_robust_tpr_at_1pct']:.4f} "
              f"at w0={top['w0']:.2f}")

    best_parent = max(s["heldout_robust_tpr_at_1pct"] for s in singles)
    best_fused = max(v["heldout_robust_tpr_at_1pct"] for v in weightings.values())
    verdict = (f"fusion beats the better parent by {best_fused - best_parent:+.4f}"
               if best_fused > best_parent else
               f"no fusion of these two beats the better parent alone "
               f"({best_fused - best_parent:+.4f})")
    print(f"\nbest single: {best_parent:.4f}   best fused: {best_fused:.4f}")
    print(f"VERDICT: {verdict}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"probe": "fuse_cropband", "off_ladder": True,
                   "metric": "heldout_robust_tpr_at_1pct",
                   "zscore_fit_splits": list(FIT_SPLITS_WHEN_FITTING_WEIGHT),
                   "weight_fit_objective": "val_robust_tpr (val_internal both classes)",
                   "weight_not_fitted_reason": fit_note,
                   "parents": singles, "complementarity": comp,
                   "weightings": weightings, "sweep": sweep,
                   "heldout_sweep_diagnostic": diagnostic,
                   "verdict": verdict}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
