"""Is the READOUT the bottleneck, or the features? (off-ladder)

    python scripts/head_capacity.py --bank data/banks/convnextt \
        --eval-bank data/banks/eval_convnextt --out docs/head_capacity.json

THE QUESTION. `readout_ceiling` showed that a closed-form readout TIES the
trained MLP on dinov3l (0.9005 vs 0.9012) but loses by 7-11 points on weaker
towers -- so the readout does real work when the features are poor. That says
training matters; it does not say the current WIDTH is enough. This sweeps it.

WHY IT IS NOT JUST CURIOSITY. Head capacity is confounded with tower quality in
every comparison already on record. `Detector` sizes its first Linear from the
bank's dim, so a 5632-d convnextv2h bank buys a 4.46M head against a 1024-d ViT
arm's 0.92M -- 4.8x -- for free. No number anywhere separates "the tower is
better" from "the head was bigger". A width sweep on ONE bank does, because the
features are held fixed and only the readout moves.

    hidden   total Detector params (at dim_feat=1024)
      256          428,941
      512          923,405     <- every number this repo has reported
     1024        2,108,941
     2048        5,266,445

READING IT. The metric is heldout_robust_tpr_at_1pct, so a win cannot be
memorisation of seen generators -- the held-out generators are absent from
training by construction. A curve that PLATEAUS says the features are the
bottleneck and a bigger readout is spent effort; one that keeps CLIMBING says
we have been under-parameterised and no fine-tuning was needed to find out.

OFF-LADDER. Width is not a rung flag: `tests/test_rung_ladder.py` enforces that
a rung differs from its base by exactly one FLAG, and this differs by a size.
It therefore writes its own JSON and is never a §6.4 selection candidate.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector, train_rung

sys.path.insert(0, "scripts")
from run_ablation import load_rung_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--config", default="configs/rungs/a3.yaml")
    ap.add_argument("--widths", default="256,512,1024,2048")
    ap.add_argument("--out-dir", default="outputs/head_capacity")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="docs/head_capacity.json")
    a = ap.parse_args()

    eval_bank = FeatureBank(a.eval_bank)
    splits = eval_bank.meta.set_index("image_idx")["split"]

    rows = []
    for w in [int(x) for x in a.widths.split(",")]:
        cfg = load_rung_config(a.config, a.bank, f"{a.out_dir}/h{w}", a.device)
        cfg.name = f"h{w}"
        cfg.head_hidden = w
        print(f"\n=== hidden={w} ===", flush=True)
        result = train_rung(cfg)
        model, ck = load_detector(result["checkpoint"], device=a.device)
        n_params = sum(p.numel() for p in model.parameters())
        df = score_grid(model, eval_bank, use_recon=cfg.use_recon, device=a.device)
        tpr = heldout_robust_tpr(df, splits, SELECTION_TARGET_FPR)
        rows.append({"hidden": w, "detector_params": int(n_params),
                     "heldout_robust_tpr_at_1pct": float(tpr),
                     "val_auc_clean_view_only": result.get("val_auc"),
                     "val_auc_mean_views": result.get("val_auc_mean_views")})
        print(f"hidden={w}  params={n_params:,}  heldout={tpr:.4f}", flush=True)

    base = next((r for r in rows if r["hidden"] == 512), None)
    print(f"\n{'hidden':>7s} {'params':>11s} {'heldout':>9s} {'vs 512':>8s}")
    for r in rows:
        d = (r["heldout_robust_tpr_at_1pct"] - base["heldout_robust_tpr_at_1pct"]
             ) if base else float("nan")
        print(f"{r['hidden']:7d} {r['detector_params']:11,} "
              f"{r['heldout_robust_tpr_at_1pct']:9.4f} {d:+8.4f}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"probe": "head_capacity", "off_ladder": True,
                   "not_eligible_reason": "width is not a rung flag; §6.4 "
                                          "chooses among a3-a6 and this is none of them",
                   "metric": "heldout_robust_tpr_at_1pct",
                   "bank": a.bank, "eval_bank": a.eval_bank,
                   "base_config": a.config, "results": rows}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
