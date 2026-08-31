#!/usr/bin/env python3
"""Train two towers in parallel into one head (experiment 2).

Mirrors `train_unfreeze.py`'s interface so the two experiments are launched the
same way; the difference is `--backbone2` and that the head sees 2*dim.

Like `train_unfreeze.py` this prints NOTHING until an epoch completes -- the
work is inside `train_dual` and there is no per-batch progress bar. Silence is
not a symptom. Unlike the original, a killed run is now recoverable: every
epoch writes an atomic checkpoint and `--resume` picks it up.
"""
from __future__ import annotations

import argparse
import os

from aigcdet.train.finetune import FinetuneConfig
from aigcdet.train.finetune_dual import DualFinetuneConfig, train_dual


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True,
                    help="supplies rows, splits, labels and degradation "
                         "targets; its cached features are NOT used")
    ap.add_argument("--root", required=True, help="where the corpus is mounted")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--backbone2", default=None,
                    help="defaults to --backbone, i.e. two independently "
                         "fine-tuned copies of one pretrained model")
    ap.add_argument("--depth", type=int, required=True,
                    help="trailing blocks unfrozen IN EACH tower")
    ap.add_argument("--perturb-tower2", type=float, default=0.0,
                    help="std of Gaussian noise on tower 2's init weights. 0 "
                         "keeps both towers the true pretrained model; the "
                         "only symmetry break is then the head's random init")
    ap.add_argument("--name", default="dual")
    ap.add_argument("--out-dir", default="outputs/dual")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tower-lr", type=float, default=1e-5)
    ap.add_argument("--n-src", type=int, default=64)
    ap.add_argument("--m-deg", type=int, default=2)
    ap.add_argument("--src-chunk", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--canon-mode", default="crop")
    ap.add_argument("--crop-side", type=int, default=200)
    ap.add_argument("--train-subsample-frac", type=float, default=1.0)
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    base = FinetuneConfig(
        name=a.name, bank_dir=a.bank, root=a.root, depth=a.depth,
        backbone=a.backbone, out_dir=a.out_dir, epochs=a.epochs, lr=a.lr,
        tower_lr=a.tower_lr, n_src=a.n_src, m_deg=a.m_deg,
        src_chunk=a.src_chunk, workers=a.workers, seed=a.seed,
        device=a.device, policy_mode=a.canon_mode, crop_side=a.crop_side,
        grad_checkpointing=not a.no_checkpointing, resume=a.resume,
        train_subsample_frac=a.train_subsample_frac)
    cfg = DualFinetuneConfig(base=base, backbone2=a.backbone2,
                             perturb_tower2=a.perturb_tower2)

    res = train_dual(cfg)
    if int(os.environ.get("RANK", "0")) != 0:
        return 0
    print(f"towers: {res['backbones']}  head input: {res['dim_feat']}")
    for i, u in enumerate(res["unfrozen"]):
        print(f"  tower{i}: depth={u['depth']} "
              f"trainable={u['trainable_params']:,} "
              f"({u['n_blocks']} blocks at {u['block_path']!r})")
    for h in res["history"]:
        print(f"  epoch {h['epoch']}: total={h['total']:.4f} "
              f"cls={h['cls']:.4f} deg={h['deg']:.4f} con={h['con']:.4f}")
    print(f"checkpoint: {res['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
