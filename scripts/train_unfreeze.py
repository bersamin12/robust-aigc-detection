#!/usr/bin/env python3
"""Train one rung of the unfreeze depth ladder (D0..D4).

One depth per invocation, so the ladder is four independent processes on four
cards rather than one long serial run. Each writes its own checkpoint carrying
BOTH the head and the tower -- at depth > 0 the head alone no longer describes
the model, and the eval bank this rung is scored on can only be extracted by
the tower that produced its training features.

D0 is the control. It unfreezes nothing, so its tower is the frozen tower and
its head is trained on exactly the pixels the cached bank holds
(`features.extract.build_view` keys every view on `(seed, row_id, view_idx)`
alone, and `LiveViewSampler` inherits `PairedSampler.draw_batch` so the batch
sequence matches too). It is still trained rather than copied from `a3`,
because a3's head was fitted on float16 cached features and this path runs the
tower in float32 -- see `FinetuneConfig.tower_dtype`. Reading D1..D4 against a3
instead of against D0 would fold that difference into the depth effect.
"""
from __future__ import annotations

import argparse
import json
import os

from aigcdet.train.finetune import FinetuneConfig, train_finetune


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True,
                    help="training bank -- read for labels, splits and the "
                         "degradation targets, never for its features")
    ap.add_argument("--root", required=True, help="where the corpus is mounted")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--name", default=None, help="defaults to d<depth>")
    ap.add_argument("--out-dir", default="outputs/unfreeze")
    ap.add_argument("--backbone", default=None, help="defaults to the bank's")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tower-lr", type=float, default=1e-5)
    ap.add_argument("--n-src", type=int, default=64)
    ap.add_argument("--m-deg", type=int, default=2)
    ap.add_argument("--src-chunk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--canon-mode", default="crop")
    ap.add_argument("--crop-side", type=int, default=200)
    ap.add_argument("--train-subsample-frac", type=float, default=1.0,
                    help="keep this fraction of train rows, stratified by "
                         "(generator, label); for data-scaling reads")
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--lr-schedule", default="constant",
                    choices=("constant", "cosine"),
                    help="cosine adds warmup + decay; OFF by default because a "
                         "scheduled arm is not one flag from the constant-LR "
                         "ladder and may not be quoted against it")
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--swa", action="store_true",
                    help="average weights over the tail; saved BESIDE the "
                         "final weights, both scorable")
    ap.add_argument("--swa-start-frac", type=float, default=0.75)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out-dir>/<name>/checkpoint.pt if it "
                         "exists; refuses if its config differs on anything "
                         "that changes the model or the data")
    a = ap.parse_args()

    cfg = FinetuneConfig(
        name=a.name or f"d{a.depth}", bank_dir=a.bank, root=a.root,
        depth=a.depth, backbone=a.backbone, out_dir=a.out_dir,
        epochs=a.epochs, lr=a.lr, tower_lr=a.tower_lr, n_src=a.n_src,
        m_deg=a.m_deg, src_chunk=a.src_chunk, workers=a.workers, seed=a.seed,
        device=a.device, policy_mode=a.canon_mode, crop_side=a.crop_side,
        grad_checkpointing=not a.no_checkpointing, resume=a.resume,
        lr_schedule=a.lr_schedule, warmup_frac=a.warmup_frac,
        min_lr_frac=a.min_lr_frac, swa=a.swa, swa_start_frac=a.swa_start_frac,
        train_subsample_frac=a.train_subsample_frac)

    res = train_finetune(cfg)
    if int(os.environ.get("RANK", "0")) != 0:
        return 0          # one writer, one reporter
    u = res["unfrozen"]
    print(f"{cfg.name}: depth={u['depth']} trainable={u['trainable_params']:,} "
          f"of {u['tower_params']:,} tower params "
          f"({u['n_blocks']} blocks at {u['block_path']!r})")
    for h in res["history"]:
        print(f"  epoch {h['epoch']}: total={h['total']:.4f} cls={h['cls']:.4f} "
              f"deg={h['deg']:.4f} con={h['con']:.4f}")
    print(f"checkpoint: {res['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
