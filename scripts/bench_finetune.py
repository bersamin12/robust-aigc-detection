#!/usr/bin/env python3
"""Measure the unfreeze ladder's real fwd+bwd throughput, per depth.

The ladder's schedule is quoted in images/second and that number has so far
been an assumption. This measures it on the tower and the crop size the run
will actually use, at each depth, so the plan is costed from the machine
rather than from a guess.

Timing excludes a warmup pass (kernel autotuning, allocator growth) and
synchronises around each step -- without both, CUDA's asynchrony reports the
launch rate rather than the compute rate, which is fast and meaningless.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from aigcdet.features.backbones import load_backbone
from aigcdet.models.heads import Detector
from aigcdet.train.finetune import _forward_tower, unfreeze_last_n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2regl")
    ap.add_argument("--depths", default="0,1,2,4")
    ap.add_argument("--n-src", type=int, default=64)
    ap.add_argument("--m-deg", type=int, default=2)
    ap.add_argument("--crop", type=int, default=200)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--src-chunk", type=int, default=4)
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    # Synthetic pixels: this measures the GPU step, and decode/augment is
    # measured separately (it is thread-pooled behind the step, so what matters
    # is which of the two is larger, not their sum).
    clean = [rng.integers(0, 256, (a.crop, a.crop, 3), dtype=np.uint8)
             for _ in range(a.n_src)]
    deg = [rng.integers(0, 256, (a.crop, a.crop, 3), dtype=np.uint8)
           for _ in range(a.n_src * a.m_deg)]
    per_step = len(clean) + len(deg)

    rows = []
    for depth in [int(d) for d in a.depths.split(",")]:
        tower, spec = load_backbone(a.backbone, device=a.device)
        tower = tower.float()
        rec = unfreeze_last_n(tower, depth)
        tower.eval()
        head = Detector(dim_feat=spec.dim, use_recon=False).to(a.device)
        params = [{"params": list(head.parameters()), "lr": 1e-3}]
        tp = [p for p in tower.parameters() if p.requires_grad]
        if tp:
            params.append({"params": tp, "lr": 1e-5})
        opt = torch.optim.AdamW(params)

        if depth and not a.no_checkpointing and hasattr(
                tower, "gradient_checkpointing_enable"):
            tower.gradient_checkpointing_enable()

        def step():
            # Mirrors `train_finetune`'s accumulation exactly, because the
            # number being measured is the schedule's, and a benchmark of a
            # different loop is a benchmark of nothing.
            opt.zero_grad(set_to_none=True)
            for s0 in range(0, len(clean), a.src_chunk):
                srcs = clean[s0:s0 + a.src_chunk]
                p0, p1 = s0 * a.m_deg, (s0 + len(srcs)) * a.m_deg
                degs = deg[p0:p1]
                share = len(degs) / len(deg)
                with torch.autocast(device_type=a.device.split(":")[0],
                                    dtype=torch.bfloat16, enabled=a.device != "cpu"):
                    f_src = _forward_tower(tower, spec, srcs, a.device,
                                           torch.float32, a.src_chunk)
                    f_clean = f_src.repeat_interleave(a.m_deg, dim=0)
                    f_deg = _forward_tower(tower, spec, degs, a.device,
                                           torch.float32, a.src_chunk)
                loss = (head(f_deg.float(), None)["logit"].mean()
                        + head(f_clean.float(), None)["logit"].mean())
                (loss * share).backward()
            opt.step()

        step()                                   # warmup, not timed
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            step()
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.steps

        rate = per_step / dt
        mem = (torch.cuda.max_memory_allocated() / 2**30
               if a.device.startswith("cuda") else 0.0)
        rows.append({"depth": depth, "trainable_params": rec["trainable_params"],
                     "sec_per_step": dt, "img_per_s": rate, "peak_gib": mem})
        print(f"depth {depth}: {rate:7.1f} img/s  {dt:6.3f} s/step  "
              f"trainable {rec['trainable_params']/1e6:7.2f}M  peak {mem:5.2f} GiB")
        del tower, head, opt
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # What the schedule actually needs: minutes per epoch for both corpora.
    print()
    for name, pos, neg in [("probe", 7866, 8134), ("full", 162091, 169166)]:
        batches = max(1, min(pos, neg) // (a.n_src // 2))
        imgs = batches * per_step
        print(f"{name:6s} {batches:5d} batches/epoch, {imgs:9,d} images/epoch")
        for r in rows:
            m = imgs / r["img_per_s"] / 60
            print(f"        depth {r['depth']}: {m:7.1f} min/epoch   "
                  f"5 epochs {5*m/60:5.2f} h")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"backbone": a.backbone, "crop": a.crop,
                       "n_src": a.n_src, "m_deg": a.m_deg, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
