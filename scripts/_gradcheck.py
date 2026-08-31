"""Is the reduced gradient the single-GPU gradient, or a scaled copy of it?

Weight comparison cannot answer this. AdamW divides by sqrt(v), so a gradient
uniformly wrong by a constant factor -- exactly what averaging instead of
summing across ranks would produce -- moves the weights to almost the same
place and hides itself. The gradient NORM is the thing that has to match.

Wraps the real `AdamW.step` rather than reimplementing the loop, so this is
measuring the production path and not a second copy of it.
"""
import os, sys, torch
import torch.optim as O
from aigcdet.train.finetune import FinetuneConfig, train_finetune

orig, seen = O.AdamW.step, {"n": 0}


def step(self, *a, **k):
    if seen["n"] == 0:
        tot = sum(float(p.grad.double().pow(2).sum())
                  for g in self.param_groups for p in g["params"]
                  if p.grad is not None)
        print(f"RANK{os.environ.get('RANK','0')} "
              f"WORLD{os.environ.get('WORLD_SIZE','1')} "
              f"GRADNORM {tot ** 0.5:.10f}", flush=True)
    seen["n"] += 1
    return orig(self, *a, **k)


O.AdamW.step = step
train_finetune(FinetuneConfig(
    name="gradchk", bank_dir="data/banks/probe_crop_dinov2regl_local",
    root="/workspace/data/probe", depth=1, epochs=1,
    train_subsample_frac=0.05, src_chunk=8, out_dir="/tmp/gradchk",
    device="cuda"))
