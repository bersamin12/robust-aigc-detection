#!/usr/bin/env python3
"""Is this backbone finite in the dtype its registry entry declares?

THE REASON THIS SCRIPT EXISTS. On 2026-08-29 a five-hour DINOv3-L bank came
back 131,116 x 11 vectors of NaN. It was produced at full speed, nothing
raised, and the only post-condition checked was the row count. DINOv3-L's
activations overflow float16 at hidden layer 1; the fix was bfloat16, and the
lesson was that a dtype is a property of the CHECKPOINT which has to be
measured rather than inherited from a neighbouring entry.

`dinov2l`'s float16 was measured that way afterwards and its numbers are in the
registry comment. This script is that measurement, generalised, so the next
entry does not have to be a bespoke session at 2am.

What it reports, per backbone:

  finite       every pooled value finite at the declared dtype
  max|diff|    against a float32 run of the same images -- the accuracy cost
  max|x|       largest pooled magnitude, against float16's 65504 ceiling.
               A run can be finite in the TOWER and still overflow on the way
               to disk: the bank stores float16 (bank.py:294-303).

A non-finite result is not a bad image and not a bug in this script. It means
the entry's `dtype` is wrong -- try bfloat16, and note that on a GPU without
native bfloat16 `run_dtype` degrades that to float32, which is slower but
correct.

Usage:

    python scripts/probe_backbone_dtype.py \\
        --backbone dinov2regl --backbone eva02l \\
        --manifest data/probe/manifest_union_probe.parquet \\
        --root /home/administrator/aigc_probe_ssd/train_root \\
        --out docs/backbone_dtype_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

#: float16's largest finite value. A pooled vector above this is finite in a
#: float32 tower and NaN once the bank writes it.
FLOAT16_MAX = 65504.0
N_IMAGES = 24
SEED = 20260827


def _load_images(manifest_path, root, n, seed):
    """`n` real images, decoded and canonicalised exactly as extraction does.

    Real images and not noise: the 2026-08-29 overflow was driven by activation
    magnitudes that a uniform-random tensor does not produce.
    """
    import cv2
    import pandas as pd

    from aigcdet.augment.canonical import canonicalise

    df = pd.read_parquet(manifest_path, columns=["rel_path", "label"])
    # Both classes, because a generated image and a photograph do not have the
    # same activation statistics and either could be the one that overflows.
    rng = np.random.default_rng(seed)
    picks = []
    for label in (0, 1):
        rows = df[df["label"] == label]
        take = rng.choice(len(rows), size=min(n // 2, len(rows)), replace=False)
        picks += [rows.iloc[int(i)]["rel_path"] for i in take]

    imgs = []
    for rel in picks:
        path = os.path.join(root, rel)
        raw = cv2.imread(path, cv2.IMREAD_COLOR)
        if raw is None:
            continue
        imgs.append(canonicalise(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)))
    if not imgs:
        raise SystemExit(f"no images decoded under {root}")
    return imgs


def probe_one(name: str, imgs, device: str) -> dict:
    import torch

    from aigcdet.features.backbones import BACKBONES, embed, load_backbone, run_dtype

    spec = BACKBONES[name]
    effective = run_dtype(spec, device)

    model, _ = load_backbone(name, device=device)
    got = embed(model, spec, imgs, device=device, batch_size=8)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # The float32 reference, through the SAME code path -- a spec with the
    # dtype overridden, not a hand-rolled forward, so a preprocessing
    # difference cannot masquerade as a dtype difference.
    from dataclasses import replace
    ref_spec = replace(spec, dtype=torch.float32)
    BACKBONES[name] = ref_spec
    try:
        ref_model, _ = load_backbone(name, device=device)
        ref = embed(ref_model, ref_spec, imgs, device=device, batch_size=8)
        del ref_model
    finally:
        BACKBONES[name] = spec
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    finite = bool(np.isfinite(got).all())
    max_abs = float(np.abs(got).max()) if finite else float("nan")
    return {
        "backbone": name,
        "hf_id": spec.hf_id,
        "declared_dtype": str(spec.dtype).replace("torch.", ""),
        "effective_dtype": str(effective).replace("torch.", ""),
        "dim": spec.dim,
        "image_size": spec.image_size,
        "n_images": len(imgs),
        "finite": finite,
        "n_nonfinite": int((~np.isfinite(got)).sum()),
        "max_abs_diff_vs_float32": (
            float(np.abs(got - ref).max()) if finite else None),
        "max_abs_pooled": max_abs,
        "fits_float16_storage": bool(finite and max_abs < FLOAT16_MAX),
        "float32_max_abs_pooled": float(np.abs(ref).max()),
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", action="append", required=True,
                   help="repeatable; a registry name")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="docs/backbone_dtype_probe.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-images", type=int, default=N_IMAGES)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    imgs = _load_images(a.manifest, a.root, a.n_images, a.seed)
    print(f"{len(imgs)} canonicalised images from {a.manifest}\n")

    results, failed = [], []
    for name in a.backbone:
        print(f"--- {name}")
        r = probe_one(name, imgs, a.device)
        results.append(r)
        verdict = "OK" if r["finite"] and r["fits_float16_storage"] else "FAIL"
        print(f"    declared {r['declared_dtype']} -> effective "
              f"{r['effective_dtype']}")
        print(f"    finite {r['finite']}  ({r['n_nonfinite']} non-finite)")
        if r["finite"]:
            print(f"    max|diff| vs float32  {r['max_abs_diff_vs_float32']:.3e}")
            print(f"    max|pooled|           {r['max_abs_pooled']:.2f}  "
                  f"(float16 stores up to {FLOAT16_MAX:.0f})")
        print(f"    {verdict}\n")
        if verdict == "FAIL":
            failed.append(name)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(imgs), "device": a.device,
                   "seed": a.seed, "results": results}, f, indent=2)
    print(f"wrote {a.out}")

    if failed:
        raise SystemExit(
            f"\n{len(failed)} backbone(s) are NOT safe at their declared dtype: "
            f"{', '.join(failed)}. Do not extract a bank until the registry "
            f"entry is fixed -- an overflow is silent and costs the whole run.")
    print("\nevery backbone probed is finite and stores in the bank's float16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
