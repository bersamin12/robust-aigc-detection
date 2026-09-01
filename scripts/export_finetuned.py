#!/usr/bin/env python3
"""Strip a fine-tuned training checkpoint down to an inference file.

    python scripts/export_finetuned.py --ckpt outputs/dual/dual_d24/checkpoint_ep1.pt \
        --out dual_d24_ep1.pt

A training checkpoint carries the AdamW state (2x the weights), both RNG
streams and the history, because its job is to survive a kill mid-run. None
of that is inference; what remains after `strip_checkpoint` is ~1/3 the size,
with the tower weights in bf16 -- which the scoring path casts them to anyway,
so the exported file scores bit-identically to the original (there is a
built-in check for exactly that: `--verify DIR` scores a directory of images
with both files and compares).

This is the file `predict.py` downloads from the Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from aigcdet.infer_finetuned import load_finetuned, score_paths, strip_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="a training checkpoint")
    ap.add_argument("--out", required=True, help="the slim file to write")
    ap.add_argument("--swa", action="store_true",
                    help="export the SWA average as THE weights of the file")
    ap.add_argument("--verify", default=None, metavar="DIR",
                    help="a directory of a few images; both files are scored "
                         "on it and the probabilities compared")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("exported"):
        raise SystemExit(f"{a.ckpt} is already an exported file")
    slim = strip_checkpoint(ck, use_swa=a.swa)
    tmp = a.out + ".tmp"
    torch.save(slim, tmp)
    os.replace(tmp, a.out)
    src_gb = os.path.getsize(a.ckpt) / 1e9
    out_gb = os.path.getsize(a.out) / 1e9
    print(f"wrote {a.out}: {out_gb:.2f} GB (from {src_gb:.2f} GB), "
          f"weights={slim['exported_weights']}, epoch={slim['epoch']}")

    if a.verify:
        verify(a.ckpt, a.out, a.verify, a.device)
    return 0


def verify(orig: str, slim: str, img_dir: str, device: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from predict import find_images  # repo root

    paths = find_images(img_dir)[:8]
    if not paths:
        raise SystemExit(f"--verify {img_dir!r} holds no images")
    got = []
    for path in (orig, slim):
        towers, specs, head, policy, _ck = load_finetuned(path, device)
        got.append(score_paths(paths, towers, specs, head, policy, device))
        del towers, head
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    diff = np.abs(got[0] - got[1]).max()
    print(f"verify: max |p_orig - p_slim| = {diff:.2e} over {len(paths)} images")
    if diff > 1e-6:
        raise SystemExit("REFUSING: the exported file does not reproduce the "
                         "original's probabilities")


if __name__ == "__main__":
    raise SystemExit(main())
