#!/usr/bin/env python3
"""Score a directory of images for the likelihood that each one is AI-generated.

This is the submission's inference entry point: a directory in, a JSON file
out, one ``{"image_path": ..., "pred": ...}`` object per image, where ``pred``
is P(AI-generated) in [0, 1].

    python predict.py --images path/to/dir --out predictions.json

With no ``--checkpoint``, the released model (two fine-tuned DINOv2 ViT-L
towers at 224px into one MLP head) is downloaded from the Hugging Face Hub on
first use and cached. Pass ``--checkpoint`` to score a local file instead --
either a training checkpoint from ``scripts/train_dual.py`` /
``scripts/run_unfreeze_ladder.sh`` or a slim export from
``scripts/export_finetuned.py``; both shapes are handled.

Everything else in this repository consumes a frozen manifest and a cached
feature bank. This does not. It is handed loose files with no manifest, no
labels and no bank, which is exactly how it will be run by someone who did
not build it. So it establishes for itself what the rest of the pipeline gets
from the manifest: which files are images, in what order, and what happens to
the one that will not decode.

**A failure is loud.** An empty directory, a missing checkpoint, or a file
PIL cannot decode are each reported by name; ``--skip-bad`` downgrades an
undecodable file to a warning on stderr and an omitted row, which is the
right behaviour for a judge's mixed directory but must be asked for, because
a silently short result file is indistinguishable from a successful run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

#: The released checkpoint: dual dinov2regl @ 224, epoch 1, slim export.
HF_REPO_ID = "justintimo/aquaforge8-aigc-dual-d24"
HF_FILENAME = "dual_d24_ep1.pt"

#: Extensions PIL can open that this project's data actually uses. A file
#: that is not one of these is skipped without comment -- a judge's directory
#: will contain a README, a .DS_Store and a licence file, and none of them
#: are errors.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})


def find_images(root: str) -> list[str]:
    """Every image under `root`, recursively, in a stable order.

    Sorted, because `os.walk` yields whatever `readdir` returned and a judge
    diffing two runs of the same directory should get identical files rather
    than identical contents in a different order.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def resolve_checkpoint(arg: str | None) -> str:
    """A local path to weights: the caller's, or the released model's."""
    if arg is not None:
        if not os.path.isfile(arg):
            raise SystemExit(f"--checkpoint {arg!r} is not a file")
        return arg
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is needed to fetch the released model "
            "(pip install huggingface_hub), or pass --checkpoint <file>")
    print(f"fetching {HF_REPO_ID}/{HF_FILENAME} (cached after first use)...",
          file=sys.stderr)
    return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score images for the likelihood that they are AI-generated.")
    ap.add_argument("--images", required=True,
                    help="directory to score, searched recursively")
    ap.add_argument("--out", default="predictions.json",
                    help="output JSON: [{image_path, pred}, ...]")
    ap.add_argument("--checkpoint", default=None,
                    help="local checkpoint to score with; default: download "
                         f"the released model ({HF_REPO_ID})")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8,
                    help="decode/canonicalise threads feeding the GPU")
    ap.add_argument("--tta", action="store_true",
                    help="8-view test-time augmentation, mean of per-view "
                         "logits (8x the compute; the reported numbers are "
                         "single-view, so this is off by default)")
    ap.add_argument("--swa", action="store_true",
                    help="score the checkpoint's SWA average instead of the "
                         "final weights (training checkpoints only)")
    ap.add_argument("--skip-bad", action="store_true",
                    help="warn and omit an undecodable file instead of "
                         "failing the run")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.images):
        raise SystemExit(f"--images {a.images!r} is not a directory")
    paths = find_images(a.images)
    if not paths:
        raise SystemExit(f"no images found under {a.images!r} "
                         f"(looked for {', '.join(sorted(IMAGE_SUFFIXES))})")

    if a.skip_bad:
        from PIL import Image
        ok = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.verify()
                ok.append(p)
            except Exception as e:  # noqa: BLE001 -- report and move on
                print(f"WARNING: skipping undecodable {p}: {e}", file=sys.stderr)
        if not ok:
            raise SystemExit(f"every file under {a.images!r} failed to decode")
        paths = ok

    from aigcdet.infer_finetuned import load_finetuned, score_paths

    ckpt = resolve_checkpoint(a.checkpoint)
    towers, specs, head, policy, ck = load_finetuned(ckpt, a.device,
                                                     use_swa=a.swa)
    names = ck.get("backbones", [ck.get("backbone")])
    print(f"model: {'+'.join(names)} (epoch {ck.get('epoch')}), "
          f"policy {policy.as_record()}, device {a.device}", file=sys.stderr)

    probs = score_paths(paths, towers, specs, head, policy, a.device,
                        batch=a.batch_size, workers=a.workers,
                        tta=a.tta, progress=True)

    rows = [{"image_path": p, "pred": float(pr)}
            for p, pr in zip(paths, probs)]
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(rows)} predictions to {a.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "src"))
    raise SystemExit(main())
