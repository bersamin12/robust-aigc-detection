"""Score a directory of images for the likelihood that each one is AI-generated.

This is the submission's inference entry point (brief §5.5.2): a directory in,
a JSON file out, one `{"image_path": ..., "pred": ...}` object per image, where
`pred` is P(AI-generated) in [0, 1].

    python scripts/predict.py --images path/to/dir \
           --checkpoint outputs/rungs/a3/checkpoint.pt --out preds.json

Everything else in this repository consumes a frozen manifest and a cached
feature bank. This does not: it is handed loose files with no manifest, no
labels and no bank, which is exactly how it will be run by someone who did not
build it. Three things follow from that, and each is a property the rest of the
pipeline gets from the manifest and this script has to establish itself.

**It is the FOURTH decode site.** `features/extract.py`, `eval/grid.py` and
`features/recon.py` all canonicalise resolution before anything else touches
the pixels (docs/resolution_shortcut.md). So does this. A decode site that
skips it embeds different pixels than the head was trained on -- no shape
error, no warning, just scores that are quietly wrong. `eval/controls.py` is
the one deliberate exception, because its whole job is to measure the shortcut.

**It scores the clean view only.** The bank's view 0 is the undegraded view and
that is what inference corresponds to. Nothing here samples a recipe; the
transforms exist to make training robust to degradation, not to degrade the
image someone asked about.

**A failure is loud.** An empty directory, a checkpoint whose head needs
features this script cannot supply, or a file PIL cannot decode are each
reported by name. The alternative -- `[]` and exit 0, or a silently short
result file -- is indistinguishable from a successful run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from aigcdet.augment.canonical import canonicalise
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.train.train_head import load_detector

#: Extensions PIL can open that this project's data actually uses. A file that
#: is not one of these is skipped without comment -- a judge's directory will
#: contain a README, a .DS_Store and a licence file, and none of them are
#: errors.
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


def load_batch(paths: list[str]) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Decode and canonicalise `paths` -> (images, kept_paths, failures).

    A file that cannot be decoded is reported rather than raised on: a judge's
    directory will contain one, and losing every score in the run to it is a
    worse outcome than losing that image's score. Skipping it *silently* is
    worse than both -- a result file with fewer rows than the directory has
    images, and nothing to say why.

    `kept_paths` is returned rather than recovered by the caller, and that is
    the whole point of the three-tuple. The obvious alternative -- filter the
    input list against the failure messages -- means parsing `"<path>: <error>"`
    back apart, and a colon is legal in a POSIX filename. Such a path survives
    the filter, `zip` truncates against the shorter image list, and every score
    lands on the WRONG image_path. Each row still looks perfectly well-formed.
    """
    images, kept, failed = [], [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception as exc:                      # noqa: BLE001 -- reported
            failed.append(f"{p}: {type(exc).__name__}: {exc}")
            continue
        # The fourth decode site. See the module docstring: this must happen
        # here, before the backbone sees anything, or the features do not
        # correspond to the ones the head was trained on.
        images.append(canonicalise(base))
        kept.append(p)
    return images, kept, failed


def score(model, backbone, spec, images: list[np.ndarray], device: str,
          batch_size: int) -> np.ndarray:
    """P(AI-generated) for each image, as float64 in [0, 1].

    One forward per batch through the frozen backbone, then the head. The
    logit is squashed with a sigmoid; it is NOT temperature-calibrated here,
    because a temperature fitted on this project's val split is a property of
    that split and shipping it as if it were a property of the model would
    make the number look more trustworthy than it is. Ranking is unaffected.
    """
    out = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        feats = embed(backbone, spec, chunk, device=device, batch_size=batch_size)
        f = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(device)
        with torch.no_grad():
            logit = model(f)["logit"]
        out.append(torch.sigmoid(logit).double().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score images for the likelihood that they are AI-generated.")
    ap.add_argument("--images", required=True,
                    help="directory to score, searched recursively")
    ap.add_argument("--checkpoint", required=True,
                    help="a trained head, e.g. outputs/rungs/a3/checkpoint.pt")
    ap.add_argument("--out", default="predictions.json",
                    help="output JSON: [{image_path, pred}, ...]")
    ap.add_argument("--backbone", default=None,
                    help="override the backbone named in the checkpoint; "
                         "normally leave unset, since a head trained on one "
                         "backbone's features cannot read another's")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)

    if not os.path.isdir(a.images):
        raise SystemExit(f"--images {a.images!r} is not a directory")

    paths = find_images(a.images)
    if not paths:
        raise SystemExit(
            f"no images found under {a.images!r} (looked for "
            f"{sorted(IMAGE_SUFFIXES)}). Check the path -- an empty result "
            "file would be indistinguishable from a model that scored nothing.")
    print(f"{len(paths)} images under {a.images}")

    model, ck = load_detector(a.checkpoint, device=a.device)
    if ck["config"].get("use_recon"):
        raise SystemExit(
            "this checkpoint's head was trained with the recon branch "
            "(use_recon=True), so it expects reconstruction features "
            "alongside the backbone embedding. This script computes the "
            "embedding only, and handing the head half its input would "
            "produce a confident, meaningless score. Use a checkpoint "
            "trained without --recon (A0-A3, A7).")

    backbone_name = a.backbone or ck["backbone"]
    print(f"backbone {backbone_name}  head {ck['config'].get('name', '?')}  "
          f"device {a.device}")
    backbone, spec = load_backbone(backbone_name, device=a.device)

    rows, failures = [], []
    for i in range(0, len(paths), a.batch_size):
        chunk = paths[i:i + a.batch_size]
        images, kept, failed = load_batch(chunk)
        failures.extend(failed)
        if not images:
            continue
        preds = score(model, backbone, spec, images, a.device, a.batch_size)
        rows.extend({"image_path": p, "pred": float(v)}
                    for p, v in zip(kept, preds))

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(rows)} predictions to {a.out}")

    if failures:
        # Non-zero exit with the file already written: the scores that were
        # produced are still good, and the caller still needs to know the run
        # was not complete.
        print(f"\n{len(failures)} file(s) could not be decoded and were skipped:",
              file=sys.stderr)
        for line in failures:
            print("  ", line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
