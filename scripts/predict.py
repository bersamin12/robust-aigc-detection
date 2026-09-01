"""Score a directory of images for the likelihood that each one is AI-generated.

This is the submission's inference entry point (brief section 5.5.2): a
directory in, a JSON file out, one `{"image_path": ..., "pred": ...}` object
per image, where `pred` is P(AI-generated) in [0, 1].

    python scripts/predict.py --images path/to/dir \
           --bundle outputs/release --out predictions.json
    python scripts/predict.py --images path/to/dir \
           --ckpt outputs/dual/checkpoint_ep1.pt --out predictions.json

**Two model formats, one contract.** `--bundle` serves the frozen-feature
pipeline's release bundle; `--ckpt` serves a finetuned checkpoint in
`_write_ckpt`'s shape (single-tower `tower_state_dict` or dual
`tower_state_dicts`), loaded and scored EXACTLY as `scripts/score_plan_splits.py`
does -- same loader, same centre-window canonicalisation, same bf16 cast -- so
`pred` from this script is the same quantity every reported AUC/TPR@1%FPR/acc
table was computed on, and a threshold read off those tables applies here
unchanged. In `--ckpt` mode `pred` is sigmoid(logit), uncalibrated: no
temperature was fitted for the finetuned arms, and dressing the raw sigmoid up
as a calibrated probability is the exact failure the paragraph below records.

**The bundle mode takes a bundle, not a checkpoint.** `pred` is the CALIBRATED probability,
so 0.9 means roughly 90% and not merely "higher than 0.8". That number needs
the temperature and the policy that `scripts/export_bundle.py` fits on
internal validation, and a bare checkpoint has neither. An earlier version of
this script squashed the head's logit with a sigmoid instead: also in [0, 1],
also varying sensibly with the image, and not the quantity the README claims.

**Everything else in this repository consumes a frozen manifest and a cached
feature bank. This does not.** It is handed loose files with no manifest, no
labels and no bank, which is exactly how it will be run by someone who did not
build it. So it establishes for itself what the rest of the pipeline gets from
the manifest: which files are images, in what order, and what happens to the
one that will not decode.

The scoring itself lives in `aigcdet.infer.Predictor`, which the dashboard
uses too. Two inference paths that agree today drift apart quietly; the
divergence shows up as a demo that reports a different number than the
submitted script, with nothing failing anywhere.

**A failure is loud.** An empty directory, a missing bundle, or a file PIL
cannot decode are each reported by name. The alternative -- `[]` and exit 0,
or a silently short result file -- is indistinguishable from a successful run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from aigcdet.infer import FAILED_PRED, RESULT_KEYS_MINIMAL, Predictor

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


def predict_paths_finetuned(ckpt: str, paths: list[str], device: str,
                            batch_size: int) -> list[dict]:
    """Score with a finetuned checkpoint instead of a bundle.

    Reuses `score_plan_splits`' loader and forward path rather than restating
    them: two inference paths that agree today drift apart quietly, and this
    one produces the numbers the results tables were quoted from.

    Failure semantics mirror `Predictor.predict_paths`: a file that will not
    decode gets `pred=FAILED_PRED` and a populated `error`, keyed by path so
    a bad file cannot shift the scores of the files after it.
    """
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_plan_splits import _load_models, _policy_from_config

    from aigcdet.augment.canonical import canonicalise
    from aigcdet.train.finetune import _forward_tower

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    towers, specs, head, _tag = _load_models(ck, device, use_swa=False)
    if head.use_recon:
        raise SystemExit(
            "REFUSING: this checkpoint's head expects recon features and this "
            "path does not compute them; score it with "
            "scripts/score_plan_splits.py instead.")
    policy = _policy_from_config(ck["config"])
    names = ck.get("backbones") or [ck.get("backbone")]
    print(f"finetuned checkpoint: towers {names}  epoch {ck.get('epoch')}  "
          f"canon {policy.mode}/{policy.crop_side}->{policy.nominal_side}")

    results: list[dict] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        imgs, kept, slots = [], [], {}
        for p in chunk:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            except Exception as exc:          # noqa: BLE001 -- reported
                print(f"skipping {p}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                slots[p] = {"pred": FAILED_PRED, "logit": 0.0,
                            "error": f"{type(exc).__name__}: {exc}"}
                continue
            # The EVAL convention: the deterministic centre window (`rng` is
            # absent), so the same file scores the same twice.
            imgs.append(canonicalise(arr, policy=policy))
            kept.append(p)
        if imgs:
            with torch.no_grad():
                feats = torch.cat(
                    [_forward_tower(t, sp, imgs, device, torch.bfloat16,
                                    batch_size)
                     for t, sp in zip(towers, specs)], dim=-1)
                logits = head(feats.float(), None)["logit"].reshape(-1)
                preds = torch.sigmoid(logits).double().cpu().numpy()
                logits = logits.double().cpu().numpy()
            for p, pr, lg in zip(kept, preds, logits):
                slots[p] = {"pred": float(pr), "logit": float(lg),
                            "error": None}
        for p in chunk:
            results.append({"image_path": p, **slots[p]})
    return results


def minimal_rows(results: list[dict]) -> list[dict]:
    """`results` reduced to the two keys the brief asks for.

    Built by picking the required keys rather than by deleting the others, so
    a new field added to `Predictor`'s output cannot leak into the submitted
    file. An extra key is how a submission fails on a technicality.
    """
    return [{k: r[k] for k in RESULT_KEYS_MINIMAL} for r in results]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score images for the likelihood that they are AI-generated.")
    ap.add_argument("--images", required=True,
                    help="directory to score, searched recursively")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundle",
                   help="a release bundle from scripts/export_bundle.py, "
                        "e.g. outputs/release -- checkpoint, calibrator, EQI "
                        "and policy together")
    g.add_argument("--ckpt",
                   help="a finetuned checkpoint (single- or dual-tower) in "
                        "_write_ckpt's shape; pred is sigmoid(logit), the "
                        "quantity the results tables were computed on")
    ap.add_argument("--out", default="predictions.json",
                    help="output JSON: [{image_path, pred}, ...]")
    ap.add_argument("--full", action="store_true",
                    help="also write logit, eqi, decision, severity, presence "
                         "and proxies per image, to <out> -- for error "
                         "analysis, NOT for submission")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)

    if not os.path.isdir(a.images):
        raise SystemExit(f"--images {a.images!r} is not a directory")
    if a.ckpt and not os.path.isfile(a.ckpt):
        raise SystemExit(f"--ckpt {a.ckpt!r} is not a file")
    if a.bundle and not os.path.isdir(a.bundle):
        raise SystemExit(
            f"--bundle {a.bundle!r} is not a directory. It should be the "
            "output of scripts/export_bundle.py, which writes checkpoint.pt, "
            "calibrator.joblib, eqi.joblib, policy.json and config.json.")

    paths = find_images(a.images)
    if not paths:
        raise SystemExit(
            f"no images found under {a.images!r} (looked for "
            f"{sorted(IMAGE_SUFFIXES)}). Check the path -- an empty result "
            "file would be indistinguishable from a model that scored nothing.")
    print(f"{len(paths)} images under {a.images}")

    if a.ckpt:
        results = predict_paths_finetuned(a.ckpt, paths, a.device,
                                          a.batch_size)
    else:
        predictor = Predictor.load(a.bundle, device=a.device)
        print(f"backbone {predictor.spec.name}  recon {predictor.use_recon}  "
              f"device {predictor.device}")
        results = predictor.predict_paths(paths, batch_size=a.batch_size)
    rows = results if a.full else minimal_rows(results)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(rows)} predictions to {a.out}")

    failures = [r for r in results if r["error"]]
    if failures:
        # Non-zero exit with the file already written: the scores that were
        # produced are still good, and the caller still needs to know the run
        # was not complete. Every image still has a row, scored 0.5.
        print(f"\n{len(failures)} file(s) could not be decoded and were "
              f"scored 0.5:", file=sys.stderr)
        for r in failures:
            print("  ", f"{r['image_path']}: {r['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
