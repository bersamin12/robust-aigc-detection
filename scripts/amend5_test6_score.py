#!/usr/bin/env python3
"""Amendment 5, Test 6, scoring step: squish logit AND predicted damage
severity of a frozen branch, per image.

    python scripts/amend5_test6_score.py --ckpt CKPT [--swa] --tag A2 --split ntire_val

The routing rule of Test 6 needs, per image and per branch, the `squish`
logit and d, the branch's predicted damage severity on the squish view (the
degradation head's output). No saved parquet carries the degradation head's
output, so this script re-runs the `squish` presentation of
scripts/ntire_crop_probe.py and keeps both. The `grid9` logit is NOT scored
here: Test 9 already saved every grid9 window's logit for both branches on
`ntire_val` and `ntire_val_hard` (outputs/test9/grid9_features_*.npz), and
the fit script averages those.

The degradation head predicts one severity per damage family (sigmoid, in
[0, 1]). All of them are written, so the fit script can form the overall
severity d from them; how it does that is recorded there.

The squish logit is checked against the frozen probe parquet
(outputs/scores/probe_<tag>_squish_<split>.parquet): same checkpoint, same
pixels, same code path, so it should agree to bf16 noise.

Output: outputs/test6/test6_squish_<tag>_<split>.parquet with path, label,
is_distorted, distortions, width, height, logit, sev_<family> (one per
family).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntire_crop_probe import views_squish  # noqa: E402
from aigcdet.augment.recipes import FAMILIES  # noqa: E402
from aigcdet.models.pooling import tower_poolers  # noqa: E402
from aigcdet.train.finetune import _forward_tower, _windowed  # noqa: E402
from score_plan_splits import _load_models, _policy_from_config  # noqa: E402

OUT_DIR = "/scratch/dion_r13/dion_extensions/outputs/test6"


def score_squish_with_severity(df, towers, specs, head, policy, device,
                               batch, workers, chunk):
    """Squish logit and per-family severity for every image in `df`, in order.

    Same loop as `ntire_crop_probe.score_arm` for the squish arm (one view
    per image, bf16 towers, float head), but it also keeps the degradation
    head's `severity` output, which `score_arm` throws away.
    """
    from PIL import Image
    from tqdm import tqdm

    def prepare(paths):
        views = []
        for p in paths:
            with Image.open(p) as im:
                decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
            views.extend(views_squish(decoded, policy))
        return views, None

    paths = df["path"].tolist()
    batches = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    poolers = tower_poolers(head, len(towers))
    logit_chunks, severity_chunks = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, batches, 2 * workers),
                           total=len(batches), desc="squish+severity"):
            views, _ = fut.result()
            pooled = torch.cat(
                [_forward_tower(t, sp, views, device, torch.bfloat16, chunk, pooler=pl)
                 for t, sp, pl in zip(towers, specs, poolers)], dim=-1)
            out = head(pooled.float(), None)
            logit_chunks.append(out["logit"].double().cpu().numpy())
            severity_chunks.append(out["severity"].double().cpu().numpy())
    logits = np.concatenate(logit_chunks)
    severities = np.concatenate(severity_chunks)
    assert len(logits) == len(df) == len(severities)
    return logits, severities


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--tag", required=True, help="A2 or B2swa, used in the file name")
    ap.add_argument("--split", required=True)
    ap.add_argument("--manifest", default="data/manifest_ntire.parquet")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert a.split != "ntire_test", "the Test 6 scoring step never reads ntire_test"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    towers, specs, head, weights_tag = _load_models(ck, a.device, a.swa)
    policy = _policy_from_config(ck["config"], True)
    assert policy.mode == "crop", f"Test 6 routes the crop-trained branches, got {policy.mode}"

    manifest = pd.read_parquet(a.manifest)
    df = manifest[manifest["split"] == a.split].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit).reset_index(drop=True)
    logits, severities = score_squish_with_severity(
        df, towers, specs, head, policy, a.device, a.batch, a.workers, a.chunk)

    family_names = list(FAMILIES)
    assert severities.shape[1] == len(family_names), (severities.shape, family_names)
    rows = df[["path", "label", "is_distorted", "distortions", "width", "height"]].copy()
    rows["logit"] = logits
    for j, family in enumerate(family_names):
        rows[f"sev_{family}"] = severities[:, j]

    # Check: the squish logit must reproduce the frozen probe's squish parquet.
    frozen_path = f"outputs/scores/probe_{a.tag}_squish_{a.split}.parquet"
    if os.path.exists(frozen_path):
        frozen = pd.read_parquet(frozen_path)[["path", "logit"]]
        both = rows[["path", "logit"]].merge(frozen, on="path", suffixes=("_new", "_frozen"),
                                             validate="1:1")
        max_diff = float(np.abs(both["logit_new"] - both["logit_frozen"]).max())
        print(f"squish reproduction vs {frozen_path}: {len(both)} rows, max |diff| {max_diff:.6f}")

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = f"{a.out_dir}/test6_squish_{a.tag}_{a.split}.parquet"
    rows.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(rows)} rows ({weights_tag} weights), families {family_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
