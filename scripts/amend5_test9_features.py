#!/usr/bin/env python3
"""Amendment 5, Test 9, step 1: pooled features of every `grid9` window.

    python scripts/amend5_test9_features.py --ckpt CKPT [--swa] --tag A2 --split ntire_val

For a frozen branch, run the `grid9` presentation of scripts/ntire_crop_probe.py
and save, per image, the pooled feature vector of each window (the head's
input, i.e. the concatenated tower poolings) together with the frozen head's
logit for that window. The mean of those per-window logits is exactly the
probe's `grid9` score, which is checked against the frozen grid9 parquet when
one exists.

Output (one .npz per branch and split, under outputs/test9/):
  features   float16 [total_windows, feature_dim]
  window_logit float64 [total_windows]
  offsets    int64   [n_images + 1]; image i owns rows offsets[i]:offsets[i+1]
  path, label, is_distorted, width, height   per image
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
from ntire_crop_probe import views_grid9  # noqa: E402
from aigcdet.models.pooling import tower_poolers  # noqa: E402
from aigcdet.train.finetune import _forward_tower, _windowed  # noqa: E402
from score_plan_splits import _load_models, _policy_from_config  # noqa: E402


def extract(df, towers, specs, head, policy, device, batch, workers, chunk):
    """Per-window pooled features and logits for every image in `df`, in order."""
    from PIL import Image
    from tqdm import tqdm

    def prepare(paths):
        windows, counts = [], []
        for p in paths:
            with Image.open(p) as im:
                decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
            views = views_grid9(decoded, policy)
            windows.extend(views)
            counts.append(len(views))
        return windows, counts

    paths = df["path"].tolist()
    batches = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    poolers = tower_poolers(head, len(towers))
    feature_chunks, logit_chunks, all_counts = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, batches, 2 * workers), total=len(batches), desc="grid9"):
            windows, counts = fut.result()
            pooled = torch.cat([_forward_tower(t, sp, windows, device, torch.bfloat16, chunk, pooler=pl)
                                for t, sp, pl in zip(towers, specs, poolers)], dim=-1)
            logits = head(pooled.float(), None)["logit"].double().cpu().numpy()
            feature_chunks.append(pooled.float().cpu().numpy().astype(np.float16))
            logit_chunks.append(logits)
            all_counts.extend(counts)
    offsets = np.concatenate([[0], np.cumsum(all_counts)]).astype(np.int64)
    return np.concatenate(feature_chunks), np.concatenate(logit_chunks), offsets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--tag", required=True, help="A2 or B2swa, used in the file name")
    ap.add_argument("--split", required=True)
    ap.add_argument("--manifest", default="data/manifest_ntire.parquet")
    ap.add_argument("--out-dir", default="/scratch/dion_r13/dion_extensions/outputs/test9")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert a.split != "ntire_test", "Test 9 never reads ntire_test"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    towers, specs, head, tag = _load_models(ck, "cuda", a.swa)
    policy = _policy_from_config(ck["config"], True)
    assert policy.mode == "crop", policy.mode

    manifest = pd.read_parquet(a.manifest)
    df = manifest[manifest["split"] == a.split].reset_index(drop=True)
    features, window_logit, offsets = extract(df, towers, specs, head, policy, "cuda",
                                              a.batch, a.workers, a.chunk)
    assert offsets[-1] == len(features) == len(window_logit)

    # Check: the mean of the window logits must be the probe's grid9 score.
    image_logit = np.array([window_logit[offsets[i]:offsets[i + 1]].mean() for i in range(len(df))])
    frozen_path = f"outputs/scores/probe_{a.tag}_grid9_{a.split}.parquet"
    if os.path.exists(frozen_path):
        frozen = pd.read_parquet(frozen_path)[["path", "logit"]]
        both = pd.DataFrame({"path": df["path"], "mine": image_logit}).merge(frozen, on="path", validate="1:1")
        print(f"grid9 reproduction vs {frozen_path}: max |diff| {np.abs(both['mine'] - both['logit']).max():.6f}")

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = f"{a.out_dir}/grid9_features_{a.tag}_{a.split}.npz"
    np.savez(out_path, features=features, window_logit=window_logit, offsets=offsets,
             path=df["path"].to_numpy(), label=df["label"].to_numpy(),
             is_distorted=df["is_distorted"].to_numpy(),
             width=df["width"].to_numpy(), height=df["height"].to_numpy())
    print(f"wrote {out_path}: {len(df)} images, {len(features)} windows, dim {features.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
