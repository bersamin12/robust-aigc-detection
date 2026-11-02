#!/usr/bin/env python3
"""Amendment 7, Test 14, scoring: shrink k x coverage f factorial (`tiles(k, f)`).

    python scripts/amend5_test14.py                        # both branches, resumable
    python scripts/amend5_test14.py --limit 8 --out-dir /tmp/t14   # smoke test

Registered design (paper_prereg.md, amendment 7, Test 14):

  View tiles(k, f), for an image of h x w:
    1. the centred rectangle of area fraction f
       (sides round(h sqrt f) x round(w sqrt f));
    2. shrunk by k with squish's downscale kernel (_DOWNSCALE_INTERP);
    3. covered by 200 px windows, ceil(L / 200) per axis, evenly spaced from
       0 to L - 200 (one centred window when ceil(L / 200) = 1);
    4. each window resized to n as views_grid9 does.
  The image score is the mean logit over windows. tiles(1, 1) = `gridall`.

  Cells: k in {1, 2} x f in {0.25, 0.5, 1}, plus tiles(4, 1), plus `center`
  and `squish` for reference.
  Rows: ntire_val + ntire_val_hard with native short side >= 800 px, so
  every shrunk region has sides of at least 200 px and no window is
  enlarged beyond the trained 200 -> n.

Output: one parquet per branch, one logit column and one window-count
column per cell. The analysis is amend5_test14_analysis.py.

`amend5_test2.py` is copied unchanged from branch dion/test2-regime-grid
for its OurBranch loader.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aigcdet.augment.canonical import _DOWNSCALE_INTERP, _UPSCALE_INTERP, _resize_short_side  # noqa: E402
from aigcdet.train.finetune import _windowed  # noqa: E402
from amend5_test2 import OurBranch  # noqa: E402
from ntire_crop_probe import views_center, views_squish  # noqa: E402
from public_detector_probe import decode  # noqa: E402

#: The registered cells, (k, f), in report order.
CELLS = [(1, 0.25), (1, 0.5), (1, 1.0), (2, 0.25), (2, 0.5), (2, 1.0), (4, 1.0)]
MIN_SHORT_SIDE = 800
SPLITS = ("ntire_val", "ntire_val_hard")
KEEP_COLUMNS = ["path", "label", "is_distorted", "distortions", "width", "height", "split"]


def cell_name(k: int, f: float) -> str:
    return f"tiles_k{k}_f{f:g}"


def window_starts(length: int, side: int) -> list[int]:
    """ceil(length / side) window starts, evenly spaced from 0 to length - side."""
    count = math.ceil(length / side)
    if count == 1:
        return [(length - side) // 2]
    return [round(i * (length - side) / (count - 1)) for i in range(count)]


def views_tiles(img: np.ndarray, policy, k: int, f: float) -> list[np.ndarray]:
    """Every 200 px window over the centred area-f region, shrunk by k."""
    h, w = img.shape[:2]
    region_h, region_w = round(h * math.sqrt(f)), round(w * math.sqrt(f))
    top, left = (h - region_h) // 2, (w - region_w) // 2
    region = img[top:top + region_h, left:left + region_w]
    if k > 1:
        region = cv2.resize(region, (round(region_w / k), round(region_h / k)),
                            interpolation=_DOWNSCALE_INTERP)
    side = policy.crop_side
    assert min(region.shape[:2]) >= side, (img.shape, k, f, region.shape)
    return [_resize_short_side(region[t:t + side, l:l + side], policy.nominal_side, _UPSCALE_INTERP)
            for t in window_starts(region.shape[0], side)
            for l in window_starts(region.shape[1], side)]


def load_rows() -> pd.DataFrame:
    """The registered rows, in manifest order."""
    ntire = pd.read_parquet("data/manifest_ntire.parquet")
    rows = ntire[ntire["split"].isin(SPLITS)]
    rows = rows[rows[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE]
    assert not rows["split"].eq("ntire_test").any(), "Test 14 never reads ntire_test"
    return rows[KEEP_COLUMNS].reset_index(drop=True)


def all_arms(img: np.ndarray, policy) -> dict[str, list[np.ndarray]]:
    arms = {"center": views_center(img, policy), "squish": views_squish(img, policy)}
    for k, f in CELLS:
        arms[cell_name(k, f)] = views_tiles(img, policy, k, f)
    return arms


def score_branch(branch: OurBranch, rows: pd.DataFrame, batch: int, workers: int):
    """Mean logit per image and arm, plus the window count per image and arm."""
    from tqdm import tqdm

    def prepare(chunk: pd.DataFrame):
        per_arm = {}
        for path in chunk["path"]:
            for arm, views in all_arms(decode(path), branch.policy).items():
                per_arm.setdefault(arm, []).append(views)
        return per_arm

    chunks = [rows.iloc[i:i + batch] for i in range(0, len(rows), batch)]
    logits, counts = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, chunks, 2 * workers), total=len(chunks)):
            for arm, views_per_image in fut.result().items():
                logits.setdefault(arm, []).append(branch.image_logits(views_per_image))
                counts.setdefault(arm, []).extend(len(v) for v in views_per_image)
    return ({arm: np.concatenate(v) for arm, v in logits.items()},
            {arm: np.array(v) for arm, v in counts.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/scratch/dion_r13/dion_extensions/outputs/test14/scores")
    ap.add_argument("--models", default="B2swa,A2")
    ap.add_argument("--limit", type=int, default=0, help="first N rows (smoke test)")
    ap.add_argument("--batch", type=int, default=1, help="images per chunk; ~200 windows each")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rows = load_rows()
    if a.limit:
        rows = rows.iloc[:a.limit]
    for model_name in a.models.split(","):
        out = f"{a.out_dir}/{model_name}_tiles.parquet"
        if os.path.exists(out):
            print(f"skip {out}", flush=True)
            continue
        branch = OurBranch(model_name, a.device)
        started = time.time()
        logits, counts = score_branch(branch, rows, a.batch, a.workers)
        frame = rows.copy()
        for arm in logits:
            frame[f"logit_{arm}"] = logits[arm]
            frame[f"n_windows_{arm}"] = counts[arm]
        frame.to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        print(f"{model_name}: {len(rows)} images, {sum(c.sum() for c in counts.values())} windows "
              f"in {time.time() - started:.0f} s -> {out}", flush=True)
        del branch
        torch.cuda.empty_cache()
    print("TEST14_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
