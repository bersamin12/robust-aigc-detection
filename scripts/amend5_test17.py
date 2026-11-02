#!/usr/bin/env python3
"""Amendment 9, Test 17, scoring: shrink then enlarge back (`roundtrip`).

    python scripts/amend5_test17.py score      # GPU: both branches, both row sets, resumable
    python scripts/amend5_test17.py score --limit 2 --out-dir /tmp/t17   # smoke test

Registered design (paper_prereg.md, amendment 9, Test 17):

  View rt(k, f): take tiles(k, f)'s centred area-f region, shrink it by k with
  squish's downscale kernel (INTER_AREA), enlarge it back to the region's own
  size with the window kernel (INTER_CUBIC), then cut tiles(1, f)'s windows.
  Each window therefore holds the same scene as tiles(1, f) (same count, same
  places), and the same averaged pixels as tiles(2, f).
  Cells: rt(2, f) for f in {0.25, 0.5, 1}. tiles(1, f) and tiles(2, f) are
  reused from Tests 14 and 16; they are not re-scored.
  Rows: Test 14's (amend5_test14.load_rows) and Test 16's (its rows.parquet,
  `_deg` rows damaged by amend5_test16.presented), unchanged.

`amend5_test14.py`, `amend5_test16.py` and `amend5_test2.py` are copied
unchanged from branch dion/test16-tile-factorial-socialrf (staged there).
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
from amend5_test14 import KEEP_COLUMNS, load_rows, window_starts  # noqa: E402
from amend5_test16 import ROWS_FILE as SOCIALRF_ROWS_FILE, presented  # noqa: E402

#: The registered cells, (k, f).
CELLS = [(2, 0.25), (2, 0.5), (2, 1.0)]
ROW_SETS = ("ntire", "socialrf")
OUT_ROOT = "/scratch/dion_r13/dion_extensions/outputs/test17"


def cell_name(k: int, f: float) -> str:
    return f"rt_k{k}_f{f:g}"


def views_roundtrip(img: np.ndarray, policy, k: int, f: float) -> list[np.ndarray]:
    """tiles(1, f)'s windows, cut after the region is shrunk by k and enlarged back."""
    h, w = img.shape[:2]
    region_h, region_w = round(h * math.sqrt(f)), round(w * math.sqrt(f))
    top, left = (h - region_h) // 2, (w - region_w) // 2
    region = img[top:top + region_h, left:left + region_w]
    # Same shrink as amend5_test14.views_tiles, so the pixels match tiles(k, f)'s.
    small = cv2.resize(region, (round(region_w / k), round(region_h / k)), interpolation=_DOWNSCALE_INTERP)
    # Back to the region's own size, so each window holds tiles(1, f)'s scene.
    region = cv2.resize(small, (region_w, region_h), interpolation=_UPSCALE_INTERP)
    side = policy.crop_side
    assert min(region.shape[:2]) >= side, (img.shape, k, f, region.shape)
    return [_resize_short_side(region[t:t + side, l:l + side], policy.nominal_side, _UPSCALE_INTERP)
            for t in window_starts(region.shape[0], side)
            for l in window_starts(region.shape[1], side)]


def load_row_set(row_set: str) -> pd.DataFrame:
    """Test 14's rows or Test 16's rows, in the order those tests scored them."""
    if row_set == "ntire":
        rows = load_rows()
    else:
        rows = pd.read_parquet(SOCIALRF_ROWS_FILE)
    assert not rows["split"].eq("ntire_test").any(), "Test 17 never reads ntire_test"
    return rows[KEEP_COLUMNS].reset_index(drop=True)


def score_branch(branch: OurBranch, rows: pd.DataFrame, batch: int, workers: int):
    """amend5_test16.score_branch, with the roundtrip cells as the only arms."""
    from tqdm import tqdm

    def prepare(chunk: pd.DataFrame):
        per_arm = {}
        for path, split in zip(chunk["path"], chunk["split"]):
            # presented() damages `_deg` rows and only decodes every other split.
            img = presented(path, split)
            for k, f in CELLS:
                per_arm.setdefault(cell_name(k, f), []).append(views_roundtrip(img, branch.policy, k, f))
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
    ap.add_argument("step", choices=["score"])
    ap.add_argument("--out-dir", default=f"{OUT_ROOT}/scores")
    ap.add_argument("--models", default="B2swa,A2")
    ap.add_argument("--row-sets", default=",".join(ROW_SETS))
    ap.add_argument("--limit", type=int, default=0, help="first and last N rows of each set (smoke test)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-gib", type=float, default=10.0, help="cap on this process's GPU memory (shared GPU)")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        # Shared GPU: past the cap, this process gets an out-of-memory error instead of
        # taking memory from other users. The CUDA context (~0.5 GB) is outside the cap.
        index = torch.device(a.device).index or 0
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gib * 2**30 / total), index)

    os.makedirs(a.out_dir, exist_ok=True)
    for model_name in a.models.split(","):
        branch = None
        for row_set in a.row_sets.split(","):
            out = f"{a.out_dir}/{model_name}_{row_set}_roundtrip.parquet"
            if os.path.exists(out):
                print(f"skip {out}", flush=True)
                continue
            rows = load_row_set(row_set)
            if a.limit:
                rows = pd.concat([rows.head(a.limit), rows.tail(a.limit)]).reset_index(drop=True)
            branch = branch or OurBranch(model_name, a.device)
            started = time.time()
            logits, counts = score_branch(branch, rows, a.batch, a.workers)
            frame = rows.copy()
            for arm in logits:
                frame[f"logit_{arm}"] = logits[arm]
                frame[f"n_windows_{arm}"] = counts[arm]
            frame.to_parquet(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            print(f"{model_name} {row_set}: {len(rows)} images, {sum(c.sum() for c in counts.values())} windows "
                  f"in {time.time() - started:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                  f"-> {out}", flush=True)
        del branch
        torch.cuda.empty_cache()
    print("TEST17_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
