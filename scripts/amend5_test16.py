#!/usr/bin/env python3
"""Amendment 8, Test 16, scoring: Test 14's tile factorial on SocialRF.

    python scripts/amend5_test16.py rows       # CPU: sizes after damage, the registered rows
    python scripts/amend5_test16.py score      # GPU: both branches, resumable
    python scripts/amend5_test16.py score --limit 4 --out-dir /tmp/t16   # smoke test

Registered design (paper_prereg.md, amendment 8, Test 16):

  Views and cells: amend5_test14 unchanged (views_tiles, CELLS, center, squish).
  Rows:
    clean      ext_aigi_socialrf, is_distorted = 0;
    distorted  ext_aigi_socialrf_deg, damaged at scoring time by
               score_plan_splits.degrade_native (seed 20261008, pool ntire,
               at most 5 ops), is_distorted = 1.
  Filter: short side as presented (after damage) >= 800 px. The `rows` step
  decodes and damages every candidate once to measure that size; it reads no
  score. Then 500 per label from each split, seed 20261019.

`amend5_test14.py` and `amend5_test2.py` are copied unchanged from branches
dion/test14-tile-factorial and dion/test2-regime-grid (staged there).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aigcdet.train.finetune import _windowed  # noqa: E402
from amend5_test2 import OurBranch  # noqa: E402
from amend5_test14 import KEEP_COLUMNS, MIN_SHORT_SIDE, all_arms  # noqa: E402
from public_detector_probe import decode  # noqa: E402
from score_plan_splits import degrade_native  # noqa: E402

CLEAN_SPLIT, DAMAGED_SPLIT = "ext_aigi_socialrf", "ext_aigi_socialrf_deg"
DEGRADE = {"seed": 20261008, "pool": "ntire", "max_ops": 5}
PER_LABEL_PER_SPLIT = 500
SUBSAMPLE_SEED = 20261019
OUT_ROOT = "/scratch/dion_r13/dion_extensions/outputs/test16"
ROWS_FILE = f"{OUT_ROOT}/rows.parquet"


def presented(path: str, split: str) -> np.ndarray:
    """The image the presentations see: the decoded file, damaged on `_deg` rows."""
    img = decode(path)
    if split == DAMAGED_SPLIT:
        img = degrade_native(img, path, DEGRADE["seed"], DEGRADE["pool"], DEGRADE["max_ops"])[0]
    return img


def build_rows(workers: int) -> pd.DataFrame:
    """The registered rows: presented short side >= 800, then 500 per label per split."""
    external = pd.read_parquet("data/manifest_external.parquet")
    picked = []
    for split in (CLEAN_SPLIT, DAMAGED_SPLIT):
        rows = external[external["split"] == split].copy()
        # Damage never enlarges an image, so a native short side < 800 cannot pass.
        rows = rows[rows[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE]
        if split == DAMAGED_SPLIT:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                shapes = list(pool.map(lambda p: presented(p, split).shape[:2], rows["path"]))
            rows["height"] = [h for h, _ in shapes]
            rows["width"] = [w for _, w in shapes]
            rows = rows[rows[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE]
        rows["is_distorted"] = int(split == DAMAGED_SPLIT)
        print(f"{split}: {len(rows)} rows pass the size filter, by label {rows['label'].value_counts().to_dict()}")
        for label in (0, 1):
            of_label = rows[rows["label"] == label]
            picked.append(of_label.sample(n=min(PER_LABEL_PER_SPLIT, len(of_label)), random_state=SUBSAMPLE_SEED))
    out = pd.concat(picked)[KEEP_COLUMNS].sort_values(["split", "path"]).reset_index(drop=True)
    assert not out["split"].eq("ntire_test").any()
    return out


def score_branch(branch: OurBranch, rows: pd.DataFrame, batch: int, workers: int):
    """amend5_test14.score_branch, with the presented (possibly damaged) image."""
    from tqdm import tqdm

    def prepare(chunk: pd.DataFrame):
        per_arm = {}
        for path, split in zip(chunk["path"], chunk["split"]):
            for arm, views in all_arms(presented(path, split), branch.policy).items():
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
    ap.add_argument("step", choices=["rows", "score"])
    ap.add_argument("--out-dir", default=f"{OUT_ROOT}/scores")
    ap.add_argument("--models", default="B2swa,A2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.step == "rows":
        os.makedirs(OUT_ROOT, exist_ok=True)
        build_rows(a.workers).to_parquet(ROWS_FILE, index=False)
        print(f"rows -> {ROWS_FILE}")
        return 0

    rows = pd.read_parquet(ROWS_FILE)
    if a.limit:
        rows = pd.concat([rows.head(a.limit), rows.tail(a.limit)])
    os.makedirs(a.out_dir, exist_ok=True)
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
    print("TEST16_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
