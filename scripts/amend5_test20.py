#!/usr/bin/env python3
"""Amendment 12, Test 20, scoring: Tests 16 and 17's views on the CLIP proxies.

    python scripts/amend5_test20.py score                        # all three seeds, resumable
    python scripts/amend5_test20.py score --models clip_crop_s1 --limit 10 --out-dir /tmp/t20  # smoke / timing

Registered design (amendment 12, Test 20):

  Models: the crop-trained CLIP proxies of amendment 4 (clip_crop_s1/_s2/_s3),
  frozen, loaded as amend5_test2.OurBranch loads A2 and B2swa (no SWA).
  Rows: Test 16's rows.parquet, unchanged (`_deg` rows damaged by
  amend5_test16.presented).
  Views, one pass per image: center, squish and Test 14's tiles(k, f) cells
  (amend5_test14.all_arms), plus Test 17's rt(2, f) cells
  (amend5_test17.views_roundtrip). Mean logit per image.

amend5_test2/14/16/17.py are copied unchanged from branch dion/test17-roundtrip.
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
import amend5_test2  # noqa: E402
from aigcdet.train.finetune import _windowed  # noqa: E402
from amend5_test2 import OurBranch  # noqa: E402
from amend5_test14 import all_arms  # noqa: E402
from amend5_test16 import ROWS_FILE, presented  # noqa: E402
from amend5_test17 import CELLS as RT_CELLS, cell_name as rt_name, views_roundtrip  # noqa: E402

CLIP_RUNS = "/scratch/justin_r13/aigcdet/outputs/paper_clip"
SEEDS = ("clip_crop_s1", "clip_crop_s2", "clip_crop_s3")
OUT_ROOT = "/scratch/dion_r13/dion_extensions/outputs/test20"

# OurBranch looks checkpoints up by tag. Register the CLIP proxies beside A2 and B2swa
# (no SWA weights: the proxies were trained without SWA).
for _seed in SEEDS:
    amend5_test2.CHECKPOINTS[_seed] = (f"{CLIP_RUNS}/{_seed}/checkpoint.pt", False)


def views(img: np.ndarray, policy) -> dict[str, list[np.ndarray]]:
    """Every registered view of one presented image."""
    arms = all_arms(img, policy)
    for k, f in RT_CELLS:
        arms[rt_name(k, f)] = views_roundtrip(img, policy, k, f)
    return arms


def score_branch(branch: OurBranch, rows: pd.DataFrame, batch: int, workers: int):
    """amend5_test16.score_branch, with the roundtrip cells added to the arms."""
    from tqdm import tqdm

    def prepare(chunk: pd.DataFrame):
        per_arm = {}
        for path, split in zip(chunk["path"], chunk["split"]):
            for arm, v in views(presented(path, split), branch.policy).items():
                per_arm.setdefault(arm, []).append(v)
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
    ap.add_argument("--models", default=",".join(SEEDS))
    ap.add_argument("--limit", type=int, default=0, help="first and last N rows (smoke test, timing)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = pd.read_parquet(ROWS_FILE)
    assert not rows["split"].eq("ntire_test").any(), "Test 20 never reads ntire_test"
    if a.limit:
        rows = pd.concat([rows.head(a.limit), rows.tail(a.limit)]).reset_index(drop=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for model_name in a.models.split(","):
        out = f"{a.out_dir}/{model_name}_views.parquet"
        if os.path.exists(out):
            print(f"skip {out}", flush=True)
            continue
        branch = OurBranch(model_name, a.device)
        policy = branch.policy
        assert (policy.crop_side, policy.nominal_side) == (200, 224), (model_name, policy)
        started = time.time()
        logits, counts = score_branch(branch, rows, a.batch, a.workers)
        frame = rows.copy()
        for arm in logits:
            frame[f"logit_{arm}"] = logits[arm]
            frame[f"n_windows_{arm}"] = counts[arm]
        frame.to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        print(f"{model_name}: {len(rows)} images, {sum(c.sum() for c in counts.values())} windows "
              f"in {time.time() - started:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"-> {out}", flush=True)
        del branch
        torch.cuda.empty_cache()
    print("TEST20_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
