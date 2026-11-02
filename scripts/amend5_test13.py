#!/usr/bin/env python3
"""Amendment 7, Test 13, scoring: damage applied AFTER squish (`squish_post`).

    python scripts/amend5_test13.py                       # both branches, resumable
    python scripts/amend5_test13.py --limit 64 --out-dir /tmp/t13   # smoke test

Registered design (paper_prereg.md, amendment 7, Test 13):

  Source: Test 2's primary rows (6,250 clean ntire_val + ntire_val_hard
  images, same row order, `amend5_test2.load_sources`).
  Arm `squish_post`: `views_squish` on the clean native image, then one of
  Test 2's nine damage conditions on the n x n view (`amend5_test2.damage`,
  same per-image noise seed). The `center` arm and native-damage `squish`
  arm are Test 2's existing parquets and are not rescored here.

Each image is decoded and squished once; the nine damaged copies of the
view are scored together. Output: one parquet per branch with one logit
column per condition, `logit_squish_post_<condition>`.

`amend5_test2.py` is copied unchanged from branch dion/test2-regime-grid
(staged, not yet committed there), so Test 13 uses Test 2's damage code.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aigcdet.train.finetune import _windowed  # noqa: E402
from amend5_test2 import CONDITIONS, OurBranch, damage, load_sources  # noqa: E402
from ntire_crop_probe import views_squish  # noqa: E402
from public_detector_probe import decode  # noqa: E402

DAMAGED_CONDITIONS = [c for c in CONDITIONS if c != "none"]  # "none" is Test 2's squish arm


def score_branch(branch: OurBranch, rows, batch: int, workers: int) -> dict[str, np.ndarray]:
    """Logits of the squished-then-damaged view, for every damage condition."""
    from tqdm import tqdm

    def prepare(chunk):
        """One squished view per image, damaged once per condition."""
        per_condition = {c: [] for c in DAMAGED_CONDITIONS}
        for path, sha in zip(chunk["path"], chunk["content_sha256"]):
            squished = views_squish(decode(path), branch.policy)[0]
            for condition in DAMAGED_CONDITIONS:
                per_condition[condition].append([damage(squished, condition, sha)])
        return per_condition

    chunks = [rows.iloc[i:i + batch] for i in range(0, len(rows), batch)]
    logits = {c: [] for c in DAMAGED_CONDITIONS}
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, chunks, 2 * workers), total=len(chunks)):
            per_condition = fut.result()
            for condition in DAMAGED_CONDITIONS:
                logits[condition].append(branch.image_logits(per_condition[condition]))
    return {c: np.concatenate(v) for c, v in logits.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/scratch/dion_r13/dion_extensions/outputs/test13/scores")
    ap.add_argument("--models", default="A2,B2swa")
    ap.add_argument("--limit", type=int, default=0, help="first N rows (smoke test)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rows = load_sources()["ntire_clean"]
    assert not rows["split"].eq("ntire_test").any(), "Test 13 never reads ntire_test"
    if a.limit:
        rows = rows.iloc[:a.limit]

    for model_name in a.models.split(","):
        out = f"{a.out_dir}/{model_name}_ntire_clean_squish_post.parquet"
        if os.path.exists(out):
            print(f"skip {out}", flush=True)
            continue
        branch = OurBranch(model_name, a.device)
        started = time.time()
        logits = score_branch(branch, rows, a.batch, a.workers)
        frame = rows.drop(columns=["content_sha256"]).copy()
        for condition, z in logits.items():
            frame[f"logit_squish_post_{condition}"] = z
        frame.to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        print(f"{model_name}: {len(rows)} images x {len(DAMAGED_CONDITIONS)} conditions in "
              f"{time.time() - started:.0f} s -> {out}", flush=True)
        del branch
        torch.cuda.empty_cache()
    print("TEST13_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
