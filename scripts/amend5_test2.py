#!/usr/bin/env python3
"""Amendment 5, Test 2, scoring: the regime grid with a damage sweep.

    python scripts/amend5_test2.py                     # full run, resumable
    python scripts/amend5_test2.py --limit 200 --out-dir .../throughput \
        --conditions none,jpeg_heavy                    # throughput check

Registered design (paper_prereg.md, amendment 5, Test 2; decisions D4, D5):

  models x presentations
    Branch A (A2 final) and Branch B (B2swa SWA): center, grid9, squish
      (the presentations of scripts/ntire_crop_probe.py, unchanged);
    B-Free (authors' checkpoint): native (authors' five-crop), train_fov,
      squish (the presentations of scripts/public_detector_probe.py).

  damage, applied to the decoded native image BEFORE any presentation,
  one op type per arm, with the project's own ops (augment/ops.py):
      none | JPEG q90 / q70 / q30 | blur sigma 0.5 / 1.0 / 2.0 |
      noise sigma 0.02 / 0.05 / 0.10            -> 10 conditions
    Noise is seeded per image by
      default_rng([20261019, int(content_sha256[:15], 16)])
    so every model and presentation sees identical pixels.

  sources (undistorted originals only)
    primary   ntire_val + ntire_val_hard, is_distorted == 0 (6,250 images);
    secondary ext_aigi_socialrf and ext_synthbuster, each subsampled to
              1,000 per label (seed 20261019), or all rows of a label with
              fewer (Synthbuster has 994 reals).

Output: one parquet per (model, source, condition) under --out-dir, with one
logit column per presentation. Existing parquets are skipped, so an
interrupted run resumes where it stopped. The analysis (D per size bucket,
the pass rule) is a separate script, amend5_test2_analysis.py.
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
from aigcdet.augment import ops  # noqa: E402
from aigcdet.models.pooling import tower_poolers  # noqa: E402
from aigcdet.train.finetune import _forward_tower, _windowed  # noqa: E402
from ntire_crop_probe import views_center, views_grid9, views_squish  # noqa: E402
from public_detector_probe import BFree, bfree_view, decode, to_tensor  # noqa: E402
from score_plan_splits import _load_models, _policy_from_config  # noqa: E402

CHECKPOINTS = {
    "A2": ("/scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt", False),
    "B2swa": ("/scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt", True),
}
MODELS = ("A2", "B2swa", "bfree")
OUR_ARMS = {"center": views_center, "grid9": views_grid9, "squish": views_squish}
BFREE_ARMS = ("native", "train_fov", "squish")

#: The registered damage table: condition name -> (op, parameter).
CONDITIONS = {
    "none": (None, None),
    "jpeg_light": ("jpeg", 90), "jpeg_medium": ("jpeg", 70), "jpeg_heavy": ("jpeg", 30),
    "blur_light": ("blur", 0.5), "blur_medium": ("blur", 1.0), "blur_heavy": ("blur", 2.0),
    "noise_light": ("noise", 0.02), "noise_medium": ("noise", 0.05), "noise_heavy": ("noise", 0.10),
}
NOISE_SEED = 20261019
SUBSAMPLE_SEED = 20261019
SECONDARY_PER_LABEL = 1000
KEEP_COLUMNS = ["path", "label", "width", "height", "content_sha256", "split"]


def damage(img: np.ndarray, condition: str, content_sha256: str) -> np.ndarray:
    """Apply one registered damage condition to a decoded native RGB image."""
    op, param = CONDITIONS[condition]
    if op is None:
        return img
    if op == "jpeg":
        return ops.jpeg(img, param)
    if op == "blur":
        return ops.blur(img, param)
    if op == "noise":
        rng = np.random.default_rng([NOISE_SEED, int(content_sha256[:15], 16)])
        return ops.noise(img, param, rng)
    raise ValueError(condition)


def load_sources() -> dict[str, pd.DataFrame]:
    """The registered source images, in a fixed row order."""
    ntire = pd.read_parquet("data/manifest_ntire.parquet")
    primary = ntire[ntire["split"].isin(["ntire_val", "ntire_val_hard"]) & (ntire["is_distorted"] == 0)]
    assert not primary["split"].eq("ntire_test").any()
    assert len(primary) == 6250, len(primary)
    sources = {"ntire_clean": primary[KEEP_COLUMNS].reset_index(drop=True)}

    external = pd.read_parquet("data/manifest_external.parquet")
    for name, split in (("socialrf", "ext_aigi_socialrf"), ("synthbuster", "ext_synthbuster")):
        rows = external[external["split"] == split]
        sampled = []
        for label in (0, 1):
            of_label = rows[rows["label"] == label]
            take = min(SECONDARY_PER_LABEL, len(of_label))
            sampled.append(of_label.sample(n=take, random_state=SUBSAMPLE_SEED))
        sources[name] = pd.concat(sampled)[KEEP_COLUMNS].sort_values("path").reset_index(drop=True)
    return sources


class OurBranch:
    """A frozen crop-trained branch scored under center / grid9 / squish."""

    arms = tuple(OUR_ARMS)

    def __init__(self, tag: str, device: str, chunk: int = 8):
        path, swa = CHECKPOINTS[tag]
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self.towers, self.specs, self.head, _ = _load_models(ck, device, swa)
        self.policy = _policy_from_config(ck["config"], True)
        assert self.policy.mode == "crop", self.policy.mode
        self.poolers = tower_poolers(self.head, len(self.towers))
        self.device, self.chunk = device, chunk

    def views(self, img: np.ndarray) -> dict[str, list[np.ndarray]]:
        return {arm: build(img, self.policy) for arm, build in OUR_ARMS.items()}

    def image_logits(self, views_per_image: list[list[np.ndarray]]) -> np.ndarray:
        """Mean logit over each image's views (one view for center/squish)."""
        flat = [v for views in views_per_image for v in views]
        pooled = torch.cat([_forward_tower(t, sp, flat, self.device, torch.bfloat16, self.chunk, pooler=pl)
                            for t, sp, pl in zip(self.towers, self.specs, self.poolers)], dim=-1)
        window_logits = self.head(pooled.float(), None)["logit"].double().cpu().numpy()
        out, start = [], 0
        for views in views_per_image:
            out.append(window_logits[start:start + len(views)].mean())
            start += len(views)
        assert start == len(window_logits)
        return np.array(out)


class BFreeDetector:
    """B-Free under native (five-crop) / train_fov / squish, one image per forward."""

    arms = BFREE_ARMS

    def __init__(self, device: str):
        self.model, self.device = BFree(device), device

    def views(self, img: np.ndarray) -> dict[str, list[np.ndarray]]:
        return {arm: [bfree_view(img, arm)] for arm in BFREE_ARMS}

    def image_logits(self, views_per_image: list[list[np.ndarray]]) -> np.ndarray:
        out = []
        for views in views_per_image:
            x = to_tensor(views[0], self.model.norm)[None].to(self.device)
            out.append(float(self.model.logits(x)[0]))
        return np.array(out)


def score(model, rows: pd.DataFrame, condition: str, batch: int, workers: int) -> dict[str, np.ndarray]:
    """Decode, damage once, build every presentation, and score them all."""
    from tqdm import tqdm

    def prepare(chunk: pd.DataFrame):
        per_arm = {arm: [] for arm in model.arms}
        for path, sha in zip(chunk["path"], chunk["content_sha256"]):
            damaged = damage(decode(path), condition, sha)
            for arm, views in model.views(damaged).items():
                per_arm[arm].append(views)
        return per_arm

    chunks = [rows.iloc[i:i + batch] for i in range(0, len(rows), batch)]
    logits = {arm: [] for arm in model.arms}
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, chunks, 2 * workers), total=len(chunks), desc=condition):
            per_arm = fut.result()
            for arm in model.arms:
                logits[arm].append(model.image_logits(per_arm[arm]))
    return {arm: np.concatenate(v) for arm, v in logits.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/scratch/dion_r13/dion_extensions/outputs/test2/scores")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sources", default="ntire_clean,socialrf,synthbuster")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=0, help="first N rows per source (throughput check)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    sources = load_sources()
    for model_name in a.models.split(","):
        model = BFreeDetector(a.device) if model_name == "bfree" else OurBranch(model_name, a.device)
        for source in a.sources.split(","):
            rows = sources[source]
            if a.limit:
                rows = rows.iloc[:a.limit]
            for condition in a.conditions.split(","):
                out = f"{a.out_dir}/{model_name}_{source}_{condition}.parquet"
                if os.path.exists(out):
                    print(f"skip {out}", flush=True)
                    continue
                started = time.time()
                logits = score(model, rows, condition, a.batch, a.workers)
                frame = rows.drop(columns=["content_sha256"]).copy()
                for arm, z in logits.items():
                    frame[f"logit_{arm}"] = z
                frame.to_parquet(out + ".tmp", index=False)
                os.replace(out + ".tmp", out)
                seconds = time.time() - started
                print(f"{model_name} {source} {condition}: {len(rows)} images in {seconds:.0f} s "
                      f"({len(rows) / seconds:.1f} img/s) -> {out}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("TEST2_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
