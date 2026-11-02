#!/usr/bin/env python3
"""Amendment 12, Test 21: Test 14's shrink x coverage factorial on B-Free.

    python scripts/amend5_test21.py rows                 # RTX 4090 box, CPU: the registered rows + pixel SHA-256
    python scripts/amend5_test21.py selfcheck            # scoring box: authors' demo logits to 1e-3
    python scripts/amend5_test21.py score --image-root ROOT   # scoring box, GPU, resumable
    python scripts/amend5_test21.py score --limit 3 --out-dir /tmp/t21 --image-root ROOT   # smoke / timing

Registered design (amendment 12, Test 21):

  Model: B-Free (BFREE_dino2reg4), public_detector_probe.BFree, unchanged.
  Views: tiles504(k, f) = amend5_test14.views_tiles with a 504 px window fed at
  504 px (no resize: _resize_short_side returns a 504 px window unchanged), the
  seven cells of Test 14; `native` (the authors' five-crop on the whole image)
  and `squish` (504), as in Test 2. Mean logit per image.
  For a 504 px input the five-crop wrapper sees five identical crops, so tiles
  are scored with one forward each (`one_crop_logits`); the smoke run asserts it
  equals BFree.logits to 1e-4.
  Rows: SocialRF clean and `_deg` (amend5_test16.presented), short side as
  presented >= 2,016 px, up to 500 per label per split (seed 20261019). The row
  step stores the SHA-256 of each presented image; the scorer asserts it, so the
  scoring box sees the pixels the row step saw.

`amend5_test14.py` and `amend5_test16.py` are copied unchanged from branch
dion/test17-roundtrip.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aigcdet.train.finetune import _windowed  # noqa: E402
from amend5_test14 import CELLS, KEEP_COLUMNS, cell_name, views_tiles  # noqa: E402
from amend5_test16 import CLEAN_SPLIT, DAMAGED_SPLIT, PER_LABEL_PER_SPLIT, SUBSAMPLE_SEED, presented  # noqa: E402
from public_detector_probe import BFree, bfree_view, to_tensor  # noqa: E402
import public_detector_probe  # noqa: E402

# The weights live under /scratch/justin_r13/ext/models on the RTX 4090 box; the second
# scoring box keeps a copy elsewhere and names it here.
public_detector_probe.MODELS_DIR = os.environ.get("BFREE_MODELS_DIR", public_detector_probe.MODELS_DIR)

WINDOW = 504
MIN_SHORT_SIDE = 4 * WINDOW  # tiles504(4, 1) and tiles504(2, 0.25) need >= 504 px sides
OUT_ROOT = "/scratch/dion_r13/dion_extensions/outputs/test21"
ROWS_FILE = f"{OUT_ROOT}/rows.parquet"
#: Where the manifest's SocialRF files live on the RTX 4090 box; --image-root replaces it.
SOURCE_ROOT = "/scratch/justin_r13/ext/aigibench"
POLICY = SimpleNamespace(crop_side=WINDOW, nominal_side=WINDOW)


def pixel_sha(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()


def build_rows(workers: int) -> pd.DataFrame:
    """Test 16's row rule with a 2,016 px floor, plus each presented image's SHA-256."""
    external = pd.read_parquet("data/manifest_external.parquet")
    picked = []
    for split in (CLEAN_SPLIT, DAMAGED_SPLIT):
        rows = external[external["split"] == split].copy()
        # Damage never enlarges an image, so a native short side below the floor cannot pass.
        rows = rows[rows[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE]
        def measure(path: str) -> tuple[int, int, str]:
            img = presented(path, split)  # one image in memory per worker
            return img.shape[0], img.shape[1], pixel_sha(img)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            measured = list(pool.map(measure, rows["path"]))
        rows["height"] = [h for h, _, _ in measured]
        rows["width"] = [w for _, w, _ in measured]
        rows["pixel_sha256"] = [sha for _, _, sha in measured]
        rows = rows[rows[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE]
        rows["is_distorted"] = int(split == DAMAGED_SPLIT)
        print(f"{split}: {len(rows)} rows pass the size filter, by label {rows['label'].value_counts().to_dict()}")
        for label in (0, 1):
            of_label = rows[rows["label"] == label]
            picked.append(of_label.sample(n=min(PER_LABEL_PER_SPLIT, len(of_label)), random_state=SUBSAMPLE_SEED))
    out = pd.concat(picked)[KEEP_COLUMNS + ["pixel_sha256"]].sort_values(["split", "path"]).reset_index(drop=True)
    assert not out["split"].eq("ntire_test").any()
    return out


def all_arms(img: np.ndarray) -> dict[str, list[np.ndarray]]:
    arms = {"native": [bfree_view(img, "native")], "squish": [bfree_view(img, "squish")]}
    for k, f in CELLS:
        arms[cell_name(k, f)] = views_tiles(img, POLICY, k, f)
    return arms


def one_crop_logits(model: BFree, x: torch.Tensor) -> torch.Tensor:
    """BFree.logits for inputs of exactly 504 px: its five crops coincide, so one forward."""
    pe = model.patch_embed
    assert x.shape[-2:] == (WINDOW, WINDOW), x.shape
    e = pe.proj(x)
    assert e.shape[-2:] == tuple(pe.grid_size), (e.shape, pe.grid_size)
    if pe.flatten:
        e = e.flatten(2).transpose(1, 2)
    return model.model(pe.norm(e))[:, 0]


class Scorer:
    def __init__(self, device: str, chunk: int):
        self.model, self.device, self.chunk = BFree(device), device, chunk

    def image_logits(self, arm: str, views_per_image: list[list[np.ndarray]]) -> np.ndarray:
        """Mean logit over each image's views. `native` keeps the authors' five-crop forward."""
        if arm == "native":
            return np.array([float(self.model.logits(to_tensor(v[0], self.model.norm)[None].to(self.device))[0])
                             for v in views_per_image])
        flat = [v for views in views_per_image for v in views]
        window_logits = []
        for i in range(0, len(flat), self.chunk):
            x = torch.stack([to_tensor(v, self.model.norm) for v in flat[i:i + self.chunk]]).to(self.device)
            window_logits.append(one_crop_logits(self.model, x).double().cpu().numpy())
        window_logits = np.concatenate(window_logits)
        out, start = [], 0
        for views in views_per_image:
            out.append(window_logits[start:start + len(views)].mean())
            start += len(views)
        assert start == len(window_logits)
        return np.array(out)


def score(scorer: Scorer, rows: pd.DataFrame, image_root: str, workers: int):
    from tqdm import tqdm

    def prepare(row) -> dict[str, list[np.ndarray]]:
        local = row.path.replace(SOURCE_ROOT, image_root, 1)
        img = presented(local, row.split)  # the damage seed is keyed on the file name only
        assert pixel_sha(img) == row.pixel_sha256, f"pixels differ from the row step: {row.path}"
        return all_arms(img)

    logits, counts = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, list(rows.itertuples()), 2 * workers), total=len(rows)):
            for arm, v in fut.result().items():
                logits.setdefault(arm, []).append(scorer.image_logits(arm, [v]))
                counts.setdefault(arm, []).append(len(v))
    return ({arm: np.concatenate(v) for arm, v in logits.items()},
            {arm: np.array(v) for arm, v in counts.items()})


def check_one_crop(scorer: Scorer, rows: pd.DataFrame, image_root: str) -> None:
    """The registered equality: one forward == the authors' five-crop forward, on 504 px windows."""
    worst = 0.0
    with torch.no_grad():
        for row in rows.head(2).itertuples():
            img = presented(row.path.replace(SOURCE_ROOT, image_root, 1), row.split)
            for window in views_tiles(img, POLICY, 2, 1.0)[:4]:
                x = to_tensor(window, scorer.model.norm)[None].to(scorer.device)
                worst = max(worst, abs(float(one_crop_logits(scorer.model, x)[0]) - float(scorer.model.logits(x)[0])))
    print(f"one-crop vs five-crop max |diff| = {worst:.2e}", flush=True)
    assert worst < 1e-4, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["rows", "selfcheck", "score"])
    ap.add_argument("--out-dir", default=f"{OUT_ROOT}/scores")
    ap.add_argument("--rows-file", default=ROWS_FILE)
    ap.add_argument("--image-root", default=SOURCE_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="first and last N rows (smoke test, timing)")
    ap.add_argument("--chunk", type=int, default=16, help="windows per forward")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.step == "rows":
        os.makedirs(OUT_ROOT, exist_ok=True)
        build_rows(a.workers).to_parquet(ROWS_FILE, index=False)
        print(f"rows -> {ROWS_FILE}")
        return 0
    if a.step == "selfcheck":
        from public_detector_probe import selfcheck
        return selfcheck(a.device)

    rows = pd.read_parquet(a.rows_file)
    assert not rows["split"].eq("ntire_test").any(), "Test 21 never reads ntire_test"
    if a.limit:
        rows = pd.concat([rows.head(a.limit), rows.tail(a.limit)]).reset_index(drop=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = f"{a.out_dir}/bfree_tiles504.parquet"
    if os.path.exists(out):
        print(f"skip {out}", flush=True)
        return 0
    scorer = Scorer(a.device, a.chunk)
    check_one_crop(scorer, rows, a.image_root)
    started = time.time()
    logits, counts = score(scorer, rows, a.image_root, a.workers)
    frame = rows.copy()
    for arm in logits:
        frame[f"logit_{arm}"] = logits[arm]
        frame[f"n_windows_{arm}"] = counts[arm]
    frame.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    print(f"bfree: {len(rows)} images, {sum(c.sum() for c in counts.values())} windows "
          f"in {time.time() - started:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {out}",
          flush=True)
    print("TEST21_SCORING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
