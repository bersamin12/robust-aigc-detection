"""The manifest is the contract every other component reads (spec §7.1)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

MANIFEST_COLUMNS = [
    "path",       # absolute path to the normalised PNG
    "label",      # 0 = authentic, 1 = AI-generated
    "generator",  # e.g. "sdxl", "midjourney"; "" for authentic images
    "source",     # dataset of origin, e.g. "wildfake", "sid_set", "coco_val2017"
    "licence",    # licence string recorded at acquisition (spec §4.5)
    "width",
    "height",
    "split",      # "train" | "val_internal" | "heldout_generator" | "benchmark"
]


def write_manifest(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df[MANIFEST_COLUMNS].to_parquet(path, index=False)


def read_manifest(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)[MANIFEST_COLUMNS]


def make_dummy_manifest(n: int, out_dir: str, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic stand-in so downstream code can be built before real data lands.

    Fakes are given a mild low-pass bias so a trivial classifier can reach
    above-chance accuracy; that makes end-to-end training smoke tests meaningful.

    Paths recorded in the manifest are absolute, so they remain valid from any
    working directory when read by downstream tasks.
    """
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    rows = []
    for i in range(n):
        label = int(i % 2)
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        if label == 1:
            arr = np.clip(arr.astype(np.float32) * 0.5 + 64, 0, 255).astype(np.uint8)
        p = os.path.abspath(os.path.join(out_dir_abs, f"dummy_{i:05d}.png"))
        Image.fromarray(arr).save(p)
        rows.append({
            "path": p,
            "label": label,
            "generator": "dummygen" if label else "",
            "source": "dummy",
            "licence": "CC0",
            "width": 64,
            "height": 64,
            "split": "train",
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
