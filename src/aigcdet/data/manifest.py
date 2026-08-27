"""The manifest is the contract every other component reads (spec §7.1)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.data.splits import assign_splits

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

SPLITS = ("train", "val_internal", "heldout_generator", "benchmark")

#: Distinct fake "generator families" in the synthetic fixture. Plural on
#: purpose: with a single generator name the fixture could not exercise the
#: held-out-generator split at all, and a bank built from it made Plan 2's
#: train_rung raise "bank has no val_internal rows".
DUMMY_GENERATORS = ("dummygen_a", "dummygen_b", "dummygen_c")


def validate_manifest(df: pd.DataFrame) -> None:
    """Fail loudly on a manifest that violates its own documented contract.

    Every one of these checks corresponds to a defect that reached the end of
    Plan 1 undetected: COCO val2017 carrying `label = 1`, a dataset name
    standing in for a generator family, and relative paths written under a
    column documented as absolute. They are cheap, and they are the last
    gate before the file is frozen — feature banks index against it
    positionally, so a manifest that is wrong is wrong for every later plan.
    """
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")

    problems: list[str] = []

    bad_labels = sorted(set(df["label"].unique()) - {0, 1})
    if bad_labels:
        problems.append(f"label must be 0 or 1, found {bad_labels}")

    bad_splits = sorted(set(df["split"].unique()) - set(SPLITS))
    if bad_splits:
        problems.append(f"split must be one of {list(SPLITS)}, found {bad_splits}")

    dupes = df["path"][df["path"].duplicated()].unique().tolist()
    if len(dupes):
        problems.append(
            f"{len(dupes)} duplicated path(s), e.g. {dupes[:3]}; rows must be "
            "one-per-image for positional indexing to be meaningful")

    relative = [p for p in df["path"] if not os.path.isabs(str(p))]
    if relative:
        problems.append(
            f"{len(relative)} relative path(s), e.g. {relative[:3]}; `path` is "
            "documented as absolute and is read from other working directories")

    if problems:
        raise ValueError("invalid manifest: " + "; ".join(problems))


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

    Fakes are spread over `DUMMY_GENERATORS`, and the splits are assigned by
    the real `assign_splits` with the last present generator held out, so a
    bank built from this fixture exercises the train / val_internal /
    heldout_generator paths rather than one uniform "train" block. n needs to
    be large enough for `val_fraction` to land at least one row in
    val_internal (a few dozen; the 500-row default is comfortable) and for
    every generator to appear at all.

    Paths recorded in the manifest are absolute, so they remain valid from any
    working directory when read by downstream tasks.

    Deterministic given `rng`: the split seed is drawn from it, not from
    global state.
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
            "generator": DUMMY_GENERATORS[(i // 2) % len(DUMMY_GENERATORS)] if label else "",
            "source": "dummy",
            "licence": "CC0",
            "width": 64,
            "height": 64,
            "split": "",
        })
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    present = [g for g in DUMMY_GENERATORS if (df["generator"] == g).any()]
    return assign_splits(df, heldout_generators=present[-1:],
                         seed=int(rng.integers(0, 2**31 - 1)))
