#!/usr/bin/env python3
"""Normalise NTIRE train shards straight from their `images/` + `labels.csv`.

Shards 0-2 went through `stage_ntire.py` -> `build_dataset.py` on another
box, which numbered every normalised file with ONE running index per bucket
across the whole union corpus (`build_dataset.py:406-407`, `{i:07d}.png`).
That index space is frozen into `manifest_union.parquet`, so a later shard
cannot be appended to it without either colliding or renumbering -- and
renumbering orphans every bank fingerprinted against the union manifest.

This script therefore writes the later shards BESIDE the existing tree,
under names that cannot collide (`s{shard}_{i:07d}.png`), and emits one
sidecar parquet per shard with everything a manifest row needs. It does not
touch `manifest_union.parquet`; `scripts/build_ntire_manifest.py` reads the
sidecars and the union manifest together.

`build_dataset.py` is deliberately not reused: it needs a `LICENCES.json` in
the raw root, a `--demo-dir`, and at least two held-out-eligible generator
families for its split step, none of which an NTIRE-only shard has.

The label polarity is `stage_ntire.BUCKET_FOR_LABEL`'s (0 = real,
1 = generated), verified against a trained head before that file was
written; the bucket names are the union tree's own (`real/`, `ntire/`), read
back through `sources.classify` so the label a row carries is derived from
the same table every other ingestion path uses.

Idempotent: an existing PNG is measured, not rewritten, so an interrupted
run resumes at the cost of a decode per finished image. Failures are
recorded to a json beside the sidecar, as `build_dataset` does, and are
never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

from aigcdet.data.manifest import compute_digests
from aigcdet.data.normalize import SHORT_SIDE, normalize_many
from aigcdet.data.sources import LICENCES, classify

#: NTIRE's own encoding, from the dataset card (see scripts/stage_ntire.py):
#: the RAW bucket `sources.classify` knows, which is how the label and the
#: pseudo-generator are derived rather than restated here.
RAW_BUCKET_FOR_LABEL = {0: "real", 1: "generated"}
#: Where the normalised file lands. `build_dataset` names a fake's directory
#: after its GENERATOR (`{generator or "real"}`), and NTIRE's pseudo-generator
#: is the source name -- so the union tree on disk is `ntire/real/` and
#: `ntire/ntire/`, and shards 3-5 must land in the same two directories.
DEST_DIR_FOR_LABEL = {0: "real", 1: "ntire"}
SOURCE = "ntire"

SIDECAR_COLUMNS = ["path", "label", "generator", "source", "licence", "width",
                   "height", "shard", "src_name", "content_sha256"]


def _read_labels(shard_dir: str) -> pd.DataFrame:
    labels = pd.read_csv(os.path.join(shard_dir, "labels.csv"))
    for col in ("image_name", "label"):
        if col not in labels.columns:
            raise SystemExit(f"{shard_dir}/labels.csv has no {col!r} column; "
                             f"it has {list(labels.columns)}")
    unknown = sorted(set(labels["label"]) - set(RAW_BUCKET_FOR_LABEL))
    if unknown:
        raise SystemExit(f"{shard_dir}/labels.csv carries label(s) {unknown}; "
                         f"this script knows only {RAW_BUCKET_FOR_LABEL}")
    return labels


def normalize_shard(shard_dir: str, shard: int, dest_root: str, *,
                    workers: int, short_side: int = SHORT_SIDE,
                    digest_workers: int = 8) -> tuple[pd.DataFrame, list]:
    """Normalise one shard into `<dest_root>/ntire/<bucket>/s<k>_<i>.png`.

    Returns `(rows, failures)`: one sidecar row per image that normalised,
    and `(src, reason)` for each that did not.
    """
    labels = _read_labels(shard_dir)
    pairs, meta = [], []
    for i, (name, label) in enumerate(zip(labels["image_name"], labels["label"])):
        src = os.path.join(shard_dir, "images", str(name))
        dst = os.path.join(dest_root, SOURCE, DEST_DIR_FOR_LABEL[int(label)],
                           f"s{shard}_{i:07d}.png")
        pairs.append((src, dst))
        meta.append((str(name), RAW_BUCKET_FOR_LABEL[int(label)], dst))

    todo = [(s, d) for s, d in pairs if not os.path.exists(d)]
    print(f"shard {shard}: {len(pairs):,} images, {len(todo):,} to normalise",
          flush=True)
    failures: list = []
    if todo:
        _, failures = normalize_many(todo, workers=workers)
    failed = {src for src, _ in failures}

    rows = []
    for (src, dst), (name, bucket, _) in tqdm(zip(pairs, meta), total=len(pairs),
                                              desc=f"measure s{shard}"):
        if src in failed or not os.path.exists(dst):
            if src not in failed:
                failures.append((src, "normalised file missing after run"))
            continue
        label, generator = classify(SOURCE, bucket)
        with Image.open(dst) as im:
            w, h = im.size
        rows.append((dst, int(label), generator, SOURCE, LICENCES[SOURCE],
                     int(w), int(h), int(shard), name, ""))
    df = pd.DataFrame(rows, columns=SIDECAR_COLUMNS)
    if len(df):
        digests = compute_digests(df["path"].tolist(), workers=digest_workers)
        df["content_sha256"] = [d for d, _ in digests]
    return df, failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="directory holding shard_<k>/{images,labels.csv}")
    ap.add_argument("--shard", type=int, required=True, action="append",
                    help="shard number; repeatable")
    ap.add_argument("--dest", required=True,
                    help="normalised tree root, e.g. .../normalized_union")
    ap.add_argument("--sidecar-dir", required=True,
                    help="where ntire_shard<k>_rows.parquet is written")
    ap.add_argument("--skipped-dir", default="docs/ntire",
                    help="where normalize_skipped_shard<k>.json is written")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)

    os.makedirs(a.sidecar_dir, exist_ok=True)
    os.makedirs(a.skipped_dir, exist_ok=True)
    for k in a.shard:
        shard_dir = os.path.join(a.src, f"shard_{k}")
        if not os.path.isdir(os.path.join(shard_dir, "images")):
            raise SystemExit(f"{shard_dir}/images is not a directory")
        df, failures = normalize_shard(shard_dir, k, a.dest, workers=a.workers)
        out = os.path.join(a.sidecar_dir, f"ntire_shard{k}_rows.parquet")
        df.to_parquet(out, index=False)
        skipped = os.path.join(a.skipped_dir, f"normalize_skipped_shard{k}.json")
        with open(skipped, "w") as f:
            json.dump([{"src": s, "reason": r} for s, r in failures], f, indent=2)
        by = df["label"].value_counts().to_dict()
        print(f"shard {k}: wrote {out} ({len(df):,} rows; label counts {by}); "
              f"{len(failures)} failure(s) -> {skipped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
