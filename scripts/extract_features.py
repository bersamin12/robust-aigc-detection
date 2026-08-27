"""Stage A CLI.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train
    # leave-one-transform-out bank for the A3-LOTO run:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l_loto --exclude noise
"""
from __future__ import annotations

import argparse

from aigcdet.data.manifest import read_manifest
from aigcdet.features.extract import extract_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="", help="filter to one manifest split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude", default="", help="comma-separated families to exclude")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    df = read_manifest(a.manifest)
    if a.split:
        # No .reset_index(): extract_bank keys its per-image RNG on this
        # index label so the same image draws the same views regardless of
        # how manifest_df was filtered or sliced to reach it (see
        # aigcdet.features.extract module docstring).
        df = df[df["split"] == a.split]
    extract_bank(df, a.backbone, a.out, seed=20260827, device=a.device,
                 limit=a.limit, batch_size=a.batch_size,
                 exclude_families=tuple(f for f in a.exclude.split(",") if f))


if __name__ == "__main__":
    main()
