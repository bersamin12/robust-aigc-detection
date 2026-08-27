"""Stage A CLI.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train
    # leave-one-transform-out bank for the A3-LOTO run:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l_loto --exclude noise
    # attach the reconstruction branch (spec section 3.3) to an existing
    # bank, for ALL of its already-cached views -- --split must select the
    # same rows the bank was originally built from, or attach_recon_to_bank
    # raises rather than silently misaligning:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --out banks/dinov3l --split train --recon
"""
from __future__ import annotations

import argparse

from aigcdet.data.manifest import read_manifest
from aigcdet.features.extract import extract_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", required=False,
                     help="required unless --recon (an existing bank already names its backbone)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="", help="filter to one manifest split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude", default="", help="comma-separated families to exclude")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--recon", action="store_true",
                     help="attach the reconstruction branch (spec section 3.3) to the "
                          "existing bank at --out for all of its cached views, instead "
                          "of building a new bank")
    a = ap.parse_args()

    df = read_manifest(a.manifest)
    if a.split:
        # No .reset_index(): extract_bank keys its per-image RNG on this
        # index label so the same image draws the same views regardless of
        # how manifest_df was filtered or sliced to reach it (see
        # aigcdet.features.extract module docstring). attach_recon_to_bank
        # (--recon) doesn't key anything on this index -- it only uses `df`
        # to check the filtered manifest is still the same rows, in the same
        # order, that the bank at --out was originally built from.
        df = df[df["split"] == a.split]

    if a.recon:
        from aigcdet.features.bank import FeatureBank
        from aigcdet.features.recon import attach_recon_to_bank

        bank = FeatureBank.open(a.out)
        attach_recon_to_bank(bank, df, device=a.device)
        return

    if not a.backbone:
        ap.error("--backbone is required unless --recon is given")
    extract_bank(df, a.backbone, a.out, seed=20260827, device=a.device,
                 limit=a.limit, batch_size=a.batch_size,
                 exclude_families=tuple(f for f in a.exclude.split(",") if f))


if __name__ == "__main__":
    main()
