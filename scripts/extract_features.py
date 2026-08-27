"""Stage A CLI.

`--split` takes a COMMA-SEPARATED list, and the training splits below are the
combination that actually works end to end. Stage B's `train_rung` evaluates
on the bank's own `val_internal` rows, so a bank extracted with `--split train`
alone is rejected -- after the full extraction has already been paid for.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train,val_internal
    # leave-one-transform-out bank for the A3-LOTO run:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l_loto --split train,val_internal \
        --exclude noise
    # attach the reconstruction branch (spec section 3.3) to an existing
    # bank, for ALL of its already-cached views -- --split must select the
    # same rows the bank was originally built from, or attach_recon_to_bank
    # raises rather than silently misaligning:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --out banks/dinov3l --split train,val_internal --recon

Stage A takes 8-13 h per bank, against Kaggle's 30 h/week free tier, so it is
built to be interrupted. `--resume` continues into the same --out, skipping
rows already written; `--workers N` runs the CPU stage (decode, augment,
proxies) in N subprocesses while this process feeds the GPU. Independent
shards -- disjoint slices of the SAME manifest, extracted in separate
sessions -- are recombined by `scripts/merge_banks.py`; because every view's
pixels depend only on (seed, row_id, view_idx), a sharded bank is identical
to one extracted in a single run.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train,val_internal \
        --resume --workers 4
"""
from __future__ import annotations

import argparse

from aigcdet.data.manifest import read_manifest
from aigcdet.features.bank import CHECKPOINT_EVERY
from aigcdet.features.extract import extract_bank


def select_splits(df, splits_arg: str):
    """Filter `df` to a comma-separated list of manifest splits.

    A list, not a single value: Stage B's `train_rung` evaluates on the bank's
    own `val_internal` rows, so a bank must carry the training AND the
    internal-validation split or it is unusable. An unknown split name is a
    typo that would otherwise produce an empty (or wrong) bank after hours of
    extraction, so it raises here instead.
    """
    wanted = [s.strip() for s in splits_arg.split(",") if s.strip()]
    if not wanted:
        return df
    present = sorted(set(df["split"].unique()))
    unknown = [s for s in wanted if s not in present]
    if unknown:
        raise ValueError(
            f"--split names {unknown}, which the manifest does not contain; "
            f"its splits are {present}")
    # No .reset_index(): extract_bank keys its per-image RNG on this index
    # label (see aigcdet.features.extract's module docstring), and
    # attach_recon_to_bank replays those views from the row_id the bank
    # stores, so the filtered frame must keep the frozen manifest's labels.
    return df[df["split"].isin(wanted)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", required=False,
                     help="required unless --recon (an existing bank already names its backbone)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="",
                     help="comma-separated manifest splits to include, e.g. "
                          "'train,val_internal' (Stage B needs both in one bank)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude", default="", help="comma-separated families to exclude")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true",
                     help="continue an interrupted extraction into --out, skipping "
                          "the rows already written (the same manifest, split, "
                          "backbone and seed must be given)")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                     help="flush meta/views parquet every N images; this is also "
                          "how much work a session timeout can cost")
    ap.add_argument("--workers", type=int, default=0,
                     help="subprocesses running the CPU stage (decode, augment, "
                          "proxies) while this process feeds the GPU; 0 = inline")
    ap.add_argument("--recon", action="store_true",
                     help="attach the reconstruction branch (spec section 3.3) to the "
                          "existing bank at --out for all of its cached views, instead "
                          "of building a new bank")
    a = ap.parse_args()

    df = read_manifest(a.manifest)
    df = select_splits(df, a.split)

    if a.recon:
        from aigcdet.features.bank import FeatureBank
        from aigcdet.features.recon import attach_recon_to_bank

        bank = FeatureBank.open(a.out)
        attach_recon_to_bank(bank, df, device=a.device)
        # Post-condition on a multi-hour job: recon.npy must cover every view
        # of every row, or A3-vs-A4 compares different augmentation budgets.
        bank.check_invariants()
        return

    if not a.backbone:
        ap.error("--backbone is required unless --recon is given")
    extract_bank(df, a.backbone, a.out, seed=20260827, device=a.device,
                 limit=a.limit, batch_size=a.batch_size,
                 exclude_families=tuple(f for f in a.exclude.split(",") if f),
                 resume=a.resume, checkpoint_every=a.checkpoint_every,
                 workers=a.workers)


if __name__ == "__main__":
    main()
