"""Extract ONE shard of a Stage A feature bank. Kaggle's per-account entry point.

`scripts/extract_features.py` extracts a whole manifest; this extracts a
contiguous slice of one, so five teammates with five free Kaggle accounts can
each pay for a fifth of the 8-13 h and `scripts/merge_banks.py` can put the
result back together.

    python notebooks/run_shard.py \
        --manifest /kaggle/input/aigcdet-manifest/manifest.parquet \
        --root /kaggle/temp/dataset \
        --backbone dinov3l --out /kaggle/working/banks/dinov3l_shard0 \
        --split train,val_internal --shard 0 --n-shards 5 \
        --expect-manifest-sha256 <from the notebook's verification cell> \
        --resume --workers 4

This exists as a FILE rather than as a call in a notebook cell for one
concrete reason: `--workers > 0` runs the CPU stage (decode, augment,
proxies -- about 200 ms per image, the part that otherwise sits serialised
behind the GPU) in a process pool using the "spawn" start method, because this
process holds a CUDA context and forking one is a documented deadlock. Spawn
re-imports the parent's `__main__`, and an IPython kernel's `__main__` is not
re-importable. A notebook cell calling `extract_bank` directly must therefore
pass `workers=0` and give up the parallel CPU stage for the whole run.

`--expect-manifest-sha256` is the verification gate crossing the process
boundary: the notebook verified the attached Datasets against the frozen
manifest and holds that manifest's fingerprint, and this process refuses to
extract anything until the manifest it reads for itself fingerprints the same.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kaggle_bootstrap as kb  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True,
                    help="where the dataset is mounted on THIS machine; the "
                         "frozen manifest's own paths are from another one")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train,val_internal")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--expect-manifest-sha256", required=True,
                    help="the fingerprint the notebook's verification cell "
                         "recorded; this run refuses to start unless the "
                         "manifest it reads matches it")
    ap.add_argument("--seed", type=int, default=20260827,
                    help="must match scripts/extract_features.py's seed, or "
                         "shards are not merge-compatible")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke path only: take the first N rows OF THIS "
                         "SHARD. A bank built with --limit is not a shard of "
                         "the real bank and must never be merged into one")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the shard, check the resume, print the plan, "
                         "and stop before the backbone is loaded")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    _, shard_df = kb.resolve_shard(
        a.manifest, a.root, a.split, a.shard, a.n_shards,
        a.expect_manifest_sha256)

    if a.limit is not None:
        # Applied HERE, not passed to extract_bank, so the bank's recorded
        # manifest fingerprint covers exactly the rows it contains. Handing
        # extract_bank a --limit would make it fingerprint the full shard
        # frame and then write a shorter bank, and `check_resume` would then
        # reject every subsequent resume of a smoke run.
        shard_df = shard_df.iloc[: a.limit]

    if not len(shard_df):
        raise SystemExit(
            f"shard {a.shard} of {a.n_shards} is empty: --n-shards exceeds the "
            f"number of rows in --split {a.split!r}. An empty shard would "
            "write a zero-row bank that merges silently and contributes "
            "nothing. Lower --n-shards.")

    # Before the backbone: a mismatched resume caught here costs a second, and
    # caught inside BankWriter costs a 1.2 GB gated model download first.
    state = kb.check_resume(a.out, shard_df, backbone=a.backbone, seed=a.seed)
    print(f"shard {a.shard}/{a.n_shards}: {len(shard_df)} rows, "
          f"row_id {int(shard_df.index[0])}..{int(shard_df.index[-1])}")
    if state.exists:
        print(f"resuming: {state.n_done}/{state.n_images} images already "
              f"written ({state.fraction_done:.1%}), {state.n_remaining} to go")
    elif a.resume:
        print("nothing to resume from -- this is the first session")

    if a.dry_run:
        print("--dry-run: stopping before the backbone is loaded")
        return 0

    from aigcdet.features.extract import extract_bank

    extract_bank(shard_df, a.backbone, a.out, seed=a.seed, device=a.device,
                 batch_size=a.batch_size, resume=a.resume,
                 checkpoint_every=a.checkpoint_every, workers=a.workers)
    print(f"shard {a.shard} complete -> {a.out}")
    return 0


if __name__ == "__main__":
    # The guard is load-bearing, not conventional: `--workers > 0` spawns
    # subprocesses that re-import this module, and without it each one would
    # start its own extraction.
    sys.exit(main())
