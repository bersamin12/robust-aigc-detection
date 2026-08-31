#!/usr/bin/env python
"""Precompute the AI-OV7 captions, in parallel, into one merged parquet.

    # one process per GPU, each to its OWN file
    python scripts/caption_ov7.py --part 0 --n-parts 4 --out data/caps_0.parquet
    # then, once:
    python scripts/caption_ov7.py --merge data/caps_*.parquet \
        --out data/ov7_captions.parquet

**Why this exists as a separate step.** `generate_ov7.py` will caption on
demand, and on one process that is correct. On four it is not:
`captions.caption_pool` rewrites the WHOLE parquet on every log tick, so four
workers pointed at one path race, and the last writer wins -- each flush
carries only that worker's captions and silently discards the others'. The
reals whose captions were lost are then generated on an empty prompt, which
`run_family` refuses, so the failure surfaces as a runaway failure rate in a
family rather than as anything about captions.

So: caption to N separate paths, merge once, and pass the merged file to every
generation worker, which then only ever READS it (`caption_pool` returns early
when nothing is missing).

A caption is an input to seed-deterministic generation. Regenerating captions
regenerates the corpus, so the merged file is frozen once written -- this
script refuses to overwrite one that already covers the reals asked for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from aigcdet.generate.captions import caption_pool
from aigcdet.generate.geometry import order_key, shard_block
from aigcdet.generate.pool import rebase_paths

DEFAULT_SEED = 20260830
PORTRAIT = "/mnt/berstorage/techjam/open_images/portrait"


def part_of(pool: pd.DataFrame, part: int, n_parts: int, seed: int) -> pd.DataFrame:
    """This worker's slice of the eligible reals.

    Ordered by `order_key` -- the same primitive `pool.select` shards on -- so
    the slice is a property of the images rather than of the directory
    listing, and re-running one part after a crash reproduces it exactly.

    These parts are a WORK SPLIT for captioning only. They are not the
    generation `--shard`s and do not have to line up with them: every part is
    merged into one file before any generation reads it.
    """
    elig = pool.loc[pool["eligible"]].copy()
    elig["order_key"] = [order_key(i, seed) for i in elig["image_id"]]
    elig = elig.sort_values("order_key", kind="mergesort").reset_index(drop=True)
    if n_parts <= 1:
        return elig
    start, stop = shard_block(len(elig), part, n_parts)
    return elig.iloc[start:stop].reset_index(drop=True)


def merge(paths: list[str], out: Path) -> pd.DataFrame:
    """Concatenate the parts, keeping one caption per image_id."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        missing = {"image_id", "caption"} - set(df.columns)
        if missing:
            raise ValueError(f"{p} has no {sorted(missing)} column")
        frames.append(df)
        print(f"  {p}: {len(df)} captions")
    if not frames:
        raise ValueError("no part files given")
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates("image_id", keep="first")

    blank = merged["caption"].isna() | merged["caption"].astype(str).str.strip().eq("")
    if blank.any():
        # An empty prompt is the failure `docs/03` §8 records: the old
        # inpaint_box arm generated on no conditioning at all. Dropping the
        # rows means generation skips those reals; keeping them means it
        # refuses them one at a time and burns its failure budget.
        print(f"  dropping {int(blank.sum())} blank captions")
        merged = merged.loc[~blank]

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"\n{before} rows from {len(paths)} parts -> {len(merged)} unique "
          f"captions -> {out}")
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="this part's parquet, or the merged one with --merge")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="part files to merge into --out; no GPU is used")
    ap.add_argument("--pool", default="data/ov7_pool.parquet")
    ap.add_argument("--portrait-dir", default=PORTRAIT)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=None,
                    help="caption only the first N of this part, for a smoke")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.merge is not None:
        merge(list(args.merge), out)
        return 0

    if not (0 <= args.part < args.n_parts):
        ap.error(f"--part must be in [0, {args.n_parts}), got {args.part}")

    pool_path = Path(args.pool)
    if not pool_path.exists():
        ap.error(f"no pool at {pool_path}. Build it once with generate_ov7.py, "
                 f"or point --pool at the staged copy.")
    pool = rebase_paths(pd.read_parquet(pool_path), args.portrait_dir)

    mine = part_of(pool, args.part, args.n_parts, args.seed)
    if args.limit:
        mine = mine.iloc[:args.limit]
    print(f"[part {args.part}/{args.n_parts}] {len(mine)} eligible reals "
          f"-> {out}", flush=True)

    caps = caption_pool(dict(zip(mine["image_id"], mine["path"])), out,
                        batch_size=args.batch_size, device=args.device)
    have = caps["caption"].astype(str).str.strip().ne("").sum()
    print(f"[part {args.part}] {have} captions of {len(mine)} reals -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
