"""Rewrite a metadata bank's absolute image paths onto another box's corpus.

`build_train_bank.py` takes 45 minutes of single-threaded work to turn a
manifest into presence/severity labels, and none of that work is
machine-specific: the only thing tying the result to the box that built it is
the absolute `path` column. Copying the bank and rewriting that column is two
minutes, so a second box should never rebuild what the first one already has.

The rewrite is a prefix substitution, and the only interesting question is
whether it was RIGHT. A wrong prefix does not raise -- it produces a bank whose
every row points at nothing, and the trainer discovers that hours later at the
first decode with the tower already loaded. So this refuses unless

  * every path matched exactly one --map prefix (an unmatched path is a root
    the caller forgot, not a path to leave alone), and
  * a random sample of the rewritten paths actually exists on this box.

Sampling randomly rather than taking the head matters: the manifest is ordered
by source, so the first 400 rows are all one root and would pass while every
other root pointed into space.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def rebase(paths: pd.Series, maps: list[tuple[str, str]]) -> tuple[pd.Series, np.ndarray]:
    """Apply the prefix maps, and report which rows matched one.

    Every test and every slice is taken against the ORIGINAL series, never
    against the partially-rewritten one. Matching on the running result lets a
    later OLD prefix match a path an earlier map already rewrote and rewrite it
    a second time -- which, since the new prefixes are all real directories on
    the target box, produces a plausible-looking path to nothing.
    """
    # object dtype, NOT numpy's fixed-width '<U*': a rewritten path is usually
    # longer than the original, and a fixed-width array truncates it silently
    # to the input's widest string -- producing a shorter path that still looks
    # like a path.
    src = np.asarray([str(p) for p in paths.to_numpy()], dtype=object)
    out = src.copy()
    matched = np.zeros(len(src), dtype=bool)
    for old, new in maps:
        hit = np.fromiter((p.startswith(old) for p in src), dtype=bool,
                          count=len(src))
        if matched[hit].any():
            overlap = src[hit & matched][:2].tolist()
            raise ValueError(
                f"--map prefixes overlap: {old!r} also matches rows an earlier "
                f"map claimed, e.g. {overlap}. Give disjoint prefixes.")
        out[hit] = [new + p[len(old):] for p in src[hit]]
        matched |= hit
    return pd.Series(out, index=paths.index), matched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True, help="bank directory to rewrite in place")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                    help="absolute prefix substitution; repeatable")
    ap.add_argument("--verify-sample", type=int, default=2000,
                    help="rewritten paths to stat before writing (0 disables)")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    maps = []
    for m in a.map:
        if "=" not in m:
            sys.exit(f"REFUSING: --map {m!r} is not OLD=NEW")
        old, new = m.split("=", 1)
        maps.append((old, new))
    if not maps:
        sys.exit("REFUSING: no --map given, so there is nothing to rebase")

    meta_path = os.path.join(a.bank, "meta.parquet")
    if not os.path.exists(meta_path):
        sys.exit(f"REFUSING: no meta.parquet at {a.bank}")
    meta = pd.read_parquet(meta_path)
    if "path" not in meta.columns:
        sys.exit("REFUSING: this bank has no absolute `path` column, so it is "
                 "root-relative already and needs no rebasing")
    if "rel_path" in meta.columns:
        sys.exit("REFUSING: this bank carries `rel_path`, which the sampler "
                 "prefers; rewriting `path` would have no effect")

    new_paths, matched = rebase(meta["path"].astype(str), maps)
    if not matched.all():
        missed = meta["path"].astype(str).to_numpy()[~matched][:3].tolist()
        sys.exit(f"REFUSING: {int((~matched).sum()):,} of {len(meta):,} paths "
                 f"matched no --map prefix, e.g. {missed}")

    n = min(int(a.verify_sample), len(new_paths))
    if n:
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(new_paths), size=n, replace=False)
        sample = new_paths.to_numpy()[idx]
        miss = [p for p in sample if not os.path.exists(p)]
        if miss:
            sys.exit(f"REFUSING: {len(miss)}/{n} randomly sampled rewritten "
                     f"paths do not exist, e.g. {miss[:2]}")
        print(f"verified {n:,} random rewritten paths exist")

    for old, new in maps:
        k = int(meta["path"].astype(str).str.startswith(old).sum())
        print(f"  {k:>9,}  {old} -> {new}")
    if a.dry_run:
        print("dry run: meta.parquet not written")
        return 0
    meta["path"] = new_paths
    tmp = meta_path + ".tmp"
    meta.to_parquet(tmp, index=False)
    os.replace(tmp, meta_path)
    print(f"rewrote {len(meta):,} paths in {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
