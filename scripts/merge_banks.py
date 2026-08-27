"""Merge shard banks into one (spec §3.1; the resumable/shardable Stage A).

Stage A is chunked across sessions because one bank takes 8-13 h against
Kaggle's 30 h/week free tier. Every view's pixels depend only on
(seed, row_id, view_idx), so shards extracted independently from disjoint
slices of the same frozen manifest are identical to one uninterrupted run --
this puts them back together.

    python scripts/merge_banks.py --out banks/dinov3l \
        banks/dinov3l_shard0 banks/dinov3l_shard1 banks/dinov3l_shard2

Shards must agree on backbone, dim, n_views and seed, and their row_id sets
must not overlap; merge_banks raises otherwise.
"""
from __future__ import annotations

import argparse

from aigcdet.features.bank import FeatureBank, merge_banks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for the merged bank")
    ap.add_argument("shards", nargs="+", help="shard bank directories, in row order")
    a = ap.parse_args()

    merge_banks(a.shards, a.out)
    bank = FeatureBank.open(a.out)
    print(f"merged {len(a.shards)} shards -> {a.out}: {len(bank.meta)} rows, "
          f"{bank.config['n_views']} views, backbone {bank.config['backbone']}")


if __name__ == "__main__":
    main()
