"""Reassemble per-GPU scoring shards into one parquet plus metrics.

Refuses gaps or overlaps: a shard scored positionally is `dim` rows of real
probabilities or it is nothing -- a missing shard must fail the merge, not
shrink the split.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from score_plan_splits import split_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", required=True,
                    help="glob for the shard parquets of ONE split, e.g. "
                         "'out/giant.shard*_demo.parquet'")
    ap.add_argument("--expect", type=int, required=True,
                    help="total rows the full split holds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        sys.exit(f"REFUSING: no shards match {a.pattern}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if len(d) != a.expect:
        sys.exit(f"REFUSING: shards hold {len(d)} rows, split has {a.expect}")
    if d["path"].duplicated().any():
        sys.exit("REFUSING: overlapping shards (duplicate paths)")
    d.to_parquet(a.out, index=False)
    ms = split_metrics(d)
    print(f"{a.out}: n={ms['n']:,} AUC={ms['auc']:.4f} "
          f"TPR@1%FPR={ms['tpr_at_1pct_fpr']:.4f} acc={ms['acc_at_0.5']:.4f}")
    if a.metrics_out:
        with open(a.metrics_out, "w") as f:
            json.dump(ms, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
