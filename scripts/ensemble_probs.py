"""Does averaging two models' probabilities beat either alone?

Consumes the per-row parquets `score_plan_splits.py` writes, joins them on
`path` (refusing anything but an exact 1:1 match -- a partial join would
silently score the ensemble on the intersection while the solo numbers came
from the whole split), and reports each solo, the plain mean, and a weight
sweep.

The sweep is labelled DIAGNOSTIC: every weight is evaluated on the rows being
reported, so its best value is an oracle bound, not a shippable number. The
one honest fitted alternative -- choosing w on val_internal -- needs both
models scored on val too; the plain 0.5 mean needs no fitting and is the
headline ablation.

Probabilities are averaged as probabilities (the user's ask), and the rank
metrics (AUC, TPR@1%FPR) are invariant to any monotone rescaling anyway; only
acc@0.5 would notice logit-space averaging.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0 or (labels == 1).sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]
    return float((scores[labels == 1] > thr).mean())


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    return {"auc": float(roc_auc_score(y, p)),
            "tpr_at_1pct_fpr": tpr_at_fpr(y, p),
            "acc_at_0.5": float(((p >= 0.5) == y).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="parquet from model A")
    ap.add_argument("--b", required=True, help="parquet from model B")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default=None, help="optional metrics json")
    args = ap.parse_args()

    da = pd.read_parquet(args.a)
    db = pd.read_parquet(args.b)
    j = da.merge(db[["path", "prob"]], on="path", suffixes=("_a", "_b"),
                 validate="1:1")
    if len(j) != len(da) or len(j) != len(db):
        raise SystemExit(
            f"REFUSING: join is not 1:1 over the whole split "
            f"({len(da)} vs {len(db)} vs {len(j)} joined); the two parquets "
            "must cover exactly the same rows")
    y = j["label"].to_numpy()
    pa, pb = j["prob_a"].to_numpy(), j["prob_b"].to_numpy()

    rows = {args.name_a: metrics(y, pa),
            args.name_b: metrics(y, pb),
            "mean(0.5/0.5)": metrics(y, (pa + pb) / 2)}
    sweep = {}
    for w in np.linspace(0, 1, 21):
        sweep[round(float(w), 2)] = metrics(y, w * pa + (1 - w) * pb)["auc"]
    best_w = max(sweep, key=sweep.get)

    print(f"n={len(j):,}  (labels {dict(zip(*np.unique(y, return_counts=True)))})")
    print(f"{'model':<18} {'AUC':>8} {'TPR@1%':>8} {'acc@.5':>8}")
    for name, ms in rows.items():
        print(f"{name:<18} {ms['auc']:>8.4f} {ms['tpr_at_1pct_fpr']:>8.4f} "
              f"{ms['acc_at_0.5']:>8.4f}")
    print(f"\nDIAGNOSTIC weight sweep (w on {args.name_a}, oracle -- fitted on "
          f"the reported rows): best w={best_w} AUC={sweep[best_w]:.4f}")

    corr = float(np.corrcoef(pa, pb)[0, 1])
    print(f"prob correlation A,B: {corr:.4f} "
          "(the lower this is, the more an average can add)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n": int(len(j)), "solo": rows,
                       "oracle_sweep_auc": {str(k): v for k, v in sweep.items()},
                       "prob_correlation": corr}, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
