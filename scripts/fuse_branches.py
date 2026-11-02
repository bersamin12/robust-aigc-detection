#!/usr/bin/env python3
"""Fuse two branches' per-row scores under the NTIRE protocol.

Reads `<prefix>_<split>.parquet` for each branch (from
`scripts/score_plan_splits.py`), fits the z-scores and the weight on the
pooled `--fit-splits`, applies them unchanged to every split named -- the
test split included, exactly once -- and writes one json with the solo,
equal-weight and fitted-weight numbers per split, the sweep, and the fitted
parameters so `scripts/predict.py` can serve the same fusion.

Refuses to fit on the test split. That is the whole reason this script
exists instead of `scripts/ensemble_probs.py`, whose sweep is labelled
DIAGNOSTIC because it is evaluated on the rows being reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from aigcdet.eval.late_fusion import evaluate, fit_fusion, join_branches


def _load(prefix: str, split: str, score_col: str) -> pd.DataFrame:
    p = f"{prefix}_{split}.parquet"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; score that split first")
    d = pd.read_parquet(p)
    if score_col not in d.columns:
        raise SystemExit(f"{p} has no {score_col!r} column; rescore with a "
                         "score_plan_splits.py that writes logits")
    return d


def _ckpt_of(prefix: str) -> str | None:
    p = f"{prefix}_metrics.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get("ckpt")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="branch A out-prefix")
    ap.add_argument("--b", required=True, help="branch B out-prefix")
    ap.add_argument("--fit-splits", default="ntire_val,ntire_val_hard")
    ap.add_argument("--eval-splits", default="",
                    help="extra splits to report on (e.g. val); the fit "
                         "splits are always reported")
    ap.add_argument("--test-split", default=None,
                    help="the sealed split, reported once and never fitted on; "
                         "omit while still selecting")
    ap.add_argument("--score-col", default="logit")
    ap.add_argument("--out", required=True, help="metrics json")
    ap.add_argument("--write-fused", action="store_true",
                    help="also write <out stem>_<split>_fused.parquet per split")
    a = ap.parse_args(argv)

    fit_splits = [s for s in a.fit_splits.split(",") if s]
    if not fit_splits:
        raise SystemExit("--fit-splits must name at least one split")
    if a.test_split and a.test_split in fit_splits:
        raise SystemExit(f"REFUSING: test split {a.test_split!r} is in --fit-splits")

    joined = {}
    for sp in fit_splits + [s for s in a.eval_splits.split(",") if s] + \
            ([a.test_split] if a.test_split else []):
        if sp in joined:
            continue
        joined[sp] = join_branches(_load(a.a, sp, a.score_col),
                                   _load(a.b, sp, a.score_col), a.score_col)
    fit = pd.concat([joined[s] for s in fit_splits], ignore_index=True)
    population = "split=" + "+".join(fit_splits)
    fusion, sweep = fit_fusion(fit, population)

    rec = {"branch_a": {"prefix": a.a, "ckpt": _ckpt_of(a.a)},
           "branch_b": {"prefix": a.b, "ckpt": _ckpt_of(a.b)},
           "score_col": a.score_col,
           "fusion": fusion.as_record(),
           "sweep_on_fit_rows": {str(k): v for k, v in sweep.items()},
           "n_fit_rows": int(len(fit)),
           "splits": {sp: evaluate(j, fusion) for sp, j in joined.items()},
           "test_split": a.test_split}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)

    stem = os.path.splitext(a.out)[0]
    print(f"fit on {population} ({len(fit):,} rows): w_a={fusion.weight:.2f}  "
          f"z_a=({fusion.z_a.mu:.3f},{fusion.z_a.sigma:.3f})  "
          f"z_b=({fusion.z_b.mu:.3f},{fusion.z_b.sigma:.3f})")
    print(f"{'split':<16}{'model':<13}{'n':>7}{'AUC':>9}{'robust':>9}{'clean':>9}{'TPR@1%':>9}")
    for sp, ms in rec["splits"].items():
        for name, m in ms.items():
            f_ = lambda v: "   n/a" if v is None else f"{v:9.4f}"  # noqa: E731
            print(f"{sp:<16}{name:<13}{m['n']:>7}{f_(m['auc'])}{f_(m['robust_auc'])}"
                  f"{f_(m['clean_auc'])}{f_(m['tpr_at_1pct_fpr'])}")
        if a.write_fused:
            j = joined[sp].copy()
            j["fused_z"] = fusion.apply(j["score_a"].to_numpy(), j["score_b"].to_numpy())
            j.to_parquet(f"{stem}_{sp}_fused.parquet", index=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
