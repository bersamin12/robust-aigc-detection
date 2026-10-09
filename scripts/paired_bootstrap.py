#!/usr/bin/env python3
"""Paired bootstrap of robust / clean / pooled ROC AUC between scored models.

Every per-row parquet the project writes (`score_plan_splits.py`,
`ntire_crop_probe.py`, `fuse_branches.py --write-fused`) carries `path`,
`label` and usually `is_distorted`, plus one score column. Two models scored
on the same split are joined on `path` and resampled with the SAME row
indices, so the interval on their difference reflects only how the rows
disagree about the two models -- not the far larger variance of either AUC on
its own. That is the interval a "B beats A" claim needs; two marginal
intervals that overlap say nothing about it.

Resampling is stratified by label x is_distorted, so every resample keeps the
split's composition and both classes in both leaderboard groups (a resample
that loses a class has no AUC; stratifying means none is ever skipped).

    python scripts/paired_bootstrap.py \
        --model crop=outputs/scores/A2_ntire_val_hard.parquet \
        --model squish=outputs/scores/probe_A2_squish_ntire_val_hard.parquet \
        --baseline crop --out docs/ntire/boot_A.json

A model spec is NAME=PARQUET or NAME=PARQUET:COLUMN. The column defaults to
`logit`, then `fused_z`, then `prob`, whichever exists first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCORE_COLUMNS = ("logit", "fused_z", "prob")
METRICS = ("auc", "robust_auc", "clean_auc")


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    if "=" not in spec:
        raise SystemExit(f"model spec {spec!r} is not NAME=PARQUET[:COLUMN]")
    name, rest = spec.split("=", 1)
    col = None
    if ":" in rest and not rest.endswith(".parquet"):
        rest, col = rest.rsplit(":", 1)
    return name, rest, col


def load(name: str, path: str, col: str | None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"{name}: missing {path}")
    d = pd.read_parquet(path)
    if col is None:
        col = next((c for c in SCORE_COLUMNS if c in d.columns), None)
        if col is None:
            raise SystemExit(f"{name}: {path} has none of {SCORE_COLUMNS}")
    elif col not in d.columns:
        raise SystemExit(f"{name}: {path} has no column {col!r}")
    keep = ["path", "label"] + (["is_distorted"] if "is_distorted" in d.columns else [])
    if d["path"].duplicated().any():
        raise SystemExit(f"{name}: duplicate paths in {path}")
    return d[keep + [col]].rename(columns={col: name})


def join(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Inner join on path; every shared label / is_distorted must agree."""
    out = frames[0]
    for f in frames[1:]:
        shared = [c for c in ("label", "is_distorted") if c in out.columns and c in f.columns]
        m = out.merge(f, on="path", how="inner", suffixes=("", "_r"))
        for c in shared:
            if (m[c] != m[f"{c}_r"]).any():
                raise SystemExit(f"REFUSING: {c} disagrees between inputs on "
                                 f"{int((m[c] != m[f'{c}_r']).sum())} rows")
            m = m.drop(columns=f"{c}_r")
        for c in ("label", "is_distorted"):
            if c not in m.columns and f"{c}_r" in m.columns:
                m = m.rename(columns={f"{c}_r": c})
        out = m
    sizes = {len(f) for f in frames}
    if len(sizes) > 1 or len(out) != min(sizes):
        print(f"note: inputs have {sorted(sizes)} rows, {len(out)} joined",
              file=sys.stderr)
    return out.reset_index(drop=True)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with midranks for ties; nan if a class is absent."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def all_metrics(y, dist, s) -> dict:
    out = {"auc": auc(y, s)}
    if dist is not None:
        out["robust_auc"] = auc(y[dist == 1], s[dist == 1])
        out["clean_auc"] = auc(y[dist == 0], s[dist == 0])
    return out


def strata_indices(y: np.ndarray, dist: np.ndarray | None) -> list[np.ndarray]:
    key = y.astype(int) * 2 + (dist.astype(int) if dist is not None else 0)
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def paired_bootstrap(df: pd.DataFrame, models: list[str], baseline: str | None,
                     n_boot: int = 2000, seed: int = 20260827,
                     alpha: float = 0.05) -> dict:
    y = df["label"].to_numpy().astype(int)
    dist = (df["is_distorted"].to_numpy().astype(int)
            if "is_distorted" in df.columns else None)
    scores = {m: df[m].to_numpy().astype(np.float64) for m in models}
    point = {m: all_metrics(y, dist, scores[m]) for m in models}
    strata = strata_indices(y, dist)
    rng = np.random.default_rng(seed)
    draws = {m: {k: np.empty(n_boot) for k in point[m]} for m in models}
    for b in range(n_boot):
        idx = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
        yb = y[idx]
        db = dist[idx] if dist is not None else None
        for m in models:
            for k, v in all_metrics(yb, db, scores[m][idx]).items():
                draws[m][k][b] = v

    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    def ci(a):
        return [float(np.nanquantile(a, lo_q)), float(np.nanquantile(a, hi_q))]

    res = {"n": int(len(df)), "n_boot": n_boot, "seed": seed, "alpha": alpha,
           "n_robust": int((dist == 1).sum()) if dist is not None else None,
           "n_clean": int((dist == 0).sum()) if dist is not None else None,
           "models": {}, "vs_baseline": {}}
    for m in models:
        res["models"][m] = {k: {"value": point[m][k], "ci": ci(draws[m][k])}
                            for k in point[m]}
    if baseline is not None:
        for m in models:
            if m == baseline:
                continue
            res["vs_baseline"][m] = {}
            for k in point[m]:
                d = draws[m][k] - draws[baseline][k]
                # Two-sided bootstrap p: how often the resampled difference
                # falls on the other side of zero, doubled.
                p = 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
                res["vs_baseline"][m][k] = {
                    "diff": point[m][k] - point[baseline][k],
                    "ci": ci(d), "p": min(1.0, p),
                    "excludes_zero": bool(ci(d)[0] > 0 or ci(d)[1] < 0)}
    return res


def to_markdown(res: dict, baseline: str | None, title: str = "") -> str:
    def f(v):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
    first = next(iter(res["models"].values()))
    # A metric with no value for any model (robust AUC on a clean-only split)
    # is a column of dashes and NaN differences; leave it out of the table.
    keys = [k for k in METRICS if k in first and any(
        not np.isnan(m[k]["value"]) for m in res["models"].values())]
    lines = [f"### {title}" if title else "",
             f"n={res['n']:,}"
             + (f" (robust {res['n_robust']:,}, clean {res['n_clean']:,})"
                if res.get("n_robust") is not None else "")
             + f", {res['n_boot']} paired resamples stratified by label x "
               f"is_distorted, {int(100 * (1 - res['alpha']))}% percentile CI", "",
             "| model | " + " | ".join(keys) + " |",
             "|---|" + "---|" * len(keys)]
    for m, mm in res["models"].items():
        lines.append(f"| `{m}` | " + " | ".join(
            f"{f(mm[k]['value'])} [{f(mm[k]['ci'][0])}, {f(mm[k]['ci'][1])}]"
            for k in keys) + " |")
    if res["vs_baseline"]:
        lines += ["", f"Paired difference against `{baseline}`:", "",
                  "| model | " + " | ".join(keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for m, dd in res["vs_baseline"].items():
            lines.append(f"| `{m}` | " + " | ".join(
                f"{dd[k]['diff']:+.4f} [{dd[k]['ci'][0]:+.4f}, {dd[k]['ci'][1]:+.4f}]"
                + (" *" if dd[k]["excludes_zero"] else "")
                for k in keys) + " |")
        lines.append("")
        lines.append("`*` = the 95% interval on the paired difference excludes zero.")
    return "\n".join(lines).strip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=PARQUET[:COLUMN], repeatable, at least two")
    ap.add_argument("--baseline", default=None,
                    help="model name every other model is differenced against")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="json; a .md twin is written beside it")
    a = ap.parse_args(argv)

    specs = [parse_spec(s) for s in a.model]
    names = [n for n, _, _ in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate model names in {names}")
    if a.baseline is not None and a.baseline not in names:
        raise SystemExit(f"--baseline {a.baseline!r} is not one of {names}")
    df = join([load(n, p, c) for n, p, c in specs])
    res = paired_bootstrap(df, names, a.baseline, a.n_boot, a.seed)
    res["inputs"] = {n: {"parquet": p, "column": c} for n, p, c in specs}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    md = to_markdown(res, a.baseline, a.title)
    with open(os.path.splitext(a.out)[0] + ".md", "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
