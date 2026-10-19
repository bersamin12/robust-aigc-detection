#!/usr/bin/env python3
"""Amendment 5, Test 3: shrink ratio vs field of view.

    python scripts/amend5_test3.py

Reads saved per-image `center` and `squish` logits of the frozen crop-trained
branches (A2 final at n=512, B2 SWA at n=224). Loads no model and no image.

Registered model (docs/ntire/paper_prereg.md, Amendment 5, Test 3):
    g = (2y - 1)(z_squish - z_center), each arm z-scored on ntire_val
    g ~ branch + log S_short + log S_long + log area_coverage + is_distorted
    S_short = short / n, S_long = long / n, area_coverage = 200^2 / (h * w)

Before fitting, this script checks the design matrix's rank. With the
registered area_coverage, log area = 2 log 200 - log S_short - log S_long
- 2 log n, and log n is a function of branch, so the matrix is rank-deficient
by construction. The script reports that and does NOT fit the registered
OLS. The Figure 1 re-plot, the other registered part, runs in full.

Writes docs/ntire/amend5/test3_shrink-ratio.json and _fig.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analyses import SEED, commit, paired_diff, with_size  # noqa: E402

OUT = "docs/ntire/amend5"
#: Same inputs as Test 4. ntire_val has no probe `center` parquet; the
#: as-trained score is the same presentation (crop_probe.md, 4 dp in AUC).
SOURCES = {
    "ntire_val_hard": {br: (f"outputs/scores/probe_{br}_center_ntire_val_hard.parquet",
                            f"outputs/scores/probe_{br}_squish_ntire_val_hard.parquet")
                       for br in ("A2", "B2swa")},
    "ntire_val": {br: (f"outputs/scores/{br}_ntire_val.parquet",
                       f"outputs/scores/probe_{br}_squish_ntire_val.parquet")
                  for br in ("A2", "B2swa")},
}
STEM = f"{OUT}/test3_shrink-ratio"
N = {"A": 512, "B": 224}
BR = {"A2": "A", "B2swa": "B"}
WINDOW = 200
N_BINS = 8


def arms(split: str) -> pd.DataFrame:
    """Long format: one row per (image, branch) with centre and squish logits."""
    assert split != "ntire_test", "Test 3 never reads ntire_test"
    out = []
    for br, (cp, sp) in SOURCES[split].items():
        c = pd.read_parquet(cp)[["path", "label", "is_distorted", "logit"]]
        s = pd.read_parquet(sp)[["path", "label", "logit"]]
        m = c.merge(s, on="path", suffixes=("_c", "_s"), validate="1:1")
        assert len(m) == len(c) == len(s) and (m["label_c"] == m["label_s"]).all()
        out.append(m.rename(columns={"label_c": "label", "logit_c": "z_center", "logit_s": "z_squish"})
                    .drop(columns="label_s").assign(branch=BR[br]))
    return with_size(pd.concat(out, ignore_index=True))


def zscore(d: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each (branch, arm) logit z-scored with mean/sd from ntire_val (all rows)."""
    stats, d = {}, d.copy()
    for b in N:
        for col in ("z_center", "z_squish"):
            r = ref.loc[ref["branch"] == b, col]
            mu, sd = float(r.mean()), float(r.std(ddof=0))
            stats[f"{b}_{col}"] = {"mu": mu, "sigma": sd}
            m = d["branch"] == b
            d.loc[m, col] = (d.loc[m, col] - mu) / sd
    return d, stats


def covariates(d: pd.DataFrame) -> pd.DataFrame:
    n = d["branch"].map(N)
    long = d[["width", "height"]].max(axis=1)
    win = np.minimum(WINDOW, d["short"])
    return d.assign(
        g=(2 * d["label"] - 1) * (d["z_squish"] - d["z_center"]),
        is_B=(d["branch"] == "B").astype(float),
        log_S_short=np.log(d["short"] / n), log_S_long=np.log(long / n),
        log_area=np.log(WINDOW ** 2 / (d["width"] * d["height"])),        # registered
        log_area_realised=np.log(win ** 2 / (d["width"] * d["height"])),  # crop_clamp window
        fov=win / d["short"], S=d["short"] / n)


def rank_check(d: pd.DataFrame, area_col: str) -> dict:
    cols = ["is_B", "log_S_short", "log_S_long", area_col, "is_distorted"]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    sv = np.linalg.svd(X, compute_uv=False)
    # residual of the area column regressed on the other regressors
    others = np.delete(X, 4, axis=1)
    beta, *_ = np.linalg.lstsq(others, X[:, 4], rcond=None)
    resid = X[:, 4] - others @ beta
    return {"columns": ["intercept"] + cols, "rank": int(np.linalg.matrix_rank(X)),
            "n_columns": X.shape[1], "singular_values": sv.tolist(),
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
            "area_on_others_coef": dict(zip(["intercept", "is_B", "log_S_short", "log_S_long",
                                             "is_distorted"], beta.tolist())),
            "area_on_others_r2": float(1 - resid.var() / X[:, 4].var()),
            "area_resid_sd": float(resid.std()),
            "rows_short_below_window": int((d["short"] < WINDOW).sum())}


def curves(d: pd.DataFrame) -> list[dict]:
    """Figure 1 per branch: robust AUC centre and squish in equal-count bins of
    native short side (distorted rows), plotted against FOV and against S."""
    dd = d[d["is_distorted"] == 1]
    img = dd[dd["branch"] == "A"]
    edges = np.quantile(img["short"], np.linspace(0, 1, N_BINS + 1))
    out = []
    for b in N:
        f = dd[dd["branch"] == b]
        bins = np.clip(np.searchsorted(edges, f["short"], side="right") - 1, 0, N_BINS - 1)
        for k in range(N_BINS):
            g = f[bins == k]
            y = g["label"].to_numpy()
            r = paired_diff(y, g["z_center"].to_numpy(), g["z_squish"].to_numpy(), y)
            out.append({"branch": b, "bin": k, "n": int(len(g)),
                        "short_median": float(g["short"].median()),
                        "fov_median": float(g["fov"].median()), "S_median": float(g["S"].median()),
                        "center": r["a"], "squish": r["b"], "gain": r["diff"], "gain_ci": r["ci"]})
    return out


def alignment(cv: list[dict]) -> dict:
    """Descriptive, not registered: how far apart the two branches' gain curves
    are when matched on FOV (same bin) vs on S (A's curve interpolated in log S
    at B's bin medians, inside the overlapping S range)."""
    a = sorted((c for c in cv if c["branch"] == "A"), key=lambda c: c["bin"])
    b = sorted((c for c in cv if c["branch"] == "B"), key=lambda c: c["bin"])
    on_fov = [abs(x["gain"] - y["gain"]) for x, y in zip(a, b)]
    la = np.log([c["S_median"] for c in a])
    ga = np.array([c["gain"] for c in a])
    on_s = [abs(np.interp(np.log(c["S_median"]), la, ga) - c["gain"])
            for c in b if la.min() <= np.log(c["S_median"]) <= la.max()]
    return {"mean_abs_gap_matched_on_fov": float(np.mean(on_fov)), "n_bins_fov": len(on_fov),
            "mean_abs_gap_matched_on_S": float(np.mean(on_s)) if on_s else None,
            "n_bins_S_overlap": len(on_s)}


def figure(res: dict, pdf: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    cv = res["splits"]["ntire_val_hard"]["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, key, lab in ((axes[0], "fov_median", "centre-window FOV = 200 / short (bin median)"),
                         (axes[1], "S_median", "squish shrink ratio S = short / n (bin median)")):
        for b, col, mk in (("A", "#b2182b", "o"), ("B", "#2166ac", "s")):
            c = [x for x in cv if x["branch"] == b]
            x = np.array([v[key] for v in c])
            g = np.array([v["gain"] for v in c])
            lo = np.array([v["gain_ci"][0] for v in c])
            hi = np.array([v["gain_ci"][1] for v in c])
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(x, g, color=col, marker=mk, ms=3, lw=1.1, label=f"branch {b} (n={N[b]})")
        ax.set_xscale("log")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("robust AUC gain, squish - centre")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    val = arms("ntire_val")
    res = {"commit": commit(), "seed": SEED, "window": WINDOW, "n": N, "splits": {}}
    zval, res["zscore"] = zscore(val, val)
    for split in ("ntire_val_hard", "ntire_val"):
        d = covariates(zval if split == "ntire_val" else zscore(arms(split), val)[0])
        cv = curves(d)
        res["splits"][split] = {
            "n_rows_long": int(len(d)), "n_images": int(d["path"].nunique()),
            "rank_registered": rank_check(d, "log_area"),
            "rank_realised_window": rank_check(d, "log_area_realised"),
            "corr": d[["log_S_short", "log_S_long", "log_area"]].corr().round(4).to_dict(),
            "aspect_long_over_short": dict(zip(("p5", "p50", "p95"), np.quantile(
                np.exp(d["log_S_long"] - d["log_S_short"]), [.05, .5, .95]).tolist())),
            "mean_g": {b: float(d.loc[d["branch"] == b, "g"].mean()) for b in N},
            "curves": cv, "alignment": alignment(cv)}
    figure(res, f"{STEM}_fig.pdf")
    with open(f"{STEM}.json", "w") as f:
        json.dump(res, f, indent=2)
    for split, r in res["splits"].items():
        for k in ("rank_registered", "rank_realised_window"):
            x = r[k]
            print(split, k, f"rank {x['rank']}/{x['n_columns']} cond {x['condition_number']:.3g} "
                  f"R2(area|others) {x['area_on_others_r2']:.6f} resid_sd {x['area_resid_sd']:.4g} "
                  f"short<200: {x['rows_short_below_window']}")
        print(split, "alignment", r["alignment"], "aspect", r["aspect_long_over_short"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
