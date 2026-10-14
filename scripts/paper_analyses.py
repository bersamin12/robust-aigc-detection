#!/usr/bin/env python3
"""Post-hoc analyses quoted in the paper, from saved per-row scores only.

    python scripts/paper_analyses.py groups    # squish gain by distortion group
    python scripts/paper_analyses.py testsize  # where read 2's clean loss sits
    python scripts/paper_analyses.py valsize   # the same cut on the fit splits
    python scripts/paper_analyses.py figsize   # robust AUC against native size

None of these is a gate. Each reads parquets that earlier, committed runs
wrote and loads no model and no image. `testsize` reads the SAVED fused
scores of the two declared test reads; it does not score ntire_test again.
Every subcommand writes docs/ntire/paper/<name>.json and .md (figsize also
writes docs/paper/fig_size.pdf).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc  # noqa: E402

MANIFEST = "data/manifest_ntire.parquet"
OUT = "docs/ntire/paper"
SEED = 20260827
#: Probe parquets of the frozen round-two crop checkpoints (crop_probe.md).
PROBE = {"A": "outputs/scores/probe_A2", "B": "outputs/scores/probe_B2swa"}
READ1 = "docs/ntire/fusion_ntire2_final_ntire_test_fused.parquet"
READ2 = "docs/ntire/paper/fusion_squish_final_ntire_test_fused.parquet"
BIG = 1100


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


def with_size(d: pd.DataFrame) -> pd.DataFrame:
    if "width" not in d.columns:
        m = pd.read_parquet(MANIFEST, columns=["path", "width", "height"]).drop_duplicates("path")
        d = d.merge(m, on="path", how="left")
        if d["width"].isna().any():
            raise SystemExit(f"{int(d['width'].isna().sum())} rows not in {MANIFEST}")
    return d.assign(short=d[["width", "height"]].min(axis=1))


def paired_diff(y, a, b, strata, n_boot=2000, seed=SEED):
    """AUC(b) - AUC(a) with a percentile CI, resampling within each stratum."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        k = np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])
        draws[i] = auc(y[k], b[k]) - auc(y[k], a[k])
    lo, hi = np.nanquantile(draws, [.025, .975])
    return {"a": auc(y, a), "b": auc(y, b), "diff": auc(y, b) - auc(y, a),
            "ci": [float(lo), float(hi)]}


def ci(r):
    return f"{r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"


# ---------------------------------------------------------------------------

def _tokens(s):
    if isinstance(s, str):
        return ast.literal_eval(s) if s.startswith("[") else [s]
    return list(s) if s is not None else []


def distortion_group(tokens) -> str:
    t = set(tokens)
    if t & {"randomcrop", "randomaspectcrop"}:
        return "crop or aspect crop"
    if "perspective" in t:
        return "perspective"
    if "downscale" in t:
        return "downscale, no crop"
    return "photometric only"


def groups() -> tuple[dict, str]:
    """Squish gain over the centre window within distortion groups of the
    distorted ntire_val_hard rows, per frozen crop-trained branch."""
    res = {"commit": commit(), "split": "ntire_val_hard", "groups": {}}
    for br, pre in PROBE.items():
        c = pd.read_parquet(f"{pre}_center_ntire_val_hard.parquet")
        s = pd.read_parquet(f"{pre}_squish_ntire_val_hard.parquet")[["path", "logit"]]
        d = with_size(c.merge(s, on="path", suffixes=("_c", "_s")))
        d = d[d["is_distorted"] == 1].copy()
        d["group"] = d["distortions"].apply(_tokens).apply(distortion_group)
        for g, f in d.groupby("group"):
            y = f["label"].to_numpy()
            r = paired_diff(y, f["logit_c"].to_numpy(), f["logit_s"].to_numpy(), y)
            e = res["groups"].setdefault(g, {"n": int(len(f)),
                                             "short_p50": float(f["short"].median())})
            e[br] = r
    order = ["photometric only", "downscale, no crop", "crop or aspect crop", "perspective"]
    L = ["# Squish gain by distortion group (ntire_val_hard, distorted rows)", "",
         "Frozen round-two crop checkpoints (A2 final, B2 SWA); robust AUC of the "
         "probe's `squish` arm minus its `center` arm within each group, 2000 "
         "paired resamples stratified by label.", "",
         "| group | n | short p50 | A centre -> squish | A gain [95% CI] | "
         "B centre -> squish | B gain [95% CI] |", "|---|---|---|---|---|---|---|"]
    for g in order:
        e = res["groups"].get(g)
        if e is None:
            continue
        L.append(f"| {g} | {e['n']} | {e['short_p50']:.0f} | "
                 f"{e['A']['a']:.4f} -> {e['A']['b']:.4f} | {ci(e['A'])} | "
                 f"{e['B']['a']:.4f} -> {e['B']['b']:.4f} | {ci(e['B'])} |")
    L += ["", f"Commit `{res['commit']}`."]
    return res, "\n".join(L) + "\n"


# ---------------------------------------------------------------------------

def testsize() -> tuple[dict, str]:
    """Clean AUC of each branch on the two test reads, split by native size."""
    r1 = pd.read_parquet(READ1)
    r2 = pd.read_parquet(READ2)[["path", "score_a", "score_b", "fused_z"]]
    d = with_size(r1.merge(r2, on="path", suffixes=("_1", "_2")))
    c = d[d["is_distorted"] == 0]
    edges = [(0, 736), (737, BIG), (BIG + 1, 10**6)]
    res = {"commit": commit(), "note": "saved scores of reads 1 and 2; no new read",
           "buckets": {}, "mean_fused_z": {}}
    L = ["# Read 2's clean loss by native size (saved test scores, no new read)", "",
         "Clean rows of `ntire_test`. Read 1 = crop fusion, read 2 = squish fusion; "
         "per-branch scores are each read's own branch models. Paired 95% CI, "
         "2000 resamples stratified by label.", "",
         "| short side | n | A read 1 -> 2 | A diff [95% CI] | B read 1 -> 2 | B diff [95% CI] |",
         "|---|---|---|---|---|---|"]
    for lo, hi in edges:
        f = c[(c["short"] >= lo) & (c["short"] <= hi)]
        y = f["label"].to_numpy()
        key = f"{lo}-{hi}" if hi < 10**6 else f">{BIG}"
        e = {"n": int(len(f)), "n_fake": int(y.sum())}
        for br in ("a", "b"):
            e[br.upper()] = paired_diff(y, f[f"score_{br}_1"].to_numpy(),
                                        f[f"score_{br}_2"].to_numpy(), y)
        res["buckets"][key] = e
        L.append(f"| {key} | {e['n']} | {e['A']['a']:.4f} -> {e['A']['b']:.4f} | "
                 f"{ci(e['A'])} | {e['B']['a']:.4f} -> {e['B']['b']:.4f} | {ci(e['B'])} |")
    L += ["", "Mean fused z by native size (higher looks more generated):", "",
          "| class | read | " + " | ".join(res["buckets"]) + " |",
          "|---|---|" + "---|" * len(edges)]
    for lab, name in ((0, "real"), (1, "generated")):
        for rd in ("1", "2"):
            vals = []
            for lo, hi in edges:
                f = c[(c["short"] >= lo) & (c["short"] <= hi) & (c["label"] == lab)]
                vals.append(float(f[f"fused_z_{rd}"].mean()))
            res["mean_fused_z"][f"{name}_read{rd}"] = vals
            L.append(f"| {name} | {rd} | " + " | ".join(f"{v:+.3f}" for v in vals) + " |")
    L += ["", f"Commit `{res['commit']}`."]
    return res, "\n".join(L) + "\n"


def valsize() -> tuple[dict, str]:
    """The same size cut on the fit splits: crop-trained against squish-trained,
    clean rows above 1100 px."""
    picks = json.load(open(f"{OUT}/picks.json"))
    crop = {"A": "outputs/scores/paper/A2", "B": "outputs/scores/paper/B2swa"}
    res = {"commit": commit(), "picks": picks, "splits": {}}
    L = [f"# Clean rows above {BIG}px on the fit splits", "",
         "Crop-trained (round two) against squish-trained (picks.json), each as "
         "trained. Paired 95% CI, 2000 resamples stratified by label.", "",
         "| split | branch | n (real/fake) | crop -> squish | diff [95% CI] |",
         "|---|---|---|---|---|"]
    for sp in ("ntire_val", "ntire_val_hard"):
        for br in ("A", "B"):
            a = pd.read_parquet(f"{crop[br]}_{sp}.parquet")[["path", "label", "is_distorted", "logit"]]
            b = pd.read_parquet(f"{picks[br]}_{sp}.parquet")[["path", "logit"]]
            d = with_size(a.merge(b, on="path", suffixes=("_c", "_s")))
            f = d[(d["is_distorted"] == 0) & (d["short"] > BIG)]
            y = f["label"].to_numpy()
            r = paired_diff(y, f["logit_c"].to_numpy(), f["logit_s"].to_numpy(), y)
            r.update(n=int(len(f)), n_fake=int(y.sum()))
            res["splits"].setdefault(sp, {})[br] = r
            L.append(f"| {sp} | {br} | {r['n']} ({r['n'] - r['n_fake']}/{r['n_fake']}) | "
                     f"{r['a']:.4f} -> {r['b']:.4f} | {ci(r)} |")
    L += ["", f"Commit `{res['commit']}`."]
    return res, "\n".join(L) + "\n"


# ---------------------------------------------------------------------------

def figsize(pdf: str, n_bins: int = 8) -> tuple[dict, str]:
    """Fused robust AUC on ntire_val_hard in equal-count bins of native short
    side: the crop fusion against the squish-trained fusion."""
    fz = {"crop": f"{OUT}/fusion_crop_ntire_val_hard_fused.parquet",
          "squish": f"{OUT}/fusion_squish_ntire_val_hard_fused.parquet"}
    a = pd.read_parquet(fz["crop"])[["path", "label", "is_distorted", "fused_z"]]
    b = pd.read_parquet(fz["squish"])[["path", "fused_z"]]
    d = with_size(a.merge(b, on="path", suffixes=("_c", "_s")))
    d = d[d["is_distorted"] == 1].reset_index(drop=True)
    edges = np.quantile(d["short"], np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, d["short"], side="right") - 1, 0, n_bins - 1)
    res = {"commit": commit(), "inputs": fz, "n_bins": n_bins, "bins": []}
    for k in range(n_bins):
        f = d[bins == k]
        y = f["label"].to_numpy()
        r = paired_diff(y, f["fused_z_c"].to_numpy(), f["fused_z_s"].to_numpy(), y)
        rng = np.random.default_rng(SEED + k)
        cells = [np.flatnonzero(y == c) for c in (0, 1)]
        dr = {"crop": [], "squish": []}
        for _ in range(2000):
            i = np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])
            dr["crop"].append(auc(y[i], f["fused_z_c"].to_numpy()[i]))
            dr["squish"].append(auc(y[i], f["fused_z_s"].to_numpy()[i]))
        res["bins"].append({
            "lo": float(f["short"].min()), "hi": float(f["short"].max()),
            "median": float(f["short"].median()), "n": int(len(f)),
            "crop": r["a"], "squish": r["b"], "diff": r["diff"], "diff_ci": r["ci"],
            "crop_ci": [float(np.quantile(dr["crop"], q)) for q in (.025, .975)],
            "squish_ci": [float(np.quantile(dr["squish"], q)) for q in (.025, .975)]})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral",
                         "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    x = np.array([b["median"] for b in res["bins"]])
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.axvspan(384, 512, color="0.88", lw=0, zorder=0)
    ax.text(443, 0.995, "train", ha="center", va="top", fontsize=7, color="0.35")
    for name, col, mk in (("crop", "#b2182b", "o"), ("squish", "#2166ac", "s")):
        y = np.array([b[name] for b in res["bins"]])
        lo = np.array([b[f"{name}_ci"][0] for b in res["bins"]])
        hi = np.array([b[f"{name}_ci"][1] for b in res["bins"]])
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, y, color=col, marker=mk, ms=3, lw=1.1,
                label="native window (crop)" if name == "crop" else "whole image (squish)")
    ax.set_xscale("log")
    ax.set_xticks([150, 300, 500, 1000, 2000])
    ax.set_xticklabels(["150", "300", "500", "1000", "2000"])
    ax.minorticks_off()
    ax.set_xlabel("native short side (px), bin median")
    ax.set_ylabel("robust AUC")
    ax.set_ylim(min(0.75, float(np.min([b["crop_ci"][0] for b in res["bins"]])) - 0.01), 1.0)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    fig.savefig(pdf)
    res["pdf"] = pdf

    L = ["# Fused robust AUC by native short side (ntire_val_hard, distorted rows)", "",
         f"{n_bins} equal-count bins. Crop fusion against the squish-trained fusion; "
         "paired 95% CI on the difference, 2000 resamples stratified by label.", "",
         "| short side | median | n | crop | squish | diff [95% CI] |",
         "|---|---|---|---|---|---|"]
    for b in res["bins"]:
        L.append(f"| {b['lo']:.0f}-{b['hi']:.0f} | {b['median']:.0f} | {b['n']} | "
                 f"{b['crop']:.4f} | {b['squish']:.4f} | {b['diff']:+.4f} "
                 f"[{b['diff_ci'][0]:+.4f}, {b['diff_ci'][1]:+.4f}] |")
    L += ["", f"Figure: `{pdf}`. Commit `{res['commit']}`."]
    return res, "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("groups", "testsize", "valsize", "figsize"))
    ap.add_argument("--pdf", default="docs/paper/fig_size.pdf")
    a = ap.parse_args(argv)
    fn = {"groups": groups, "testsize": testsize, "valsize": valsize,
          "figsize": lambda: figsize(a.pdf)}[a.what]
    res, md = fn()
    stem = f"{OUT}/analysis_{a.what}"
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=2)
    with open(stem + ".md", "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
