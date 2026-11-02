#!/usr/bin/env python3
"""Amendment 5, Test 4: squish gain over the centre window by damage family.

    python scripts/amend5_test4.py

Reads saved per-image logits of the frozen crop-trained branches (A2 final,
B2 SWA), `center` and `squish` arms, distorted rows only. Loads no model and
no image. Family map, counting rules and pass conditions are the ones
registered in docs/ntire/paper_prereg.md, "Amendment 5", Test 4.

Writes docs/ntire/amend5/test4_damage-family.json, the per-image join as
test4_damage-family_rows.parquet, and test4_damage-family_fig.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, _tokens, commit, paired_diff, with_size  # noqa: E402

OUT = "docs/ntire/amend5"
STEM = f"{OUT}/test4_damage-family"
#: Registered family map (amendment 5, Test 4), written before any regrouping.
FAMILIES = {
    "JPEG": {"jpeg", "jpeg_ai", "jpeg_recompression_1", "jpeg_recompression_2",
             "jpeg_recompression_comb_jpegai", "quantization"},
    "blur": {"gausblur", "lensblur", "motionblur", "glassblur"},
    "noise": {"whitenoise", "impulsenoise", "multnoise", "isonoise"},
    "colour": {"lincontrchange", "darken", "brighten", "rgbshift", "colorshift",
               "colorsat", "jitter", "randomtonecurve", "clahe"},
    "downscale": {"downscale", "pixelate"},
    "geometric": {"perspective", "randomcrop", "randomaspectcrop"},
    "adversarial": {"adv_embed_resnet", "adv_embed_clip"},
}
TOKEN_FAMILY = {t: f for f, ts in FAMILIES.items() for t in ts}
ORDER = list(FAMILIES) + ["other"]
#: `center` on ntire_val_hard is the probe's own arm. ntire_val has no probe
#: `center` parquet; the as-trained score is the same presentation
#: (crop_probe.md: the probe arm reproduces it to 4 dp in AUC).
SOURCES = {
    "ntire_val_hard": {br: (f"outputs/scores/probe_{br}_center_ntire_val_hard.parquet",
                            f"outputs/scores/probe_{br}_squish_ntire_val_hard.parquet")
                       for br in ("A2", "B2swa")},
    "ntire_val": {br: (f"outputs/scores/{br}_ntire_val.parquet",
                       f"outputs/scores/probe_{br}_squish_ntire_val.parquet")
                  for br in ("A2", "B2swa")},
}
BRANCH = {"A2": "A", "B2swa": "B"}
MIN_N = 100


def families_of(tokens) -> set[str]:
    return {TOKEN_FAMILY.get(t, "other") for t in tokens}


def load(split: str) -> pd.DataFrame:
    assert split != "ntire_test", "Test 4 never reads ntire_test"
    rows = None
    for br, (cp, sp) in SOURCES[split].items():
        c = pd.read_parquet(cp)[["path", "label", "is_distorted", "distortions", "logit"]]
        s = pd.read_parquet(sp)[["path", "label", "is_distorted", "logit"]]
        m = c.merge(s, on="path", suffixes=("_c", "_s"), validate="1:1")
        assert len(m) == len(c) == len(s), f"{br} {split}: arms cover different rows"
        assert (m["label_c"] == m["label_s"]).all() and (m["is_distorted_c"] == m["is_distorted_s"]).all()
        m = m.rename(columns={"label_c": "label", "is_distorted_c": "is_distorted",
                              "logit_c": f"{BRANCH[br]}_center", "logit_s": f"{BRANCH[br]}_squish"})
        m = m.drop(columns=["label_s", "is_distorted_s"])
        rows = m if rows is None else rows.merge(
            m[["path", f"{BRANCH[br]}_center", f"{BRANCH[br]}_squish"]], on="path", validate="1:1")
    d = with_size(rows)
    d = d[d["is_distorted"] == 1].reset_index(drop=True)
    d["tokens"] = d["distortions"].apply(_tokens)
    d["families"] = d["tokens"].apply(lambda t: sorted(families_of(t)))
    return d


def joint_gain_draws(d: pd.DataFrame, fams: list[str], br: str, n_boot=2000, seed=SEED):
    """Per-family squish gain under ONE resample of all distorted rows, so the
    difference of two families' gains is paired. Same draw scheme as
    paper_analyses.paired_diff (seed, label strata, percentile CI); rows that
    sit in two families move together, which is what makes it paired."""
    y = d["label"].to_numpy()
    a, b = d[f"{br}_center"].to_numpy(), d[f"{br}_squish"].to_numpy()
    member = {f: d["families"].apply(lambda s, f=f: f in s).to_numpy() for f in fams}
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(y == s) for s in np.unique(y)]
    draws = {f: np.empty(n_boot) for f in fams}
    for i in range(n_boot):
        k = np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])
        for f in fams:
            kk = k[member[f][k]]
            draws[f][i] = auc(y[kk], b[kk]) - auc(y[kk], a[kk])
    return draws


def family_table(d: pd.DataFrame, single: bool) -> dict:
    out = {}
    for f in ORDER:
        mask = d["families"].apply(lambda s: f in s if not single else s == [f])
        g = d[mask]
        if len(g) == 0:
            continue
        y = g["label"].to_numpy()
        e = {"n": int(len(g)), "n_fake": int(y.sum()), "short_p50": float(g["short"].median()),
             "mean_ops": float(g["tokens"].apply(len).mean())}
        if y.min() != y.max():
            for br in ("A", "B"):
                e[br] = paired_diff(y, g[f"{br}_center"].to_numpy(), g[f"{br}_squish"].to_numpy(), y)
        out[f] = e
    # quantization on its own row, as registered
    g = d[d["tokens"].apply(lambda t: "quantization" in t and (not single or families_of(t) == {"JPEG"}))]
    if len(g):
        y = g["label"].to_numpy()
        e = {"n": int(len(g)), "n_fake": int(y.sum()), "short_p50": float(g["short"].median()),
             "mean_ops": float(g["tokens"].apply(len).mean())}
        for br in ("A", "B"):
            e[br] = paired_diff(y, g[f"{br}_center"].to_numpy(), g[f"{br}_squish"].to_numpy(), y)
        out["quantization (in JPEG)"] = e
    return out


def verdict(table: dict, d: pd.DataFrame) -> dict:
    res = {}
    fams = [f for f in ORDER if f in table and table[f]["n"] >= MIN_N and "A" in table[f]]
    for br in ("A", "B"):
        gains = {f: table[f][br]["diff"] for f in fams}
        ci_above = {f: table[f][br]["ci"][0] > 0 for f in fams}
        lo, hi = min(gains.values()), max(gains.values())
        ratio = hi / lo if lo > 0 else float("inf")
        coverage = all(ci_above.values()) and ratio < 2
        draws = joint_gain_draws(d, ["JPEG", "noise", "blur", "colour"], br)
        diffs = {}
        for x in ("JPEG", "noise"):
            for z in ("blur", "colour"):
                dd = draws[x] - draws[z]
                diffs[f"{x}-{z}"] = {"diff": table[x][br]["diff"] - table[z][br]["diff"],
                                     "ci": [float(q) for q in np.nanquantile(dd, [.025, .975])]}
        cleaning = all(v["diff"] > 0 and v["ci"][0] > 0 for v in diffs.values())
        res[br] = {"families_n_ge_100": fams, "ci_above_zero": ci_above,
                   "min_gain": lo, "max_gain": hi, "max_over_min": ratio,
                   "supports_coverage": bool(coverage), "cleaning_diffs": diffs,
                   "supports_cleaning": bool(cleaning),
                   "reading": "coverage" if coverage and not cleaning else
                              "cleaning" if cleaning and not coverage else
                              "both conditions met" if coverage else "mixed"}
    return res


def figure(res: dict, pdf: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    t = res["splits"]["ntire_val_hard"]["table_a"]
    fams = [f for f in ORDER + ["quantization (in JPEG)"] if f in t and "A" in t[f]]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    y = np.arange(len(fams))
    for off, br, col, mk in ((-0.15, "A", "#b2182b", "o"), (0.15, "B", "#2166ac", "s")):
        g = np.array([t[f][br]["diff"] for f in fams])
        lo = g - np.array([t[f][br]["ci"][0] for f in fams])
        hi = np.array([t[f][br]["ci"][1] for f in fams]) - g
        ax.errorbar(g, y + off, xerr=[lo, hi], fmt=mk, ms=3, color=col, lw=0.9,
                    capsize=1.5, label=f"branch {br}")
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f} (n={t[f]['n']})" for f in fams])
    ax.invert_yaxis()
    ax.set_xlabel("robust AUC gain, squish - centre (95% CI)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = {"commit": commit(), "seed": SEED, "n_boot": 2000, "families": {k: sorted(v) for k, v in FAMILIES.items()},
           "sources": SOURCES, "splits": {}}
    parts = []
    for split in ("ntire_val_hard", "ntire_val"):
        d = load(split)
        unmapped = sorted({t for ts in d["tokens"] for t in ts if t not in TOKEN_FAMILY})
        ta, tb = family_table(d, False), family_table(d, True)
        res["splits"][split] = {"n_distorted": int(len(d)), "unmapped_tokens": unmapped,
                                "n_families_per_row": d["families"].apply(len).value_counts().sort_index().to_dict(),
                                "table_a": ta, "table_b": tb, "verdict_a": verdict(ta, d)}
        parts.append(d.assign(split=split, families=d["families"].apply(",".join),
                              tokens=d["tokens"].apply(",".join))
                     [["split", "path", "label", "short", "tokens", "families",
                       "A_center", "A_squish", "B_center", "B_squish"]])
    pd.concat(parts).to_parquet(f"{STEM}_rows.parquet", index=False)
    figure(res, f"{STEM}_fig.pdf")
    with open(f"{STEM}.json", "w") as f:
        json.dump(res, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else str(o))
    for split, r in res["splits"].items():
        print(split, "n", r["n_distorted"], "unmapped", r["unmapped_tokens"])
        for br, v in r["verdict_a"].items():
            print(" ", br, v["reading"], "cov", v["supports_coverage"], "clean", v["supports_cleaning"],
                  f"ratio {v['max_over_min']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
