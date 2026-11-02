#!/usr/bin/env python3
"""Amendment 5, Test 5: training field of view after augmentation.

    python scripts/amend5_test5.py

On a seeded 20,000-row sample of `train` (seed 20261019), rebuilds every
view's recipe exactly as the live trainer does (`recipe_for_view` under the
bank's seed, pool and max_ops) and follows the image's shape through
`build_view`: pre-canonical ops on the decoded image, then the crop window
(`crop_side` 200, `crop_clamp` on). Logs per view the native short side, the
short side when the window is taken, the realised FOV and the area coverage.
No model is loaded. Pixel content never matters here, so pre-canonical ops
run on a zero array of the file's true shape: the output shape is the same.

Writes docs/ntire/amend5/test5_train-fov.json and _fig.pdf; per-view rows go
to /scratch/dion_r13/dion_extensions/outputs/test5/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.augment.recipes import PRE_CANONICAL_OPS
from aigcdet.features.extract import recipe_for_view

BANK = "data/banks/ntire2_meta"
OUT = "docs/ntire/amend5"
STEM = f"{OUT}/test5_train-fov"
ROWS_OUT = "/scratch/dion_r13/dion_extensions/outputs/test5/test5_train-fov_views.parquet"
SAMPLE_SEED = 20261019
N_SAMPLE = 20000
CROP_SIDE = 200          # scripts/ntire_v2.sh: --crop-side 200 --crop-clamp, both branches
NOMINAL = (512, 224)     # branch A, branch B; the window never reads nominal_side
#: Table 1 of the paper (docs/paper/main.tex, tab:fov), train row.
TABLE1_TRAIN = {"n": 407223, "p5": 0.391, "p50": 0.391, "p95": 0.521}
THRESH = 0.10
POST_CROP_OPS = {"crop", "aspect_crop", "perspective"}
QS = (("p5", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


def quant(x) -> dict:
    x = np.asarray(x, float)
    return {"n": int(len(x)), **{k: float(np.quantile(x, q)) for k, q in QS},
            "min": float(x.min()), "max": float(x.max())}


def main() -> int:
    cfg = json.load(open(f"{BANK}/config.json"))
    seed, pool, max_ops, n_views = cfg["seed"], cfg["recipe_pool"], cfg["max_ops"], cfg["n_views"]
    root = cfg["manifest_root"]
    meta = pd.read_parquet(f"{BANK}/meta.parquet",
                           columns=["image_idx", "row_id", "rel_path", "split", "label"])
    train = meta[meta["split"] == "train"]
    assert "ntire_test" not in set(meta["split"])
    pick = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(train), N_SAMPLE, replace=False))
    s = train.iloc[pick].reset_index(drop=True)

    stored = pd.read_parquet(f"{BANK}/views.parquet")
    stored = stored[stored["image_idx"].isin(set(s["image_idx"]))].set_index(["image_idx", "view_idx"])["recipe_json"]

    rows, mismatched = [], 0
    for r in s.itertuples(index=False):
        with Image.open(os.path.join(root, r.rel_path)) as im:
            w, h = im.size                      # header only; no decode
        native_short = min(h, w)
        for v in range(n_views):
            rec = recipe_for_view(v, int(r.row_id), seed, (), pool=pool, max_ops=max_ops)
            mine = [{"name": o.name, "params": o.params} for o in rec.ops]
            if json.dumps(mine, sort_keys=True) != json.dumps(json.loads(stored[(r.image_idx, v)]), sort_keys=True):
                mismatched += 1
            pre, post = rec.split()
            hh, ww = (pre.apply_pre(np.zeros((h, w, 3), np.uint8)).shape[:2] if pre.ops else (h, w))
            short = min(hh, ww)
            win = min(CROP_SIDE, short)             # crop_clamp
            rows.append((r.image_idx, v, r.label, h, w, native_short, hh, ww, short, win,
                         min(CROP_SIDE, native_short) / native_short, win / short,
                         win * win / (hh * ww),
                         min(CROP_SIDE, native_short) ** 2 / (h * w),
                         bool(pre.ops), bool({o.name for o in post.ops} & POST_CROP_OPS)))
    d = pd.DataFrame(rows, columns=["image_idx", "view_idx", "label", "h", "w", "native_short",
                                    "h_win", "w_win", "short_win", "window", "fov_native",
                                    "fov", "area", "area_native", "has_pre_op", "has_post_crop"])
    os.makedirs(os.path.dirname(ROWS_OUT), exist_ok=True)
    d.to_parquet(ROWS_OUT, index=False)

    img = d[d["view_idx"] == 0]
    deg = d[d["view_idx"] > 0]
    # Trainer weighting: each step uses view 0 once and m_deg=2 views drawn
    # uniformly from 1..10, so view 0 carries 1/3 and each degraded view 1/15.
    tw = lambda c: np.concatenate([np.repeat(img[c].to_numpy(), 5), deg[c].to_numpy()])  # noqa: E731
    fov = {"native_per_image": quant(img["fov_native"]),
           "post_all11_views": quant(d["fov"]),
           "post_degraded_views": quant(deg["fov"]),
           "post_trainer_weighted": quant(tw("fov"))}
    area = {"native_per_image": quant(img["area_native"]),
            "post_all11_views": quant(d["area"]),
            "post_degraded_views": quant(deg["area"]),
            "post_trainer_weighted": quant(tw("area"))}
    post = fov["post_all11_views"]
    lo_ext = TABLE1_TRAIN["p5"] - post["p5"]
    hi_ext = post["p95"] - TABLE1_TRAIN["p95"]
    res = {"commit": commit(), "bank": BANK, "bank_seed": seed, "pool": pool, "max_ops": max_ops,
           "n_views": n_views, "sample_seed": SAMPLE_SEED, "n_images": int(len(s)),
           "n_views_total": int(len(d)), "crop_side": CROP_SIDE, "crop_clamp": True,
           "nominal_sides": list(NOMINAL),
           "nominal_note": "the crop window reads crop_side only, so FOV and area are identical at n=512 and n=224",
           "recipe_mismatches_vs_bank": mismatched,
           "frac_views_with_pre_op": float(d["has_pre_op"].mean()),
           "frac_degraded_views_with_pre_op": float(deg["has_pre_op"].mean()),
           "frac_views_clamped": float((d["short_win"] < CROP_SIDE).mean()),
           "frac_views_with_post_crop_op": float(d["has_post_crop"].mean()),
           "native_short": quant(img["native_short"]),
           "short_at_window": quant(d["short_win"]),
           "fov": fov, "area": area, "table1_train": TABLE1_TRAIN,
           "reading": {"low_end_extension": lo_ext, "high_end_extension": hi_ext,
                       "threshold": THRESH,
                       "rewrite_sentence": bool(lo_ext > THRESH or hi_ext > THRESH)}}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    bins = np.linspace(0, 1, 51)
    ax.hist(img["fov_native"], bins=bins, density=True, histtype="step", color="0.3", lw=1.1,
            label="native (per image)")
    ax.hist(d["fov"], bins=bins, density=True, histtype="step", color="#b2182b", lw=1.1,
            label="after augmentation (all 11 views)")
    ax.axvspan(TABLE1_TRAIN["p5"], TABLE1_TRAIN["p95"], color="0.88", lw=0, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("FOV of the 200 px window (window / short side)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{STEM}_fig.pdf")
    with open(f"{STEM}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("recipe_mismatches_vs_bank", "frac_views_with_pre_op",
                                          "frac_views_clamped", "frac_views_with_post_crop_op",
                                          "reading")}, indent=1))
    for k, v in fov.items():
        print("fov", k, {q: round(v[q], 3) for q in ("p5", "p25", "p50", "p75", "p95", "min", "max")})
    for k, v in area.items():
        print("area", k, {q: round(v[q], 3) for q in ("p5", "p50", "p95")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
