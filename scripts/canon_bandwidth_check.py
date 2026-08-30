"""Why some EVAL_GRID conditions barely do anything after standardisation.

Standardisation runs BEFORE the condition (`eval/grid.py:153`), so it cannot
erase a degradation that has not happened yet. What it does instead is cap the
CONTENT: both policies place `band_side`/`crop_side` of real detail inside a
`nominal_side` frame -- 200 in 512, or 0.39 of Nyquist. A `resize` whose own
cut-off sits ABOVE that cap therefore removes almost nothing, and no head can
detect a transform that is very nearly the identity.

This measures that directly, with no bank, no backbone and no checkpoint: it
compares the degraded view against the SAME canonicalised image, so a high
PSNR means the condition changed nothing. The AUC column is the companion
question -- orientation-corrected separability of `laplacian_var`, clean vs
degraded, over the same images -- which is roughly the best a single pixel
statistic could do and therefore a floor to read the A2 head against.

Both policies are run because the obvious hypothesis is that this is band's
fault. It is not: the cap is `side/nominal` either way, so the policy is
irrelevant to this effect and only the RATIO matters. That ratio is
load-bearing -- `CanonPolicy.__post_init__` requires `band_side < nominal_side`
so step 2 is always an upscale, which is what stops native resolution being
re-recorded as an interpolation signature. The pipeline destroys this evidence
on purpose.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from aigcdet.augment.canonical import (CANON_NOMINAL_SIDE, MODE_BAND, MODE_CROP,
                                       CanonPolicy, canonicalise)
from aigcdet.augment.ops import OP_FUNCS
from aigcdet.features.proxies import laplacian_variance

SEED = 20260827
# `cuts_at` is the fraction of Nyquist a resize round-trip preserves; None for
# conditions that are not band-limiting and have no such number.
CONDITIONS = {
    "resize_0.5":  (lambda x, r: OP_FUNCS["resize"](x, 0.5), 0.50),
    "resize_0.25": (lambda x, r: OP_FUNCS["resize"](x, 0.25), 0.25),
    "blur_s2.0":   (lambda x, r: OP_FUNCS["blur"](x, 2.0), None),
    "noise_s0.05": (lambda x, r: OP_FUNCS["noise"](x, 0.05, r), None),
    "jpeg_q30":    (lambda x, r: OP_FUNCS["jpeg"](x, 30), None),
}


def run(manifest: str, root: str, n: int, crop_side: int) -> dict:
    m = pd.read_parquet(manifest)
    m = m[np.minimum(m.width, m.height) >= crop_side]
    rows = m.sample(min(n, len(m)), random_state=SEED)
    policies = {"band": CanonPolicy(mode=MODE_BAND, jitter=0.0),
                "crop": CanonPolicy(mode=MODE_CROP, crop_side=crop_side)}
    acc = {p: {c: {"psnr": [], "clean": [], "deg": []} for c in CONDITIONS}
           for p in policies}
    rng = np.random.default_rng(SEED)
    for _, r in rows.iterrows():
        with Image.open(f"{root}/{r.rel_path}") as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        for pname, pol in policies.items():
            canon = canonicalise(base, policy=pol, rng=None)
            lap_clean = laplacian_variance(canon)
            for cname, (fn, _) in CONDITIONS.items():
                d = fn(canon, rng)
                mse = float(np.mean((canon.astype(np.float64) - d.astype(np.float64)) ** 2))
                acc[pname][cname]["psnr"].append(
                    99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
                acc[pname][cname]["clean"].append(lap_clean)
                acc[pname][cname]["deg"].append(laplacian_variance(d))
    return {"n": len(rows), "crop_side": crop_side, "acc": acc,
            "policies": list(policies)}


def render(out: dict) -> None:
    acc, pols = out["acc"], out["policies"]
    cap = out["crop_side"] / CANON_NOMINAL_SIDE
    print(f"n = {out['n']} images   side {out['crop_side']} in "
          f"{CANON_NOMINAL_SIDE}  ->  cap = {cap:.2f} of Nyquist\n")
    print(f"{'condition':13s} {'cuts at':>8s} | " +
          " | ".join(f"{p} PSNR  {p} AUC" for p in pols))
    for cname, (_, cuts) in CONDITIONS.items():
        cells = []
        for p in pols:
            d = acc[p][cname]
            y = np.r_[np.zeros(len(d["clean"])), np.ones(len(d["deg"]))]
            a = roc_auc_score(y, np.r_[d["clean"], d["deg"]])
            cells.append(f"{np.mean(d['psnr']):8.1f}dB  {max(a, 1 - a):7.4f}")
        print(f"{cname:13s} {(f'{cuts:.2f}' if cuts else '-'):>8s} | " +
              " | ".join(cells))
    print("\nPSNR is the degraded view against the SAME canonicalised image: high")
    print("means the condition changed almost nothing. A condition whose cut-off")
    print(f"sits above {cap:.2f} removes content that standardisation already took.")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--crop-side", type=int, default=200)
    a = ap.parse_args(argv)
    out = run(a.manifest, a.root, a.n, a.crop_side)
    render(out)
    return out


if __name__ == "__main__":
    main()
