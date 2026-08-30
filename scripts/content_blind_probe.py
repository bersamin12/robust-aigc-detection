"""The content-blind control, run on the CANONICALISED view rather than the file.

Why the existing control is not enough here
-------------------------------------------
`controls.thumbnail_features` reads each image straight off disk. That is
correct for the section 4.2 headline -- "do the two classes differ in gross
appearance as published" -- and it is blind to the confound that crop
standardisation introduces, because that confound is created BY the
standardisation and does not exist in the file.

Crop takes a 200x200 window at native resolution. On a 200px WildFake image
that window is the whole frame; on an 800px NTIRE photograph it is a detail.
So under crop, FIELD OF VIEW becomes a function of native resolution, native
resolution is a property of the source, and two of the union's sources (COCO,
Open Images) are authentic-only. "Small detail, therefore real" is then a route
to the right answer that has nothing to do with generation -- and every proxy
in `gate_confounds.py` is blind to it, because it is content, not spectrum.

This runs the same cross-validated 16x16 thumbnail control over the view the
model actually receives: decode -> canonicalise(policy) -> 16x16. A high AUC
means the classes are separable by gross composition AFTER standardisation.

Read it as a DIFFERENCE, not a level
------------------------------------
A single arm's number is close to meaningless: these corpora differ in
appearance for reasons that predate any policy. What is diagnostic is crop's
figure against band's on identical rows. Band collapses every image to one
bandwidth and preserves whole-frame composition; crop preserves detail and
varies the frame. If crop's content-blind AUC is materially higher, crop
bought its spectral improvement with a content shortcut.

The within-source breakdown is the second half. If the gap disappears once
source is held constant, the shortcut IS source identity -- which is
actionable (it is what handoff 02's generated half fixes) rather than
mysterious.

CPU only, so it costs no wall clock: run it beside the GPU arms.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.augment.canonical import (
    CANON_CROP_SIDE, MODES, MODE_BAND, MODE_CROP, CanonPolicy, canonical_rng,
    canonicalise)
from aigcdet.eval.controls import NO_QUALITY_COLUMN, content_blind_auc

#: The view whose crop window this control reproduces. View 0 is the clean
#: view -- no degradation recipe -- so its thumbnail isolates what
#: standardisation alone did to the frame. Any other view would mix the
#: recipe's own resizing and cropping into the measurement.
CLEAN_VIEW = 0

#: Matches `controls.thumbnail_features`. Not a tunable: 16x16 is small enough
#: to destroy every band a generator fingerprint lives in and large enough to
#: keep composition, which is the whole balance the control depends on.
THUMB = 16


def canonicalised_thumbnail(path: str, policy: CanonPolicy, seed: int,
                            row_id: int, size: int = THUMB) -> np.ndarray:
    """One row's clean view, as a `size x size x 3` thumbnail in [0, 1].

    BILINEAR, matching `controls.thumbnail_features`, and load-bearing for the
    same reason: a point-sampling filter would ALIAS fine structure into the
    thumbnail rather than destroying it, and the control would stop being
    blind to the band the detector actually uses.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    view = canonicalise(arr, policy=policy,
                        rng=canonical_rng(seed, row_id, CLEAN_VIEW))
    t = Image.fromarray(view).resize((size, size), Image.BILINEAR)
    return np.asarray(t, dtype=np.float32).reshape(-1) / 255.0


def features_for(df: pd.DataFrame, policy: CanonPolicy, seed: int,
                 workers: int = 16) -> np.ndarray:
    """`(N, 768)` thumbnails, keyed on each row's manifest INDEX LABEL.

    The index label and not a positional index, because that is the key the
    extraction derives its crop offset from. Using `range(len(df))` here would
    thumbnail a different window than the model sees, and the control would
    silently measure a corpus nobody trained on.
    """
    rows = list(zip(df["path"], df.index))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(
            lambda pr: canonicalised_thumbnail(str(pr[0]), policy, seed, int(pr[1])),
            rows))
    return np.stack(out).astype(np.float32)


def within_source(features: np.ndarray, labels: np.ndarray,
                  sources: np.ndarray, seed: int, min_rows: int = 400) -> dict:
    """The control again, per source, so a pooled figure driven purely by
    source identity is visible as such.

    A source with one class present cannot produce an AUC and is reported as
    such rather than skipped silently -- for the union that is COCO and Open
    Images, which is exactly the asymmetry under investigation.
    """
    out = {}
    for src in sorted(set(sources.tolist())):
        m = sources == src
        y = labels[m]
        if len(np.unique(y)) < 2:
            out[src] = {"n": int(m.sum()), "note": "one class only -- no AUC"}
            continue
        if m.sum() < min_rows:
            out[src] = {"n": int(m.sum()), "note": f"under {min_rows} rows"}
            continue
        r = content_blind_auc(features[m], y, seed=seed,
                              quality_branches=NO_QUALITY_COLUMN)
        out[src] = {"n": int(m.sum()), "auc": r["auc"], "verdict": r["verdict"]}
    return out


def run(manifest_path: str, mode: str, seed: int, limit: int | None,
        crop_side: int, workers: int) -> dict:
    df = pd.read_parquet(manifest_path)
    if limit:
        df = df.iloc[:limit]
    policy = (CanonPolicy(mode=MODE_CROP, crop_side=crop_side)
              if mode == MODE_CROP else CanonPolicy(mode=MODE_BAND))
    print(f"{mode}: {len(df):,} rows through {policy.as_record()}")
    feats = features_for(df, policy, seed, workers=workers)
    labels = df["label"].to_numpy()
    pooled = content_blind_auc(feats, labels, seed=seed,
                               quality_branches=NO_QUALITY_COLUMN)
    per_source = within_source(feats, labels, df["source"].to_numpy(), seed)
    return {"mode": mode, "n_rows": int(len(df)), "policy": policy.as_record(),
            "pooled": pooled, "within_source": per_source}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", action="append", choices=MODES, default=None,
                    help="repeatable; defaults to BOTH, because the number is "
                         "only meaningful as a difference")
    ap.add_argument("--out", default="docs/content_blind_probe.json")
    ap.add_argument("--seed", type=int, default=20260827,
                    help="must match the extraction's seed, or this "
                         "thumbnails a different crop than the model sees")
    ap.add_argument("--crop-side", type=int, default=CANON_CROP_SIDE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    modes = a.mode or list(MODES)

    results = {m: run(a.manifest, m, a.seed, a.limit, a.crop_side, a.workers)
               for m in modes}

    print("\n=========== CONTENT-BLIND CONTROL (16x16 after standardisation) ===========")
    for m, r in results.items():
        p = r["pooled"]
        auc = p.get("auc", p.get("auc_unverified_branch_provenance"))
        lo, hi = p.get("auc_ci", (float("nan"),) * 2)
        print(f"  {m:5s} pooled AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]  {p['verdict']}")
    if len(results) == 2:
        a_crop = results[MODE_CROP]["pooled"]["auc"]
        a_band = results[MODE_BAND]["pooled"]["auc"]
        print(f"\n  crop - band = {a_crop - a_band:+.4f}")
        print("  Positive means crop bought its spectral gain with a content")
        print("  shortcut. Check the within-source rows before acting: if the")
        print("  gap vanishes there, the shortcut is source identity.")
    print("\n  within source:")
    for m, r in results.items():
        for src, s in sorted(r["within_source"].items()):
            note = s.get("note")
            print(f"    {m:5s} {src:16s} n={s['n']:6,}  "
                  + (note if note else f"AUC {s['auc']:.4f}  {s['verdict']}"))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {a.out}")
    return results


if __name__ == "__main__":
    main()
