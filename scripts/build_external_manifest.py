#!/usr/bin/env python3
"""External held-out benchmarks for the squish paper -> data/manifest_external.parquet.

Nothing in here was ever trained on or selected on. The NTIRE splits chose
the policy; these sets test it. Three populations, each emitted three times
under different split names so one scorer call per split covers every
reading the paper reports:

    ext_<set>          the files as distributed
    ext_<set>_deg      one seeded ntire-pool chain per file on the native
                       image (`score_plan_splits.degrade_native`); scored
                       only with --degrade-seed
    ext_<set>_jpeg95   every file re-encoded as JPEG q95 before scoring, which
                       equalises container format across the classes

The `_jpeg95` reading exists because of a confound found on arrival:
AIGIBench CommunityAI ships every real as JPEG and every fake as PNG, so a
detector that reads JPEG traces separates it perfectly without detecting
anything (the "Fake or JPEG?" bias, Grommelt et al. 2024). Each set's
format-vs-label and short-side-vs-label AUC are printed and written to the
provenance json, and the paper reports them beside every number.

Layouts (verified on download, 2026-10-08):
    aigibench/SocialRF/{0_real,1_fake}/*.png         3000 / 3000
    aigibench/CommunityAI/{0_real,1_fake}/*          6000 jpg / 6000 png
    synthbuster/synthbuster/<generator>/*  with the RAISE-1k reals in the
                                           folder whose name contains "raise"

Rows whose content digest also occurs in data/manifest_ntire.parquet are
dropped and counted: a file the model trained on is not held out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.data.manifest import compute_digests
from aigcdet.data.sources import LICENCES

COLS = ["path", "label", "generator", "source", "licence", "width", "height",
        "split", "rel_path", "family", "assign_rule", "image_id", "variant",
        "content_sha256", "pixel_sha256", "is_distorted", "distortions",
        "distortion_scales", "file_format"]

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")

#: source name -> (path under --ext-root, layout)
EXT_SETS = {
    "aigibench_socialrf": ("aigibench/SocialRF", "binary"),
    "aigibench_community": ("aigibench/CommunityAI", "binary"),
    "synthbuster": ("synthbuster/synthbuster", "by_generator"),
}
VARIANTS = ("", "_deg", "_jpeg95")
SPLIT_NAMES = {s: f"ext_{s.replace('aigibench_', 'aigi_')}" for s in EXT_SETS}


def _images(d: str) -> list[str]:
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(IMAGE_EXT))


def list_set(root: str, layout: str) -> list[tuple[str, int, str]]:
    """[(path, label, generator)] for one set; raises on an unexpected layout
    rather than guessing which folder is real."""
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    out = []
    if layout == "binary":
        for sub, lab in (("0_real", 0), ("1_fake", 1)):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                raise SystemExit(f"{root}: expected {sub}/ (binary layout)")
            out += [(p, lab, "" if lab == 0 else os.path.basename(root))
                    for p in _images(d)]
    elif layout == "by_generator":
        subs = sorted(s for s in os.listdir(root)
                      if os.path.isdir(os.path.join(root, s)))
        real = [s for s in subs if "raise" in s.lower()]
        if len(real) != 1:
            raise SystemExit(f"{root}: expected exactly one RAISE folder among "
                             f"{subs}, found {real}")
        for s in subs:
            lab = 0 if s in real else 1
            out += [(p, lab, "" if lab == 0 else s)
                    for p in _images(os.path.join(root, s))]
    else:
        raise ValueError(layout)
    if not out:
        raise SystemExit(f"{root}: no images")
    return out


def _dims(path: str) -> tuple[int, int, str]:
    with Image.open(path) as im:
        return im.size[0], im.size[1], (im.format or "").lower()


def leak_auc(y: np.ndarray, x: np.ndarray) -> float:
    """max(AUC, 1-AUC) of a one-dimensional cue: how well it separates the
    classes in EITHER direction, i.e. how much a shortcut could buy."""
    from sklearn.metrics import roc_auc_score
    if len(set(y.tolist())) < 2 or len(set(x.tolist())) < 2:
        return 0.5 if len(set(x.tolist())) < 2 else float("nan")
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def build_set(source: str, root: str, layout: str, workers: int) -> pd.DataFrame:
    items = list_set(root, layout)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        dims = list(ex.map(_dims, [p for p, _, _ in items]))
    base = SPLIT_NAMES[source]
    rows = []
    for (path, lab, gen), (w, h, fmt) in zip(items, dims):
        rows.append(dict(path=path, label=lab, generator=gen, source=source,
                         licence=LICENCES[source], width=w, height=h,
                         split=base, rel_path="", family=gen,
                         assign_rule=f"{base}_external", image_id="",
                         variant=source, content_sha256="", pixel_sha256="",
                         is_distorted=0, distortions="", distortion_scales="",
                         file_format=fmt))
    df = pd.DataFrame(rows, columns=COLS)
    df["content_sha256"] = [d for d, _ in compute_digests(df.path.tolist(),
                                                          workers=workers)]
    return df


def provenance(df: pd.DataFrame) -> dict:
    y = df.label.to_numpy()
    short = df[["width", "height"]].min(axis=1).to_numpy()
    fmt_codes = pd.factorize(df.file_format)[0]
    by_fmt = (df.groupby(["file_format", "label"]).size()
                .unstack(fill_value=0).to_dict(orient="index"))
    return {"n": int(len(df)), "n_real": int((y == 0).sum()),
            "n_fake": int((y == 1).sum()),
            "short_side": {q: float(np.percentile(short, p)) for q, p in
                           (("p5", 5), ("p50", 50), ("p95", 95))},
            "short_side_max": int(short.max()),
            "frac_above_512": float((short > 512).mean()),
            "shortside_vs_label_auc": leak_auc(y, short),
            "format_vs_label_auc": leak_auc(y, fmt_codes),
            "format_by_label": {k: {str(l): int(n) for l, n in v.items()}
                                for k, v in by_fmt.items()},
            "generators": sorted(g for g in df.generator.unique() if g)}


def build(ext_root: str, sets: list[str], train_manifest: str | None,
          workers: int = 12) -> tuple[pd.DataFrame, dict]:
    blocks, prov = [], {}
    for s in sets:
        sub, layout = EXT_SETS[s]
        df = build_set(s, os.path.join(ext_root, sub), layout, workers)
        dup_within = int(df.content_sha256.duplicated().sum())
        df = df.drop_duplicates("content_sha256").reset_index(drop=True)
        dropped_train = 0
        if train_manifest:
            seen = set(pd.read_parquet(train_manifest, columns=["content_sha256"])
                       .content_sha256.dropna())
            hit = df.content_sha256.isin(seen)
            dropped_train = int(hit.sum())
            df = df[~hit].reset_index(drop=True)
        if df.label.nunique() != 2:
            raise SystemExit(f"{s}: only label(s) {sorted(df.label.unique())} "
                             "present after dedup; an AUC needs both")
        p = provenance(df)
        p.update(dropped_duplicate_within=dup_within,
                 dropped_overlap_with_ntire_manifest=dropped_train,
                 root=os.path.join(ext_root, sub))
        prov[SPLIT_NAMES[s]] = p
        for v in VARIANTS:
            blocks.append(df.assign(split=SPLIT_NAMES[s] + v))
    out = pd.concat(blocks, ignore_index=True)
    out["label"] = out.label.astype(int)
    out["is_distorted"] = out.is_distorted.astype(int)
    return out, prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ext-root", default="/scratch/justin_r13/ext")
    ap.add_argument("--sets", default=",".join(EXT_SETS),
                    help="comma list; a set whose folder is absent is skipped "
                         "with a warning unless --require-all")
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--train-manifest", default="data/manifest_ntire.parquet")
    ap.add_argument("--out", default="data/manifest_external.parquet")
    ap.add_argument("--provenance", default="docs/ntire/external_sets.json")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)

    want = [s for s in a.sets.split(",") if s]
    unknown = [s for s in want if s not in EXT_SETS]
    if unknown:
        raise SystemExit(f"unknown sets {unknown}; known {list(EXT_SETS)}")
    present = [s for s in want
               if os.path.isdir(os.path.join(a.ext_root, EXT_SETS[s][0]))]
    missing = sorted(set(want) - set(present))
    if missing:
        if a.require_all:
            raise SystemExit(f"missing sets {missing} under {a.ext_root}")
        print(f"WARNING: skipping absent sets {missing}", file=sys.stderr)
    if not present:
        raise SystemExit("no external set is present")
    df, prov = build(a.ext_root, present, a.train_manifest, a.workers)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, a.out)
    with open(a.provenance, "w") as f:
        json.dump({"sets": prov, "missing": missing, "variants": list(VARIANTS)},
                  f, indent=2)
    for sp, p in prov.items():
        print(f"{sp:22s} n={p['n']:6d} real={p['n_real']:5d} fake={p['n_fake']:5d}"
              f"  short p50={p['short_side']['p50']:.0f} >512={p['frac_above_512']:.0%}"
              f"  shortside-AUC={p['shortside_vs_label_auc']:.3f}"
              f"  format-AUC={p['format_vs_label_auc']:.3f}"
              f"  dropped(ntire overlap)={p['dropped_overlap_with_ntire_manifest']}")
    print(f"wrote {a.out} ({len(df):,} rows over {df.split.nunique()} splits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
