#!/usr/bin/env python3
"""Build the NTIRE-stream manifest: NTIRE's labelled sets are the only test.

The plan manifest (`scripts/build_plan_manifest.py`) is a lineage split for
zero-shot transfer. This one asks a narrower question -- how does the
detector score on the NTIRE 2026 challenge's own labelled sets, under the
challenge's own metric -- so its splits are the challenge's:

    train           every corpus on this box except the organisers' sets
    val             a seeded ~10% slice of train, stratified per source and
                    label, for in-distribution health and calibration
    ntire_val       the organisers' 10k validation set (5k clean, 5k distorted)
    ntire_val_hard  their 2.5k hard validation set
    ntire_test      their 2.5k public test set -- read ONCE, at the end

Model selection, the fusion weight and any threshold are fitted on
`ntire_val` + `ntire_val_hard` (+ `val`); `ntire_test` is the number that
gets quoted beside the leaderboard, and nothing selects on it.

Every assignment is data, not a heuristic, and every row carries an
`assign_rule` saying which line put it where. `rel_path` is relative to
`--exp`, so one `--root` serves every tree (`normalized_union/...`,
`normalized_ov7/...`, `raw_ov7_42k/...`, `raw_ntire_val/...`) and
`scripts/build_train_bank.py --root $EXP` needs no `--path-map`.

Sources on THIS box (2026-09-05): NTIRE train shards 0-2 as normalised PNGs
inside `manifest_union.parquet`; shards 3-5 as sidecars from
`scripts/normalize_ntire_shards.py`; COCO train2017 and Open Images reals from
the union manifest; the 11,978-pair OV7 build via `manifest_ov7.parquet`; the
42k-pair OV7 build as raw JPEG pairs; the three organisers' sets as raw JPEGs.
`sid_set` and `wildfake` were deleted from disk and are dropped by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.data.manifest import compute_digests, read_manifest, relative_to_root
from aigcdet.data.sources import LICENCES, SOURCES

#: Sources whose files were removed from this box; dropped from the union
#: manifest by name rather than by a failed existence check, so the reason is
#: recorded and a stray survivor cannot sneak back in.
DROPPED_SOURCES = ("sid_set", "wildfake")

COLS = ["path", "label", "generator", "source", "licence", "width", "height",
        "split", "rel_path", "family", "assign_rule", "image_id", "variant",
        "content_sha256", "pixel_sha256", "is_distorted", "distortions",
        "distortion_scales"]

#: The organisers' sets: (source name, labels csv, image zip stem, split).
#: The zip stem is resolved against the two layouts a zip can extract to
#: (`<stem>/<stem>/<name>` or `<stem>/<name>`) by `_resolve_image_dir`.
NTIRE_SETS = (
    ("ntire_val", "raw_ntire_val/val_labels.csv", "raw_ntire_val/val_images",
     "ntire_val"),
    ("ntire_val_hard", "raw_ntire_val/val_hard_labels.csv",
     "raw_ntire_val/val_images_hard", "ntire_val_hard"),
    ("ntire_test_public", "raw_ntire_test_public/test_labels.csv",
     "raw_ntire_test_public/test_images", "ntire_test"),
)
FIXED_SPLITS = ("ntire_val", "ntire_val_hard", "ntire_test")
TRAIN_SPLITS = ("train", "val")

OV7_42K_PAIR_COLUMNS = ("image_id", "family", "fake_rel", "real_rel", "width", "height")


def _blank(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df:
            df[c] = 0 if c in ("label", "width", "height", "is_distorted") else ""
    return df[COLS]


def _val_draw(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """A boolean mask selecting round(n * frac) rows, seeded."""
    k = int(round(n * frac))
    mask = np.zeros(n, dtype=bool)
    if k:
        mask[rng.choice(n, size=k, replace=False)] = True
    return mask


# --- blocks ----------------------------------------------------------------

def from_union(repo: str, exp: str) -> pd.DataFrame:
    u = read_manifest(f"{repo}/manifest_union.parquet",
                      root=f"{exp}/normalized_union")
    u = u[~u.source.isin(DROPPED_SOURCES)].copy()
    stray = u[u.split == "heldout_generator"]
    if len(stray):
        raise SystemExit(f"{len(stray)} heldout_generator rows survive dropping "
                         f"{DROPPED_SOURCES}: {stray.source.unique().tolist()}")
    u["split"] = u.split.map({"train": "train", "val_internal": "val"})
    if u.split.isna().any():
        raise SystemExit(f"union manifest has an unmapped split: "
                         f"{u.split[u.split.isna()].unique()}")
    u["family"] = u.generator.fillna("")
    u["assign_rule"] = "union_" + u.split
    u["variant"] = "normalized_union"
    u["image_id"] = ""
    return _blank(u)


def from_shard_sidecars(sidecar_dir: str, shards: tuple[int, ...], frac: float,
                        rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for k in shards:
        p = os.path.join(sidecar_dir, f"ntire_shard{k}_rows.parquet")
        if not os.path.exists(p):
            raise SystemExit(f"missing sidecar {p}; run "
                             f"scripts/normalize_ntire_shards.py --shard {k}")
        parts.append(pd.read_parquet(p))
    if not parts:
        return _blank(pd.DataFrame(columns=COLS))
    s = pd.concat(parts, ignore_index=True)
    s["split"] = "train"
    # Stratified per label, so val's real:fake ratio is the shard's.
    for label in (0, 1):
        idx = np.flatnonzero(s.label.to_numpy() == label)
        s.loc[idx[_val_draw(len(idx), frac, rng)], "split"] = "val"
    s["family"] = s.generator
    s["assign_rule"] = "ntire_shard_" + s.split
    s["variant"] = "normalized_union"
    s["image_id"] = ""
    return _blank(s.drop(columns=["shard", "src_name"]))


def from_ov7_small(repo: str, exp: str) -> pd.DataFrame:
    m = read_manifest(f"{repo}/manifest_ov7.parquet",
                      root=f"{exp}/normalized_ov7/open_images_v7")
    m = m.copy()
    # `heldout_generator` (the flux2_vae lineage) was assigned per image_id
    # group alongside train, so folding it into train keeps every pair on one
    # side; `val_internal` was drawn the same way.
    m["split"] = m.split.map({"train": "train", "heldout_generator": "train",
                              "val_internal": "val"})
    if m.split.isna().any():
        raise SystemExit("manifest_ov7 has an unmapped split")
    m["family"] = m.generator.fillna("")
    m["assign_rule"] = "ov7_11978_" + m.split
    m["variant"] = "open_images_v7_11978"
    m["image_id"] = ""
    return _blank(m)


def from_ov7_42k(root: str, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    """The raw 42k-pair build: one row per image, a real following its fake.

    Read from the image_id-keyed RAW tree, as the plan manifest did: the pair
    correspondence is what lets the split be drawn per photograph, so a real
    never lands opposite its own generated counterpart.
    """
    pairs = pd.read_csv(os.path.join(root, "pairs.csv"))
    missing = [c for c in OV7_42K_PAIR_COLUMNS if c not in pairs.columns]
    if missing:
        raise SystemExit(f"{root}/pairs.csv lacks {missing}; it has "
                         f"{list(pairs.columns)}")
    lic = LICENCES["open_images_v7"]
    ids = pairs.image_id.astype(str).unique()
    val_ids = set(ids[_val_draw(len(ids), frac, rng)])
    rows = []
    for r in pairs.itertuples():
        split = "val" if str(r.image_id) in val_ids else "train"
        rule = f"ov7_42k_pair_{split}"
        rows.append((f"{root}/{r.fake_rel}", 1, r.family, "open_images_v7_42k",
                     lic, int(r.width), int(r.height), split, "", r.family, rule,
                     str(r.image_id), "open_images_v7_42k", "", "", 0, "", ""))
        rows.append((f"{root}/{r.real_rel}", 0, "", "open_images_v7_42k", lic,
                     int(r.width), int(r.height), split, "", "", rule,
                     str(r.image_id), "open_images_v7_42k", "", "", 0, "", ""))
    return pd.DataFrame(rows, columns=COLS)


COMMERCIAL_COLUMNS = ("path", "label", "source", "image_id", "family", "kind", "licence")


def from_commercial(root: str, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    """The commercial-API pairs (Kaggle `selennnnn/commercial-api-dataset`):
    closed-source generators (Ideogram, Seedream, Nano Banana, GPT Image,
    Muse) over Open Images V7 validation reals, one `manifest.csv` with a
    `kind` of `real` or the API name and an `image_id` pairing key.

    The only closed-source samples on hand, and NTIRE's val/test lean on
    exactly that class of generator, so they train. Split is drawn per
    `image_id`, real following its fakes, the same rule as the OV7 pairs. The
    csv's own `split` column (the plan's held-out assignment) is ignored on
    purpose: that split answered a lineage-transfer question this stream does
    not ask.
    """
    m = pd.read_csv(os.path.join(root, "manifest.csv"))
    missing = [c for c in COMMERCIAL_COLUMNS if c not in m.columns]
    if missing:
        raise SystemExit(f"{root}/manifest.csv lacks {missing}; it has "
                         f"{list(m.columns)}")
    ids = m.image_id.astype(str).unique()
    val_ids = set(ids[_val_draw(len(ids), frac, rng)])
    rows = []
    for r in m.itertuples():
        path = os.path.join(root, str(r.path))
        with Image.open(path) as im:
            w, h = im.size
        fake = int(r.label) == 1
        fam = str(r.family) if fake and isinstance(r.family, str) else ""
        split = "val" if str(r.image_id) in val_ids else "train"
        rows.append((path, int(r.label), fam, str(r.source), str(r.licence), w, h,
                     split, "", fam, f"commercial_pair_{split}", str(r.image_id),
                     "commercial_api", "", "", 0, "", ""))
    return pd.DataFrame(rows, columns=COLS)


def _resolve_image_dir(exp: str, stem: str, first_name: str) -> str:
    """Where a zip's images landed: `<stem>/<stem>/` or `<stem>/`."""
    base = os.path.basename(stem)
    for cand in (f"{exp}/{stem}/{base}", f"{exp}/{stem}"):
        if os.path.isfile(os.path.join(cand, first_name)):
            return cand
    raise SystemExit(f"cannot find {first_name!r} under {exp}/{stem} (tried "
                     f"'{stem}/{base}/' and '{stem}/'); extract the zip first")


def from_ntire_sets(exp: str, digest_workers: int = 8) -> pd.DataFrame:
    """The organisers' val / val_hard / test_public, read with their labels.

    Already class-balanced and carrying the challenge's own distortion
    metadata: `is_distorted` marks the degraded half, `distortions` and
    `distortion_scales` name what was applied, so a number can be quoted per
    half (the leaderboard's Robust vs Clean AUC) and per distortion.
    """
    out = []
    for name, labels, stem, split in NTIRE_SETS:
        lic = LICENCES[name]
        L = pd.read_csv(f"{exp}/{labels}")
        need = {"image_name", "label", "is_distorted"}
        if not need <= set(L.columns):
            raise SystemExit(f"{labels} lacks {sorted(need - set(L.columns))}")
        imgdir = _resolve_image_dir(exp, stem, str(L.image_name.iloc[0]))
        rows = []
        for r in L.itertuples():
            path = os.path.join(imgdir, str(r.image_name))
            with Image.open(path) as im:
                w, h = im.size
            fam = name if int(r.label) else ""
            rows.append((path, int(r.label), fam, name, lic, w, h, split, "",
                         fam, f"{split}_pooled", "", name, "", "",
                         int(r.is_distorted),
                         str(getattr(r, "distortions", "")),
                         str(getattr(r, "distortion_scales", ""))))
        df = pd.DataFrame(rows, columns=COLS)
        df["content_sha256"] = [d for d, _ in compute_digests(
            df.path.tolist(), workers=digest_workers)]
        out.append(df)
    return pd.concat(out, ignore_index=True)


# --- assembly --------------------------------------------------------------

def build(exp: str, repo: str, sidecar_dir: str, shards: tuple[int, ...],
          seed: int, val_frac: float, ov7_42k_root: str | None,
          require_ov7_42k: bool = True, commercial_root: str | None = None,
          require_commercial: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = [from_union(repo, exp),
              from_shard_sidecars(sidecar_dir, shards, val_frac, rng),
              from_ov7_small(repo, exp)]
    if ov7_42k_root and os.path.exists(os.path.join(ov7_42k_root, "pairs.csv")):
        blocks.append(from_ov7_42k(ov7_42k_root, val_frac, rng))
    elif require_ov7_42k:
        raise SystemExit(f"no pairs.csv under {ov7_42k_root!r}; pass "
                         "--no-ov7-42k to build without the 42k OV7 pairs")
    else:
        print("WARNING: building without the 42k OV7 pairs", flush=True)
    if commercial_root and os.path.exists(os.path.join(commercial_root, "manifest.csv")):
        blocks.append(from_commercial(commercial_root, val_frac, rng))
    elif require_commercial:
        raise SystemExit(f"no manifest.csv under {commercial_root!r}; pass "
                         "--no-commercial to build without the commercial-API pairs")
    else:
        print("WARNING: building without the commercial-API pairs", flush=True)
    blocks.append(from_ntire_sets(exp))
    df = pd.concat(blocks, ignore_index=True)
    df["rel_path"] = relative_to_root(df.path.tolist(), exp)
    df["is_distorted"] = df.is_distorted.astype(int)
    df["label"] = df.label.astype(int)
    check(df)
    return df


def check(df: pd.DataFrame) -> None:
    """Every guard that makes a number from this manifest mean something."""
    bad = df[~df.label.isin([0, 1])]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows with a label outside {{0,1}}")
    unassigned = df[~df.split.isin(TRAIN_SPLITS + FIXED_SPLITS)]
    if len(unassigned):
        raise SystemExit(f"{len(unassigned)} rows with an unknown split: "
                         f"{unassigned.split.unique().tolist()}")
    excluded = {n for n, s in SOURCES.items() if s.exclude_from_training}
    leak = df[df.split.isin(TRAIN_SPLITS) & df.source.isin(excluded)]
    if len(leak):
        raise SystemExit(f"{len(leak)} rows from a never-train source sit in "
                         f"train/val: {leak.source.unique().tolist()}")
    dropped = df[df.source.isin(DROPPED_SOURCES)]
    if len(dropped):
        raise SystemExit(f"{len(dropped)} rows from {DROPPED_SOURCES} survived")
    dup = df[df.duplicated("path", keep=False)]
    if len(dup):
        both = dup.groupby("path").split.nunique()
        raise SystemExit(f"{len(dup)} duplicated paths, "
                         f"{int((both > 1).sum())} of them in two splits")
    for sp in FIXED_SPLITS:
        g = df[df.split == sp]
        if not len(g):
            raise SystemExit(f"split {sp!r} is empty")
        for d in (0, 1):
            h = g[g.is_distorted == d]
            if set(h.label.unique()) != {0, 1}:
                raise SystemExit(f"{sp}: is_distorted={d} half lacks a class; "
                                 "robust/clean AUC would be undefined")
    # image_id pairs never straddle train/val (OV7 42k).
    paired = df[df.image_id != ""]
    if len(paired):
        straddle = paired.groupby(["variant", "image_id"]).split.nunique()
        if (straddle > 1).any():
            raise SystemExit(f"{int((straddle > 1).sum())} image_id pairs "
                             "straddle train and val")


def verify_paths(df: pd.DataFrame, sample: int | None = None,
                 seed: int = 0) -> None:
    rows = df if sample is None else df.sample(min(sample, len(df)), random_state=seed)
    miss = [p for p in rows.path if not os.path.exists(p)]
    if miss:
        raise SystemExit(f"{len(miss)}/{len(rows)} paths do not exist, e.g. "
                         f"{miss[:3]}")


def provenance(df: pd.DataFrame, seed: int, shards, out_path: str) -> dict:
    counts = (df.groupby(["source", "split", "label"]).size()
              .reset_index(name="n").to_dict("records"))
    fp = hashlib.sha256("\n".join(df.rel_path.astype(str)).encode()).hexdigest()
    short = {}
    for sp in FIXED_SPLITS:
        g = df[df.split == sp]
        ss = g[["width", "height"]].min(axis=1)
        short[sp] = {str(d): {"min": int(ss[g.is_distorted == d].min()),
                              "p5": int(np.percentile(ss[g.is_distorted == d], 5)),
                              "n_below_200": int((ss[g.is_distorted == d] < 200).sum())}
                     for d in (0, 1)}
    train = df[df.split == "train"]
    rec = {"manifest": os.path.abspath(out_path), "rows": int(len(df)),
           "seed": seed, "ntire_shards_from_sidecars": list(shards),
           "dropped_sources": list(DROPPED_SOURCES),
           "rel_path_fingerprint_sha256": fp,
           "train_real": int((train.label == 0).sum()),
           "train_fake": int((train.label == 1).sum()),
           "split_totals": df.split.value_counts().to_dict(),
           "eval_short_side_by_is_distorted": short,
           "counts": counts}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="experiments/data root")
    ap.add_argument("--repo", required=True,
                    help="directory holding manifest_union.parquet and "
                         "manifest_ov7.parquet (usually <repo>/data)")
    ap.add_argument("--sidecar-dir", default=None,
                    help="where normalize_ntire_shards.py wrote its parquets "
                         "(default: --exp)")
    ap.add_argument("--shards", default="3,4,5",
                    help="shard numbers with sidecars; '' for none")
    ap.add_argument("--ov7-42k-root", default=None,
                    help="raw 42k OV7 build (default <exp>/raw_ov7_42k)")
    ap.add_argument("--no-ov7-42k", action="store_true",
                    help="build without the 42k OV7 pairs (recorded)")
    ap.add_argument("--commercial-root", default=None,
                    help="commercial-API pairs (default <exp>/raw_commercial_api)")
    ap.add_argument("--no-commercial", action="store_true",
                    help="build without the commercial-API pairs (recorded)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits-json", default="docs/ntire/splits.json")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--verify-sample", type=int, default=0,
                    help="check this many random paths exist instead of all")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out")
    a = ap.parse_args(argv)

    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"{a.out} exists. A frozen manifest is what every bank "
                         "and score indexes against; pass --force only if "
                         "nothing has been built from it.")
    shards = tuple(int(s) for s in a.shards.split(",") if s.strip())
    df = build(a.exp, a.repo, a.sidecar_dir or a.exp, shards, a.seed, a.val_frac,
               a.ov7_42k_root or f"{a.exp}/raw_ov7_42k",
               require_ov7_42k=not a.no_ov7_42k,
               commercial_root=a.commercial_root or f"{a.exp}/raw_commercial_api",
               require_commercial=not a.no_commercial)
    verify_paths(df, None if a.verify_sample == 0 else a.verify_sample)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)
    rec = provenance(df, a.seed, shards, a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.splits_json)), exist_ok=True)
    with open(a.splits_json, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {a.out}: {len(df):,} rows")
    print("  splits:", rec["split_totals"])
    print(f"  train real:fake = {rec['train_real']:,}:{rec['train_fake']:,}")
    print("  eval short sides:", json.dumps(rec["eval_short_side_by_is_distorted"]))
    print(f"  provenance -> {a.splits_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
