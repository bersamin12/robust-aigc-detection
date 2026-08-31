#!/usr/bin/env python3
"""A bank that carries a manifest's TRAINING METADATA and no features.

`train_unfreeze.py` and `train_dual.py` need a `FeatureBank`, but they never
read `feats.npy`: the tower runs live, so every embedding in a cached bank is
recomputed from pixels anyway. Building one the normal way over 350k rows would
spend ~5 GPU-hours computing embeddings that are then discarded.

What the trainers actually read is `meta` (rel_path, row_id, label, split,
generator), `presence`/`severity`, and `config["seed"]`. Presence and severity
are pure functions of the sampled recipe -- `recipe_for_view(view, row_id,
seed).labels()` -- and a recipe is a pure function of `(seed, row_id,
view_idx)`. So none of it needs an image decoded or a GPU touched, and this
script writes the whole thing on CPU in minutes.

**The features are NaN, deliberately, and the config says so.** Zeros would be
the obvious filler and the wrong one: this project has already lost five hours
to a bank of 131,116 NaN rows that passed every check because the only
post-condition was the row count, and a bank of silent ZEROS is that failure
with the alarm removed. NaN propagates, `check_invariants` refuses it by name,
and `features_absent: true` in config.json states the intent for anything that
looks. `dim` is 1 rather than the tower's width so the placeholder costs
megabytes instead of gigabytes -- nothing reads it, and `train_finetune` takes
its head width from the backbone SPEC, never from the bank.

This bank is NOT usable for scoring, rung training, or fusion. It is an input
to live-tower fine-tuning and nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from aigcdet.augment.recipes import FAMILIES
from aigcdet.features.bank import BankWriter
from aigcdet.features.extract import recipe_for_view

N_VIEWS = 11
PROXY_DIM = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True,
                    help="where this box mounts the corpus; recorded so "
                         "rel_path is right for the machine that trains")
    ap.add_argument("--splits", default="train,val",
                    help="manifest splits to include. The trainer selects "
                         "split=='train' itself; val is carried so the same "
                         "bank can back a val pass without a second build")
    ap.add_argument("--n-views", type=int, default=N_VIEWS)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--backbone", default="metadata-only")
    ap.add_argument("--exclude-families", default="")
    ap.add_argument("--path-map", action="append", default=[],
                    metavar="OLD=NEW",
                    help="prefix rewrite applied to the absolute `path` "
                         "column, repeatable. The plan manifest spans FIVE "
                         "roots (normalized_union, ov7_upload, raw_ov7_42k, "
                         "raw_commercial_api, and experiments/data itself), so "
                         "no single --root can serve it and rel_path is not "
                         "portable across boxes. Rewriting `path` and dropping "
                         "rel_path makes the trainer read absolute paths, "
                         "which is honest: a live-tower training bank is tied "
                         "to the box whose corpus it names.")
    ap.add_argument("--verify-sample", type=int, default=500,
                    help="paths sampled for existence before anything is "
                         "written; 0 disables (do not)")
    a = ap.parse_args()

    absolute = False
    df = pd.read_parquet(a.manifest)
    keep = [s for s in a.splits.split(",") if s]
    df = df[df["split"].isin(keep)].reset_index(drop=True)
    if not len(df):
        raise SystemExit(f"no rows with split in {keep}")
    for col in ("path", "label", "split", "generator"):
        if col not in df.columns:
            raise SystemExit(f"manifest has no {col!r} column; the trainer "
                             "reads it directly")
    excl = tuple(f for f in a.exclude_families.split(",") if f)

    if a.path_map:
        pairs = []
        for spec in a.path_map:
            old, new = spec.split("=", 1)
            pairs.append((old, new))
        # pandas .str, not np.char: the column is object dtype and
        # np.char.replace has no loop for it (it raises _UFuncNoLoopError).
        paths = df["path"].astype(str)
        for old, new in pairs:
            paths = paths.str.replace(old, new, regex=False)
        df = df.assign(path=paths.to_numpy())
        # rel_path described the ORIGINAL root and is now a lie. Dropping it
        # makes `LiveViewSampler` fall through to the absolute `path`, which is
        # the branch that is actually correct here.
        df = df.drop(columns=[c for c in ("rel_path",) if c in df.columns])
        # `write_image` RE-DERIVES rel_path from manifest_root whenever the row
        # does not carry one, so dropping the column is not enough: the writer
        # would helpfully reconstruct a rel_path against the wrong root and the
        # sampler would prefer it over the absolute path we just fixed.
        absolute = True
        print(f"applied {len(pairs)} path rewrite(s); rel_path dropped")

    if a.verify_sample:
        n = min(int(a.verify_sample), len(df))
        samp = df.sample(n, random_state=0)
        col = "path" if "rel_path" not in df.columns else "rel_path"
        def _full(v):
            return v if col == "path" else os.path.join(a.root, v)
        miss = [v for v in samp[col].astype(str) if not os.path.exists(_full(v))]
        if miss:
            raise SystemExit(
                f"REFUSING: {len(miss)}/{n} sampled rows do not exist, e.g.\n  "
                + "\n  ".join(miss[:3])
                + "\nA bank whose paths do not resolve fails hours into "
                  "training, at the first decode, with the towers already "
                  "loaded.")
        print(f"path check OK: {n}/{n} sampled rows exist")

    sha = hashlib.sha256(open(a.manifest, "rb").read()).hexdigest()
    writer = BankWriter(
        a.out, len(df), a.n_views, 1, a.backbone, a.seed,
        manifest_sha256=sha, manifest_root=None if absolute else a.root,
        extra_config={
            "features_absent": True,
            "purpose": "live-tower fine-tuning metadata only; feats.npy is "
                       "NaN by construction and must never be read",
            "source_manifest": os.path.abspath(a.manifest),
            "splits_included": keep,
            "paths_are_absolute": absolute,
        })

    nan_feats = np.full((a.n_views, 1), np.nan, dtype=np.float32)
    nan_prox = np.full((a.n_views, PROXY_DIM), np.nan, dtype=np.float32)
    presence = np.zeros((a.n_views, len(FAMILIES)), dtype=np.float32)
    severity = np.zeros((a.n_views, len(FAMILIES)), dtype=np.float32)

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df),
                                 desc="metadata")):
        # row_id is the position in THIS bank's manifest slice, and it is the
        # RNG key every view's pixels are rebuilt from at train time. It must
        # match what `LiveViewSampler` will pass, which is `bank.row_ids[i]`.
        recipes = []
        for v in range(a.n_views):
            r = recipe_for_view(v, i, a.seed, excl)
            lab = r.labels()
            presence[v] = lab["presence"]
            severity[v] = lab["severity"]
            recipes.append(r.to_json())
        writer.write_image(
            i, {c: getattr(row, c) for c in df.columns}, nan_feats,
            presence.copy(), severity.copy(), nan_prox, recipes, row_id=i)
    writer.close()

    if absolute:
        # `BankWriter.write_image` fills a missing rel_path from manifest_root,
        # and with manifest_root=None that yields the ABSOLUTE path under the
        # rel_path name. The sampler would then take its rel_path branch and
        # `os.path.join(root, "/abs/path")` -- which happens to return the
        # absolute path, because join discards everything before an absolute
        # component. It works, and it works for a reason nobody reading
        # `_path()` would see. Dropping the column makes the sampler take its
        # `_absolute` branch, so the code path matches the intent.
        import pandas as _pd
        mp = os.path.join(a.out, "meta.parquet")
        md = _pd.read_parquet(mp).drop(columns=["rel_path"])
        md.to_parquet(mp, index=False)
        print("dropped rel_path from meta.parquet (absolute-path bank)")

    print(f"wrote {a.out}: {len(df):,} rows x {a.n_views} views")
    print("  splits:", df["split"].value_counts().to_dict())
    print("  labels:", df["label"].value_counts().to_dict())
    print("  feats.npy is NaN by construction -- this bank is for live-tower "
          "training only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
