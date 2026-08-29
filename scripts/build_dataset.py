"""audit -> normalise -> dedupe -> split -> manifest (spec §4).

This is the seam where every Plan 1 module meets: it is run once, by a human,
against real acquired data, to produce the manifest every later plan indexes
against positionally. The manifest is frozen the moment it is written —
re-running the split after feature banks exist would silently misalign
labels against features (spec §4.6, project constraints). `build_dataset`
therefore refuses to overwrite an existing manifest unless `force=True`.

Usage:
    python scripts/build_dataset.py --raw data/raw --out data/normalized \
        --demo-dir data/demo --manifest data/manifest.parquet

`--demo-dir` must sit OUTSIDE `--raw`: it is the organisers' benchmark, and
anything under `--raw` is scanned as training data.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from aigcdet.data.audit import audit_flags, audit_table
from aigcdet.data.dedupe import build_hash_index, find_leaks
from aigcdet.data.manifest import MANIFEST_COLUMNS, validate_manifest, write_manifest
from aigcdet.data.normalize import normalize_many
from aigcdet.data.sources import (
    classify, is_excluded_from_training, is_restricted_bucket, restriction_reason,
)
from aigcdet.data.splits import (
    DEFAULT_SEED,
    assign_splits,
    choose_heldout_generators,
    split_report,
)

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def _scan(root: str) -> list[str]:
    out = []
    for ext in IMG_EXT:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(out)


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without the
    `tabulate` package, which is not part of this project's dependency set
    (pandas.DataFrame.to_markdown requires it and would raise ImportError)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def _load_licences(raw_dir: str) -> dict[str, str]:
    """Read <raw_dir>/LICENCES.json, written by scripts/acquire_data.py at
    acquisition time (JSON-lines, one `{dataset: licence_string}` object per
    line; the licence string embeds the source URL, e.g. "CC BY 4.0 ... -
    https://...").

    Spec §4.5 requires every dataset's licence and source URL to reach the
    manifest. A Task 8 review flagged that nothing verified the hand-off
    actually happens, so a missing file fails loudly here (FileNotFoundError)
    rather than the caller silently writing blank provenance someone has to
    reconstruct later. Missing or blank *entries* for a specific source being
    ingested are checked separately, by the caller, once it knows which
    sources are actually present.
    """
    path = os.path.join(raw_dir, "LICENCES.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/acquire_data.py first so every "
            "dataset's licence and source URL are recorded at acquisition "
            "time (spec §4.5) — this script refuses to fabricate them.")
    licences: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                licences.update(json.loads(line))
    return licences


def _cap_per_generator(
    df: pd.DataFrame, cap: int, seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep at most `cap` rows per generator family, drawn from `seed`.

    Returns the thinned frame in its original row order and the per-family
    dropped counts. Authentic rows (generator "") are never touched.

    Row order is preserved rather than regrouped because `_scan` sorts, and
    that sorted order is what makes a rebuild with the same seed produce the
    same manifest -- which matters more here than usual, since the manifest is
    indexed positionally by every feature bank built against it.
    """
    keep = np.ones(len(df), dtype=bool)
    dropped: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    generated = df["generator"].to_numpy()
    for g in sorted({x for x in generated if x}):
        idx = np.flatnonzero(generated == g)
        if len(idx) <= cap:
            continue
        # `choice` on the POSITIONS, then set the complement False: sampling
        # which to keep rather than which to drop keeps the draw's size fixed
        # at `cap` however large the family is.
        chosen = rng.choice(idx, size=cap, replace=False)
        keep[idx] = False
        keep[chosen] = True
        dropped[g] = int(len(idx) - cap)
    return df[keep].reset_index(drop=True), dropped


def build_dataset(
    raw: str,
    out: str,
    demo_dir: str,
    manifest: str,
    workers: int = 16,
    docs_dir: str = "docs",
    seed: int = DEFAULT_SEED,
    force: bool = False,
    heldout_generators: list[str] | None = None,
    max_per_generator: int = 0,
) -> pd.DataFrame:
    """Run the full audit -> normalise -> dedupe -> split -> manifest
    pipeline and write the manifest. Returns the DataFrame it wrote.

    Refuses to run at all if `manifest` already exists and `force` is not
    set: the manifest is frozen the instant it is written, because Plan 2's
    feature banks index against it positionally, and a silent re-split
    would misalign labels against features without ever raising an error.

    `heldout_generators` pins the held-out families instead of drawing them
    with `choose_heldout_generators`; the automatic choice is restricted to
    genuine generator families (spec §4.6), so a human who wants a specific
    pair says so here rather than reseeding until the draw obliges.

    `max_per_generator` keeps at most that many images per generator family,
    drawn deterministically from `seed`. It exists because the corpus is
    lopsided by construction: barring WildFake's authentic bucket (see
    `aigcdet.data.sources`) leaves every real image coming from SID_Set while
    the generated side keeps every WildFake family. Thinning the families
    rebalances the classes WITHOUT deleting a family -- `heldout_generator`
    and the LOTO rung both need every family to survive -- and without
    touching the raw tree, so the decision stays a rebuild flag rather than an
    irreversible `rm`. Authentic images are never capped: they are the scarce
    side.
    """
    if os.path.exists(manifest) and not force:
        raise FileExistsError(
            f"{manifest} already exists and would be overwritten. Any "
            "feature bank built against it indexes rows positionally, so "
            "silently rebuilding the manifest would misalign labels against "
            "features without raising an error. Pass force=True "
            "(--force on the CLI) only if you intend to discard it and "
            "everything built against it.")

    licences = _load_licences(raw)

    # `raw/<source>/<bucket>/...`, mapped to label/generator by the registry
    # in aigcdet.data.sources -- the same module acquire_data.py writes the
    # tree with. Inferring the label from the directory name here is what
    # labelled all ~5,000 COCO val2017 photographs AI-generated: their bucket
    # is `val2017`, not `real`. An unregistered source or bucket now raises.
    rows = []
    restricted: dict[str, int] = {}
    for p in _scan(raw):
        rel = os.path.relpath(p, raw).split(os.sep)
        source = rel[0]
        bucket = rel[1] if len(rel) > 1 else ""
        # classify FIRST, so an unregistered source or bucket still raises
        # even when the bucket would then have been barred: a tree we cannot
        # read is a different problem from one we may not use.
        label, generator = classify(source, bucket)
        if is_restricted_bucket(source, bucket):
            # Barred here, before normalisation, not filtered out of the
            # manifest afterwards. Normalising images we may not use costs an
            # hour and ~17 GB, and the licence is about the copy on our disk,
            # not about the manifest row that points at it.
            key = f"{source}/{bucket}"
            restricted[key] = restricted.get(key, 0) + 1
            continue
        rows.append({"src": p, "label": label, "generator": generator, "source": source})
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError(f"no images found under {raw}")
    for key, n in sorted(restricted.items()):
        print(f"barred {n} images in {key} by licence: "
              f"{restriction_reason(key.split('/')[0])}")
    capped: dict[str, int] = {}
    if max_per_generator:
        raw_df, capped = _cap_per_generator(raw_df, max_per_generator, seed)
        for g, n in sorted(capped.items()):
            print(f"capped {g}: dropped {n} to {max_per_generator}")
    print(f"scanned {len(raw_df)} raw images")

    # Reject a missing key, JSON `null`, `""`, AND a whitespace-only value
    # (e.g. `"   "`) -- all four flow through to a blank-in-substance
    # `licence` column otherwise. Falsiness is tested on the RAW value
    # first, before any str() coercion: str(None) == "None" is truthy, so
    # coercing first (as an earlier round of this fix did) would silently
    # accept a JSON-null licence. Only a truthy value is then checked for
    # being whitespace-only once stringified.
    missing_licences = sorted(
        s for s in set(raw_df["source"])
        if not licences.get(s) or not str(licences.get(s)).strip()
    )
    if missing_licences:
        raise ValueError(
            f"LICENCES.json has no usable (non-blank) entry for source(s) "
            f"{missing_licences}. Record every dataset's licence at "
            "acquisition time (spec §4.5) before building the manifest.")

    os.makedirs(docs_dir, exist_ok=True)

    # 1. Normalise. Unreadable files are skipped and recorded rather than
    # ending the run: at ~100k images streamed from third-party hosts, at
    # least one truncated or zero-byte file is near-certain.
    pairs, dsts = [], []
    for i, r in raw_df.iterrows():
        # Absolute: manifest.py documents `path` as absolute, and Plans 2
        # and 3 open row["path"] directly from notebooks and from Kaggle,
        # where the working directory is not this one.
        dst = os.path.abspath(
            os.path.join(out, r["source"], r["generator"] or "real", f"{i:07d}.png"))
        pairs.append((r["src"], dst))
        dsts.append(dst)
    sizes, failures = normalize_many(pairs, workers=workers)
    if failures:
        skipped_path = os.path.join(docs_dir, "normalize_skipped.json")
        with open(skipped_path, "w") as f:
            json.dump([{"src": src, "reason": reason} for src, reason in failures],
                      f, indent=2)
        print(f"skipped {len(failures)} of {len(pairs)} images that could not "
              f"be read; the full list is in {skipped_path}")
    ok = [i for i, size in enumerate(sizes) if size is not None]
    if not ok:
        raise ValueError(
            f"every one of the {len(pairs)} images under {raw} failed to "
            f"normalise; see {docs_dir}/normalize_skipped.json")
    raw_df = raw_df.iloc[ok].reset_index(drop=True)
    raw_df["path"] = [dsts[i] for i in ok]
    raw_df["width"] = [sizes[i][0] for i in ok]
    raw_df["height"] = [sizes[i][1] for i in ok]

    # 2. Audit the RAW files -- the table profiles the two classes before
    # normalisation removes their container differences, and it is the
    # figure for the README. It runs after step 1 only so that it reads the
    # raw files normalisation has just proved decodable: audit_table opens
    # every image, so an undecodable one would otherwise abort the ~20-minute
    # audit pass before the skip list above could ever be produced.
    at = audit_table(raw_df["src"].tolist(), raw_df["label"].tolist(),
                     raw_df["source"].tolist())
    flags = audit_flags(at)
    with open(os.path.join(docs_dir, "data_audit.md"), "w") as f:
        f.write("# Pre-normalisation data audit\n\n")
        f.write(_to_markdown_table(at) + "\n\n## Flags\n\n")
        f.write("\n".join(f"- {x}" for x in flags) if flags else "- none")
    print(f"audit flags: {flags}")

    # 3. Leakage guard against the demo set (spec §4.1). find_leaks removes
    # nothing itself; we drop only the training-side matches it reports,
    # never a demo path — the demo set is never added to raw_df at all.
    demo = build_hash_index(_scan(demo_dir))
    cand = build_hash_index(raw_df["path"].tolist())
    leaks = find_leaks(cand, demo, max_distance=4)
    print(f"dropping {len(leaks)} images that near-duplicate the demo set")
    raw_df = raw_df[~raw_df["path"].isin(leaks)].reset_index(drop=True)

    # Spec §4.1(2): the organisers' demo benchmark may never be trained on.
    # Keyed on the SOURCE alone, never on the label: gating this on
    # `label == 0` meant a source mislabelled upstream slipped straight
    # through the exclusion it exists to enforce.
    excluded = raw_df["source"].map(is_excluded_from_training)
    if excluded.any():
        by_source = raw_df.loc[excluded, "source"].value_counts().to_dict()
        print(f"excluding {int(excluded.sum())} images from demo-benchmark "
              f"sources (spec §4.1): {by_source}")
    raw_df = raw_df[~excluded].reset_index(drop=True)

    raw_df["licence"] = raw_df["source"].map(licences)
    raw_df["split"] = ""

    df = raw_df[MANIFEST_COLUMNS]
    if heldout_generators:
        held = sorted(heldout_generators)
        print(f"held-out generators (pinned): {held}")
    else:
        held = choose_heldout_generators(df, n=2, seed=seed)
        print(f"held-out generators: {held}")
    df = assign_splits(df, heldout_generators=held, seed=seed)
    validate_manifest(df)
    write_manifest(df, manifest)
    print(split_report(df).to_string(index=False))
    with open(os.path.join(docs_dir, "splits.json"), "w") as f:
        json.dump(
            {"heldout_generators": held, "seed": seed, "leaked_dropped": len(leaks),
             "normalize_skipped": len(failures),
             # The licence audit trail. The counts say what was dropped; the
             # reasons say why, which is the half a reader six months from now
             # cannot reconstruct.
             "restricted_dropped": restricted,
             "restriction_reasons": {
                 src: restriction_reason(src)
                 for src in sorted({k.split("/")[0] for k in restricted})},
             "max_per_generator": max_per_generator,
             "capped_dropped": capped},
            f, indent=2,
        )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true",
                     help="overwrite an existing manifest at --manifest")
    ap.add_argument("--max-per-generator", type=int, default=0,
                    help="keep at most N images per generator family, drawn "
                         "deterministically from --seed; 0 (default) keeps "
                         "every image. Authentic images are never capped.")
    ap.add_argument("--heldout-generators", default="",
                    help="comma-separated generator families to hold out, "
                         "pinning the choice instead of drawing it")
    a = ap.parse_args()
    build_dataset(a.raw, a.out, a.demo_dir, a.manifest, workers=a.workers,
                  seed=a.seed, force=a.force,
                  max_per_generator=a.max_per_generator,
                  heldout_generators=[g for g in a.heldout_generators.split(",") if g])


if __name__ == "__main__":
    main()
