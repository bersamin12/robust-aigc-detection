"""audit -> normalise -> dedupe -> split -> manifest (spec §4).

This is the seam where every Plan 1 module meets: it is run once, by a human,
against real acquired data, to produce the manifest every later plan indexes
against positionally. The manifest is frozen the moment it is written —
re-running the split after feature banks exist would silently misalign
labels against features (spec §4.6, project constraints).

Usage:
    python scripts/build_dataset.py --raw data/raw --out data/normalized \
        --demo-dir data/raw/demo --manifest data/manifest.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from aigcdet.data.audit import audit_flags, audit_table
from aigcdet.data.dedupe import build_hash_index, find_leaks
from aigcdet.data.manifest import MANIFEST_COLUMNS, write_manifest
from aigcdet.data.normalize import normalize_many
from aigcdet.data.splits import assign_splits, choose_heldout_generators, split_report

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
    actually happens, so a missing file or a missing entry fails loudly here
    rather than writing blank provenance someone has to reconstruct later.
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


def build_dataset(
    raw: str,
    out: str,
    demo_dir: str,
    manifest: str,
    workers: int = 16,
    docs_dir: str = "docs",
) -> pd.DataFrame:
    """Run the full audit -> normalise -> dedupe -> split -> manifest
    pipeline and write the manifest. Returns the DataFrame it wrote."""
    licences = _load_licences(raw)

    # Directory convention from acquire_data.py: raw/<source>/<real|fake>/...
    # and for WildFake, raw/wildfake/<generator>/...
    rows = []
    for p in _scan(raw):
        rel = os.path.relpath(p, raw).split(os.sep)
        source = rel[0]
        bucket = rel[1] if len(rel) > 1 else ""
        label = 0 if bucket == "real" else 1
        generator = "" if label == 0 else (bucket if source == "wildfake" else source)
        rows.append({"src": p, "label": label, "generator": generator, "source": source})
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError(f"no images found under {raw}")
    print(f"scanned {len(raw_df)} raw images")

    missing_licences = sorted(set(raw_df["source"]) - set(licences))
    if missing_licences:
        raise ValueError(
            f"LICENCES.json has no entry for source(s) {missing_licences}. "
            "Record every dataset's licence at acquisition time (spec §4.5) "
            "before building the manifest.")

    # 1. Audit BEFORE normalising: the table is the figure for the README.
    at = audit_table(raw_df["src"].tolist(), raw_df["label"].tolist(), raw_df["source"].tolist())
    flags = audit_flags(at)
    os.makedirs(docs_dir, exist_ok=True)
    with open(os.path.join(docs_dir, "data_audit.md"), "w") as f:
        f.write("# Pre-normalisation data audit\n\n")
        f.write(_to_markdown_table(at) + "\n\n## Flags\n\n")
        f.write("\n".join(f"- {x}" for x in flags) if flags else "- none")
    print(f"audit flags: {flags}")

    # 2. Normalise.
    pairs, dsts = [], []
    for i, r in raw_df.iterrows():
        dst = os.path.join(out, r["source"], r["generator"] or "real", f"{i:07d}.png")
        pairs.append((r["src"], dst))
        dsts.append(dst)
    sizes = normalize_many(pairs, workers=workers)
    raw_df["path"] = dsts
    raw_df["width"] = [s[0] for s in sizes]
    raw_df["height"] = [s[1] for s in sizes]

    # 3. Leakage guard against the demo set (spec §4.1). find_leaks removes
    # nothing itself; we drop only the training-side matches it reports,
    # never a demo path — the demo set is never added to raw_df at all.
    demo = build_hash_index(_scan(demo_dir))
    cand = build_hash_index(raw_df["path"].tolist())
    leaks = find_leaks(cand, demo, max_distance=4)
    print(f"dropping {len(leaks)} images that near-duplicate the demo set")
    raw_df = raw_df[~raw_df["path"].isin(leaks)].reset_index(drop=True)

    # Spec §4.1(2): no COCO-derived authentic source may be trained on.
    raw_df = raw_df[
        ~((raw_df["label"] == 0) & (raw_df["source"].str.contains("coco")))
    ].reset_index(drop=True)

    raw_df["licence"] = raw_df["source"].map(licences)
    raw_df["split"] = ""

    df = raw_df[MANIFEST_COLUMNS]
    held = choose_heldout_generators(df, n=2)
    print(f"held-out generators: {held}")
    df = assign_splits(df, heldout_generators=held)
    write_manifest(df, manifest)
    print(split_report(df).to_string(index=False))
    with open(os.path.join(docs_dir, "splits.json"), "w") as f:
        json.dump(
            {"heldout_generators": held, "seed": 20260827, "leaked_dropped": len(leaks)},
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
    a = ap.parse_args()
    build_dataset(a.raw, a.out, a.demo_dir, a.manifest, workers=a.workers)


if __name__ == "__main__":
    main()
