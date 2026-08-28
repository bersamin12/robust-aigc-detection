"""Freeze the organisers' demo benchmark as a manifest (spec §4.1(2), §7.1).

    python scripts/build_benchmark_manifest.py --demo-dir data/demo \
        --manifest data/benchmark_manifest.parquet

Why this is a SEPARATE script from `build_dataset.py`, and not a flag on it
----------------------------------------------------------------------------

`build_dataset.py` already takes `--demo-dir`, but only as a LEAKAGE
REFERENCE: it pHashes the demo images and drops training-pool rows that
near-duplicate them. The demo images themselves never enter its manifest,
and they must not — spec §4.1(2) forbids training on either half, and
`sources.is_excluded_from_training` drops them by SOURCE, before any label is
consulted. Adding a "…and also emit them" branch to that script would put the
one set that may never be trained on into the same frame as the training
pool, one `assign_splits` call away from being relabelled `train`.

The two also have different LIFECYCLES, which is the operational reason they
are different commands:

* The benchmark is FIXED. It is 4,998 COCO val2017 photographs and 8,843
  DALL·E 3 Advanced images, the organisers' own counts, and it does not grow.
  So it can be frozen the moment `acquire_data.py --dataset
  wildfake_benchmark` finishes — days before the training pool has finished
  downloading, and without waiting on normalisation, the audit or the
  pHash leakage pass that `build_dataset` spends most of its runtime on.
* The training manifest is REBUILT whenever the pool changes, and its split
  assignment is drawn fresh each time. The benchmark manifest must survive
  those rebuilds untouched: feature banks index it positionally, so a
  benchmark that was re-frozen alongside a training rebuild would silently
  misalign every reported benchmark number against its features.

Every row this writes carries `split = "benchmark"` — the fourth member of
`manifest.SPLITS`, which until now nothing produced. `scripts/extract_eval_bank.py`
selects on exactly that value, and `scripts/run_ablation.py` scores against
what it extracts.

Nothing here is authored by hand
--------------------------------

`aigcdet.data.wildfake.BENCHMARK_HALVES` is the source of truth for which
directories hold which half and how many images each must contain, and
`aigcdet.data.sources.classify` is the source of truth for what
`<source>/<bucket>` means. This script derives label, generator, licence,
directory name and expected count from those two modules and states none of
them itself, so a registry change cannot leave a stale literal here.

The count check is FATAL, per half and in total. It is not decoration: it is
what caught a benchmark marker that matched 0 of `real_coco.csv`'s 163,846
rows. A benchmark of the wrong size does not fail loudly downstream; it
silently changes what every reported number means.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
from PIL import Image

from aigcdet.data.manifest import MANIFEST_COLUMNS, validate_manifest, write_manifest
from aigcdet.data.sources import LICENCES, classify, is_excluded_from_training, raw_subdir
from aigcdet.data.wildfake import BENCHMARK_HALVES

#: Same set `build_dataset._scan` accepts. The materialised benchmark is
#: `.jpg` throughout (`wildfake.dest_filename` keeps the upstream extension),
#: but a re-encoded copy must scan identically or the two scripts would
#: disagree about which files exist.
IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

#: The split value every row gets. Named here so the one place that writes it
#: is greppable; `manifest.SPLITS` is what declares it legal.
BENCHMARK_SPLIT = "benchmark"


def half_bucket(half) -> str:
    """The bucket directory this half's images sit in, under its source.

    The SAME expression `wildfake.benchmark_dest` uses to place them there, so
    the writer and this reader cannot drift: `raw_subdir` is the declared
    inverse of `classify`, and `classify` is what turns the directory back
    into `(label, generator)` below.
    """
    return (raw_subdir(half.source, 0) if not half.generator
            else raw_subdir(half.source, 1, half.generator))


def half_dir(demo_dir: str, half) -> str:
    """`<demo_dir>/<source>/<bucket>` — where `benchmark_dest` writes."""
    return os.path.join(os.path.abspath(str(demo_dir)), half.source,
                        half_bucket(half))


def scan_half(demo_dir: str, half) -> list[str]:
    """Every image of one benchmark half, in a stable order.

    Sorted because the manifest's row order IS its identity: the fingerprint
    every feature-bank shard records is taken over `rel_path` in order, so a
    filesystem-dependent enumeration order would make two machines freeze two
    manifests that disagree while describing the same files.
    """
    d = half_dir(demo_dir, half)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} does not exist, so the {half.source!r} half of the demo "
            f"benchmark ({half.expected} images) cannot be manifested. "
            "Materialise it with `python scripts/acquire_data.py --dataset "
            "wildfake_benchmark --benchmark-dir <dir>` and point --demo-dir "
            "at <dir>.")
    out: list[str] = []
    for ext in IMG_EXT:
        out += glob.glob(os.path.join(d, "**", ext), recursive=True)
    return sorted(os.path.abspath(p) for p in out)


def check_count(half, n: int) -> None:
    """Fatal unless `n` is exactly this half's organiser-stated count.

    Deliberately per-half rather than only on the total: two halves that are
    wrong in opposite directions sum to the right number, and 4,998 + 8,843
    is a figure the eye reads as correct.
    """
    if n != half.expected:
        raise ValueError(
            f"{half.source}: found {n} images under "
            f"{os.path.join(half.source, half_bucket(half))}, but the "
            f"organisers' benchmark half is {half.expected} images "
            f"({n - half.expected:+d}). A benchmark of a different size does "
            "not fail downstream — it silently changes what every reported "
            "benchmark number means. Re-run scripts/acquire_data.py --dataset "
            "wildfake_benchmark rather than adjusting this number.")


def build_benchmark_manifest(
    demo_dir: str,
    manifest: str,
    halves=None,
    workers: int = 8,
    digests: str | None = "bytes",
    force: bool = False,
) -> pd.DataFrame:
    """Scan `demo_dir`, verify the counts, and freeze the manifest.

    Refuses to overwrite an existing `manifest` without `force`, for the same
    reason `build_dataset` does: a manifest is frozen the instant it is
    written and feature banks index it positionally, so re-freezing one that
    banks already exist against misaligns labels and features without ever
    raising.

    `root` is pinned to `demo_dir` rather than left to `derive_root`. The
    derived root is the deepest directory containing every path, which is
    `demo_dir` only while BOTH halves are present — with one half it would be
    that half's own directory, and the resulting `rel_path` would not survive
    the other half being added. It is also the directory a teammate attaches
    the published Kaggle Dataset at, which is the whole point of `rel_path`.
    """
    halves = tuple(halves) if halves is not None else BENCHMARK_HALVES
    if os.path.exists(manifest) and not force:
        raise FileExistsError(
            f"{manifest} already exists and would be overwritten. The demo "
            "benchmark is frozen once: any feature bank built against this "
            "manifest indexes its rows positionally, so re-freezing it would "
            "misalign labels against features without raising an error. Pass "
            "force=True (--force on the CLI) only if you intend to discard it "
            "and every bank extracted against it.")

    rows = []
    for half in halves:
        # The guarantee that actually holds end to end, asserted here for the
        # same reason `benchmark_dest` asserts it: whatever a row claims to
        # be, `build_dataset` drops these rows by SOURCE before it looks at a
        # label. A half filed under a trainable source would be a §4.1(2)
        # violation dressed as a manifest.
        if not is_excluded_from_training(half.source):
            raise ValueError(
                f"source {half.source!r} is not marked exclude_from_training "
                "in aigcdet.data.sources, so build_dataset would train on "
                "these rows if they ever reached --raw. Refusing to manifest "
                "the demo benchmark under it.")
        paths = scan_half(demo_dir, half)
        check_count(half, len(paths))
        # Both derived from the registry, never stated here: `half_bucket` is
        # how the images were placed and `classify` is the declared inverse,
        # so COCO val2017 reads back label 0 and DALLE3 reads back
        # (1, "dalle3") without this script holding an opinion.
        label, generator = classify(half.source, half_bucket(half))
        licence = LICENCES[half.source]
        for p in paths:
            with Image.open(p) as im:
                width, height = im.size
            rows.append({
                "path": p,
                "label": label,
                "generator": generator,
                "source": half.source,
                "licence": licence,
                "width": int(width),
                "height": int(height),
                "split": BENCHMARK_SPLIT,
            })
        print(f"{half.source}: {len(paths)} images (label {label}, "
              f"generator {generator or '-'})")

    expected_total = sum(h.expected for h in halves)
    if len(rows) != expected_total:
        raise ValueError(
            f"the demo benchmark is {expected_total} images "
            f"({', '.join(f'{h.source} {h.expected}' for h in halves)}), but "
            f"{len(rows)} rows were collected. Refusing to freeze a benchmark "
            "of the wrong size.")

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    validate_manifest(df)
    write_manifest(df, manifest, root=os.path.abspath(str(demo_dir)),
                   digests=digests, workers=workers)
    print(f"froze {len(df)} benchmark rows to {manifest} "
          f"(split={BENCHMARK_SPLIT!r}, root={os.path.abspath(str(demo_dir))})")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Freeze the organisers' demo benchmark as a manifest "
                    "whose every row carries split='benchmark'. Never train "
                    "on these rows (spec §4.1(2)).")
    ap.add_argument("--demo-dir", required=True,
                    help="the directory acquire_data.py --benchmark-dir "
                         "wrote; the same one build_dataset.py takes as "
                         "--demo-dir. Must be OUTSIDE --raw.")
    ap.add_argument("--manifest", required=True,
                    help="output parquet path")
    ap.add_argument("--workers", type=int, default=8,
                    help="threads used to digest the images")
    ap.add_argument("--digests", default="bytes",
                    choices=("bytes", "pixels", "none"),
                    help="identity digests to stamp at freeze time")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest at --manifest")
    return ap


def main(argv=None) -> pd.DataFrame:
    a = build_parser().parse_args(argv)
    return build_benchmark_manifest(
        a.demo_dir, a.manifest, workers=a.workers,
        digests=None if a.digests == "none" else a.digests, force=a.force)


if __name__ == "__main__":
    main()
