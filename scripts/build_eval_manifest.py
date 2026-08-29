"""Join the training and benchmark manifests into the one the eval bank needs.

`extract_eval_bank.py --tier ablation` is defined over three splits --
`val_internal`, `heldout_generator` and `benchmark` -- but the first two live
in the training manifest and the third in the benchmark manifest, and
`--manifest` takes one file. Nothing joined them, so the ablation tier could
not be extracted at all.

    python scripts/build_eval_manifest.py \
        --manifest data/normalized/manifest.parquet \
        --benchmark-manifest data/demo/benchmark_manifest.parquet \
        --out data/eval_manifest.parquet

This is not `pd.concat`. Two things break silently first, and a third loudly:

**Two dataset roots.** The training tree is `data/normalized`, the benchmark
tree is `data/demo`. `dataset_root` refuses a frame implying both -- correctly,
since a manifest describes one tree that can be rebased onto one mount point --
and `extract_eval_bank` calls it *after* loading the backbone. So the combined
frame is re-rooted onto the two trees' common ancestor, which rewrites every
`rel_path`. That is a new identity, deliberately: this is a new manifest, not a
view of either input, and its fingerprint should not match either.

**Colliding index labels.** Both inputs are indexed 0..N. The index label is
the per-view RNG key (`extract_bank`), so a naive concat would hand two
different images the same key. Freezing through `write_manifest`, which writes
`index=False`, is what makes the labels unique -- and it must happen exactly
once, here, at freeze time. Every later consumer slices with `.iloc` and must
never call `reset_index`.

**Digests are recomputed, not copied.** The eval bank is about to be extracted
from these exact files. Re-reading them proves they still are what each source
manifest was frozen against; a divergence is reported by name and refused,
because features extracted from a changed file do not correspond to its
recorded label and nothing downstream could tell.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from aigcdet.data.manifest import (
    MANIFEST_COLUMNS,
    derive_root,
    read_manifest,
    write_manifest,
)

#: The splits the ablation tier is defined over, minus `benchmark`, which comes
#: from the other file. Kept in step with `extract_eval_bank.TIERS["ablation"]`.
DEFAULT_SPLITS = ("val_internal", "heldout_generator")


def select(df: pd.DataFrame, splits: tuple[str, ...], where: str) -> pd.DataFrame:
    """`df` restricted to `splits`, in the frame's own row order.

    An unknown split is a typo that would otherwise produce a manifest quietly
    missing a whole population -- and the eval bank built from it would look
    fine until `run_ablation` refused the selection population three hours
    later.
    """
    present = sorted(set(df["split"]))
    missing = [s for s in splits if s not in present]
    if missing:
        raise SystemExit(
            f"{where} has no rows for split(s) {missing}; it contains {present}")
    return df[df["split"].isin(splits)]


def check_no_overlap(train: pd.DataFrame, bench: pd.DataFrame) -> None:
    """Refuse an image that appears on both sides.

    COCO val2017 is benchmark data and may never be trained on (spec §4.1);
    `build_dataset` enforces that by source. This is the second net, on
    content: if a row did reach both frames, the eval bank would score the
    same image twice under two different splits, and every per-split metric
    would be computed over a population that overlaps itself.
    """
    both = set(train["content_sha256"]) & set(bench["content_sha256"])
    both.discard("")
    if both:
        example = sorted(both)[0]
        row = train[train["content_sha256"] == example].iloc[0]
        raise SystemExit(
            f"{len(both)} image(s) appear in both manifests, e.g. "
            f"{row['path']} (split {row['split']!r}). The benchmark must not "
            "overlap the training tree -- check the --raw/--demo-dir split in "
            "scripts/build_dataset.py.")


def check_digests(frozen: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    """Every file must still hash to what its source manifest recorded."""
    was = dict(zip(frozen["path"], frozen["content_sha256"]))
    changed = [p for p, d in zip(rebuilt["path"], rebuilt["content_sha256"])
               if was.get(p) and was[p] != d]
    if changed:
        raise SystemExit(
            f"{len(changed)} file(s) differ from the digest their manifest was "
            f"frozen against, e.g.\n  {changed[0]}\n"
            "Features extracted from a changed file do not correspond to its "
            "recorded label, and nothing downstream can detect that. Restore "
            "the files, or re-freeze the source manifest deliberately.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Combine the training and benchmark manifests for the "
                    "ablation-tier eval bank.")
    ap.add_argument("--manifest", required=True, help="the training manifest")
    ap.add_argument("--benchmark-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                    help="comma-separated splits to take from --manifest; "
                         "`benchmark` always comes from --benchmark-manifest")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    train = read_manifest(a.manifest)
    bench = read_manifest(a.benchmark_manifest)

    stray = sorted(set(bench["split"]) - {"benchmark"})
    if stray:
        raise SystemExit(
            f"--benchmark-manifest holds non-benchmark split(s) {stray}. That "
            "is almost certainly the training manifest passed twice, which "
            "would double-count its rows in the eval bank.")

    train = select(train, splits, "--manifest")
    check_no_overlap(train, bench)

    # Fixed order: the eval splits in the training manifest's own order, then
    # the benchmark rows in theirs. Deterministic given the inputs, and the
    # thing that must never change afterwards -- row order IS the key space.
    combined = pd.concat([train, bench], ignore_index=True)[MANIFEST_COLUMNS]

    # The root is derived from the IMAGE paths, by `write_manifest`'s own
    # `derive_root` -- the deepest directory containing every image, which for
    # `data/normalized/...` plus `data/demo/...` is `data`. Deriving it from
    # the two manifest FILES instead would land a level too shallow whenever a
    # manifest sits outside the tree it describes, and every `rel_path` would
    # carry a spurious leading component.
    root = derive_root(combined["path"])
    print(f"{len(train)} eval rows + {len(bench)} benchmark rows -> {len(combined)}")
    print(f"derived root: {root}")

    write_manifest(combined, a.out, root=root, digests="bytes",
                   workers=a.workers)
    check_digests(pd.concat([read_manifest(a.manifest),
                             read_manifest(a.benchmark_manifest)]), combined)

    print(combined["split"].value_counts().to_string())
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
