"""Assemble one scannable `<source>/<bucket>/` root out of trees that live apart.

WHY THIS EXISTS
---------------
`build_dataset.py` scans ONE raw root, and the `coco_crop` corpus is spread
across two filesystems' worth of directories:

    data/raw/wildfake                    the generated families, and real/
    data/raw/sid_set                     58,757 images
    /mnt/berstorage/coco/train2017       118,287 photographs, outside data/

Linking COCO into `data/raw` directly would have been simpler and is wrong:
`data/raw` is the frozen stream's corpus, `configs/datasets/max_data.yaml`
describes it as "every image on disk", and `_scan` reads the tree rather than
the manifest. A new source dropped in there silently changes what that preset
means the next time anyone builds it. A stream gets its own root.

WHAT IT COSTS
-------------
Nothing. It LINKS rather than copies, so a 48 GB tree costs inodes, and
nothing is ever re-encoded on the way in -- which matters because re-encoding
one source and not another is precisely how the two classes start differing by
container.

A NOTE ON `--mode`
------------------
Hardlinks are the default: they cost one inode reference, cannot be broken by
moving the target, and leave the staged file indistinguishable from a real one
to everything downstream. They require one filesystem. `symlink` is the
cross-device fallback (`build_dataset._scan` globs for image extensions and a
symlink to a file matches, which is why this links FILES and not bucket
directories -- `glob` with `**` does not descend into a symlinked directory).
`copy` is the last resort and the only mode that spends disk.
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import shutil

from aigcdet.data.sources import classify

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

MODES = ("hardlink", "symlink", "copy")


def bucket_files(bucket_dir: str) -> list[str]:
    """Every image under a bucket, at any depth.

    Recursive because WildFake's authentic images are nested one level BELOW
    the bucket (`wildfake/real/<subset>/`), which is what lets `classify` read
    bucket "real" while the directory keeps which upstream they came from --
    and what `DatasetPreset.exclude_subpaths` then addresses.
    """
    out: list[str] = []
    for ext in IMG_EXT:
        out += glob.glob(os.path.join(bucket_dir, "**", ext), recursive=True)
    return sorted(out)


def link_one(src: str, dst: str, mode: str) -> None:
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            raise OSError(
                errno.EXDEV,
                f"cannot hardlink {src} -> {dst}: they are on different "
                "filesystems. Re-run with --mode symlink (free, but the "
                "staged root breaks if the source moves) or --mode copy "
                "(spends the full size of the tree).") from None
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return
    shutil.copy2(src, dst)


def stage_source(out_root: str, source: str, tree: str, mode: str) -> dict[str, int]:
    """Link every bucket of `tree` into `out_root/<source>/<bucket>/`.

    Each bucket is classified BEFORE anything is linked, so a tree whose
    layout `build_dataset` could not read fails here -- in a second, with the
    bucket named -- rather than after a 200,000-file link pass.

    The relative path below the bucket is PRESERVED, because it carries
    meaning: `wildfake/real/real_ffhq/x.png` and
    `wildfake/real/real_laion5b/y.png` are the same bucket and the same label,
    and the only thing that distinguishes their licences is that directory.
    Flattening them would make `exclude_subpaths` unable to name either.
    """
    buckets = sorted(d for d in os.listdir(tree)
                     if os.path.isdir(os.path.join(tree, d)))
    if not buckets:
        raise ValueError(f"{tree} has no bucket directories to stage")
    for b in buckets:
        classify(source, b)          # raises on a layout build_dataset cannot read

    counts: dict[str, int] = {}
    for b in buckets:
        src_dir = os.path.join(tree, b)
        dst_dir = os.path.join(out_root, source, b)
        os.makedirs(dst_dir, exist_ok=True)
        files = bucket_files(src_dir)
        for f in files:
            link_one(f, os.path.join(dst_dir, os.path.relpath(f, src_dir)), mode)
        counts[b] = len(files)
        label, generator = classify(source, b)
        print(f"  {source}/{b:28s} {len(files):7d}  -> label={label} "
              f"generator={generator or '(authentic)'}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="root to create, e.g. data/raw_coco_crop")
    ap.add_argument("--source", action="append", default=[], metavar="NAME=TREE",
                    help="registered source name and the directory holding its "
                         "bucket subdirectories, e.g. "
                         "wildfake=data/raw/wildfake. Repeatable.")
    ap.add_argument("--licences", default="data/raw/LICENCES.json",
                    help="LICENCES.json to place at the staged root; "
                         "build_dataset refuses to run without one (spec §4.5)")
    ap.add_argument("--mode", choices=MODES, default="hardlink")
    ap.add_argument("--force", action="store_true",
                    help="stage into an existing root instead of refusing")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        raise SystemExit(
            f"{a.out} already exists. A half-staged root that a later run tops "
            "up is a corpus nobody can describe; delete it or pass --force if "
            "you are resuming the same staging.")
    if not a.source:
        raise SystemExit("--source is required at least once")
    os.makedirs(a.out, exist_ok=True)

    staged: dict[str, dict[str, int]] = {}
    for spec in a.source:
        if "=" not in spec:
            raise SystemExit(f"--source {spec!r} is not NAME=TREE")
        name, tree = spec.split("=", 1)
        if not os.path.isdir(tree):
            raise SystemExit(f"--source {name}: {tree} is not a directory")
        print(f"staging {name} from {tree} ({a.mode})")
        staged[name] = stage_source(a.out, name, tree, a.mode)

    shutil.copyfile(a.licences, os.path.join(a.out, "LICENCES.json"))
    missing = sorted(set(staged) - set(_licence_names(a.out)))
    if missing:
        raise SystemExit(
            f"{a.licences} has no entry for {missing}, and build_dataset "
            "refuses to fabricate provenance for a source it is about to "
            "ingest (spec §4.5). Append it with acquire_data._record_licences "
            "or add the line by hand before building.")

    # The receipt. Which tree each bucket came from is exactly what a staged
    # root cannot be asked afterwards, since a hardlink carries no provenance.
    receipt = os.path.join(a.out, "STAGED_FROM.json")
    with open(receipt, "w") as f:
        json.dump({"mode": a.mode,
                   "sources": {s.split("=", 1)[0]: os.path.abspath(s.split("=", 1)[1])
                               for s in a.source},
                   "counts": staged}, f, indent=2)
    total = sum(sum(c.values()) for c in staged.values())
    print(f"\nstaged {total} images into {a.out}; receipt {receipt}")


def _licence_names(root: str) -> set[str]:
    """Source names with an entry in the staged root's LICENCES.json.

    Same JSON-lines format `acquire_data._record_licences` appends and
    `build_dataset._load_licences` reads; checked here so a missing entry
    fails in a second rather than after the scan.
    """
    names: set[str] = set()
    with open(os.path.join(root, "LICENCES.json")) as f:
        for line in f:
            line = line.strip()
            if line:
                names.update(k for k, v in json.loads(line).items()
                             if v and str(v).strip())
    return names


if __name__ == "__main__":
    main()
