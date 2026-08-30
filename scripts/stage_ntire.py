"""Restage NTIRE's shards into the bucket layout `build_dataset` ingests.

NTIRE ships every image of every class in one `images/` directory, with the
class in a sibling `labels.csv`:

    shard_0/images/<20-char name>.jpg      labels.csv: image_name,label

`data.sources.classify` reads the class off the BUCKET DIRECTORY, which is the
whole point of that design -- a writer and a reader that agree on directory
names cannot drift apart the way a writer and a reader of a CSV column can
(the C1 failure). So the CSV has to become directories before ingestion:

    <dest>/ntire/real/<name>.jpg           label 0
    <dest>/ntire/generated/<name>.jpg      label 1

Hardlinks, not copies. 150,000 images is 60 GB, and every one of them is
already on the same filesystem; a copy would cost an hour and 60 GB to produce
bytes that already exist. The source tree is never modified.

Idempotent: an existing correct link is left alone, so a re-run after an
interrupted pass costs a stat per image and nothing else. A name that already
exists pointing at DIFFERENT bytes is an error, not something to overwrite --
NTIRE's names are random 20-character strings and a genuine collision across
shards would mean two images share a name, which silently halves one of them.

Polarity is not assumed. `--verify-polarity` re-states the mapping this script
was built against, and the mapping was checked against a trained model before
it was written (see the `ntire` SourceSpec). An inverted corpus of this size
does not fail loudly.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

#: NTIRE's own encoding, from the dataset card: "0 corresponds to a real
#: image, and 1 to a generated one". Identical to this project's convention,
#: which is exactly why it is written down -- a mapping that needs no
#: translation is the easiest one to get silently wrong later.
BUCKET_FOR_LABEL = {0: "real", 1: "generated"}


def shard_dirs(root: str) -> list[str]:
    """Every `shard_*` directory under `root`, in numeric order.

    Sorted numerically rather than lexically so `shard_10` follows `shard_9`
    if the remaining published shards are ever fetched.
    """
    found = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.startswith("shard_") and os.path.isdir(path):
            try:
                found.append((int(name.split("_", 1)[1]), path))
            except ValueError:
                raise ValueError(
                    f"{path}: expected shard_<int>, and a directory that looks "
                    "like a shard but is not one would be silently skipped")
    if not found:
        raise ValueError(f"no shard_* directories under {root}")
    return [p for _, p in sorted(found)]


def stage_shard(shard: str, dest_root: str, dry_run: bool = False) -> dict:
    """Link one shard's images into `<dest_root>/ntire/<bucket>/`."""
    labels = pd.read_csv(os.path.join(shard, "labels.csv"))
    for col in ("image_name", "label"):
        if col not in labels.columns:
            raise ValueError(f"{shard}/labels.csv has no {col!r} column; "
                             f"it has {list(labels.columns)}")
    unknown = sorted(set(labels["label"]) - set(BUCKET_FOR_LABEL))
    if unknown:
        raise ValueError(
            f"{shard}/labels.csv carries label(s) {unknown}, which this script "
            f"has no bucket for (it knows {BUCKET_FOR_LABEL}). SID_Set uses a "
            "third value for TAMPERED images; if NTIRE has done the same, that "
            "class needs a decision, not a default.")

    counts = {b: 0 for b in BUCKET_FOR_LABEL.values()}
    linked = skipped = 0
    for bucket in BUCKET_FOR_LABEL.values():
        os.makedirs(os.path.join(dest_root, "ntire", bucket), exist_ok=True)

    for name, label in zip(labels["image_name"], labels["label"]):
        src = os.path.join(shard, "images", str(name))
        bucket = BUCKET_FOR_LABEL[int(label)]
        dst = os.path.join(dest_root, "ntire", bucket, str(name))
        counts[bucket] += 1
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"{src} is named in labels.csv but is not on disk; the shard "
                "is incomplete and staging it would produce a manifest with "
                "rows that cannot be decoded")
        if os.path.exists(dst):
            # Same inode = this script's own earlier pass. Different inode =
            # two different images claiming one name, which would silently
            # drop one of them.
            if os.path.samefile(src, dst):
                skipped += 1
                continue
            raise FileExistsError(
                f"{dst} already exists and is NOT the same file as {src}. Two "
                "images share a name across shards; resolve it rather than "
                "overwriting, or one of them vanishes from the corpus.")
        if not dry_run:
            os.link(src, dst)
        linked += 1
    return {"shard": os.path.basename(shard), "rows": len(labels),
            "linked": linked, "already_present": skipped, "by_bucket": counts}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="directory holding shard_0/, shard_1/, ...")
    ap.add_argument("--dest", required=True,
                    help="staging root; images land in <dest>/ntire/<bucket>/")
    ap.add_argument("--dry-run", action="store_true",
                    help="count and validate without creating any link")
    args = ap.parse_args(argv)

    print(f"label mapping: {BUCKET_FOR_LABEL}  "
          "(NTIRE card: 0 = real, 1 = generated; verified against the dinov3l "
          "a3 head at AUC 0.9454 in this direction)")
    totals = {b: 0 for b in BUCKET_FOR_LABEL.values()}
    linked = skipped = 0
    for shard in shard_dirs(args.src):
        r = stage_shard(shard, args.dest, dry_run=args.dry_run)
        print(f"  {r['shard']}: {r['rows']:,} rows -> "
              f"{r['linked']:,} linked, {r['already_present']:,} already there, "
              f"{r['by_bucket']}")
        for b, n in r["by_bucket"].items():
            totals[b] += n
        linked += r["linked"]
        skipped += r["already_present"]

    total = sum(totals.values())
    print(f"\n{total:,} images: " + ", ".join(f"{b} {n:,}" for b, n in totals.items()))
    print(f"  {linked:,} linked{' (dry run)' if args.dry_run else ''}, "
          f"{skipped:,} already present")
    print(f"  staged at {os.path.join(args.dest, 'ntire')}")
    return {"total": total, "by_bucket": totals, "linked": linked}


if __name__ == "__main__":
    main()
    sys.exit(0)
