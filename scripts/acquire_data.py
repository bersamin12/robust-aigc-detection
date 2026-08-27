"""Subset download for WildFake, SID_Set, and COCO val2017 (spec §4.1).

Pulls selected generator folders and parquet shards rather than whole
repositories — download volume is the binding risk, not compute (spec §4.4).
Records each dataset's licence, which the manifest and README require (§4.5).

Usage:
    python scripts/acquire_data.py --dataset sid_set --limit 30000 --out data/raw
    python scripts/acquire_data.py --dataset wildfake --generators sdxl,sd15,midjourney \
        --limit 30000 --out data/raw
    python scripts/acquire_data.py --dataset coco_val2017 --out data/raw

This script is not exercised in CI: it pulls tens of gigabytes from
third-party hosts and is meant to be run once by a human, on day 1, who can
confirm each source's licence before the download starts.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from aigcdet.data.sources import LICENCES, SOURCES, raw_subdir

#: Record fields SID_Set may carry the generating model under. Where one is
#: present the image is filed under that generator, so it stays a real
#: generator family in the manifest instead of collapsing into the
#: dataset-level pseudo-generator "sid_set" (which spec §4.6 cannot hold out).
_GENERATOR_FIELDS = ("generator", "model", "generator_name", "source_model")
_SAFE_GENERATOR = re.compile(r"^[A-Za-z0-9._-]+$")


def _record_generator(rec: dict) -> str:
    """The generator this record names, or "" if it does not name one.

    Rejects anything that is not a plain identifier: the value becomes a
    directory name, and a value containing a separator would write outside
    the source's own tree.
    """
    for key in _GENERATOR_FIELDS:
        value = rec.get(key)
        if isinstance(value, str) and _SAFE_GENERATOR.match(value.strip()):
            return value.strip()
    return ""


def acquire_sid_set(out: str, limit: int) -> None:
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    os.makedirs(out, exist_ok=True)
    n = 0
    for rec in ds:
        if n >= limit:
            break
        # SID_Set labels: 0 real, 1 fully synthetic, 2 tampered.
        # Tampered is out of scope for the binary task (spec §4.1).
        if rec.get("label") == 2:
            continue
        label = 0 if rec["label"] == 0 else 1
        # raw_subdir is the inverse of the mapping build_dataset.py reads the
        # tree back with, so the two scripts cannot drift apart.
        sub = raw_subdir("sid_set", label,
                         "" if label == 0 else _record_generator(rec))
        d = os.path.join(out, "sid_set", sub)
        os.makedirs(d, exist_ok=True)
        rec["image"].save(os.path.join(d, f"{n:07d}.png"))
        n += 1
    print(f"sid_set: wrote {n}")


def acquire_wildfake(out: str, limit: int, generators: list[str]) -> None:
    try:
        from modelscope.msdatasets import MsDataset  # noqa: F401  # pip install modelscope
    except ImportError:
        pass  # the SystemExit below is the message either way
    raise SystemExit(
        "WildFake layout must be inspected before subsetting. Run:\n"
        "  python -c \"from modelscope.hub.api import HubApi; "
        "print(HubApi().get_dataset_files('hy2628982280/WildFake'))\"\n"
        f"then pull only the folders for: {generators}, writing to {out}/wildfake/<generator>/."
    )


def acquire_coco_val2017(out: str) -> None:
    import urllib.request
    import zipfile
    os.makedirs(out, exist_ok=True)
    zp = os.path.join(out, "val2017.zip")
    if not os.path.exists(zp):
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", zp)
    dest = os.path.join(out, "coco_val2017")
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    # The zip's own top-level directory is the bucket build_dataset.py reads
    # this tree back through. It is "val2017", not "real" -- assuming
    # otherwise is what labelled every COCO photograph AI-generated. If the
    # archive layout ever changes, say so here rather than 5,000 mislabelled
    # rows later.
    buckets = {e for e in os.listdir(dest) if os.path.isdir(os.path.join(dest, e))}
    declared = SOURCES["coco_val2017"].real_buckets
    if not buckets <= declared:
        raise SystemExit(
            f"coco_val2017 extracted to {sorted(buckets)}, but "
            f"aigcdet.data.sources declares {sorted(declared)}. Update the "
            "registry before building a manifest from this tree.")
    print(f"coco_val2017: extracted to {sorted(buckets)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--generators", default="")
    a = ap.parse_args()

    if a.dataset == "sid_set":
        acquire_sid_set(a.out, a.limit)
    elif a.dataset == "wildfake":
        acquire_wildfake(a.out, a.limit,
                         [g for g in a.generators.split(",") if g])
    else:
        acquire_coco_val2017(a.out)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "LICENCES.json"), "a") as f:
        f.write(json.dumps({a.dataset: LICENCES[a.dataset]}) + "\n")


if __name__ == "__main__":
    main()
