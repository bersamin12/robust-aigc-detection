"""Materialise exactly the images a manifest references, as an uploadable tree.

Why this exists
---------------
The union corpus normalises to ~146 GB. A 20,000-row probe of it needs ~8 GB.
Uploading the corpus to run the probe would cost eighteen times the transfer
for none of the benefit, and the alternative -- pointing a Kaggle notebook at
a subset of a Dataset that does not exist -- is not an alternative.

So: read a manifest, hardlink each row's image into `<dest>/<rel_path>`, and
publish that. `rel_path` is preserved EXACTLY, because it is the manifest's
half of the bank identity (`features/bank.py`): a tree whose paths differ from
the manifest's by so much as a directory level resolves to nothing on the far
side, and the failure surfaces as "0 of 200 sampled rows resolve" an hour into
a session rather than here in a second.

Hardlinks, so staging a probe of an already-normalised corpus costs inodes and
not bytes. `--mode copy` exists for a destination on a different filesystem,
where a hardlink is impossible rather than merely wasteful.

More than one manifest may be staged into one tree (`--manifest` repeats): a
probe needs its training rows AND its eval rows, the eval rows include the
organisers' benchmark from a different source root, and both halves have to
arrive in the same Dataset or the notebook's two-link farm has nothing to
link. Rows shared between manifests are staged once.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import pandas as pd

MODES = ("hardlink", "copy", "symlink")


def link_one(src: str, dst: str, mode: str) -> bool:
    """Place `src` at `dst`. Returns True if it did work, False if already there.

    An existing destination is left alone only when it is genuinely the same
    file. Two different images at one rel_path means the manifests disagree
    about what lives there, and silently keeping the first is how a corpus
    ends up with a row whose pixels are not the ones its digest was taken
    over.
    """
    if os.path.exists(dst):
        if mode != "symlink" and os.path.samefile(src, dst):
            return False
        if mode == "symlink" and os.path.realpath(dst) == os.path.realpath(src):
            return False
        raise FileExistsError(
            f"{dst} exists and is not {src}. Two manifests disagree about what "
            "lives at this rel_path; staging either one silently would make "
            "the tree disagree with at least one manifest's digests.")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)
    return True


def stage(manifest_path: str, roots: list[str], dest: str, mode: str) -> dict:
    """Stage every row of one manifest into `dest`, preserving `rel_path`.

    `roots` is searched in order for each row, because one manifest can span
    two trees: an eval manifest's rel_paths start either with the normalised
    corpus's own top-level name or with `demo/`, and those live under
    different roots on this machine. Searching rather than requiring one root
    is what lets a single call stage both.
    """
    df = pd.read_parquet(manifest_path)
    for col in ("rel_path",):
        if col not in df.columns:
            raise ValueError(
                f"{manifest_path} has no {col!r} column, so nothing records "
                "where each image sits inside the dataset; it cannot be staged "
                "portably. Rebuild it with a current write_manifest.")
    staged = present = 0
    missing: list[str] = []
    for rel in df["rel_path"]:
        rel = str(rel)
        src = next((os.path.join(r, rel) for r in roots
                    if os.path.exists(os.path.join(r, rel))), None)
        if src is None:
            missing.append(rel)
            if len(missing) > 20:
                break
            continue
        if link_one(src, os.path.join(dest, rel), mode):
            staged += 1
        else:
            present += 1
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}+ rows of {manifest_path} do not resolve under any "
            f"of {roots}, e.g. {missing[:3]}. Staging a partial tree would "
            "produce a Dataset whose manifest names images it does not carry.")
    return {"manifest": os.path.basename(manifest_path), "rows": len(df),
            "staged": staged, "already_present": present}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True,
                    help="manifest whose rows to stage; repeatable")
    ap.add_argument("--root", action="append", required=True,
                    help="directory a manifest's rel_paths are relative to; "
                         "repeatable and searched in order")
    ap.add_argument("--dest", required=True, help="tree to create")
    ap.add_argument("--mode", choices=MODES, default="hardlink")
    ap.add_argument("--copy-manifest", action="store_true",
                    help="also copy each manifest file into --dest, so the "
                         "published Dataset carries the manifest it describes")
    a = ap.parse_args(argv)

    os.makedirs(a.dest, exist_ok=True)
    results = []
    for m in a.manifest:
        r = stage(m, a.root, a.dest, a.mode)
        print(f"  {r['manifest']}: {r['rows']:,} rows -> {r['staged']:,} staged, "
              f"{r['already_present']:,} already present")
        results.append(r)
        if a.copy_manifest:
            shutil.copy2(m, os.path.join(a.dest, os.path.basename(m)))

    total = sum(r["staged"] for r in results)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(a.dest) for f in fs)
    print(f"\n{total:,} images staged into {a.dest} ({a.mode})")
    print(f"  tree holds {size / 1024**3:.2f} GiB of image bytes"
          + ("  (hardlinked: no new bytes on disk)" if a.mode == "hardlink" else ""))
    return {"staged": total, "bytes": size, "results": results}


if __name__ == "__main__":
    main()
    sys.exit(0)
