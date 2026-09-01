"""Harvest portrait-orientation Open Images V7 photographs as authentic rows.

WHY THUMBNAILS AND NOT ORIGINALS
--------------------------------
Open Images ships no width/height in its metadata, so orientation cannot be
filtered before fetching something. Measured 2026-08-30 over 1,374 sampled
thumbnails: 18.0% portrait (<0.8), 8.7% at <=0.7, 0.80% within +-0.03 of 9:16,
thumbnail short side median 457 / p05 360.

`OriginalURL` points at full-size Flickr statics, several MB each. 60k of those
is ~180 GB against 92 GB of free disk, with NTIRE landing on the same volume.
`Thumbnail300KURL` is ~30 KB and its short side clears `crop_side=200` with
room to spare, so thumbnails are what fits.

THE COST OF THAT, STATED PLAINLY: a thumbnail is a re-encoded JPEG. This
project has already measured that JPEG history leaks the label
(docs/low_level_confounds.md), so introducing an authentic source with its own
distinct compression history is exactly the kind of shortcut the confound work
exists to catch. `scripts/gate_confounds.py` MUST be run over any preset built
on this source before it is trained on, and a `jpeg_quality` AUC materially
above the frozen corpus's 0.6721 laplacian baseline means these rows are
teaching compression rather than provenance.

ATTRIBUTION
-----------
CC BY 2.0 is a permissive licence but not a free-for-all: it requires
attribution. Every kept row records Author, AuthorProfileURL, Title,
OriginalURL and the licence URL in `attribution.csv`, so the obligation can
still be met after normalisation strips the metadata.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

META_URL = ("https://storage.googleapis.com/openimages/v6/"
            "oidv6-train-images-with-labels-with-rotation.csv")
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
UA = "Mozilla/5.0 (compatible; aigcdet-research/1.0)"

_lock = threading.Lock()
_kept = 0
_seen = 0
_fail = 0
_noattr = 0


def fetch_metadata(path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000_000:
        print(f"metadata already present: {path}", flush=True)
        return path
    print(f"downloading metadata -> {path}", flush=True)
    req = urllib.request.Request(META_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    print(f"metadata done: {os.path.getsize(path) / 1e9:.2f} GB", flush=True)
    return path


def candidates(meta_path: str, limit_rows: int):
    """Rows that are CC BY 2.0 and actually have a thumbnail to measure."""
    with open(meta_path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit_rows and i >= limit_rows:
                return
            if row.get("License") != CC_BY_2:
                continue
            if not row.get("Thumbnail300KURL"):
                continue
            if not _profile_url(row):
                # Not shippable: CC BY needs attribution and this file is the
                # only record of it after normalisation. Dropped here rather
                # than written blank -- the previous behaviour -- and rather
                # than raised inside `handle`, which runs under the lock after
                # `_kept` is incremented and whose exceptions ThreadPoolExecutor
                # discards, so a raise there would desync image and row counts
                # without saying so.
                global _noattr
                with _lock:
                    _noattr += 1
                continue
            yield row


#: Open Images spells it `AuthorProfileURL` -- URL upper-cased. This was read
#: as `row.get("AuthorProfileUrl", "")`, and because `csv.DictReader` does not
#: raise on a missing key and `.get` has a default, it returned "" for every
#: row: 60,000 blank profile URLs and a corpus that cannot be shipped, since
#: CC BY 2.0 requires attribution and normalisation strips image metadata, so
#: this file is the only surviving record. Both spellings are tried, so a
#: rename upstream degrades to the other rather than back to silence.
PROFILE_KEYS = ("AuthorProfileURL", "AuthorProfileUrl")


def _profile_url(row):
    """The row's author profile URL, or "" if it carries none."""
    for key in PROFILE_KEYS:
        v = (row.get(key) or "").strip()
        if v:
            return v
    return ""


def handle(row, out_dir, max_ratio, min_short, target, writer):
    global _kept, _seen, _fail
    with _lock:
        if _kept >= target:
            return
        _seen += 1
    try:
        req = urllib.request.Request(row["Thumbnail300KURL"],
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25) as r:
            blob = r.read()
        im = Image.open(io.BytesIO(blob))
        w, h = im.size
    except Exception:
        with _lock:
            _fail += 1
        return

    # Portrait AND big enough that a 200px crop is a crop, not an upscale.
    if w >= h or (w / h) > max_ratio or min(w, h) < min_short:
        return

    iid = row["ImageID"]
    dest = os.path.join(out_dir, f"{iid}.jpg")
    try:
        with open(dest, "wb") as f:
            f.write(blob)
    except OSError:
        return

    with _lock:
        if _kept >= target:
            os.remove(dest)
            return
        _kept += 1
        writer.writerow([iid, f"{w}x{h}", f"{w / h:.4f}", row.get("Author", ""),
                         _profile_url(row), row.get("Title", ""),
                         row.get("OriginalURL", ""), CC_BY_2])
        if _kept % 250 == 0:
            print(f"kept {_kept}/{target}  checked {_seen}  dead {_fail}  "
                  f"yield {_kept / max(_seen, 1):.1%}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/berstorage/techjam/open_images")
    ap.add_argument("--target", type=int, default=60000)
    ap.add_argument("--max-ratio", type=float, default=0.7,
                    help="width/height ceiling. 0.7 spans 9:16 through 2:3; "
                         "strict 9:16 yields only ~74k over all of Open "
                         "Images, which leaves no room to filter further.")
    ap.add_argument("--min-short", type=int, default=400)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--limit-rows", type=int, default=0)
    a = ap.parse_args(argv)

    img_dir = os.path.join(a.out, "portrait")
    os.makedirs(img_dir, exist_ok=True)
    meta = fetch_metadata(os.path.join(a.out, "oidv6-train-with-rotation.csv"))

    att = open(os.path.join(a.out, "attribution.csv"), "w", newline="",
               encoding="utf-8")
    writer = csv.writer(att)
    writer.writerow(["ImageID", "size", "ratio", "Author", "AuthorProfileURL",
                     "Title", "OriginalURL", "License"])

    print(f"target {a.target} portrait images, ratio <= {a.max_ratio}, "
          f"short side >= {a.min_short}", flush=True)
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for row in candidates(meta, a.limit_rows):
            if _kept >= a.target:
                break
            ex.submit(handle, row, img_dir, a.max_ratio, a.min_short,
                      a.target, writer)
    att.close()
    print(f"DONE kept={_kept} checked={_seen} dead={_fail} "
          f"no_attribution={_noattr}", flush=True)

    # The invariant the corpus is shipped on: one attribution row per image,
    # every one of them attributable. Checked here rather than trusted, because
    # the blank-profile-URL bug survived a whole 60,000-image harvest.
    with open(os.path.join(a.out, "attribution.csv"), newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    blank = [r["ImageID"] for r in rows if not (r["AuthorProfileURL"] or "").strip()]
    n_img = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    print(f"attribution rows={len(rows)} images={n_img} blank_profile={len(blank)}",
          flush=True)
    if blank or len(rows) != n_img:
        print(f"ATTRIBUTION INCOMPLETE -- {len(blank)} blank profile URLs, "
              f"{len(rows)} rows against {n_img} images. The corpus cannot be "
              f"redistributed in this state.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
