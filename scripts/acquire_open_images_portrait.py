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
attribution. Every kept row records Author, AuthorProfileUrl, Title,
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

#: The validation split's metadata: same columns, same licences, **15 MB
#: against 2.7 GB**, over ~41k images of which ~40k are CC BY 2.0.
#:
#: Two reasons a pilot should prefer it. It is a 15 MB download rather than a
#: 2.7 GB one for a few dozen images. And task 03's eval reals must not be
#: reals the detector trained on (`docs/03` §5.5) -- drawing the pilot from
#: validation while task 02 harvests from train makes the two disjoint by
#: construction rather than by bookkeeping.
#:
#: Too small for the real corpus: ~5% of candidates survive the portrait
#: filter, so it tops out near 1,900 images. Use the default for 60k.
VALIDATION_META_URL = ("https://storage.googleapis.com/openimages/2018_04/"
                       "validation/validation-images-with-rotation.csv")
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
UA = "Mozilla/5.0 (compatible; aigcdet-research/1.0)"

_lock = threading.Lock()
_kept = 0
_seen = 0
_fail = 0


def fetch_metadata(path: str, url: str = META_URL, min_bytes: int = 1_000_000_000) -> str:
    """Download `url` to `path`, skipping a copy that is already complete.

    `min_bytes` is what "complete" means, and it has to track the URL: the
    train metadata is 2.7 GB and validation is 15 MB, so a single hardcoded
    floor would either re-download the big one forever or accept a truncated
    small one. Writes through `.part` and renames -- an interrupted download
    otherwise leaves a short file that the size check on the next run may
    accept, and a silently truncated metadata file yields a silently truncated
    harvest.
    """
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        print(f"metadata already present: {path} "
              f"({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)
        return path
    print(f"downloading metadata -> {path}", flush=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        if os.path.getsize(tmp) < min_bytes:
            raise OSError(f"metadata truncated: {os.path.getsize(tmp)} < {min_bytes} bytes")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"metadata done: {os.path.getsize(path) / 1e6:.0f} MB", flush=True)
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
            yield row


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
                         row.get("AuthorProfileUrl", ""), row.get("Title", ""),
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
    ap.add_argument("--split", choices=("train", "validation"), default="train",
                    help="which Open Images metadata to walk. `validation` is "
                         "15 MB against train's 2.7 GB and tops out near 1,900 "
                         "portrait images -- right for a pilot, too small for "
                         "the corpus. It also keeps task 03's eval reals "
                         "disjoint from task 02's training reals (docs/03 §5.5).")
    a = ap.parse_args(argv)

    img_dir = os.path.join(a.out, "portrait")
    os.makedirs(img_dir, exist_ok=True)
    if a.split == "validation":
        meta = fetch_metadata(os.path.join(a.out, "validation-with-rotation.csv"),
                              VALIDATION_META_URL, 10_000_000)
    else:
        meta = fetch_metadata(os.path.join(a.out, "oidv6-train-with-rotation.csv"),
                              META_URL, 1_000_000_000)

    att = open(os.path.join(a.out, "attribution.csv"), "w", newline="",
               encoding="utf-8")
    writer = csv.writer(att)
    writer.writerow(["ImageID", "size", "ratio", "Author", "AuthorProfileUrl",
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
    print(f"DONE kept={_kept} checked={_seen} dead={_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
