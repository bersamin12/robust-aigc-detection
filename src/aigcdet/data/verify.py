"""Check a materialised dataset against the manifest it was frozen from.

This is the ten-minute check a teammate runs on Kaggle after attaching the
published Dataset and BEFORE paying 8-13 GPU-hours to extract a feature-bank
shard from it. Its job is not to say "something is wrong"; it is to say which
of three different things is wrong, because the fixes are different and
expensive to guess at:

  missing rows        the manifest names files that are not here
                      -> the Dataset is attached somewhere else, or the copy
                         is incomplete: re-attach / re-download.
  extra files         files are here that the manifest does not name
                      -> harmless for extraction (nothing reads them), but
                         this is not exactly the published Dataset.
  divergent content   the files here are not the bytes the manifest was
                      frozen against
                      -> re-download; if it persists, this copy was
                         re-encoded, and whether that matters is answered by
                         the pixel digest, not the byte digest.

The last distinction is the one that costs money. A byte digest changes under
any re-encode, including a lossless PNG re-save that leaves every pixel
identical -- which is a false alarm for a pipeline that only ever looks at
decoded pixels. A pixel digest changes only when what the model sees changes.
So the cheap byte digest is compared first (one streamed read, no decode, and
it can never MISS a pixel change, because different pixels in the same file
means different bytes), and any row it flags is escalated to the pixel digest
when the manifest carries one. "Re-encoded but pixel-identical" is then
reported as a warning, and "the pixels differ" as a stop.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import pandas as pd

from aigcdet.data.manifest import (
    dataset_root,
    digest_row,
    read_manifest,
    rebase_manifest,
)

#: Extensions counted when scanning the tree for files the manifest does not
#: name. Restricted to images on purpose: a README, a LICENCE or a checksum
#: file shipped alongside the data is not an "extra image".
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: Cap on how many examples of each problem a report keeps. The counts are
#: exact; the lists are for a human to read.
MAX_EXAMPLES = 20


@dataclass
class VerifyReport:
    """What `verify_images` found. `describe()` is the part a human reads."""

    root: str | None
    n_rows: int
    n_digested: int
    digest_kind: str | None
    missing: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    divergent: list[str] = field(default_factory=list)
    #: Rows whose stored byte digest disagrees but whose stored PIXEL digest
    #: still matches -- a re-encode that changed no pixel.
    reencoded: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    #: True when a byte mismatch was re-checked against the stored PIXEL
    #: digest. It changes what a remaining divergence means: not "the bytes
    #: differ, which may or may not matter" but "the pixels differ".
    escalated: bool = False
    n_missing: int = 0
    n_unreadable: int = 0
    n_divergent: int = 0
    n_reencoded: int = 0
    n_extra: int = 0
    sampled: bool = False

    @property
    def n_fatal(self) -> int:
        """Problems that make extracting from this copy wrong, as opposed to
        merely untidy. A pixel-identical re-encode and an unlisted extra file
        are not in here: neither changes a single feature."""
        return self.n_missing + self.n_unreadable + (self.n_divergent
                                                     - self.n_reencoded)

    @property
    def ok(self) -> bool:
        return self.n_fatal == 0

    def raise_for_status(self) -> "VerifyReport":
        if not self.ok:
            raise ValueError(self.describe())
        return self

    def describe(self) -> str:
        head = "verify_images: OK" if self.ok else "verify_images: FAILED"
        kind = self.digest_kind or "none (manifest carries no digests)"
        lines = [
            f"{head}  root={self.root}",
            f"  manifest rows      {self.n_rows}",
            f"  content-checked    {self.n_digested}"
            + ("  (SAMPLE -- a clean sample does not prove the rest is clean)"
               if self.sampled else "")
            + f"  digest={kind}",
            f"  missing            {self.n_missing}",
            f"  unreadable         {self.n_unreadable}",
            f"  content-divergent  {self.n_divergent}"
            + (f"  (of which {self.n_reencoded} re-encoded but pixel-identical)"
               if self.n_reencoded else ""),
            f"  extra files        {self.n_extra}",
        ]
        for name, items in (("missing", self.missing),
                            ("divergent", self.divergent),
                            ("extra", self.extra)):
            if items:
                shown = ", ".join(items[:5])
                lines.append(f"  e.g. {name}: {shown}"
                             + (" ..." if len(items) > 5 else ""))
        if self.unreadable:
            rel, why = self.unreadable[0]
            lines.append(f"  e.g. unreadable: {rel}: {why}")
        lines.append("")
        lines.append("what to do:")
        lines += [f"  - {a}" for a in self.advice()]
        return "\n".join(lines)

    def advice(self) -> list[str]:
        """The actionable half of the report: re-download, re-normalise, or
        stop. Ordered worst-first, so the first line is the one to act on."""
        out: list[str] = []
        hard_divergent = self.n_divergent - self.n_reencoded
        if hard_divergent and (self.digest_kind == "pixels" or self.escalated):
            out.append(
                f"STOP. {hard_divergent} image(s) DECODE to different pixels "
                "than the manifest was frozen against. Features extracted here "
                "would not correspond to this manifest's labels. Re-download "
                "the published Dataset; if it still differs, this copy was "
                "re-normalised on another machine (a different Pillow/libjpeg) "
                "-- use the published one, do not re-normalise locally.")
        elif hard_divergent:
            out.append(
                f"{hard_divergent} file(s) differ in BYTES from the manifest. "
                "Re-download the Dataset first. If the mismatch survives a "
                "re-download, re-run with digest='pixels' (the manifest must "
                "have been frozen with digests='pixels') to find out whether "
                "the decoded pixels differ too: if they do, stop; if they do "
                "not, it is a re-encode and extraction is safe.")
        if self.n_missing == self.n_rows and self.n_rows:
            out.append(
                "EVERY row is missing: the dataset is not at this root. Check "
                "the mount point (Kaggle attaches under /kaggle/input/<slug>, "
                "often with one extra directory level inside) and pass it as "
                "--root / AIGCDET_DATA_ROOT. Nothing here needs re-normalising.")
        elif self.n_missing:
            near = self.n_missing == self.n_extra and self.n_extra > 0
            out.append(
                f"{self.n_missing} file(s) named by the manifest are absent"
                + (" while the same number of unlisted files are present, so "
                   "this tree is the right size but a different layout or a "
                   "different normalisation run -- re-attach the exact "
                   "published Dataset version."
                   if near else
                   " -- the copy is incomplete. Re-download / re-attach the "
                   "Dataset; do not extract a partial shard, the manifest is "
                   "indexed positionally."))
        if self.n_unreadable:
            out.append(
                f"{self.n_unreadable} file(s) are present but could not be "
                "read or decoded -- almost always a truncated download. "
                "Re-download.")
        if self.n_reencoded:
            out.append(
                f"{self.n_reencoded} file(s) were re-encoded but decode to "
                "identical pixels. Extraction is safe; the copy is simply not "
                "byte-identical to the published one.")
        if self.n_extra:
            out.append(
                f"{self.n_extra} image file(s) here are not named by the "
                "manifest. Nothing reads them, so extraction is unaffected, "
                "but check you attached the version this manifest was frozen "
                "against.")
        if self.digest_kind is None:
            out.append(
                "No content digests were compared -- this manifest was frozen "
                "without them (write_manifest(digests=None)), so presence is "
                "all that could be checked. Re-freezing the manifest with "
                "digests invalidates every bank already built against it.")
        if not out:
            out.append("nothing: this tree is the dataset the manifest was "
                       "frozen against. Safe to extract.")
        return out


def _sample_positions(n: int, sample: int | None) -> list[int]:
    """Evenly spaced positions, deterministic and reproducible across
    machines -- a random subsample would make two teammates' reports
    incomparable."""
    if sample is None or sample >= n:
        return list(range(n))
    if sample <= 0:
        return []
    step = n / sample
    return sorted({int(i * step) for i in range(sample)})


def _scan_tree(root: str) -> set[str]:
    found = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXT):
                found.add(os.path.relpath(os.path.join(dirpath, fn), root))
    return found


def verify_images(manifest_df: pd.DataFrame, root: str | None = None,
                  digest: str | None = "auto", sample: int | None = None,
                  check_extra: bool = True, workers: int = 8) -> VerifyReport:
    """Check the files under `root` against `manifest_df`.

    `root` rebases the manifest's absolute paths onto wherever the data
    actually is (see `aigcdet.data.manifest.rebase_manifest`); omit it to
    check the manifest where it points.

    `digest` picks what "the same file" means:
      * `"auto"` (default) -- the byte digest if the manifest carries one,
        else the pixel digest, else presence only. Bytes first because they
        are one streamed read with no decode and cannot miss a pixel change;
        rows they flag are escalated to the pixel digest automatically when
        the manifest has one.
      * `"bytes"` / `"pixels"` -- force one. `"pixels"` requires a manifest
        frozen with `digests="pixels"`.
      * `None` -- presence only.

    `sample` digests only that many evenly spaced rows (presence is always
    checked for every row, since it is one stat call). Use it for a first
    look at a 100k-image Dataset; a clean sample is evidence, not proof, and
    the report says so.
    """
    df = manifest_df if root is None else rebase_manifest(manifest_df, root)
    if root is None:
        try:
            root = dataset_root(df)
        except ValueError:
            root = None
    else:
        root = os.path.abspath(str(root))

    has_bytes = ("content_sha256" in df.columns
                 and bool(len(df)) and all(str(v) for v in df["content_sha256"]))
    has_pixels = ("pixel_sha256" in df.columns
                  and bool(len(df)) and all(str(v) for v in df["pixel_sha256"]))
    if digest == "auto":
        digest = "bytes" if has_bytes else ("pixels" if has_pixels else None)
    elif digest == "bytes" and not has_bytes:
        raise ValueError(
            "this manifest carries no content_sha256, so byte digests cannot "
            "be compared. It was frozen with write_manifest(digests=None); "
            "pass digest=None to check presence only.")
    elif digest == "pixels" and not has_pixels:
        raise ValueError(
            "this manifest carries no pixel_sha256, so decoded pixels cannot "
            "be compared. Freeze it with write_manifest(digests='pixels') -- "
            "which costs a full decode of every image -- or pass "
            "digest='bytes'.")
    elif digest not in ("bytes", "pixels", None):
        raise ValueError(f"digest must be 'auto', 'bytes', 'pixels' or None, "
                         f"got {digest!r}")

    ids = ([str(r) for r in df["rel_path"]] if "rel_path" in df.columns
           else [str(p) for p in df["path"]])
    paths = [str(p) for p in df["path"]]

    missing, present_pos = [], []
    for i, p in enumerate(paths):
        if os.path.isfile(p):
            present_pos.append(i)
        else:
            missing.append(ids[i])

    to_digest = ([] if digest is None
                 else [present_pos[k]
                       for k in _sample_positions(len(present_pos), sample)])
    unreadable: list[tuple[str, str]] = []
    divergent: list[str] = []
    reencoded: list[str] = []
    escalated = False
    if to_digest:
        want_pixels = digest == "pixels"
        expected = df["content_sha256" if digest == "bytes"
                      else "pixel_sha256"].to_numpy()
        results = _digest_many([paths[i] for i in to_digest], want_pixels, workers)
        for i, (err, content, pixel) in zip(to_digest, results):
            if err is not None:
                unreadable.append((ids[i], err))
                continue
            got = content if digest == "bytes" else pixel
            if got == str(expected[i]):
                continue
            divergent.append(ids[i])
            if digest == "bytes" and has_pixels:
                # Escalate only the rows that already failed: this is the
                # difference between "re-download, it is only a re-encode" and
                # "stop, the pixels are not the same".
                escalated = True
                try:
                    _, _, px = _digest_one(paths[i], want_pixels=True)
                except Exception as exc:                     # pragma: no cover
                    unreadable.append((ids[i], repr(exc)))
                    continue
                if px == str(df["pixel_sha256"].to_numpy()[i]):
                    reencoded.append(ids[i])

    extra: list[str] = []
    if check_extra and root is not None and os.path.isdir(root):
        named = set(ids) if "rel_path" in df.columns else set()
        if named:
            extra = sorted(_scan_tree(root) - named)

    return VerifyReport(
        root=root, n_rows=len(df), n_digested=len(to_digest),
        digest_kind=digest,
        missing=missing[:MAX_EXAMPLES], unreadable=unreadable[:MAX_EXAMPLES],
        divergent=divergent[:MAX_EXAMPLES], reencoded=reencoded[:MAX_EXAMPLES],
        extra=extra[:MAX_EXAMPLES],
        n_missing=len(missing), n_unreadable=len(unreadable),
        n_divergent=len(divergent), n_reencoded=len(reencoded),
        n_extra=len(extra),
        sampled=sample is not None and len(to_digest) < len(present_pos),
        escalated=escalated,
    )


def _digest_one(path: str, want_pixels: bool):
    try:
        content, pixel = digest_row(path, want_pixels=want_pixels)
    except Exception as exc:
        return repr(exc), "", ""
    return None, content, pixel


def _digest_many(paths: list[str], want_pixels: bool, workers: int):
    if workers <= 1 or len(paths) < 2:
        return [_digest_one(p, want_pixels) for p in paths]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: _digest_one(p, want_pixels), paths))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check a materialised dataset against its frozen manifest, "
                    "before spending GPU-hours extracting features from it.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None,
                    help="where the dataset actually is on this machine "
                         "(Kaggle: /kaggle/input/<slug>/...)")
    ap.add_argument("--digest", default="auto",
                    choices=["auto", "bytes", "pixels", "none"])
    ap.add_argument("--sample", type=int, default=None,
                    help="digest only this many evenly spaced rows")
    ap.add_argument("--no-extra-scan", action="store_true",
                    help="skip the walk of the tree looking for unlisted files")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    df = read_manifest(a.manifest, root=a.root)
    report = verify_images(df, root=a.root,
                           digest=None if a.digest == "none" else a.digest,
                           sample=a.sample, check_extra=not a.no_extra_scan,
                           workers=a.workers)
    print(report.describe())
    return 0 if report.ok else 1


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
