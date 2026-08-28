"""verify_images: is the dataset on THIS machine the one the manifest was
frozen against, and if not, in which of the three ways that matter.

This is the check a teammate runs on Kaggle after attaching the published
Dataset and before spending 8-13 GPU-hours extracting a feature-bank shard
from it. Each test below pins one of the answers its output has to
distinguish, because the remedies differ: re-attach, re-download,
re-normalise, or stop.
"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import (
    MANIFEST_COLUMNS,
    read_manifest,
    write_manifest,
)
from aigcdet.data.verify import verify_images

BASE = (np.arange(16 * 16 * 3) % 256).astype(np.uint8).reshape(16, 16, 3)


def _shifted(arr, by):
    return ((arr.astype(np.int32) + by) % 256).astype(np.uint8)


def _save(path, arr, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, **kw)


def _dataset(root, n=4):
    """A small nested dataset on disk plus its authored manifest rows."""
    rows = []
    for i in range(n):
        bucket = "real" if i % 2 == 0 else "sdxl"
        p = os.path.abspath(os.path.join(root, bucket, f"img_{i:03d}.png"))
        arr = _shifted(BASE, i)
        _save(p, arr)
        rows.append({"path": p, "label": i % 2,
                     "generator": "" if i % 2 == 0 else "sdxl",
                     "source": "wildfake", "licence": "CC0",
                     "width": 16, "height": 16, "split": "train"})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _frozen(tmp_path, n=4, digests="bytes"):
    root = str(tmp_path / "normalized")
    df = _dataset(root, n)
    m = str(tmp_path / "m.parquet")
    write_manifest(df, m, digests=digests)
    return root, m, read_manifest(m)


# --- the happy path --------------------------------------------------------

def test_an_untouched_dataset_verifies_clean(tmp_path):
    root, _, df = _frozen(tmp_path)
    r = verify_images(df, root=root)
    assert r.ok and r.n_fatal == 0
    assert (r.n_missing, r.n_divergent, r.n_extra, r.n_unreadable) == (0, 0, 0, 0)
    assert r.n_digested == 4 and r.digest_kind == "bytes"
    assert "Safe to extract" in r.describe()
    r.raise_for_status()


def test_a_dataset_moved_to_another_root_verifies_clean_there(tmp_path):
    """The Kaggle case: the frozen manifest's absolute paths do not exist on
    this machine, and that is not a problem with the data.

    The tree is MOVED, not copied. With a copy, `root=` could be ignored
    entirely and the check would still pass against the original files -- the
    property would be unreachable and the test would prove nothing.
    """
    root, _, df = _frozen(tmp_path)
    other = str(tmp_path / "kaggle_input" / "slug")
    shutil.move(root, other)
    assert not os.path.exists(root)

    assert verify_images(df, root=other).ok
    # Without the root, the frozen manifest's absolute paths are stale, and
    # that is reported as every row missing rather than passing quietly.
    stale = verify_images(df)
    assert not stale.ok and stale.n_missing == stale.n_rows


# --- one changed pixel -----------------------------------------------------

def test_a_single_changed_pixel_is_reported_as_divergent(tmp_path):
    root, _, df = _frozen(tmp_path)
    target = df.loc[2, "path"]
    original = np.array(Image.open(target))
    changed = original.copy()
    changed[7, 9, 2] = (int(changed[7, 9, 2]) + 1) % 256

    # The fixture must actually reach the property under test: exactly one
    # pixel, one channel, differs by one -- and the file's bytes differ too.
    assert (changed != original).sum() == 1
    before = open(target, "rb").read()
    _save(target, changed)
    assert open(target, "rb").read() != before

    r = verify_images(df, root=root)
    assert not r.ok
    assert r.n_divergent == 1 and r.divergent == [df.loc[2, "rel_path"]]
    assert r.n_missing == 0 and r.n_extra == 0
    assert "content-divergent  1" in r.describe()
    with pytest.raises(ValueError, match="differ in BYTES"):
        r.raise_for_status()


def test_a_changed_pixel_is_reported_against_the_pixel_digest_too(tmp_path):
    """With a manifest frozen on pixels, the same corruption is named as what
    it is -- the pixels a model would see are not the manifest's pixels -- and
    the advice is to stop, not to re-download."""
    root, _, df = _frozen(tmp_path, digests="pixels")
    target = df.loc[1, "path"]
    changed = np.array(Image.open(target))
    changed[0, 0, 0] = (int(changed[0, 0, 0]) + 17) % 256
    _save(target, changed)

    r = verify_images(df, root=root, digest="pixels")
    assert not r.ok and r.divergent == [df.loc[1, "rel_path"]]
    assert r.n_reencoded == 0
    assert "STOP" in r.describe() and "DECODE to different pixels" in r.describe()


def test_a_lossless_re_encode_is_a_warning_not_a_stop(tmp_path):
    """Byte-divergent, pixel-identical. The byte digest flags it (that is the
    price of not decoding); the escalation to the stored pixel digest is what
    stops it from sending a teammate off to re-normalise for nothing."""
    root, _, df = _frozen(tmp_path, digests="pixels")
    target = df.loc[0, "path"]
    arr = np.array(Image.open(target))
    before = open(target, "rb").read()
    _save(target, arr, compress_level=9, optimize=True)
    assert open(target, "rb").read() != before      # the fixture reaches it

    r = verify_images(df, root=root)                # auto -> bytes first
    assert r.digest_kind == "bytes"
    assert r.n_divergent == 1 and r.n_reencoded == 1
    assert r.ok, r.describe()
    assert "re-encoded but pixel-identical" in r.describe()


def test_a_byte_mismatch_that_is_also_a_pixel_mismatch_is_escalated_to_a_stop(
        tmp_path):
    """The other half of the escalation. The cheap byte check flags the row;
    the pixel digest then says the pixels really did change, so the report
    must say STOP rather than "re-run with digest='pixels' to find out" --
    it already found out."""
    root, _, df = _frozen(tmp_path, digests="pixels")
    target = df.loc[3, "path"]
    arr = np.array(Image.open(target))
    arr[5, 5, 0] = (int(arr[5, 5, 0]) + 9) % 256
    _save(target, arr)

    r = verify_images(df, root=root)               # auto -> bytes, then escalate
    assert r.digest_kind == "bytes" and r.escalated
    assert r.n_divergent == 1 and r.n_reencoded == 0
    assert not r.ok
    assert "STOP" in r.describe()


def test_without_pixel_digests_a_byte_divergence_cannot_be_triaged(tmp_path):
    """The honest failure: a bytes-only manifest cannot tell a harmless
    re-encode from a real change, and the advice says so instead of guessing."""
    root, _, df = _frozen(tmp_path, digests="bytes")
    arr = np.array(Image.open(df.loc[0, "path"]))
    _save(df.loc[0, "path"], arr, compress_level=9, optimize=True)
    r = verify_images(df, root=root)
    assert not r.ok and r.n_reencoded == 0
    assert "digest='pixels'" in r.describe()


# --- missing and extra are different problems ------------------------------

def test_a_missing_file_is_reported_as_missing(tmp_path):
    root, _, df = _frozen(tmp_path)
    os.remove(df.loc[3, "path"])
    r = verify_images(df, root=root)
    assert not r.ok
    assert r.n_missing == 1 and r.missing == [df.loc[3, "rel_path"]]
    assert r.n_extra == 0 and r.n_divergent == 0
    assert "the copy is incomplete" in r.describe()


def test_an_extra_file_is_reported_as_extra_and_is_not_fatal(tmp_path):
    root, _, df = _frozen(tmp_path)
    _save(os.path.join(root, "real", "stowaway.png"), BASE)
    r = verify_images(df, root=root)
    assert r.n_extra == 1 and r.extra == [os.path.join("real", "stowaway.png")]
    assert r.n_missing == 0 and r.n_divergent == 0
    # Nothing reads it, so it cannot change a feature: a warning, not a stop.
    assert r.ok
    assert "not named by the manifest" in r.describe()


def test_a_rename_inside_the_dataset_is_one_missing_and_one_extra(tmp_path):
    root, _, df = _frozen(tmp_path)
    old = df.loc[1, "path"]
    shutil.move(old, os.path.join(os.path.dirname(old), "renamed.png"))
    r = verify_images(df, root=root)
    assert r.missing == [df.loc[1, "rel_path"]]
    assert r.extra == [os.path.join("sdxl", "renamed.png")]
    assert not r.ok
    assert "different layout" in r.describe()


def test_a_dataset_attached_at_the_wrong_root_says_so(tmp_path):
    """All rows missing means "you pointed me at the wrong directory", not
    "your data is corrupt" -- and the two remedies are nothing alike."""
    root, _, df = _frozen(tmp_path)
    wrong = str(tmp_path / "empty")
    os.makedirs(wrong)
    r = verify_images(df, root=wrong)
    assert r.n_missing == r.n_rows == 4 and not r.ok
    assert "the dataset is not at this root" in r.describe()
    assert "Nothing here needs re-normalising" in r.describe()


def test_a_truncated_file_is_unreadable_rather_than_divergent(tmp_path):
    root, _, df = _frozen(tmp_path, digests="pixels")
    with open(df.loc[2, "path"], "r+b") as f:
        f.truncate(20)
    r = verify_images(df, root=root, digest="pixels")
    assert r.n_unreadable == 1 and r.unreadable[0][0] == df.loc[2, "rel_path"]
    assert not r.ok
    assert "truncated download" in r.describe()


# --- what can and cannot be checked ----------------------------------------

def test_a_manifest_frozen_without_digests_checks_presence_only(tmp_path):
    root, _, df = _frozen(tmp_path, digests=None)
    arr = np.array(Image.open(df.loc[0, "path"]))
    _save(df.loc[0, "path"], _shifted(arr, 3))      # content change, unseen
    r = verify_images(df, root=root)
    assert r.digest_kind is None and r.n_digested == 0
    assert r.ok
    assert "No content digests were compared" in r.describe()


def test_asking_for_pixels_against_a_bytes_only_manifest_is_refused(tmp_path):
    root, _, df = _frozen(tmp_path, digests="bytes")
    with pytest.raises(ValueError, match="carries no pixel_sha256"):
        verify_images(df, root=root, digest="pixels")


def test_asking_for_bytes_against_a_digestless_manifest_is_refused(tmp_path):
    root, _, df = _frozen(tmp_path, digests=None)
    with pytest.raises(ValueError, match="carries no content_sha256"):
        verify_images(df, root=root, digest="bytes")


def test_sampling_checks_fewer_rows_and_says_the_rest_is_unproven(tmp_path):
    root, _, df = _frozen(tmp_path, n=8)
    r = verify_images(df, root=root, sample=3)
    assert r.n_digested == 3 and r.sampled
    assert r.n_rows == 8 and r.ok
    assert "SAMPLE" in r.describe()
    # Presence is still checked for every row, sample or not.
    os.remove(df.loc[7, "path"])
    assert verify_images(df, root=root, sample=1).n_missing == 1


def test_sampling_is_deterministic_so_two_teammates_compare_notes(tmp_path):
    root, _, df = _frozen(tmp_path, n=8)
    a = verify_images(df, root=root, sample=3)
    b = verify_images(df, root=root, sample=3)
    assert a.n_digested == b.n_digested == 3


def test_the_extra_scan_can_be_skipped(tmp_path):
    root, _, df = _frozen(tmp_path)
    _save(os.path.join(root, "real", "stowaway.png"), BASE)
    assert verify_images(df, root=root, check_extra=False).n_extra == 0


def test_the_cli_returns_non_zero_when_the_dataset_is_wrong(tmp_path, capsys):
    from aigcdet.data.verify import main

    root, m, df = _frozen(tmp_path)
    assert main(["--manifest", m, "--root", root]) == 0
    os.remove(df.loc[0, "path"])
    assert main(["--manifest", m, "--root", root]) == 1
    assert "FAILED" in capsys.readouterr().out
