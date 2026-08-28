"""The manifest's portable identity: rel_path, the derived root, and the
per-row content digests write_manifest stamps at freeze time.

The defect these exist for: `manifest_fingerprint` hashed the ABSOLUTE `path`
column, so a feature-bank shard extracted on Kaggle -- where the same images
live under /kaggle/input/<slug>/ -- fingerprinted differently from the
manifest it was built from, and `FeatureBank.verify_against_manifest` refused
it. The identity of a manifest row is which file it names INSIDE the dataset,
not which machine the dataset happens to be mounted on.
"""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import (
    MANIFEST_COLUMNS,
    MANIFEST_IDENTITY_COLUMNS,
    ROOT_ENV_VAR,
    add_identity,
    compute_digests,
    dataset_root,
    derive_root,
    file_digest,
    pixel_digest,
    read_manifest,
    rebase_manifest,
    relative_to_root,
    root_of,
    validate_manifest,
    write_manifest,
)


def _tree(root, n=4, nested=True):
    """A tiny two-bucket dataset on disk, returning an authored manifest."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        bucket = "real" if i % 2 == 0 else "sdxl"
        d = os.path.join(root, bucket) if nested else root
        os.makedirs(d, exist_ok=True)
        p = os.path.abspath(os.path.join(d, f"img_{i:03d}.png"))
        Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(p)
        rows.append({"path": p, "label": i % 2, "generator": "" if i % 2 == 0 else "sdxl",
                     "source": "wildfake", "licence": "CC0", "width": 8,
                     "height": 8, "split": "train"})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


# --- rel_path and the derived root -----------------------------------------

def test_add_identity_records_paths_relative_to_the_derived_root(tmp_path):
    root = str(tmp_path / "normalized")
    df = add_identity(_tree(root))
    assert dataset_root(df) == os.path.abspath(root)
    assert sorted(df["rel_path"]) == [
        os.path.join("real", "img_000.png"), os.path.join("real", "img_002.png"),
        os.path.join("sdxl", "img_001.png"), os.path.join("sdxl", "img_003.png"),
    ]
    for p, r in zip(df["path"], df["rel_path"]):
        assert os.path.isabs(p) and not os.path.isabs(r)


def test_derive_root_is_the_deepest_directory_holding_every_image(tmp_path):
    root = str(tmp_path / "normalized")
    df = _tree(root)
    assert derive_root(df["path"]) == os.path.abspath(root)
    # One bucket only -> the root derived is that bucket, and the identity is
    # still consistent, just rooted deeper.
    only_real = df[df["generator"] == ""]
    assert derive_root(only_real["path"]) == os.path.join(os.path.abspath(root), "real")


def test_an_explicit_root_pins_a_shallower_one(tmp_path):
    root = str(tmp_path / "normalized")
    df = _tree(root)
    pinned = add_identity(df[df["generator"] == ""], root=str(tmp_path))
    assert list(pinned["rel_path"]) == [
        os.path.join("normalized", "real", "img_000.png"),
        os.path.join("normalized", "real", "img_002.png"),
    ]


def test_root_of_refuses_a_path_that_does_not_end_with_its_rel_path():
    assert root_of("/data/norm/real/a.png", "real/a.png") == "/data/norm"
    with pytest.raises(ValueError, match="does not end with rel_path"):
        root_of("/data/norm/real/a.png", "fake/a.png")


def test_dataset_root_refuses_rows_that_imply_two_roots():
    df = pd.DataFrame({"path": ["/a/real/x.png", "/b/real/y.png"],
                       "rel_path": ["real/x.png", "real/y.png"]})
    with pytest.raises(ValueError, match="different dataset roots"):
        dataset_root(df)


def test_dataset_root_is_none_without_rel_path():
    assert dataset_root(pd.DataFrame({"path": ["/a/x.png"]})) is None


def test_relative_to_root_refuses_an_image_outside_the_root():
    with pytest.raises(ValueError, match="not under the dataset root"):
        relative_to_root(["/elsewhere/x.png"], "/data/norm")


def test_rebase_manifest_moves_path_and_leaves_identity_alone(tmp_path):
    df = add_identity(_tree(str(tmp_path / "normalized")))
    moved = rebase_manifest(df, "/kaggle/input/slug")
    assert list(moved["rel_path"]) == list(df["rel_path"])
    assert all(p.startswith("/kaggle/input/slug/") for p in moved["path"])
    assert list(df["path"]) != list(moved["path"])


def test_rebase_refuses_a_frame_with_no_identity():
    with pytest.raises(ValueError, match="no rel_path column"):
        rebase_manifest(pd.DataFrame({"path": ["/a/x.png"]}), "/b")


# --- freeze / read ---------------------------------------------------------

def test_write_manifest_stamps_identity_on_disk_and_on_the_callers_frame(tmp_path):
    df = _tree(str(tmp_path / "normalized"))
    out = str(tmp_path / "m.parquet")
    write_manifest(df, out)
    # In place, so build_dataset's "returns the DataFrame it wrote" is true.
    assert list(df.columns) == MANIFEST_COLUMNS + MANIFEST_IDENTITY_COLUMNS
    back = read_manifest(out)
    assert list(back["rel_path"]) == list(df["rel_path"])
    assert list(back["content_sha256"]) == list(df["content_sha256"])
    validate_manifest(back)


def test_read_manifest_rebases_onto_a_root_given_explicitly(tmp_path):
    df = _tree(str(tmp_path / "normalized"))
    out = str(tmp_path / "m.parquet")
    write_manifest(df, out)
    back = read_manifest(out, root="/kaggle/input/slug")
    assert list(back["path"]) == [os.path.join("/kaggle/input/slug", r)
                                  for r in df["rel_path"]]


def test_read_manifest_rebases_onto_the_environments_root(tmp_path, monkeypatch):
    """scripts/extract_features.py takes no --root flag; a Kaggle session must
    still be able to point it at /kaggle/input/<slug>."""
    df = _tree(str(tmp_path / "normalized"))
    out = str(tmp_path / "m.parquet")
    write_manifest(df, out)
    monkeypatch.setenv(ROOT_ENV_VAR, "/mnt/other")
    assert all(p.startswith("/mnt/other/") for p in read_manifest(out)["path"])
    monkeypatch.delenv(ROOT_ENV_VAR)
    assert list(read_manifest(out)["path"]) == list(df["path"])


def test_read_manifest_refuses_to_rebase_a_manifest_without_identity(tmp_path):
    """Matched on read_manifest's OWN message, which names the file and the
    fix. Matching the generic "no rel_path column" that rebase_manifest raises
    would pass with this guard deleted, and the caller would be told a
    DataFrame is wrong rather than that their manifest predates portability."""
    df = _tree(str(tmp_path / "normalized"))
    out = str(tmp_path / "old.parquet")
    df[MANIFEST_COLUMNS].to_parquet(out, index=False)
    with pytest.raises(ValueError, match="Rebuild it with a current write_manifest"):
        read_manifest(out, root="/kaggle/input/slug")


# --- content digests -------------------------------------------------------

def test_byte_and_pixel_digests_answer_different_questions(tmp_path):
    """A lossless re-save changes the bytes and not one pixel. That is the
    whole reason both digests exist: the byte digest is the cheap check that
    can never MISS a pixel change, and the pixel digest is what says whether a
    byte change actually matters."""
    arr = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    a = str(tmp_path / "a.png")
    b = str(tmp_path / "b.png")
    Image.fromarray(arr).save(a, compress_level=1)
    Image.fromarray(arr).save(b, compress_level=9)

    assert file_digest(a) != file_digest(b)        # the fixture reaches it
    assert pixel_digest(a) == pixel_digest(b)

    changed = arr.copy()
    changed[3, 4, 1] = (int(changed[3, 4, 1]) + 40) % 256
    c = str(tmp_path / "c.png")
    Image.fromarray(changed).save(c, compress_level=1)
    assert pixel_digest(c) != pixel_digest(a)
    assert file_digest(c) != file_digest(a)        # bytes never miss it


def test_pixel_digest_covers_shape_and_mode_not_just_bytes(tmp_path):
    """Two images whose raw pixel buffers could be confused must not share a
    digest: 4x2 and 2x4 grey hold the same bytes in the same order."""
    arr = np.arange(8, dtype=np.uint8).reshape(4, 2)
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    Image.fromarray(arr, mode="L").save(a)
    Image.fromarray(arr.reshape(2, 4), mode="L").save(b)
    assert pixel_digest(a) != pixel_digest(b)


def test_write_manifest_with_pixel_digests_fills_both_columns(tmp_path):
    df = _tree(str(tmp_path / "normalized"))
    write_manifest(df, str(tmp_path / "m.parquet"), digests="pixels")
    assert all(len(v) == 64 for v in df["content_sha256"])
    assert all(len(v) == 64 for v in df["pixel_sha256"])


def test_the_default_freeze_is_bytes_only_because_pixels_cost_a_decode(tmp_path):
    df = _tree(str(tmp_path / "normalized"))
    write_manifest(df, str(tmp_path / "m.parquet"))
    assert all(len(v) == 64 for v in df["content_sha256"])
    assert set(df["pixel_sha256"]) == {""}


def test_digests_none_leaves_both_columns_empty_and_still_validates(tmp_path):
    df = _tree(str(tmp_path / "normalized"))
    write_manifest(df, str(tmp_path / "m.parquet"), digests=None)
    assert set(df["content_sha256"]) == {""} and set(df["pixel_sha256"]) == {""}
    validate_manifest(df)


def test_add_identity_rejects_an_unknown_digest_kind(tmp_path):
    with pytest.raises(ValueError, match="digests must be"):
        add_identity(_tree(str(tmp_path / "n")), digests="md5")


def test_digesting_a_path_that_does_not_exist_is_a_loud_failure(tmp_path):
    with pytest.raises(FileNotFoundError, match="do not exist and cannot be"):
        compute_digests([str(tmp_path / "nope.png")])


def test_compute_digests_preserves_row_order_under_threading(tmp_path):
    root = str(tmp_path / "normalized")
    df = _tree(root, n=12)
    serial = compute_digests(df["path"], workers=1)
    threaded = compute_digests(df["path"], workers=8)
    assert threaded == serial
    assert len(set(c for c, _ in serial)) == 12


# --- validation of the identity columns ------------------------------------

def _frozen(tmp_path, **overrides):
    df = _tree(str(tmp_path / "normalized"))
    write_manifest(df, str(tmp_path / "m.parquet"))
    for k, v in overrides.items():
        df[k] = v
    return df


def test_validate_rejects_an_absolute_rel_path(tmp_path):
    df = _frozen(tmp_path)
    df.loc[0, "rel_path"] = os.path.abspath(df.loc[0, "path"])
    with pytest.raises(ValueError, match="rel_path.*are absolute"):
        validate_manifest(df)


def test_validate_rejects_a_rel_path_escaping_the_root(tmp_path):
    df = _frozen(tmp_path)
    df.loc[0, "rel_path"] = os.path.join("..", "elsewhere", "x.png")
    with pytest.raises(ValueError, match=r"contain '\.\.'"):
        validate_manifest(df)


def test_validate_rejects_duplicated_rel_paths(tmp_path):
    df = _frozen(tmp_path)
    df.loc[1, "rel_path"] = df.loc[0, "rel_path"]
    df.loc[1, "path"] = os.path.join(os.path.dirname(str(df.loc[1, "path"])),
                                     "..", str(df.loc[0, "rel_path"]))
    with pytest.raises(ValueError, match="duplicated rel_path"):
        validate_manifest(df)


def test_validate_rejects_a_rel_path_that_disagrees_with_its_path(tmp_path):
    df = _frozen(tmp_path)
    df.loc[0, "rel_path"] = os.path.join("sdxl", "not_this_one.png")
    with pytest.raises(ValueError, match="does not end with rel_path"):
        validate_manifest(df)


def test_validate_rejects_partial_identity_columns(tmp_path):
    df = _frozen(tmp_path).drop(columns=["pixel_sha256"])
    with pytest.raises(ValueError, match="identity columns are partial"):
        validate_manifest(df)


def test_validate_rejects_a_malformed_digest(tmp_path):
    df = _frozen(tmp_path)
    df.loc[0, "content_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="malformed content_sha256"):
        validate_manifest(df)


def test_validate_rejects_a_digest_column_filled_on_only_some_rows(tmp_path):
    """Half a digest column is worse than none: verify_images would report a
    clean run having silently skipped the rows without one."""
    df = _frozen(tmp_path)
    df.loc[0, "content_sha256"] = ""
    with pytest.raises(ValueError, match="content_sha256 is set on 3 of 4 rows"):
        validate_manifest(df)


def test_validate_still_accepts_an_authored_frame_with_no_identity_yet(tmp_path):
    """build_dataset validates the frame BEFORE freezing it, and at that point
    the identity columns do not exist yet."""
    validate_manifest(_tree(str(tmp_path / "normalized")))
