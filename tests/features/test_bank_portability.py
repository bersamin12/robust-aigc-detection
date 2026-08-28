"""The manifest fingerprint is an identity, not an address.

The bank indexes the manifest POSITIONALLY, and `manifest_sha256` is what
makes "this bank was built from that manifest" checkable. It used to be a
hash of the absolute `path` column, which made it an address: the same rows,
the same order, the same images, mounted somewhere else, fingerprinted
differently. That is fatal for this project's plan -- normalise once, publish
as Kaggle Datasets, extract bank shards in several Kaggle sessions where the
data lives under /kaggle/input/<slug>/ -- because every shard would be
refused by `verify_against_manifest` and `merge_banks` would produce a merged
fingerprint matching nothing.

These tests pin the four properties that distinction rests on: reorder,
rename and insert/delete still change the fingerprint; moving the dataset
does not; and shards extracted under different roots merge.
"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import (
    MANIFEST_COLUMNS,
    add_identity,
    read_manifest,
    write_manifest,
)
from aigcdet.features.bank import (
    FeatureBank,
    manifest_fingerprint,
    merge_banks,
)


def _tree(root, n=4):
    """A nested dataset on disk with an authored (identity-free) manifest."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        bucket = "real" if i % 2 == 0 else "sdxl"
        p = os.path.abspath(os.path.join(root, bucket, f"img_{i:03d}.png"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
        rows.append({"path": p, "label": i % 2,
                     "generator": "" if i % 2 == 0 else "sdxl",
                     "source": "wildfake", "licence": "CC0", "width": 64,
                     "height": 64, "split": "train"})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _frozen(tmp_path, name="normalized", n=4):
    root = str(tmp_path / name)
    df = _tree(root, n)
    m = str(tmp_path / f"{name}.parquet")
    write_manifest(df, m)
    return root, m, read_manifest(m)


def _fake_backbone(monkeypatch):
    """No GPU, no weights: a stub embedder, so the real extract_bank path --
    the one that records manifest_root -- is what these tests exercise."""
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        extract, "embed",
        lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim, float(np.asarray(i).mean()), np.float32)
                      for i in imgs]))
    return extract


# --- what the fingerprint must and must not notice -------------------------

def test_reordering_two_rows_changes_the_fingerprint(tmp_path):
    _, _, df = _frozen(tmp_path)
    swapped = df.iloc[[1, 0, 2, 3]]
    assert manifest_fingerprint(swapped) != manifest_fingerprint(df)
    # ...and it is the ORDER, not the index labels, that did it.
    assert manifest_fingerprint(df.set_index(pd.Index([9, 8, 7, 6]))) == \
        manifest_fingerprint(df)


def test_inserting_or_deleting_a_row_changes_the_fingerprint(tmp_path):
    _, _, df = _frozen(tmp_path)
    assert manifest_fingerprint(df.iloc[:3]) != manifest_fingerprint(df)
    doubled = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert manifest_fingerprint(doubled) != manifest_fingerprint(df)


def test_renaming_a_file_inside_the_dataset_changes_the_fingerprint(tmp_path):
    """The fingerprint stops caring WHERE the dataset is; it must not stop
    caring what is in it."""
    root, _, df = _frozen(tmp_path)
    authored = df[MANIFEST_COLUMNS].copy()
    old = str(authored.loc[1, "path"])
    new = os.path.join(os.path.dirname(old), "renamed.png")
    shutil.move(old, new)
    authored.loc[1, "path"] = new
    renamed = add_identity(authored, root=root)

    assert list(renamed["path"]) != list(df["path"])
    assert manifest_fingerprint(renamed) != manifest_fingerprint(df)


def test_moving_the_whole_dataset_does_not_change_the_fingerprint(tmp_path):
    """The headline property. Checked two ways: rebasing the frozen manifest,
    and freezing a fresh manifest over a copy of the tree at another root --
    the second is what a teammate would actually end up with if they rebuilt
    it, and it must land on the same identity."""
    root, m, df = _frozen(tmp_path)
    other = str(tmp_path / "kaggle_input" / "slug")
    shutil.copytree(root, other)

    rebased = read_manifest(m, root=other)
    assert list(rebased["path"]) != list(df["path"])
    assert manifest_fingerprint(rebased) == manifest_fingerprint(df)

    refrozen = add_identity(rebased[MANIFEST_COLUMNS])
    assert manifest_fingerprint(refrozen) == manifest_fingerprint(df)

    # The control: this is exactly what the old, path-based fingerprint did,
    # and it is why every Kaggle shard was refused.
    assert manifest_fingerprint(df.drop(columns=["rel_path"])) != \
        manifest_fingerprint(rebased.drop(columns=["rel_path"]))


def test_two_row_sequences_that_concatenate_alike_fingerprint_differently():
    """The identity strings are hashed with a separator, so the hash is of a
    LIST of paths and not of one long string. Without it, `ab/` + `c.png` and
    `a/` + `bc.png` are the same bytes -- two different datasets, one
    fingerprint, and a bank that would verify against the wrong manifest."""
    a = pd.DataFrame({"rel_path": ["ab", "c.png"]})
    b = pd.DataFrame({"rel_path": ["a", "bc.png"]})
    assert manifest_fingerprint(a) != manifest_fingerprint(b)


def test_a_frame_with_no_rel_path_still_fingerprints_by_path(tmp_path):
    """Ad-hoc frames (and banks written without a manifest_root) fall back to
    absolute paths: as strong on one machine, merely not portable."""
    a = pd.DataFrame({"path": ["/a.png", "/b.png"]})
    assert manifest_fingerprint(a) == manifest_fingerprint(a.copy())
    assert manifest_fingerprint(a) != manifest_fingerprint(
        pd.DataFrame({"path": ["/b.png", "/a.png"]}))


# --- banks -----------------------------------------------------------------

def test_a_bank_records_the_root_it_was_extracted_under(tmp_path, monkeypatch):
    extract = _fake_backbone(monkeypatch)
    root, _, df = _frozen(tmp_path)
    b = FeatureBank.open(extract.extract_bank(
        df, "fake", str(tmp_path / "bank"), n_views=3, seed=5, device="cpu"))
    assert b.config["manifest_root"] == os.path.abspath(root)
    assert list(b.rel_paths) == list(df["rel_path"])
    assert all(os.path.isabs(p) for p in b.meta["path"])


def test_a_bank_extracted_under_one_root_verifies_against_another(
        tmp_path, monkeypatch):
    extract = _fake_backbone(monkeypatch)
    root, m, df = _frozen(tmp_path)
    other = str(tmp_path / "elsewhere")
    shutil.copytree(root, other)

    b = FeatureBank.open(extract.extract_bank(
        df, "fake", str(tmp_path / "bank"), n_views=3, seed=5, device="cpu"))
    moved = read_manifest(m, root=other)
    b.verify_against_manifest(moved)              # must not raise
    b.verify_against_manifest(df)

    # It must still refuse a manifest that is genuinely a different one.
    other_rows = df.iloc[[1, 0, 2, 3]]
    with pytest.raises(ValueError, match="not the manifest the bank was built from"):
        b.verify_against_manifest(other_rows)


def test_shards_extracted_under_different_roots_merge_into_one_bank(
        tmp_path, monkeypatch):
    """The whole point: one frozen manifest, one published dataset, two
    teammates whose copies are mounted at different absolute paths, one merged
    bank that verifies against the manifest either of them holds."""
    extract = _fake_backbone(monkeypatch)
    root_a, m, df_a = _frozen(tmp_path, n=4)
    root_b = str(tmp_path / "kaggle_input" / "slug")
    shutil.copytree(root_a, root_b)
    df_b = read_manifest(m, root=root_b)

    s0 = extract.extract_bank(df_a.iloc[:2], "fake", str(tmp_path / "shard_a"),
                              n_views=3, seed=7, device="cpu")
    s1 = extract.extract_bank(df_b.iloc[2:], "fake", str(tmp_path / "shard_b"),
                              n_views=3, seed=7, device="cpu")
    b0, b1 = FeatureBank.open(s0), FeatureBank.open(s1)

    # The two shards really were extracted from two different mounts...
    assert b0.config["manifest_root"] != b1.config["manifest_root"]
    assert all(str(p).startswith(root_a) for p in b0.meta["path"])
    assert all(str(p).startswith(root_b) for p in b1.meta["path"])
    # ...and each verifies against its slice of the manifest as either
    # teammate holds it.
    b1.verify_against_manifest(df_a.iloc[2:])
    b1.verify_against_manifest(df_b.iloc[2:])

    merged = FeatureBank.open(merge_banks([s0, s1], str(tmp_path / "merged")))
    merged.check_invariants()
    assert len(merged.meta) == 4
    assert merged.config["manifest_sha256"] == manifest_fingerprint(df_a)
    assert manifest_fingerprint(df_a) == manifest_fingerprint(df_b)
    merged.verify_against_manifest(df_a)
    merged.verify_against_manifest(df_b)
    # The merged bank has no single root -- and does not need one, because
    # every row carries its own identity.
    assert merged.config["manifest_root"] is None
    assert list(merged.rel_paths) == list(df_a["rel_path"])


def test_merging_preserves_each_shards_own_rel_path(tmp_path, monkeypatch):
    """A merged bank must not re-derive rel_path from its own (absent) root:
    that would overwrite four correct identities with four absolute paths and
    silently make the merged fingerprint unmatchable."""
    extract = _fake_backbone(monkeypatch)
    root_a, m, df_a = _frozen(tmp_path, n=4)
    root_b = str(tmp_path / "b_root")
    shutil.copytree(root_a, root_b)
    df_b = read_manifest(m, root=root_b)

    s0 = extract.extract_bank(df_a.iloc[:2], "fake", str(tmp_path / "sa"),
                              n_views=3, seed=7, device="cpu")
    s1 = extract.extract_bank(df_b.iloc[2:], "fake", str(tmp_path / "sb"),
                              n_views=3, seed=7, device="cpu")
    merged = FeatureBank.open(merge_banks([s0, s1], str(tmp_path / "m2")))
    assert all(not os.path.isabs(r) for r in merged.rel_paths)
    assert list(merged.meta["rel_path"]) == list(df_a["rel_path"])


def test_a_bank_written_without_a_root_still_verifies_against_its_manifest(
        tmp_path):
    """`aigcdet.eval.grid` builds its eval banks with BankWriter directly and
    passes no manifest_root, so their rows hold ABSOLUTE paths. Verifying such
    a bank against a frozen manifest must compare absolute against absolute --
    comparing the manifest's rel_path against the bank's absolute path would
    report every row misaligned when nothing at all is wrong."""
    from aigcdet.features.bank import N_VIEWS, BankWriter

    _, _, df = _frozen(tmp_path, n=3)
    out = str(tmp_path / "eval_bank")
    w = BankWriter(out, n_images=3, n_views=N_VIEWS, dim=3, backbone="test",
                   seed=0, manifest_sha256=manifest_fingerprint(df))
    for i, row in enumerate(df.itertuples()):
        pres = np.zeros((N_VIEWS, 6), np.float32)
        pres[1:, 0] = 1.0
        w.write_image(i, {"path": row.path, "label": int(row.label),
                          "generator": row.generator, "source": row.source,
                          "split": row.split},
                      feats=np.zeros((N_VIEWS, 3), np.float32), presence=pres,
                      severity=np.zeros((N_VIEWS, 6), np.float32),
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS, row_id=i)
    w.close()

    b = FeatureBank.open(out)
    assert b.config["manifest_root"] is None
    b.verify_against_manifest(df)                 # must not raise
    with pytest.raises(ValueError, match="misaligned"):
        b.verify_against_manifest(df.iloc[[1, 0, 2]].assign(
            rel_path=list(df["rel_path"])))


def test_shards_from_the_same_root_keep_it_on_the_merged_bank(
        tmp_path, monkeypatch):
    extract = _fake_backbone(monkeypatch)
    root, _, df = _frozen(tmp_path, n=4)
    s0 = extract.extract_bank(df.iloc[:2], "fake", str(tmp_path / "s0"),
                              n_views=3, seed=7, device="cpu")
    s1 = extract.extract_bank(df.iloc[2:], "fake", str(tmp_path / "s1"),
                              n_views=3, seed=7, device="cpu")
    merged = FeatureBank.open(merge_banks([s0, s1], str(tmp_path / "m3")))
    assert merged.config["manifest_root"] == os.path.abspath(root)
