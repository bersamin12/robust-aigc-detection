"""Staging a manifest's images: rel_path is preserved, and nothing is invented.

The tree this produces is published as a Kaggle Dataset and read back by a
notebook that resolves `<mount>/<rel_path>`. Two failures would survive to
that point and cost a session: a path shifted by a directory level (every row
fails to resolve, an hour in), and a tree that is quietly short of what its
manifest claims (a bank whose rows do not exist).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import pandas as pd
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smi = _load_script("stage_manifest_images")


def _corpus(root, rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())


def _manifest(path, rels):
    pd.DataFrame({"rel_path": rels, "label": [0] * len(rels)}).to_parquet(
        path, index=False)
    return str(path)


def test_rel_path_is_preserved_exactly(tmp_path):
    """The manifest's half of the bank identity. A tree that differs by a
    directory level resolves to nothing on the far side of an upload."""
    root, dest = tmp_path / "src", tmp_path / "dest"
    rels = ["ntire/real/a.png", "wildfake/styleGAN/b.png"]
    _corpus(root, rels)
    m = _manifest(tmp_path / "m.parquet", rels)
    smi.main(["--manifest", m, "--root", str(root), "--dest", str(dest)])
    for rel in rels:
        assert (dest / rel).exists(), rel


def test_images_are_hardlinked_by_default(tmp_path):
    root, dest = tmp_path / "src", tmp_path / "dest"
    _corpus(root, ["a/b.png"])
    m = _manifest(tmp_path / "m.parquet", ["a/b.png"])
    smi.main(["--manifest", m, "--root", str(root), "--dest", str(dest)])
    assert os.path.samefile(root / "a" / "b.png", dest / "a" / "b.png")


def test_two_manifests_stage_into_one_tree_and_share_rows(tmp_path):
    """A probe needs its training rows and its eval rows in ONE Dataset, and
    the two overlap on val_internal. The overlap must be staged once, not
    raise as a collision with itself."""
    root, dest = tmp_path / "src", tmp_path / "dest"
    _corpus(root, ["x/1.png", "x/2.png", "x/3.png"])
    m1 = _manifest(tmp_path / "m1.parquet", ["x/1.png", "x/2.png"])
    m2 = _manifest(tmp_path / "m2.parquet", ["x/2.png", "x/3.png"])
    out = smi.main(["--manifest", m1, "--manifest", m2,
                    "--root", str(root), "--dest", str(dest)])
    assert out["staged"] == 3
    assert out["results"][1]["already_present"] == 1


def test_roots_are_searched_in_order_so_one_manifest_can_span_two_trees(tmp_path):
    """An eval manifest's rel_paths start with the corpus's own top-level name
    OR with `demo/`, and those live under different roots here."""
    corpus, demo, dest = tmp_path / "c", tmp_path / "d", tmp_path / "dest"
    _corpus(corpus, ["normalized_union/a.png"])
    _corpus(demo, ["demo/b.png"])
    m = _manifest(tmp_path / "m.parquet", ["normalized_union/a.png", "demo/b.png"])
    out = smi.main(["--manifest", m, "--root", str(corpus), "--root", str(demo),
                    "--dest", str(dest)])
    assert out["staged"] == 2
    assert (dest / "normalized_union" / "a.png").exists()
    assert (dest / "demo" / "b.png").exists()


def test_a_row_that_resolves_nowhere_raises_rather_than_staging_a_short_tree(tmp_path):
    root, dest = tmp_path / "src", tmp_path / "dest"
    _corpus(root, ["x/1.png"])
    m = _manifest(tmp_path / "m.parquet", ["x/1.png", "x/missing.png"])
    with pytest.raises(FileNotFoundError, match="do not resolve"):
        smi.main(["--manifest", m, "--root", str(root), "--dest", str(dest)])


def test_two_different_images_at_one_rel_path_raise(tmp_path):
    """Silently keeping the first would leave the tree disagreeing with at
    least one manifest's content digests."""
    r1, r2, dest = tmp_path / "r1", tmp_path / "r2", tmp_path / "dest"
    (r1 / "x").mkdir(parents=True); (r1 / "x" / "a.png").write_bytes(b"one")
    (r2 / "x").mkdir(parents=True); (r2 / "x" / "a.png").write_bytes(b"two")
    m1 = _manifest(tmp_path / "m1.parquet", ["x/a.png"])
    smi.main(["--manifest", m1, "--root", str(r1), "--dest", str(dest)])
    with pytest.raises(FileExistsError, match="disagree"):
        smi.main(["--manifest", m1, "--root", str(r2), "--dest", str(dest)])


def test_rerunning_stages_nothing_twice(tmp_path):
    root, dest = tmp_path / "src", tmp_path / "dest"
    _corpus(root, ["x/1.png", "x/2.png"])
    m = _manifest(tmp_path / "m.parquet", ["x/1.png", "x/2.png"])
    smi.main(["--manifest", m, "--root", str(root), "--dest", str(dest)])
    again = smi.main(["--manifest", m, "--root", str(root), "--dest", str(dest)])
    assert again["staged"] == 0


def test_copy_manifest_puts_the_manifest_beside_its_images(tmp_path):
    """The Dataset must carry the manifest it describes, or the notebook has
    images and no way to know what they are."""
    root, dest = tmp_path / "src", tmp_path / "dest"
    _corpus(root, ["x/1.png"])
    m = _manifest(tmp_path / "probe.parquet", ["x/1.png"])
    smi.main(["--manifest", m, "--root", str(root), "--dest", str(dest),
              "--copy-manifest"])
    assert (dest / "probe.parquet").exists()


def test_a_manifest_without_rel_path_is_refused(tmp_path):
    root, dest = tmp_path / "src", tmp_path / "dest"
    _corpus(root, ["x/1.png"])
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"path": [str(root / "x" / "1.png")]}).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="rel_path"):
        smi.main(["--manifest", str(p), "--root", str(root), "--dest", str(dest)])


def test_copy_mode_is_idempotent(tmp_path):
    """Re-staging a copy tree must be a no-op, not an error.

    `os.path.samefile` is False for a copy by construction, so the
    already-present branch could only ever recognise a hardlink -- and copy
    mode is precisely the expensive, cross-filesystem case where resuming a
    half-finished stage matters. `copy2` preserves mtime, so size and mtime
    identify a copy this script made; a DIFFERENT file at that path still
    collides.
    """
    src = tmp_path / "src"
    rels = [f"s/b/{i}.png" for i in range(3)]
    _corpus(src, rels)
    man = _manifest(tmp_path / "m.parquet", rels)

    dest = tmp_path / "dest"
    first = smi.stage(man, [str(src)], str(dest), "copy")
    assert (first["staged"], first["already_present"]) == (3, 0)
    second = smi.stage(man, [str(src)], str(dest), "copy")
    assert (second["staged"], second["already_present"]) == (0, 3)

    # A different image at the same rel_path is still a collision.
    (dest / rels[0]).write_bytes(b"something else entirely, and longer")
    with pytest.raises(FileExistsError):
        smi.stage(man, [str(src)], str(dest), "copy")
