"""NTIRE restaging: the CSV becomes directories, and nothing is lost doing it.

NTIRE's class lives in a `labels.csv` column; every reader in this project
takes the class off the bucket DIRECTORY. That translation happens exactly
once, here, over 150,000 images -- so the failures worth pinning are the ones
that would produce a plausible corpus rather than an error: a silently dropped
image, a name collision that halves a shard, an unrecognised third label.
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


sn = _load_script("stage_ntire")


def _shard(tmp_path, idx, rows):
    """rows: list of (name, label). Writes images/ + labels.csv."""
    d = tmp_path / f"shard_{idx}"
    (d / "images").mkdir(parents=True)
    for name, _ in rows:
        (d / "images" / name).write_bytes(name.encode())
    pd.DataFrame({"image_name": [n for n, _ in rows],
                  "label": [l for _, l in rows]}).to_csv(d / "labels.csv", index=False)
    return d


def test_labels_become_buckets_and_every_row_is_staged(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("a.jpg", 0), ("b.jpg", 1), ("c.jpg", 1)])
    out = sn.main(["--src", str(src), "--dest", str(dest)])
    assert out["by_bucket"] == {"real": 1, "generated": 2}
    assert sorted(os.listdir(dest / "ntire" / "real")) == ["a.jpg"]
    assert sorted(os.listdir(dest / "ntire" / "generated")) == ["b.jpg", "c.jpg"]


def test_the_mapping_is_the_cards_and_is_stated_not_inferred():
    """0 = real, 1 = generated. An inverted 150,000-row corpus trains happily
    and produces a confidently wrong detector, so the constant is asserted
    rather than left to be read off an if-statement."""
    assert sn.BUCKET_FOR_LABEL == {0: "real", 1: "generated"}


def test_images_are_hardlinked_not_copied(tmp_path):
    """60 GB already on the filesystem must not be duplicated."""
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("a.jpg", 0)])
    sn.main(["--src", str(src), "--dest", str(dest)])
    assert os.path.samefile(src / "shard_0" / "images" / "a.jpg",
                            dest / "ntire" / "real" / "a.jpg")


def test_rerun_is_idempotent_and_relinks_nothing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("a.jpg", 0), ("b.jpg", 1)])
    sn.main(["--src", str(src), "--dest", str(dest)])
    again = sn.main(["--src", str(src), "--dest", str(dest)])
    assert again["linked"] == 0
    assert again["total"] == 2


def test_a_name_collision_within_a_bucket_raises_instead_of_dropping_an_image(tmp_path):
    """Two different images under one name in one bucket would leave the
    corpus an image short with nothing in any log to say so. Same LABEL in
    both shards is what makes this a collision -- one name reused across two
    different buckets is two distinct rel_paths and is not a problem."""
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("dup.jpg", 1)])
    d1 = src / "shard_1"
    (d1 / "images").mkdir(parents=True)
    (d1 / "images" / "dup.jpg").write_bytes(b"DIFFERENT BYTES")
    pd.DataFrame({"image_name": ["dup.jpg"], "label": [1]}).to_csv(
        d1 / "labels.csv", index=False)
    with pytest.raises(FileExistsError, match="share a name"):
        sn.main(["--src", str(src), "--dest", str(dest)])


def test_one_name_reused_across_two_buckets_is_allowed(tmp_path):
    """The complement of the test above, so the collision check cannot be
    tightened into refusing a legitimate corpus: `ntire/real/x.jpg` and
    `ntire/generated/x.jpg` are two different rel_paths and both must survive."""
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("same.jpg", 0)])
    d1 = src / "shard_1"
    (d1 / "images").mkdir(parents=True)
    (d1 / "images" / "same.jpg").write_bytes(b"OTHER")
    pd.DataFrame({"image_name": ["same.jpg"], "label": [1]}).to_csv(
        d1 / "labels.csv", index=False)
    out = sn.main(["--src", str(src), "--dest", str(dest)])
    assert out["by_bucket"] == {"real": 1, "generated": 1}
    assert os.path.exists(dest / "ntire" / "real" / "same.jpg")
    assert os.path.exists(dest / "ntire" / "generated" / "same.jpg")


def test_a_row_whose_image_is_missing_raises(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    d = _shard(src, 0, [("a.jpg", 0), ("gone.jpg", 1)])
    (d / "images" / "gone.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="not on disk"):
        sn.main(["--src", str(src), "--dest", str(dest)])


def test_an_unknown_label_raises_rather_than_defaulting(tmp_path):
    """SID_Set uses 2 for TAMPERED. If NTIRE ever does, that class needs a
    decision; bucketing it as 'generated' by default would be one made
    silently."""
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("a.jpg", 0), ("t.jpg", 2)])
    with pytest.raises(ValueError, match=r"label\(s\) \[2\]"):
        sn.main(["--src", str(src), "--dest", str(dest)])


def test_dry_run_validates_without_writing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _shard(src, 0, [("a.jpg", 0), ("b.jpg", 1)])
    out = sn.main(["--src", str(src), "--dest", str(dest), "--dry-run"])
    assert out["total"] == 2
    assert not os.listdir(dest / "ntire" / "real")


def test_shards_are_ordered_numerically_not_lexically(tmp_path):
    """shard_10 must follow shard_9, or a later fetch of the remaining
    published shards silently reorders the corpus."""
    src = tmp_path / "src"
    src.mkdir()
    for i in (0, 2, 10, 9):
        _shard(src, i, [(f"{i}.jpg", 0)])
    got = [os.path.basename(d) for d in sn.shard_dirs(str(src))]
    assert got == ["shard_0", "shard_2", "shard_9", "shard_10"]


def test_the_staged_layout_is_what_classify_reads(tmp_path):
    """The point of the whole script: `<source>/<bucket>` must round-trip
    through the registry to the label the CSV carried."""
    from aigcdet.data.sources import classify

    assert classify("ntire", "real") == (0, "")
    label, generator = classify("ntire", "generated")
    assert label == 1 and generator == "ntire"
    for bucket, expected in sn.BUCKET_FOR_LABEL.items():
        assert classify("ntire", expected)[0] == bucket
