"""`scripts/stage_corpus_root.py` -- one scannable root from trees that live apart.

`build_dataset` scans ONE raw root, and the `coco_crop` corpus is spread over
`data/raw/wildfake`, `data/raw/sid_set` and `/mnt/berstorage/coco/train2017`.
Linking COCO into `data/raw` instead would be simpler and wrong: `_scan` reads
the tree rather than the manifest, so a new source dropped there silently
changes what `configs/datasets/max_data.yaml` means the next time anyone
builds it.

So the tests that matter are: does the staged root read back with the right
labels, does it preserve the sub-bucket structure the licence exclusions
address, and does it cost nothing?
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from aigcdet.data.sources import classify

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load("stage_corpus_root")


def _tree(root, buckets: dict[str, int], rng, ext="png"):
    """`buckets` maps a path relative to the source root to a file count, so a
    nested `real/real_ffhq` is expressible."""
    for bucket, n in buckets.items():
        d = os.path.join(str(root), bucket)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(d, f"{i:04d}.{ext}"))
    return str(root)


def _licences(path, names=("wildfake", "sid_set", "coco_train2017")) -> str:
    from aigcdet.data.sources import LICENCES
    with open(str(path), "w") as f:
        json.dump({n: LICENCES[n] for n in names}, f)
    return str(path)


def _run(argv):
    old, sys.argv = sys.argv, ["stage_corpus_root.py", *argv]
    try:
        st.main()
    finally:
        sys.argv = old


# --------------------------------------------------------------------------
# the layouts that have to survive the round trip
# --------------------------------------------------------------------------

def test_coco_train2017_stages_from_outside_the_data_directory(tmp_path):
    """The case the script exists for. COCO ships as `train2017/train2017/`,
    i.e. already `<tree>/<bucket>/`, so it stages unchanged -- and it must read
    back as authentic, never as a generator family."""
    coco = _tree(tmp_path / "elsewhere" / "coco", {"train2017": 6},
                 np.random.default_rng(0), ext="jpg")
    out = str(tmp_path / "root")
    st.stage_source(out, "coco_train2017", coco, "hardlink")

    assert classify("coco_train2017", "train2017") == (0, "")
    assert len(os.listdir(os.path.join(out, "coco_train2017", "train2017"))) == 6


def test_the_subset_directory_below_a_bucket_is_preserved(tmp_path):
    """WildFake's authentic images are nested one level BELOW the bucket, which
    is what lets `classify` read bucket "real" while the directory records the
    upstream. Flattening them here would leave `exclude_subpaths` unable to
    name either subset -- and dropping the five non-commercial ones while
    keeping LAION is the whole point of this stream."""
    wf = _tree(tmp_path / "wildfake",
               {"real/real_ffhq": 3, "real/real_laion5b": 4, "BigGAN": 5},
               np.random.default_rng(1))
    out = str(tmp_path / "root")
    counts = st.stage_source(out, "wildfake", wf, "hardlink")

    staged = os.path.join(out, "wildfake", "real")
    assert sorted(os.listdir(staged)) == ["real_ffhq", "real_laion5b"]
    assert len(os.listdir(os.path.join(staged, "real_ffhq"))) == 3
    # The bucket count is the whole bucket, at any depth.
    assert counts["real"] == 7


def test_sid_sets_pseudo_bucket_reads_back_as_a_generated_row(tmp_path):
    sid = _tree(tmp_path / "sid_set", {"real": 3, "fake": 4},
                np.random.default_rng(2))
    out = str(tmp_path / "root")
    st.stage_source(out, "sid_set", sid, "hardlink")
    assert classify("sid_set", "fake") == (1, "sid_set")
    assert classify("sid_set", "real") == (0, "")
    assert len(os.listdir(os.path.join(out, "sid_set", "fake"))) == 4


# --------------------------------------------------------------------------
# it must cost nothing
# --------------------------------------------------------------------------

def test_hardlinked_files_are_the_same_inode(tmp_path):
    """The premise: a 48 GB tree costs inodes, not gigabytes, and nothing is
    re-encoded on the way in -- re-encoding one source and not another is how
    two classes start differing by container."""
    src = _tree(tmp_path / "src" / "wildfake", {"real/real_ffhq": 2},
                np.random.default_rng(3))
    out = str(tmp_path / "root")
    st.stage_source(out, "wildfake", src, "hardlink")

    for name in os.listdir(os.path.join(src, "real", "real_ffhq")):
        a = os.stat(os.path.join(src, "real", "real_ffhq", name))
        b = os.stat(os.path.join(out, "wildfake", "real", "real_ffhq", name))
        assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
        assert b.st_nlink >= 2


def test_symlink_mode_produces_files_the_scan_finds(tmp_path):
    """The cross-device fallback, and the reason files are linked rather than
    bucket directories: `glob` with `**` does not descend into a symlinked
    directory, but a symlink to a FILE matches an extension pattern."""
    bd = _load("build_dataset")
    src = _tree(tmp_path / "src" / "wildfake", {"real/real_ffhq": 3},
                np.random.default_rng(4))
    out = str(tmp_path / "root")
    st.stage_source(out, "wildfake", src, "symlink")

    found = bd._scan(out)
    assert len(found) == 3
    assert all(os.path.islink(p) for p in found)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_an_unreadable_bucket_layout_raises_before_anything_is_linked(tmp_path):
    """Classifying every bucket first turns a surprise 200,000 files into a
    failure in a second that names the bucket."""
    src = _tree(tmp_path / "src" / "wildfake", {"real": 2},
                np.random.default_rng(5))
    out = str(tmp_path / "root")
    with pytest.raises(ValueError, match="not_a_registered_source"):
        st.stage_source(out, "not_a_registered_source", src, "hardlink")
    assert not os.path.exists(os.path.join(out, "not_a_registered_source"))


def test_a_source_with_no_buckets_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no bucket directories"):
        st.stage_source(str(tmp_path / "root"), "wildfake", str(empty), "hardlink")


def test_staging_refuses_an_existing_root_without_force(tmp_path):
    """A half-staged root that a later run tops up is a corpus nobody can
    describe."""
    a = _tree(tmp_path / "a" / "wildfake", {"real": 1}, np.random.default_rng(6))
    out = tmp_path / "root"
    out.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        _run(["--out", str(out), "--source", f"wildfake={a}",
              "--licences", _licences(tmp_path / "L.json")])


def test_a_source_with_no_licence_entry_is_refused_here_not_after_the_scan(tmp_path):
    """`build_dataset` refuses to fabricate provenance (spec §4.5). Catching it
    at staging costs a second instead of a full scan."""
    coco = _tree(tmp_path / "coco", {"train2017": 2}, np.random.default_rng(7))
    with pytest.raises(SystemExit, match="coco_train2017"):
        _run(["--out", str(tmp_path / "root"),
              "--source", f"coco_train2017={coco}",
              "--licences", _licences(tmp_path / "L.json", names=("wildfake",))])


# --------------------------------------------------------------------------
# the receipt
# --------------------------------------------------------------------------

def test_staging_writes_a_receipt_naming_where_each_bucket_came_from(tmp_path):
    """A hardlink carries no provenance, so which tree a bucket came from is
    exactly what a staged root cannot be asked afterwards."""
    a = _tree(tmp_path / "a" / "wildfake", {"real/real_laion5b": 2, "BigGAN": 3},
              np.random.default_rng(8))
    b = _tree(tmp_path / "b" / "coco", {"train2017": 4}, np.random.default_rng(9))
    out = str(tmp_path / "root")

    _run(["--out", out, "--source", f"wildfake={a}",
          "--source", f"coco_train2017={b}",
          "--licences", _licences(tmp_path / "L.json")])

    with open(os.path.join(out, "STAGED_FROM.json")) as f:
        rec = json.load(f)
    assert rec["mode"] == "hardlink"
    assert rec["counts"] == {"wildfake": {"real": 2, "BigGAN": 3},
                             "coco_train2017": {"train2017": 4}}
    assert rec["sources"]["coco_train2017"] == os.path.abspath(b)
    # build_dataset refuses to run without this (spec §4.5).
    assert os.path.exists(os.path.join(out, "LICENCES.json"))
