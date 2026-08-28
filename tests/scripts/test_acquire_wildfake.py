"""`scripts/acquire_data.py`'s WildFake routines, against archives planted on
disk — never the network.

The acquisition functions skip a download when the file is already in the
cache directory, so a synthetic `label_csv_files/<name>.csv` plus a synthetic
`Images/....zip` exercises the REAL code path: selection, the benchmark gate,
selective extraction, archive deletion, and resume. `fetch` is injected only
where a test needs to prove that nothing was downloaded at all.

The round trip that matters is at the end of every acquisition test: whatever
this writes, `aigcdet.data.sources.classify` must read back as the intended
`(label, generator)` — the C1 contract.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

from aigcdet.data import wildfake as wf
from aigcdet.data.sources import classify

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "scripts")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ad = _load_script("acquire_data")

_HEADER = "Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num"


def _jpeg_bytes(seed: int, size=8) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _plant(cache, subset: wf.WildFakeSubset, rels, *, archive=None,
           member_prefix="", extra_members=()):
    """Write `<cache>/label_csv_files/<name>.csv` and the archive holding
    `rels`, exactly where `_hub_download` looks for them."""
    csv_path = os.path.join(cache, "label_csv_files", f"{subset.name}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(_HEADER + "\n")
        for rel in rels:
            f.write(f"g,a,w,c,1,{subset.label},./{rel},1\n")
    archive = archive or subset.zips[0]
    zp = os.path.join(cache, *wf.split_segments(archive))
    os.makedirs(os.path.dirname(zp), exist_ok=True)
    mode = "a" if os.path.exists(zp) else "w"
    with zipfile.ZipFile(zp, mode) as z:
        for i, rel in enumerate(list(rels) + list(extra_members)):
            z.writestr(member_prefix + rel, _jpeg_bytes(i))
    return zp


def _fake_subset(name, label, rels, prefix, archive="Images/Fixture.zip"):
    return wf.WildFakeSubset(name, label, len(rels), prefix, (archive,))


def _images_under(root):
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.join(d, f) for f in files if f.endswith((".jpg", ".png"))]
    return sorted(out)


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------

def test_acquired_fakes_read_back_as_their_generator(tmp_path):
    subset = wf.SUBSETS["ddim"]
    rels = [f"Diffusion_based/DDIM/ddim/{i:04d}/{i:04d}.jpg" for i in range(6)]
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    # The real registry declares 65,713 rows; the fixture has 6, so point the
    # registry at the fixture's count for this subset only.
    patched = wf.WildFakeSubset(subset.name, subset.label, len(rels),
                                subset.prefix, subset.zips)
    out = str(tmp_path / "raw")
    with _registry({subset.name: patched}):
        ad.acquire_wildfake(out, 0, ["ddim"], cache_dir=cache)

    written = _images_under(os.path.join(out, "wildfake"))
    assert len(written) == 6
    for p in written:
        rel = os.path.relpath(p, out).split(os.sep)
        assert rel[0] == "wildfake"
        assert classify(rel[0], rel[1]) == (1, "ddim")


def test_acquired_reals_read_back_as_authentic_and_keep_their_source(tmp_path):
    subset = wf.SUBSETS["real_ffhq"]
    rels = [f"Real/ffhq/ffhq/{i:04d}/{i:04d}.jpg" for i in range(4)]
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    patched = wf.WildFakeSubset(subset.name, 0, len(rels), subset.prefix, subset.zips)
    out = str(tmp_path / "raw")
    with _registry({subset.name: patched}):
        ad.acquire_wildfake(out, 0, ["real_ffhq"], cache_dir=cache)

    written = _images_under(os.path.join(out, "wildfake"))
    assert len(written) == 4
    for p in written:
        rel = os.path.relpath(p, out).split(os.sep)
        assert classify(rel[0], rel[1]) == (0, "")
        assert rel[2] == "real_ffhq"   # provenance kept below the bucket


# --------------------------------------------------------------------------
# the benchmark gate, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", ["dalle3", "DALLE-3", "dalle_advanced",
                                      "dalle3.csv", "DALLE/Advanced/DALLE3"])
def test_acquire_wildfake_refuses_dalle3_into_the_training_tree(tmp_path, spelling):
    with pytest.raises(ValueError, match="demo benchmark|benchmark data"):
        ad.acquire_wildfake(str(tmp_path / "raw"), 10, [spelling],
                            cache_dir=str(tmp_path / "cache"))
    assert not os.path.exists(tmp_path / "raw" / "wildfake")


@pytest.mark.parametrize("spelling", ["real_coco", "coco", "coco_val2017",
                                      "coco/val2017", "REAL-COCO"])
def test_acquire_wildfake_refuses_coco_into_the_training_tree(tmp_path, spelling):
    with pytest.raises(ValueError, match="COCO-derived|benchmark data"):
        ad.acquire_wildfake(str(tmp_path / "raw"), 10, [spelling],
                            cache_dir=str(tmp_path / "cache"))


def test_a_benchmark_image_hiding_in_a_trainable_subsets_csv_raises(tmp_path):
    """dalle2 is trainable and shares DALLE.zip with the benchmark. A CSV row
    of its that points into Advanced/DALLE3 must stop the run, not be written:
    layer 2 reads the PATH, so no registry flag has to be right."""
    rels = ["Diffusion_based/DALLE/Typical/DALLE2/a/a.jpg",
            "Diffusion_based/DALLE/Advanced/DALLE3/dalle3/b/b.jpg"]
    # The real registry declares dalle2 at `DALLE/Typical/DALLE2/`, so the CSV
    # prefix check would reject the second row first and this test would pass
    # without ever reaching the gate it names. The fixture deliberately
    # LOOSENS the prefix to the parent, which is what a mis-declared registry
    # looks like, so the per-path gate is the thing under test.
    subset = _fake_subset("dalle2", 1, rels, "Diffusion_based/DALLE/",
                          archive="Images/Diffusion_based/DALLE.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    assert wf.read_subset_csv(
        subset, os.path.join(cache, "label_csv_files", "dalle2.csv")) == rels
    out = str(tmp_path / "raw")
    with _registry({"dalle2": subset}):
        with pytest.raises(ValueError, match="demo benchmark"):
            ad.acquire_wildfake(out, 0, ["dalle2"], cache_dir=cache)


def test_a_benchmark_member_hiding_in_the_archive_raises(tmp_path):
    """The other half of layer 2: the CSV was clean, but the ARCHIVE member
    that matched is benchmark data. Nothing is written."""
    # The CSV row is innocuous; the archive member with the SAME matching
    # tail sits inside the benchmark subtree.
    rels = ["Diffusion_based/DALLE/dalle3/a/a.jpg"]
    cache = str(tmp_path / "cache")
    subset = _fake_subset("dalle2", 1, rels, "Diffusion_based/DALLE/",
                          archive="Images/Diffusion_based/DALLE.zip")
    _plant(cache, subset, rels)
    assert wf.forbidden_reason(rels[0]) == ""   # the CSV path itself is clean
    # Re-plant the archive with the member renamed into the benchmark subtree
    # while keeping the tail the CSV matches on.
    zp = os.path.join(cache, "Images", "Diffusion_based", "DALLE.zip")
    os.remove(zp)
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("Diffusion_based/DALLE/Advanced/DALLE3/dalle3/a/a.jpg",
                   _jpeg_bytes(0))
    out = str(tmp_path / "raw")
    with _registry({"dalle2": subset}):
        with pytest.raises(SystemExit, match="demo benchmark"):
            ad.acquire_wildfake(out, 0, ["dalle2"], cache_dir=cache)
    assert _images_under(out) == []


def test_the_benchmark_verb_writes_under_excluded_sources(tmp_path):
    """The demo benchmark IS acquirable — through its own function, with its
    own destination argument, into sources build_dataset excludes by name."""
    coco = wf.BenchmarkHalf("real_coco", "coco_val2017", "", ("coco", "val2017"), 3)
    dalle = wf.BenchmarkHalf("dalle3", "dalle_advanced", "dalle3",
                             ("DALLE", "Advanced", "DALLE3"), 2)
    coco_rels = ([f"Real/coco/val2017/{i}/{i}.jpg" for i in range(3)]
                 + [f"Real/coco/train2017/{i}/{i}.jpg" for i in range(4)])
    dalle_rels = [f"Diffusion_based/DALLE/Advanced/DALLE3/dalle3/{i}/{i}.jpg"
                  for i in range(2)]
    cache = str(tmp_path / "cache")
    coco_subset = _fake_subset("real_coco", 0, coco_rels, "Real/coco/",
                               archive="Images/Real/coco.zip")
    dalle_subset = _fake_subset("dalle3", 1, dalle_rels,
                                "Diffusion_based/DALLE/Advanced/DALLE3/",
                                archive="Images/Diffusion_based/DALLE.zip")
    _plant(cache, coco_subset, coco_rels)
    _plant(cache, dalle_subset, dalle_rels)

    demo = str(tmp_path / "demo")
    with _registry({"real_coco": coco_subset, "dalle3": dalle_subset}):
        totals = ad.acquire_wildfake_benchmark(demo, halves=(coco, dalle),
                                               cache_dir=cache)
    assert totals == {"coco_val2017": 3, "dalle_advanced": 2}
    # train2017 rows are NOT the benchmark and were left in the archive.
    assert len(_images_under(os.path.join(demo, "coco_val2017"))) == 3
    for p in _images_under(os.path.join(demo, "dalle_advanced")):
        rel = os.path.relpath(p, demo).split(os.sep)
        assert classify(rel[0], rel[1]) == (1, "dalle3")


def test_the_benchmark_verb_refuses_an_archive_member_that_is_not_the_benchmark(tmp_path):
    """The inverted gate. `real_coco.csv` spans train2017 and test2017 as well,
    and an archive member that is not val2017 must not be materialised as
    though it were: a benchmark that quietly grew is a benchmark whose every
    published number means something else."""
    half = wf.BenchmarkHalf("real_coco", "coco_val2017", "", ("coco", "val2017"), 1)
    rels = ["Real/coco/val2017/a/a.jpg"]
    subset = _fake_subset("real_coco", 0, rels, "Real/coco/",
                          archive="Images/Real/coco.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    zp = os.path.join(cache, "Images", "Real", "coco.zip")
    os.remove(zp)
    # Same matching tail as the CSV row, but the member's own path does not
    # say `coco/val2017` — the case tail matching exists to tolerate, and the
    # case the guard exists to catch when the tolerance is wrong.
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("Real/coco/unsorted/val2017/a/a.jpg", _jpeg_bytes(0))
    demo = str(tmp_path / "demo")
    with _registry({"real_coco": subset}):
        with pytest.raises(SystemExit, match="not benchmark data"):
            ad.acquire_wildfake_benchmark(demo, halves=(half,), cache_dir=cache)
    assert _images_under(demo) == []


def test_the_benchmark_verb_refuses_a_directory_that_is_the_training_tree(tmp_path):
    demo = tmp_path / "demo"
    (demo / "wildfake" / "ddim").mkdir(parents=True)
    with pytest.raises(SystemExit, match="looks like the --raw training tree"):
        ad.acquire_wildfake_benchmark(str(demo), cache_dir=str(tmp_path / "c"))


# --------------------------------------------------------------------------
# cap, determinism, resume, disk
# --------------------------------------------------------------------------

def test_the_per_generator_cap_is_respected_and_deterministic(tmp_path):
    rels = [f"Diffusion_based/DDIM/ddim/{i:04d}/{i:04d}.jpg" for i in range(40)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")

    def run(root):
        cache = str(tmp_path / f"cache_{root}")
        _plant(cache, subset, rels)
        out = str(tmp_path / root)
        with _registry({"ddim": subset}):
            ad.acquire_wildfake(out, 9, ["ddim"], seed=99, cache_dir=cache)
        return sorted(os.path.basename(p)
                      for p in _images_under(os.path.join(out, "wildfake")))

    first, second = run("a"), run("b")
    assert len(first) == 9
    assert first == second          # same seed, same nine images


def test_a_resumed_run_agrees_with_a_fresh_one_and_downloads_nothing(tmp_path):
    """The property that makes an hour-scale download survivable: re-running
    must skip what is on disk, choose the same images, and not re-fetch the
    53 GB archive to discover it has nothing to do."""
    rels = [f"Diffusion_based/DDIM/ddim/{i:04d}/{i:04d}.jpg" for i in range(40)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    out = str(tmp_path / "raw")
    with _registry({"ddim": subset}):
        ad.acquire_wildfake(out, 9, ["ddim"], seed=5, cache_dir=cache)
        after_first = _images_under(os.path.join(out, "wildfake"))
        stamps = {p: os.stat(p).st_mtime_ns for p in after_first}

        def refuse(member, cache_dir):
            local = os.path.join(cache_dir, *wf.split_segments(member))
            if os.path.exists(local):
                return local
            raise AssertionError(f"a resumed run must not download {member!r}")

        ad.acquire_wildfake(out, 9, ["ddim"], seed=5, cache_dir=cache, fetch=refuse)

    assert _images_under(os.path.join(out, "wildfake")) == after_first
    assert {p: os.stat(p).st_mtime_ns for p in after_first} == stamps


def test_a_partial_run_is_completed_rather_than_restarted(tmp_path):
    rels = [f"Diffusion_based/DDIM/ddim/{i:04d}/{i:04d}.jpg" for i in range(20)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    zp = _plant(cache, subset, rels)
    out = str(tmp_path / "raw")
    with _registry({"ddim": subset}):
        ad.acquire_wildfake(out, 8, ["ddim"], seed=3, cache_dir=cache)
        full = _images_under(os.path.join(out, "wildfake"))
        assert len(full) == 8
        # Simulate an interruption after three images, and re-plant the
        # archive the finished run deleted.
        for p in full[3:]:
            os.remove(p)
        _plant(cache, subset, rels)
        ad.acquire_wildfake(out, 8, ["ddim"], seed=3, cache_dir=cache)
    assert _images_under(os.path.join(out, "wildfake")) == full


def test_the_archive_is_deleted_once_extracted(tmp_path):
    """Disk is the binding constraint: 268 GB free against archives up to
    53 GB, so no archive may outlive its own extraction."""
    rels = [f"Diffusion_based/DDIM/ddim/{i}/{i}.jpg" for i in range(3)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    zp = _plant(cache, subset, rels)
    with _registry({"ddim": subset}):
        ad.acquire_wildfake(str(tmp_path / "raw"), 0, ["ddim"], cache_dir=cache)
    assert not os.path.exists(zp)
    # The label CSV is kept: it is small and a resume needs it again.
    assert os.path.exists(os.path.join(cache, "label_csv_files", "ddim.csv"))


def test_only_the_csvs_own_images_are_extracted(tmp_path):
    """Selective extraction, not `extractall`: the archive holds images this
    subset's CSV does not list, and none of them reach the tree."""
    rels = [f"Diffusion_based/DDIM/ddim/{i}/{i}.jpg" for i in range(3)]
    extra = [f"Diffusion_based/DDIM/other/{i}/{i}.jpg" for i in range(50)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels, extra_members=extra)
    out = str(tmp_path / "raw")
    with _registry({"ddim": subset}):
        ad.acquire_wildfake(out, 0, ["ddim"], cache_dir=cache)
    assert len(_images_under(out)) == 3


def test_a_member_is_never_used_for_a_subset_that_lives_in_another_archive(tmp_path):
    """Members are matched by their path tail, and `wanted` spans every
    requested subset, so a member of DDIM.zip can match a row belonging to
    ddpm. Using it would write one generator's image into the other's bucket —
    with the image COUNT still looking exactly right, which is why this test
    compares bytes rather than counts."""
    a_rel = "Diffusion_based/DDIM/aa/bb/cc.jpg"
    b_rel = "Diffusion_based/DDPM/pp/qq/rr.jpg"
    impostor = "Diffusion_based/DDIM/pp/qq/rr.jpg"   # same tail as b_rel
    assert wf.tail_key(impostor) == wf.tail_key(b_rel)
    ddim = _fake_subset("ddim", 1, [a_rel], "Diffusion_based/DDIM/",
                        archive="Images/Diffusion_based/DDIM.zip")
    ddpm = _fake_subset("ddpm", 1, [b_rel], "Diffusion_based/DDPM/",
                        archive="Images/Diffusion_based/DDPM.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, ddim, [a_rel])
    _plant(cache, ddpm, [b_rel])
    with zipfile.ZipFile(os.path.join(cache, "Images", "Diffusion_based",
                                      "DDIM.zip"), "a") as z:
        z.writestr(impostor, b"WRONG-ARCHIVE")
    right = zipfile.ZipFile(os.path.join(cache, "Images", "Diffusion_based",
                                         "DDPM.zip")).read(b_rel)

    out = str(tmp_path / "raw")
    with _registry({"ddim": ddim, "ddpm": ddpm}):
        ad.acquire_wildfake(out, 0, ["ddim", "ddpm"], cache_dir=cache)
    written = _images_under(os.path.join(out, "wildfake", "ddpm"))
    assert len(written) == 1
    with open(written[0], "rb") as f:
        assert f.read() == right


def test_an_interrupted_extraction_leaves_nothing_a_resume_would_trust(tmp_path,
                                                                       monkeypatch):
    """Resume decides by `os.path.exists(dest)`, so a copy that dies partway
    must leave neither a truncated image nor a stray `.part` — a half image
    that a later run counts as finished reaches the manifest."""
    rels = [f"Diffusion_based/DDIM/ddim/{i}/{i}.jpg" for i in range(3)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)

    def die(src, dst, length=None):
        dst.write(b"half an image")
        raise OSError("connection reset by peer")

    monkeypatch.setattr(ad.shutil, "copyfileobj", die)
    out = str(tmp_path / "raw")
    with _registry({"ddim": subset}):
        with pytest.raises(OSError, match="connection reset"):
            ad.acquire_wildfake(out, 0, ["ddim"], cache_dir=cache)
    leftovers = [os.path.join(d, f) for d, _, fs in os.walk(out) for f in fs]
    assert leftovers == []


def test_an_archive_holding_none_of_the_subsets_images_raises(tmp_path):
    """The loud failure for a wrong archive in the registry: extracting zero
    images must not look like a successful acquisition of zero images."""
    rels = [f"Diffusion_based/DDIM/ddim/{i}/{i}.jpg" for i in range(3)]
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    zp = os.path.join(cache, "Images", "Diffusion_based", "DDIM.zip")
    os.remove(zp)
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("Diffusion_based/DDIM/somethingelse/x/x.jpg", _jpeg_bytes(0))
    with _registry({"ddim": subset}):
        with pytest.raises(SystemExit, match="held none of the images"):
            ad.acquire_wildfake(str(tmp_path / "raw"), 0, ["ddim"], cache_dir=cache)


def test_the_wrong_tree_is_abandoned_after_the_first_archive(tmp_path):
    """The version axis makes this the expensive mistake: a subset declared
    against the wrong tree matches nothing in ANY part of it. Warning about a
    shortfall at the end would mean paying for all seven parts of a ~372 GB
    glob first, so the first barren archive is fatal."""
    rels = [f"Diffusion_based/Midjourney/Typical/mj_v4/{i}/{i}.jpg"
            for i in range(3)]
    subset = wf.WildFakeSubset(
        "mjv4", 1, 3, "Diffusion_based/Midjourney/Typical/mj_v4/",
        ("Images/Diffusion_based/Midjourney/Advanced/part_*.zip",))  # wrong tree
    cache = str(tmp_path / "cache")
    # Plant three Advanced-tree parts, none of which holds an mj_v4 image.
    parts = [f"Images/Diffusion_based/Midjourney/Advanced/part_{i}.zip"
             for i in range(3)]
    _plant(cache, subset, rels, archive=parts[0], extra_members=())
    zp = os.path.join(cache, *wf.split_segments(parts[0]))
    os.remove(zp)
    for part in parts:
        p = os.path.join(cache, *wf.split_segments(part))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("Diffusion_based/Midjourney/Advanced/mj_v5/a/a.jpg",
                       _jpeg_bytes(0))
    fetched = []

    def fetch(member, cache_dir):
        fetched.append(member)
        return ad._hub_download(member, cache_dir)

    with _registry({"mjv4": subset}):
        with pytest.raises(SystemExit, match="held none of the images"):
            ad.acquire_wildfake(str(tmp_path / "raw"), 0, ["mjv4"],
                                cache_dir=cache, allow_large=True,
                                fetch=fetch, list_files=lambda: parts)
    assert fetched == ["label_csv_files/mjv4.csv", parts[0]]


def test_a_later_part_holding_nothing_is_not_an_error(tmp_path):
    """The exemption that keeps the rule from firing on ordinary sparseness:
    once a subset has produced images, a later part of the same tree holding
    none of its capped selection is expected, not a misdeclaration."""
    rels = [f"Diffusion_based/Midjourney/Typical/mj_v4/{i}/{i}.jpg"
            for i in range(3)]
    subset = wf.WildFakeSubset(
        "mjv4", 1, 3, "Diffusion_based/Midjourney/Typical/mj_v4/",
        ("Images/Diffusion_based/Midjourney/Typical/part_*.zip",))
    cache = str(tmp_path / "cache")
    parts = [f"Images/Diffusion_based/Midjourney/Typical/part_{i}.zip"
             for i in range(2)]
    # The CSV lists all three rows; part_0 holds only the first two and part_1
    # holds none of them, so the run reaches part_1 with work outstanding and
    # gets nothing from it.
    _plant(cache, subset, rels, archive=parts[0])
    zp0 = os.path.join(cache, *wf.split_segments(parts[0]))
    os.remove(zp0)
    with zipfile.ZipFile(zp0, "w") as z:
        for i, rel in enumerate(rels[:2]):
            z.writestr(rel, _jpeg_bytes(i))
    with zipfile.ZipFile(os.path.join(cache, *wf.split_segments(parts[1])),
                         "w") as z:
        z.writestr("Diffusion_based/Midjourney/Typical/mj_v4_filler/a/a.jpg",
                   _jpeg_bytes(9))
    out = str(tmp_path / "raw")
    with _registry({"mjv4": subset}):
        ad.acquire_wildfake(out, 0, ["mjv4"], cache_dir=cache, allow_large=True,
                            list_files=lambda: parts)
    assert len(_images_under(out)) == 2


def test_a_multi_part_archive_stops_once_the_subset_is_complete(tmp_path):
    """`part_*.zip` trees run to ~51 GB per part; once the cap is met the
    remaining parts must not be fetched at all."""
    rels = [f"Diffusion_based/SD/originalSD/Typical/o/{i}/{i}.jpg"
            for i in range(4)]
    subset = wf.WildFakeSubset(
        "originsd", 1, 4, "Diffusion_based/SD/originalSD/Typical/",
        ("Images/Diffusion_based/SD/originalSD/Typical/part_*.zip",))
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels,
           archive="Images/Diffusion_based/SD/originalSD/Typical/part_0.zip")
    parts = ["Images/Diffusion_based/SD/originalSD/Typical/part_0.zip",
             "Images/Diffusion_based/SD/originalSD/Typical/part_1.zip"]
    fetched = []

    def fetch(member, cache_dir):
        fetched.append(member)
        local = os.path.join(cache_dir, *wf.split_segments(member))
        if not os.path.exists(local):
            # part_1 was never planted. Failing here rather than delegating to
            # the real downloader keeps this test offline whatever the code
            # under it does.
            raise AssertionError(f"{member!r} should not have been fetched")
        return local

    out = str(tmp_path / "raw")
    with _registry({"originsd": subset}):
        # originsd is ~119 GB, over the budget: the override is what makes
        # the multi-part path reachable at all.
        ad.acquire_wildfake(out, 0, ["originsd"], cache_dir=cache,
                            allow_large=True, fetch=fetch,
                            list_files=lambda: parts)
    assert parts[1] not in fetched          # part_1 was never downloaded
    assert len(_images_under(out)) == 4


def test_two_rows_sharing_a_matching_tail_raise_rather_than_overwrite(tmp_path):
    """Members are matched to CSV rows by their last few path components. Two
    rows sharing that tail would make the match ambiguous and put one image in
    the other's place, so the run stops instead of guessing."""
    tail = "/".join(f"s{i}" for i in range(wf.TAIL_COMPONENTS - 1)) + "/y.jpg"
    rels = [f"Diffusion_based/DDIM/a/{tail}", f"Diffusion_based/DDIM/b/{tail}"]
    assert wf.tail_key(rels[0]) == wf.tail_key(rels[1])
    subset = _fake_subset("ddim", 1, rels, "Diffusion_based/DDIM/",
                          archive="Images/Diffusion_based/DDIM.zip")
    cache = str(tmp_path / "cache")
    _plant(cache, subset, rels)
    with _registry({"ddim": subset}):
        with pytest.raises(SystemExit, match="share the archive-matching key"):
            ad.acquire_wildfake(str(tmp_path / "raw"), 0, ["ddim"], cache_dir=cache)


def test_an_over_budget_subset_is_refused_before_anything_is_fetched(tmp_path):
    """The gate has to be wired into the acquisition entry point, not merely
    available beside it: `mjv5` is ~372 GB across seven parts, and the caller
    must be told that before the first byte, not an hour in."""
    def never(member, cache_dir):
        raise AssertionError(f"nothing should be fetched, got {member!r}")

    with pytest.raises(ValueError, match="~372 GB"):
        ad.acquire_wildfake(str(tmp_path / "raw"), 10, ["mjv5"],
                            cache_dir=str(tmp_path / "cache"), fetch=never)
    assert not os.path.exists(tmp_path / "raw")


@pytest.mark.parametrize(("argv", "expected"), [
    ([], False),
    (["--allow-large-archives"], True),
])
def test_the_cli_passes_the_large_archive_override_through(tmp_path, monkeypatch,
                                                           argv, expected):
    """The override has to be opt-in from the command line too. A flag that is
    parsed but not forwarded reads as a working guard while every subset is
    acquirable."""
    seen = {}

    def record(out, limit, generators, *, seed, allow_large):
        seen.update(out=out, limit=limit, generators=generators, seed=seed,
                    allow_large=allow_large)
        return {}

    monkeypatch.setattr(ad, "acquire_wildfake", record)
    monkeypatch.setattr(sys, "argv", [
        "acquire_data.py", "--dataset", "wildfake", "--out", str(tmp_path),
        "--generators", "ddim", "--limit", "12", "--seed", "7", *argv])
    ad.main()
    assert seen["allow_large"] is expected
    assert seen["generators"] == ["ddim"] and seen["limit"] == 12
    assert seen["seed"] == 7
    # Spec §4.5: the licence is recorded at acquisition time, by main().
    with open(os.path.join(str(tmp_path), "LICENCES.json")) as f:
        assert "wildfake" in json.load(f)


def _plant_split_parts(cache, subset, rels, parts, split):
    """CSV listing every row, with the rows dealt out across `parts`."""
    _plant(cache, subset, rels, archive=parts[0])
    for part in parts:
        p = os.path.join(cache, *wf.split_segments(part))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p):
            os.remove(p)
    for part, members in zip(parts, split):
        with zipfile.ZipFile(os.path.join(cache, *wf.split_segments(part)),
                             "w") as z:
            for i, rel in enumerate(members):
                z.writestr(rel, _jpeg_bytes(i))


def test_a_resumed_subset_is_exempt_from_the_barren_check(tmp_path):
    """The interrupted multi-part case the exemption exists for. A run that
    already extracted part_0's images and died is resumed; part_0 now yields
    nothing new, because everything still outstanding lives in part_1. Without
    the exemption that reads as "wrong tree" and the resume dies on the first
    archive — turning an ordinary interruption into an unrecoverable one."""
    rels = [f"Diffusion_based/Midjourney/Typical/mj_v4/{i}/{i}.jpg"
            for i in range(3)]
    subset = wf.WildFakeSubset(
        "mjv4", 1, 3, "Diffusion_based/Midjourney/Typical/mj_v4/",
        ("Images/Diffusion_based/Midjourney/Typical/part_*.zip",))
    cache = str(tmp_path / "cache")
    parts = [f"Images/Diffusion_based/Midjourney/Typical/part_{i}.zip"
             for i in range(2)]
    out = str(tmp_path / "raw")

    # Phase 1: the interrupted run only ever sees part_0.
    _plant_split_parts(cache, subset, rels, parts, [rels[:2], rels[2:]])
    with _registry({"mjv4": subset}):
        ad.acquire_wildfake(out, 0, ["mjv4"], cache_dir=cache, allow_large=True,
                            list_files=lambda: parts[:1])
    assert len(_images_under(out)) == 2

    # Phase 2: the resume sees both parts. part_0 yields nothing new.
    _plant_split_parts(cache, subset, rels, parts, [rels[:2], rels[2:]])
    fetched = []

    def fetch(member, cache_dir):
        fetched.append(member)
        return ad._hub_download(member, cache_dir)

    with _registry({"mjv4": subset}):
        ad.acquire_wildfake(out, 0, ["mjv4"], cache_dir=cache, allow_large=True,
                            fetch=fetch, list_files=lambda: parts)
    assert parts[0] in fetched and parts[1] in fetched
    assert len(_images_under(out)) == 3


def test_the_volume_report_names_archives_of_unrecorded_size(capsys):
    """Archives are shared, so the total is over DISTINCT archives — asking
    for four GAN families is one 47.3 GB download, not four. And the eight
    `Images/Real/*.zip` sizes were never published: unrecorded must be SAID,
    because a total that silently omits them reads as the whole cost."""
    gan = [wf.SUBSETS[n] for n in ("BigGAN", "styleGAN", "VQGAN")]
    ad._report_download_volume(gan)
    out = capsys.readouterr().out
    assert "~47 GB across 1 archive(s)" in out
    assert "unrecorded" not in out

    ad._report_download_volume(gan + [wf.SUBSETS["real_ffhq"]])
    out = capsys.readouterr().out
    assert "size unrecorded for ['Images/Real/ffhq.zip']" in out


def test_requesting_no_generators_raises(tmp_path):
    with pytest.raises(SystemExit, match="--generators is required"):
        ad.acquire_wildfake(str(tmp_path / "raw"), 10, [])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _registry:
    """Temporarily replace registry entries, so fixtures can declare 6 rows
    where the real registry declares 65,713 without weakening the count check
    the production path performs."""

    def __init__(self, entries: dict[str, wf.WildFakeSubset]):
        self.entries = entries
        self.saved: dict[str, wf.WildFakeSubset] = {}

    def __enter__(self):
        for name, subset in self.entries.items():
            self.saved[name] = wf.SUBSETS[name]
            wf.SUBSETS[name] = subset
        return self

    def __exit__(self, *exc):
        wf.SUBSETS.update(self.saved)
        return False
