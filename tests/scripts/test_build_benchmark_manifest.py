"""End-to-end test of scripts/build_benchmark_manifest.py.

The fixtures are built so that every property under test can actually FAIL:

* the two halves get DIFFERENT counts (3 and 5), so a count guard that
  compares a half against the other half's expectation is detectable. Equal
  counts would make that mutation invisible.
* the two halves get DIFFERENT pixel content and different image SIZES, so a
  row that ended up attributed to the wrong half is detectable from the row
  itself rather than only from the path.
* the halves are the REAL `BenchmarkHalf` records with only `expected`
  reduced, so the source names, bucket names, labels and generators exercised
  here are the ones the real registry produces. A fabricated source would let
  every test pass against a scanner that looks in the wrong directory —
  which is the failure `test_scanned_directory_is_where_benchmark_dest_writes`
  and `test_real_demo_layout_matches_benchmark_halves` exist to prevent.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data import wildfake as wf
from aigcdet.data.manifest import (
    MANIFEST_COLUMNS,
    MANIFEST_IDENTITY_COLUMNS,
    SPLITS,
    read_manifest,
)
from aigcdet.data.sources import LICENCES

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")


def _load_script(name: str):
    path = os.path.join(_SCRIPTS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbm = _load_script("build_benchmark_manifest")

#: Real upstream `Image_path` values, one per half — the same two strings
#: `tests/data/test_wildfake.py` pins the markers against. Used here to place
#: the fixture through the REAL `benchmark_dest`, so the directory layout under
#: test is the one acquisition actually produces.
REAL_CSV_PATHS = {
    "real_coco": "./Real/coco/coco2017/val2017/000000000139.jpg",
    "dalle3": ("./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/"
               "202311011943129901ca391019566e/"
               "0000bc251bd2e98239266f18c7422f00.jpg"),
}

#: Deliberately UNEQUAL, and small. Unequal is what makes a count guard that
#: consults the wrong half's expectation detectable at all.
COCO_N = 3
DALLE_N = 5


def _small_halves():
    """The real halves with only `expected` shrunk to the fixture sizes."""
    by_subset = {h.subset: h for h in wf.BENCHMARK_HALVES}
    return (
        dataclasses.replace(by_subset["real_coco"], expected=COCO_N),
        dataclasses.replace(by_subset["dalle3"], expected=DALLE_N),
    )


def _plant(demo_dir, half, n, size, seed):
    """Write `n` distinct images into the directory `benchmark_dest` names.

    Distinct because a fixture of identical images cannot detect a digest that
    was computed from the wrong file: `content_sha256` would be the same value
    either way.
    """
    d = os.path.dirname(
        wf.benchmark_dest(str(demo_dir), half, REAL_CSV_PATHS[half.subset]))
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        p = os.path.join(d, f"{seed:04d}{i:04d}.jpg")
        Image.fromarray(arr).save(p, format="JPEG", quality=95)
        paths.append(p)
    return sorted(paths)


@pytest.fixture
def demo(tmp_path):
    """A demo tree in the real layout, with unequal halves of unequal size."""
    halves = _small_halves()
    coco, dalle = halves
    planted = {
        coco.source: _plant(tmp_path / "demo", coco, COCO_N, size=32, seed=1),
        dalle.source: _plant(tmp_path / "demo", dalle, DALLE_N, size=48, seed=2),
    }
    return {"dir": str(tmp_path / "demo"), "halves": halves, "planted": planted}


def _build(demo, tmp_path, **kw):
    out = kw.pop("manifest", str(tmp_path / "bench.parquet"))
    return bbm.build_benchmark_manifest(
        demo["dir"], out, halves=demo["halves"], **kw), out


# --- layout pins -----------------------------------------------------------

@pytest.mark.parametrize("subset", sorted(REAL_CSV_PATHS))
def test_scanned_directory_is_where_benchmark_dest_writes(subset):
    """The scanner must look exactly where the materialiser writes.

    Asserted against `benchmark_dest` applied to a REAL upstream CSV path,
    not against a string this test composes: a test that fabricated the path
    would agree with a scanner that consults the wrong marker, and every other
    test here would then pass against a scanner that finds nothing real.
    """
    half = {h.subset: h for h in wf.BENCHMARK_HALVES}[subset]
    expected = os.path.dirname(
        wf.benchmark_dest("/demo", half, REAL_CSV_PATHS[subset]))
    assert bbm.half_dir("/demo", half) == expected


def test_half_dirs_are_the_declared_source_and_bucket_names():
    """The two directory names, verbatim, against BENCHMARK_HALVES."""
    dirs = {h.source: bbm.half_dir("/demo", h) for h in wf.BENCHMARK_HALVES}
    assert dirs == {
        "coco_val2017": os.path.join("/demo", "coco_val2017", "val2017"),
        "dalle_advanced": os.path.join("/demo", "dalle_advanced", "dalle3"),
    }


_REAL_DEMO = os.path.join(_REPO_ROOT, "data", "demo")


@pytest.mark.skipif(not os.path.isdir(_REAL_DEMO),
                    reason="the real demo benchmark is not materialised here")
def test_real_demo_layout_matches_benchmark_halves():
    """Read-only pin on the benchmark ACTUALLY on disk.

    A previous bug on this project had every test compose a fabricated path
    that agreed with a wrong marker, so the suite was green against a scanner
    that would have found nothing in production. This is the one assertion
    that consults the real tree.
    """
    for half in wf.BENCHMARK_HALVES:
        d = bbm.half_dir(_REAL_DEMO, half)
        assert os.path.isdir(d), d
        assert len(bbm.scan_half(_REAL_DEMO, half)) == half.expected


# --- the manifest it writes ------------------------------------------------

def test_writes_one_row_per_image_with_the_manifest_contract(demo, tmp_path):
    df, out = _build(demo, tmp_path)
    assert len(df) == COCO_N + DALLE_N
    frozen = pd.read_parquet(out)
    assert list(frozen.columns) == MANIFEST_COLUMNS + MANIFEST_IDENTITY_COLUMNS
    assert len(frozen) == COCO_N + DALLE_N


def test_every_row_is_the_benchmark_split(demo, tmp_path):
    df, out = _build(demo, tmp_path)
    assert set(df["split"]) == {"benchmark"}
    assert set(pd.read_parquet(out)["split"]) == {"benchmark"}
    assert "benchmark" in SPLITS


def test_coco_is_authentic_and_dalle3_is_generated(demo, tmp_path):
    """The mapping the whole benchmark score depends on.

    Checked per SOURCE and in both directions: asserting only "both labels
    appear" would pass on a manifest with the two halves swapped, which is
    exactly the defect (C1) that labelled 5,000 COCO photographs
    AI-generated.
    """
    df, _ = _build(demo, tmp_path)
    coco = df[df["source"] == "coco_val2017"]
    dalle = df[df["source"] == "dalle_advanced"]
    assert len(coco) == COCO_N and len(dalle) == DALLE_N
    assert set(coco["label"]) == {0}
    assert set(dalle["label"]) == {1}
    assert set(coco["generator"]) == {""}
    assert set(dalle["generator"]) == {"dalle3"}


def test_licence_comes_from_the_source_registry(demo, tmp_path):
    df, _ = _build(demo, tmp_path)
    for source, licence in df.groupby("source")["licence"].first().items():
        assert licence == LICENCES[source]


def test_dimensions_are_read_from_the_images(demo, tmp_path):
    """The two halves were planted at different sizes, so a row carrying the
    other half's dimensions is visible here."""
    df, _ = _build(demo, tmp_path)
    assert set(df[df["source"] == "coco_val2017"]["width"]) == {32}
    assert set(df[df["source"] == "dalle_advanced"]["width"]) == {48}


def test_paths_are_absolute_and_rel_paths_are_rooted_at_demo_dir(demo, tmp_path):
    """`rel_path` is the identity the fingerprint is taken over, and it must be
    relative to --demo-dir — the directory a teammate attaches the published
    Kaggle Dataset at, where the absolute `path` column is meaningless."""
    df, out = _build(demo, tmp_path)
    frozen = pd.read_parquet(out)
    assert all(os.path.isabs(p) for p in frozen["path"])
    rels = sorted(frozen["rel_path"])
    assert rels[0].split(os.sep)[0] in ("coco_val2017", "dalle_advanced")
    for p, r in zip(frozen["path"], frozen["rel_path"]):
        assert os.path.abspath(os.path.join(demo["dir"], r)) == p
    # Two top-level components survive, which is what pins the root at
    # demo_dir rather than at either half's own directory.
    assert {r.split(os.sep)[0] for r in rels} == {"coco_val2017", "dalle_advanced"}


def test_root_is_pinned_to_demo_dir_even_with_a_single_half(demo, tmp_path):
    """`derive_root` is the DEEPEST directory containing every path, so with
    only one half present it would return that half's own directory and
    `rel_path` would lose the `<source>/<bucket>/` prefix -- an identity that
    silently changes the moment the other half is added. The root is pinned
    to --demo-dir instead.
    """
    coco = demo["halves"][0]
    df = bbm.build_benchmark_manifest(
        demo["dir"], str(tmp_path / "one.parquet"), halves=(coco,))
    assert len(df) == COCO_N
    frozen = pd.read_parquet(tmp_path / "one.parquet")
    assert all(r.startswith("coco_val2017" + os.sep) for r in frozen["rel_path"])


def test_rebases_onto_another_root(demo, tmp_path):
    """The Kaggle move: same identity, different mount point."""
    _, out = _build(demo, tmp_path)
    rebased = read_manifest(out, root="/kaggle/input/demo")
    assert all(p.startswith("/kaggle/input/demo" + os.sep) for p in rebased["path"])
    assert list(rebased["rel_path"]) == list(pd.read_parquet(out)["rel_path"])


def test_digests_are_distinct_per_image(demo, tmp_path):
    """Every fixture image has different pixels, so every digest must differ;
    a scanner that digested one file repeatedly would collapse them."""
    _, out = _build(demo, tmp_path)
    digests = pd.read_parquet(out)["content_sha256"]
    assert len(set(digests)) == COCO_N + DALLE_N
    assert all(len(d) == 64 for d in digests)


def test_row_order_is_stable_across_runs(demo, tmp_path):
    """Row order IS the manifest's identity; two freezes of the same tree must
    produce the same order."""
    a, _ = _build(demo, tmp_path, manifest=str(tmp_path / "a.parquet"))
    b, _ = _build(demo, tmp_path, manifest=str(tmp_path / "b.parquet"))
    assert list(a["path"]) == list(b["path"])


# --- the count guard -------------------------------------------------------

def test_extra_image_in_a_half_is_fatal(demo, tmp_path):
    coco = demo["halves"][0]
    _plant(demo["dir"], coco, 1, size=32, seed=7)
    with pytest.raises(ValueError) as exc:
        _build(demo, tmp_path)
    msg = str(exc.value)
    assert "coco_val2017" in msg
    assert str(COCO_N + 1) in msg and str(COCO_N) in msg


def test_missing_image_in_a_half_is_fatal(demo, tmp_path):
    os.remove(demo["planted"]["dalle_advanced"][0])
    with pytest.raises(ValueError) as exc:
        _build(demo, tmp_path)
    assert "dalle_advanced" in str(exc.value)
    assert str(DALLE_N) in str(exc.value)
    assert str(DALLE_N - 1) in str(exc.value)


def test_count_guard_names_the_half_it_checked(demo, tmp_path):
    """A guard that compared a half against the OTHER half's expectation would
    accept the wrong half's count. The two halves differ (3 vs 5), so growing
    coco to dalle's expected size must still raise.
    """
    coco = demo["halves"][0]
    _plant(demo["dir"], coco, DALLE_N - COCO_N, size=32, seed=8)
    assert len(bbm.scan_half(demo["dir"], coco)) == DALLE_N
    with pytest.raises(ValueError) as exc:
        _build(demo, tmp_path)
    msg = str(exc.value)
    assert "coco_val2017" in msg and str(COCO_N) in msg


def test_halves_wrong_in_opposite_directions_are_fatal_despite_a_right_total(
        demo, tmp_path):
    """The reason the count check is PER HALF and not only on the total.

    Move two images' worth of count from dalle to coco: the tree still holds
    exactly `expected_total` images, so a script that checked only the sum
    would freeze a benchmark whose two halves are both wrong. 4,998 + 8,843 is
    a figure the eye reads as correct.
    """
    coco, dalle = demo["halves"]
    for p in demo["planted"]["dalle_advanced"][:DALLE_N - COCO_N]:
        os.remove(p)
    _plant(demo["dir"], coco, DALLE_N - COCO_N, size=32, seed=12)
    assert len(bbm.scan_half(demo["dir"], coco)) == DALLE_N
    assert len(bbm.scan_half(demo["dir"], dalle)) == COCO_N
    assert (len(bbm.scan_half(demo["dir"], coco))
            + len(bbm.scan_half(demo["dir"], dalle))
            == COCO_N + DALLE_N)          # the total alone is still correct
    with pytest.raises(ValueError) as exc:
        _build(demo, tmp_path)
    assert "coco_val2017" in str(exc.value)


def test_no_manifest_is_written_when_a_count_is_wrong(demo, tmp_path):
    """The guard must fire BEFORE the freeze, or a wrong-sized benchmark is on
    disk for anything that ignores the traceback."""
    _plant(demo["dir"], demo["halves"][0], 1, size=32, seed=9)
    out = str(tmp_path / "bench.parquet")
    with pytest.raises(ValueError):
        _build(demo, tmp_path, manifest=out)
    assert not os.path.exists(out)


def test_missing_half_directory_names_the_half(demo, tmp_path):
    for p in demo["planted"]["dalle_advanced"]:
        os.remove(p)
    os.rmdir(bbm.half_dir(demo["dir"], demo["halves"][1]))
    with pytest.raises(FileNotFoundError, match="dalle_advanced"):
        _build(demo, tmp_path)


def test_check_count_accepts_the_exact_count_and_rejects_neighbours():
    """Off-by-one in either direction, on a half whose expected count is not
    shared with the other half."""
    coco, dalle = _small_halves()
    bbm.check_count(coco, COCO_N)
    for n in (COCO_N - 1, COCO_N + 1, DALLE_N):
        with pytest.raises(ValueError):
            bbm.check_count(coco, n)


def test_real_halves_declare_the_organisers_counts():
    """The numbers this manifest is verified against are the organisers' own,
    read off the registry rather than restated here."""
    counts = {h.source: h.expected for h in wf.BENCHMARK_HALVES}
    assert counts == {"coco_val2017": 4998, "dalle_advanced": 8843}
    assert sum(counts.values()) == 13841


# --- overwrite protection --------------------------------------------------

def test_refuses_to_overwrite_an_existing_manifest_without_force(demo, tmp_path):
    out = str(tmp_path / "bench.parquet")
    _build(demo, tmp_path, manifest=out)
    before = open(out, "rb").read()
    with pytest.raises(FileExistsError, match="already exists"):
        _build(demo, tmp_path, manifest=out)
    assert open(out, "rb").read() == before


def test_force_overwrites_an_existing_manifest(demo, tmp_path):
    out = str(tmp_path / "bench.parquet")
    _build(demo, tmp_path, manifest=out)
    df, _ = _build(demo, tmp_path, manifest=out, force=True)
    assert len(df) == COCO_N + DALLE_N


def test_refuses_before_scanning_so_a_broken_tree_still_cannot_clobber(demo, tmp_path):
    """The refusal must not depend on the scan succeeding: a corrupted demo
    tree must leave the existing manifest alone rather than raising a
    count error that a human might read as 'nothing was written'."""
    out = str(tmp_path / "bench.parquet")
    _build(demo, tmp_path, manifest=out)
    _plant(demo["dir"], demo["halves"][0], 1, size=32, seed=11)
    with pytest.raises(FileExistsError):
        _build(demo, tmp_path, manifest=out)


# --- CLI -------------------------------------------------------------------

def test_cli_writes_the_manifest(demo, tmp_path, monkeypatch):
    """`main` takes no `halves` argument -- it uses the real BENCHMARK_HALVES,
    whose counts are 4,998 and 8,843. Planting 13,841 images in a test is not
    viable, so the module-level default is patched to the fixture halves; the
    argument parsing, the digest choice and the freeze are all still the real
    ones."""
    monkeypatch.setattr(bbm, "BENCHMARK_HALVES", demo["halves"])
    out = str(tmp_path / "cli.parquet")
    df = bbm.main(["--demo-dir", demo["dir"], "--manifest", out])
    assert os.path.exists(out)
    assert len(df) == COCO_N + DALLE_N
    assert set(pd.read_parquet(out)["split"]) == {"benchmark"}


def test_cli_force_flag(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(bbm, "BENCHMARK_HALVES", demo["halves"])
    out = str(tmp_path / "cli.parquet")
    bbm.main(["--demo-dir", demo["dir"], "--manifest", out])
    with pytest.raises(FileExistsError):
        bbm.main(["--demo-dir", demo["dir"], "--manifest", out])
    bbm.main(["--demo-dir", demo["dir"], "--manifest", out, "--force"])


def test_cli_digests_none_leaves_the_digest_columns_empty(demo, tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(bbm, "BENCHMARK_HALVES", demo["halves"])
    out = str(tmp_path / "cli.parquet")
    bbm.main(["--demo-dir", demo["dir"], "--manifest", out, "--digests", "none"])
    frozen = pd.read_parquet(out)
    assert set(frozen["content_sha256"]) == {""}
    assert list(frozen["rel_path"]) == sorted(frozen["rel_path"])
