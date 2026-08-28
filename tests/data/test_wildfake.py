"""The WildFake subset registry and its benchmark gate.

The gate is the point of this module. The organisers' demo benchmark is INSIDE
WildFake — `dalle3.csv` is exactly the 8,843 DALL·E Advanced images, and 4,998
of `real_coco.csv`'s rows are exactly COCO val2017 — so an ordinary-looking
acquisition of a neighbouring subset can put benchmark data in the training
tree. These tests exercise the refusal from both directions: by NAME (however
it is spelled) and by PATH (whatever the registry says).
"""
from __future__ import annotations

import pytest

from aigcdet.data import wildfake as wf
from aigcdet.data.sources import SOURCES, classify, is_safe_generator


# --------------------------------------------------------------------------
# registry consistency
# --------------------------------------------------------------------------

def test_every_subset_names_itself():
    for key, subset in wf.SUBSETS.items():
        assert subset.name == key


@pytest.mark.parametrize("name", sorted(wf.SUBSETS))
def test_fake_subset_names_are_usable_generator_buckets(name):
    """A subset name becomes a directory under `raw/wildfake/`, so `classify`
    must read it back as the same generator — the sources.py contract."""
    subset = wf.SUBSETS[name]
    if subset.label == 1:
        assert is_safe_generator("wildfake", name)
        assert classify("wildfake", name) == (1, name)


@pytest.mark.parametrize("name", sorted(wf.SUBSETS))
def test_declared_prefix_agrees_with_the_declared_archive(name):
    """Internal consistency, since the real `label_csv_files/*.csv` cannot be
    consulted offline: a subset's path prefix and each archive it names must
    lie on one path — one a prefix of the other, segment by segment. A subset
    whose archive is `Images/Real/afhq.zip` but whose prefix says
    `Diffusion_based/...` would download 30 GB and extract nothing."""
    subset = wf.SUBSETS[name]
    prefix = wf.split_segments(subset.prefix)
    for archive in subset.zips:
        segments = wf.split_segments(archive)
        assert segments[0] == "Images", archive
        # Drop "Images/" and the archive's own filename.
        zip_path = segments[1:-1] + [segments[-1][:-len(".zip")]]
        shared = min(len(prefix), len(zip_path))
        assert prefix[:shared] == zip_path[:shared], (subset.prefix, archive)


def test_subsets_sharing_an_archive_agree_on_it():
    """Every archive named by more than one subset is named identically, and
    every subset's archives are declared, not derived — so `_archive_order`
    can group downloads without a fuzzy match."""
    by_archive: dict[str, set[str]] = {}
    for subset in wf.SUBSETS.values():
        for archive in subset.zips:
            by_archive.setdefault(archive, set()).add(subset.prefix)
    assert by_archive["Images/Diffusion_based/DALLE.zip"] == {
        "Diffusion_based/DALLE/", "Diffusion_based/DALLE/Advanced/DALLE3/"}
    assert len(by_archive["Images/GAN_based.zip"]) == 1


def test_declared_row_counts_are_the_ones_counted_upstream():
    """Spot-checks of the counts the registry claims, including the two that
    the benchmark identity rests on: dalle3 is 8,843 (the organisers' AIGC
    half exactly) and real_coco is 163,846 (of which 4,998 are val2017)."""
    assert wf.SUBSETS["dalle3"].rows == 8843
    assert wf.SUBSETS["real_coco"].rows == 163846
    assert wf.SUBSETS["mjv5"].rows == 236578
    assert sum(s.rows for s in wf.SUBSETS.values()) == 3570724


def test_only_the_two_benchmark_subsets_are_training_forbidden():
    forbidden = {n for n, s in wf.SUBSETS.items() if s.training_forbidden}
    assert forbidden == {"dalle3", "real_coco"}


def test_subsets_with_no_established_archive_refuse_rather_than_guess():
    assert wf.SUBSETS["sdxl"].zips == ()
    with pytest.raises(ValueError, match="no established archive path"):
        wf.resolve_for_training("sdxl")


# --------------------------------------------------------------------------
# name resolution
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spelling", ["dalle3", "DALLE3", "DALLE-3", "dalle_3", " dalle3 ",
                 "dalle3.csv", "dalle_advanced", "DALLE_Advanced"])
def test_every_spelling_of_dalle3_resolves_to_the_one_entry(spelling):
    assert wf.resolve(spelling).name == "dalle3"


@pytest.mark.parametrize(
    "spelling", ["dalle3", "DALLE3", "DALLE-3", "dalle_3", "dalle3.csv",
                 "dalle_advanced", "DALLE/Advanced/DALLE3",
                 "./Diffusion_based/DALLE/Advanced/DALLE3/"])
def test_dalle3_into_the_training_tree_raises_however_it_is_spelled(spelling):
    """The organisers' AIGC half. Every alias reaches the SAME registry entry,
    and a path-shaped request is refused by the path gate, so there is no
    spelling that reaches the files."""
    with pytest.raises(ValueError, match="demo benchmark|benchmark data"):
        wf.resolve_for_training(spelling)


@pytest.mark.parametrize("spelling", ["real_coco", "REAL-COCO", "coco",
                                      "coco_val2017", "coco/val2017"])
def test_coco_into_the_training_tree_raises(spelling):
    """Spec §4.1(2) excludes COCO-derived reals from training ENTIRELY, not
    only the 4,998 val2017 photographs the benchmark uses."""
    with pytest.raises(ValueError, match="COCO-derived|benchmark data"):
        wf.resolve_for_training(spelling)


def test_an_unknown_subset_is_refused_not_skipped():
    with pytest.raises(ValueError, match="unknown WildFake subset"):
        wf.resolve("sd15")


def test_resolution_is_unambiguous():
    """Built at import; this asserts the built index rather than re-deriving
    it, so a future alias that collides with a benchmark subset is a startup
    error rather than a silent redirect."""
    assert len(wf.NAME_INDEX) == len(set(wf.NAME_INDEX))
    for spelling, canonical in wf.NAME_INDEX.items():
        assert canonical in wf.SUBSETS
        assert wf.resolve(spelling).name == canonical


# --------------------------------------------------------------------------
# the per-path gate (layer 2)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/abc/abc.jpg",
    "Diffusion_based/DALLE/Advanced/DALLE3/x.jpg",
    "diffusion_based/dalle/advanced/dalle3/x.jpg",     # mis-cased archive
    r"Diffusion_based\DALLE\Advanced\DALLE3\x.jpg",    # windows separators
    "WildFake/Images/Diffusion_based/DALLE/Advanced/DALLE3/x.jpg",
])
def test_forbidden_reason_catches_the_dalle3_subtree(path):
    assert "demo benchmark" in wf.forbidden_reason(path)


@pytest.mark.parametrize("path", [
    "./Real/coco/val2017/000000397133.jpg",
    "./Real/coco/train2017/000000000009.jpg",
    "Real/COCO/test2017/x.jpg",
])
def test_forbidden_reason_catches_every_coco_derived_real(path):
    assert "COCO-derived" in wf.forbidden_reason(path)


@pytest.mark.parametrize("path", [
    "./Diffusion_based/DALLE/Typical/DALLE2/x/y.jpg",
    "./Real/ffhq/00000.png",
    "./GAN_based/styleGAN/a/b.png",
    "./Diffusion_based/Midjourney/Advanced/mjv5/a/b.jpg",
])
def test_forbidden_reason_leaves_trainable_paths_alone(path):
    assert wf.forbidden_reason(path) == ""


def test_training_dest_refuses_a_benchmark_path_under_a_safe_subset():
    """The gate that a mis-declared registry cannot route around: dalle2 is a
    perfectly trainable subset sharing DALLE.zip with the benchmark, and a row
    of its CSV that pointed into Advanced/DALLE3 is still refused."""
    dalle2 = wf.SUBSETS["dalle2"]
    assert dalle2.training_forbidden == ""
    with pytest.raises(ValueError, match="demo benchmark"):
        wf.training_dest("/out", dalle2,
                         "./Diffusion_based/DALLE/Advanced/DALLE3/x/y.jpg")


def test_training_dest_refuses_a_forbidden_subset_directly():
    with pytest.raises(ValueError, match="demo benchmark"):
        wf.training_dest("/out", wf.SUBSETS["dalle3"],
                         "./Diffusion_based/DALLE/Advanced/DALLE3/x/y.jpg")


# --------------------------------------------------------------------------
# destinations round-trip through sources.classify
# --------------------------------------------------------------------------

def test_training_dest_of_a_fake_reads_back_as_that_generator():
    dest = wf.training_dest("/out", wf.SUBSETS["mjv5"],
                            "./Diffusion_based/Midjourney/Advanced/mjv5/a/b.jpg")
    parts = dest.split("/")
    assert parts[2] == "wildfake" and parts[3] == "mjv5"
    assert classify("wildfake", parts[3]) == (1, "mjv5")


def test_training_dest_of_a_real_reads_back_as_authentic():
    """Nested one level below the `real` bucket so `classify` still sees
    `real` (label 0) while the directory name keeps WHICH real source the
    images came from — spec §4.1(2) asks for that to be recorded."""
    dest = wf.training_dest("/out", wf.SUBSETS["real_ffhq"], "./Real/ffhq/1.png")
    parts = dest.split("/")
    assert parts[2] == "wildfake" and parts[3] == "real" and parts[4] == "real_ffhq"
    assert classify("wildfake", parts[3]) == (0, "")


def test_dest_filename_is_a_pure_function_of_the_source_path():
    a = wf.dest_filename("./Real/ffhq/1.png")
    assert a == wf.dest_filename("./Real/ffhq/1.png")   # same input, same name
    assert a.endswith(".png") and a != wf.dest_filename("./Real/ffhq/2.png")
    assert wf.dest_filename("./x/y.JPG").endswith(".jpg")


def test_dest_filename_never_carries_a_path_from_the_archive():
    """A hostile or merely odd archive member cannot steer the write: the
    name is a hash, so no separator survives into the destination."""
    name = wf.dest_filename("../../../etc/passwd")
    assert "/" not in name and ".." not in name


# --------------------------------------------------------------------------
# the benchmark verb's own gate
# --------------------------------------------------------------------------

def test_every_benchmark_half_is_filed_under_an_excluded_source():
    """The guarantee that actually holds end to end: wherever a human points
    the benchmark directory, `build_dataset` drops these rows by SOURCE."""
    for half in wf.BENCHMARK_HALVES:
        assert SOURCES[half.source].exclude_from_training


def test_benchmark_markers_are_exactly_the_forbidden_markers():
    """One table drives both directions: what the training tree refuses is
    what the benchmark requires, so the two cannot drift apart."""
    forbidden = {marker for marker, _ in wf.FORBIDDEN_PATH_MARKERS}
    for half in wf.BENCHMARK_HALVES:
        assert half.marker in forbidden


def test_benchmark_dest_refuses_a_source_that_is_not_excluded_from_training():
    """The assertion that carries the whole end-to-end guarantee, exercised
    directly — with the real registry it can never fire, so without this test
    the branch that makes `exclude_from_training` load-bearing here is dead
    code that nothing would notice being deleted."""
    half = wf.BenchmarkHalf("dalle3", "wildfake", "dalle3",
                            ("DALLE", "Advanced", "DALLE3"), 1)
    assert not SOURCES["wildfake"].exclude_from_training
    with pytest.raises(ValueError, match="not marked exclude_from_training"):
        wf.benchmark_dest("/demo", half,
                          "./Diffusion_based/DALLE/Advanced/DALLE3/a/b.jpg")


def test_benchmark_dest_requires_the_path_to_be_benchmark_data():
    half = {h.subset: h for h in wf.BENCHMARK_HALVES}["real_coco"]
    with pytest.raises(ValueError, match="not part of the coco_val2017"):
        wf.benchmark_dest("/demo", half, "./Real/coco/train2017/1.jpg")
    dest = wf.benchmark_dest("/demo", half, "./Real/coco/val2017/1.jpg")
    assert dest.split("/")[2] == "coco_val2017" and dest.split("/")[3] == "val2017"


def test_benchmark_dest_of_the_aigc_half_reads_back_as_dalle3():
    half = {h.subset: h for h in wf.BENCHMARK_HALVES}["dalle3"]
    dest = wf.benchmark_dest(
        "/demo", half, "./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/a/b.jpg")
    parts = dest.split("/")
    assert parts[2] == "dalle_advanced" and parts[3] == "dalle3"
    assert classify("dalle_advanced", parts[3]) == (1, "dalle3")


def test_benchmark_rows_verifies_the_organisers_count():
    half = {h.subset: h for h in wf.BENCHMARK_HALVES}["real_coco"]
    paths = ([f"Real/coco/val2017/{i}.jpg" for i in range(half.expected)]
             + [f"Real/coco/train2017/{i}.jpg" for i in range(10)])
    assert len(wf.benchmark_rows(half, paths)) == half.expected
    with pytest.raises(ValueError, match="organisers' benchmark"):
        wf.benchmark_rows(half, paths[:-11])


# --------------------------------------------------------------------------
# deterministic selection
# --------------------------------------------------------------------------

def _paths(n, prefix="Real/ffhq/"):
    return [f"{prefix}{i:06d}.png" for i in range(n)]


def test_select_paths_respects_the_cap():
    subset = wf.SUBSETS["real_ffhq"]
    assert len(wf.select_paths(subset, _paths(100), 7, 1)) == 7
    assert len(wf.select_paths(subset, _paths(100), 0, 1)) == 100
    assert len(wf.select_paths(subset, _paths(5), 50, 1)) == 5


def test_select_paths_is_deterministic_across_processes():
    """A resumed run must choose the SAME images as the run it resumes, or the
    cap silently grows. Python's `hash()` is salted per process, so the seed
    derivation must not use it — this asserts the values, not just that two
    calls in one process agree."""
    subset = wf.SUBSETS["real_ffhq"]
    a = wf.select_paths(subset, _paths(1000), 12, 20260827)
    b = wf.select_paths(subset, _paths(1000), 12, 20260827)
    assert a == b
    assert a == ['Real/ffhq/000001.png', 'Real/ffhq/000013.png',
                 'Real/ffhq/000038.png', 'Real/ffhq/000047.png',
                 'Real/ffhq/000073.png', 'Real/ffhq/000240.png',
                 'Real/ffhq/000397.png', 'Real/ffhq/000411.png',
                 'Real/ffhq/000670.png', 'Real/ffhq/000687.png',
                 'Real/ffhq/000781.png', 'Real/ffhq/000882.png']


def test_select_paths_differs_between_seeds_and_between_subsets():
    a = wf.select_paths(wf.SUBSETS["real_ffhq"], _paths(1000), 12, 1)
    assert a != wf.select_paths(wf.SUBSETS["real_ffhq"], _paths(1000), 12, 2)
    assert a != wf.select_paths(wf.SUBSETS["real_afhq"], _paths(1000), 12, 1)


def test_select_paths_is_a_prefix_stable_subset_of_its_input():
    subset = wf.SUBSETS["real_ffhq"]
    chosen = wf.select_paths(subset, _paths(1000), 12, 7)
    assert chosen == sorted(chosen)          # input order preserved
    assert set(chosen) <= set(_paths(1000))
    assert len(set(chosen)) == len(chosen)   # no duplicates


# --------------------------------------------------------------------------
# CSV reading
# --------------------------------------------------------------------------

_HEADER = "Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num"


def _csv(tmp_path, name, rows):
    p = tmp_path / f"{name}.csv"
    p.write_text("\n".join([_HEADER] + rows) + "\n")
    return str(p)


def _row(path, is_fake=1):
    return f"g,a,w,c,0,{is_fake},{path},1"


def test_read_subset_csv_returns_normalised_paths(tmp_path, monkeypatch):
    subset = wf.WildFakeSubset("t", 1, 2, "Diffusion_based/DDIM/", ("Images/x.zip",))
    p = _csv(tmp_path, "t", [_row("./Diffusion_based/DDIM/a/b.jpg"),
                             _row("./Diffusion_based/DDIM/c/d.jpg")])
    assert wf.read_subset_csv(subset, p) == ["Diffusion_based/DDIM/a/b.jpg",
                                             "Diffusion_based/DDIM/c/d.jpg"]


def test_read_subset_csv_rejects_a_changed_row_count(tmp_path):
    subset = wf.WildFakeSubset("t", 1, 3, "Diffusion_based/DDIM/", ("Images/x.zip",))
    p = _csv(tmp_path, "t", [_row("./Diffusion_based/DDIM/a/b.jpg")])
    with pytest.raises(ValueError, match="declares 3"):
        wf.read_subset_csv(subset, p)


def test_read_subset_csv_rejects_a_row_outside_the_declared_prefix(tmp_path):
    """The prefix ties a subset to the archive downloaded for it; a row
    outside it means the registry describes something else."""
    subset = wf.WildFakeSubset("t", 1, 1, "Diffusion_based/DDIM/", ("Images/x.zip",))
    p = _csv(tmp_path, "t", [_row("./Diffusion_based/DDPM/a/b.jpg")])
    with pytest.raises(ValueError, match="outside the prefix"):
        wf.read_subset_csv(subset, p)


def test_read_subset_csv_rejects_a_row_whose_isfake_contradicts_the_label(tmp_path):
    """Writing it would put a fake in the authentic bucket — the C1 defect."""
    subset = wf.WildFakeSubset("t", 0, 1, "Real/ffhq/", ("Images/x.zip",))
    p = _csv(tmp_path, "t", [_row("./Real/ffhq/a.png", is_fake=1)])
    with pytest.raises(ValueError, match="IsFake=1"):
        wf.read_subset_csv(subset, p)


def test_read_subset_csv_rejects_a_csv_without_the_path_column(tmp_path):
    subset = wf.WildFakeSubset("t", 1, 1, "x/", ("Images/x.zip",))
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="no 'Image_path' column"):
        wf.read_subset_csv(subset, str(p))


def test_read_subset_csv_tolerates_an_unparseable_isfake(tmp_path):
    """An unrecognised IsFake spelling must not abort a 200k-row CSV; only a
    value that clearly CONTRADICTS the declared label does."""
    subset = wf.WildFakeSubset("t", 1, 1, "Real/ffhq/", ("Images/x.zip",))
    p = _csv(tmp_path, "t", [_row("./Real/ffhq/a.png", is_fake="?")])
    assert wf.read_subset_csv(subset, p) == ["Real/ffhq/a.png"]
