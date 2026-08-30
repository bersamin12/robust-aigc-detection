"""End-to-end test of scripts/build_dataset.py: the seam where every Plan 1
module (audit, normalize, dedupe, splits, manifest) meets.

The raw tree is built from the paths `scripts/acquire_data.py` ACTUALLY
writes -- COCO's through the real `acquire_coco_val2017` (against a locally
planted zip, so nothing is downloaded), the rest through
`aigcdet.data.sources.raw_subdir`, the same function the acquisition script
uses. The previous version of this file fabricated `raw/coco_val2017/real/`,
a directory the acquisition script never creates, and so encoded the absence
of C1: in the real layout the bucket is `val2017`, and every COCO photograph
was labelled AI-generated.

Runs entirely against synthetic fixtures under tmp_path -- never against real
data (a real run is a human decision).
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import read_manifest
from aigcdet.data.sources import LICENCES, raw_subdir
from aigcdet.data.splits import DEFAULT_SEED, MIN_HELDOUT_IMAGES

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")


def _load_script(name: str):
    path = os.path.join(_SCRIPTS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load_script("build_dataset")
ad = _load_script("acquire_data")


def _write_images(raw_root, source, bucket, n, rng, size=32):
    d = os.path.join(str(raw_root), source, bucket)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        p = os.path.join(d, f"{i:05d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _acquire_coco(raw_dir, n, rng, size=32):
    """Produce the COCO tree by running the REAL acquisition function.

    `acquire_coco_val2017` skips the download when the zip is already on
    disk, so planting a synthetic val2017.zip exercises its actual
    extraction layout with no network access. If that layout ever changes,
    this test -- not a production run -- is what notices.
    """
    raw_dir = str(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    zp = os.path.join(raw_dir, "val2017.zip")
    with zipfile.ZipFile(zp, "w") as z:
        for i in range(n):
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            tmp = os.path.join(raw_dir, "_stage.jpg")
            Image.fromarray(arr).save(tmp, format="JPEG", quality=92)
            z.write(tmp, arcname=f"val2017/{i:012d}.jpg")
    os.remove(os.path.join(raw_dir, "_stage.jpg"))
    ad.acquire_coco_val2017(raw_dir)


N_PER_GEN = MIN_HELDOUT_IMAGES + 2   # clear the threshold with a small margin
GENS = ("sdxl", "sd15", "midjourney", "flux")
N_REAL = 60
N_COCO = 12
# Enough unattributed SID_Set fakes to clear MIN_HELDOUT_IMAGES: the pseudo
# generator "sid_set" is therefore *selectable* on count alone, and must be
# rejected on eligibility instead (I2).
N_SID_FAKE = MIN_HELDOUT_IMAGES + 2


def _write_licences(raw_dir, licences=None):
    licences = LICENCES if licences is None else licences
    with open(os.path.join(str(raw_dir), "LICENCES.json"), "w") as f:
        for dataset, licence in licences.items():
            f.write(json.dumps({dataset: licence}) + "\n")


def _build_raw_tree(raw_dir, rng):
    for g in GENS:
        _write_images(raw_dir, "wildfake", raw_subdir("wildfake", 1, g), N_PER_GEN, rng)
    _write_images(raw_dir, "sid_set", raw_subdir("sid_set", 1), N_SID_FAKE, rng)
    real_paths = _write_images(raw_dir, "sid_set", raw_subdir("sid_set", 0), N_REAL, rng)
    _acquire_coco(raw_dir, N_COCO, rng)
    _write_licences(raw_dir)
    return real_paths


def test_missing_licences_file_raises_loudly(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _write_images(raw_dir, "sid_set", "real", 5, rng)
    with pytest.raises(FileNotFoundError, match="LICENCES.json"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        )


def test_source_missing_from_licences_raises_loudly(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _write_images(raw_dir, "sid_set", "real", 5, rng)
    # LICENCES.json exists but has no entry for "sid_set".
    _write_licences(raw_dir, {"wildfake": "CC0"})
    with pytest.raises(ValueError, match="sid_set"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        )


def test_unregistered_raw_source_directory_raises(tmp_path):
    """A source nobody declared must stop the build, not be guessed at."""
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _write_images(raw_dir, "some_new_dataset", "real", 3, rng)
    _write_licences(raw_dir, {"some_new_dataset": "CC0"})
    with pytest.raises(ValueError, match="unregistered raw source"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        )


def test_end_to_end_pipeline(tmp_path):
    raw_dir = str(tmp_path / "raw")
    demo_dir = str(tmp_path / "demo")
    out_dir = str(tmp_path / "normalized")
    manifest_path = str(tmp_path / "manifest.parquet")
    docs_dir = str(tmp_path / "docs")
    os.makedirs(demo_dir, exist_ok=True)

    rng = np.random.default_rng(20260827)
    real_paths = _build_raw_tree(raw_dir, rng)

    # Plant a near-duplicate of a training image in the demo set: an exact
    # byte-identical copy is the simplest possible collision at Hamming
    # distance 0, well within max_distance=4. This is the leak the guard
    # must catch and drop from the TRAINING side, never from the demo side.
    duplicated_src = real_paths[0]
    with Image.open(duplicated_src) as im:
        im.save(os.path.join(demo_dir, "demo_planted_dup.png"))
    # A handful of unrelated demo images so the demo set isn't trivially
    # just the one planted duplicate.
    for i in range(4):
        arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(demo_dir, f"demo_other_{i}.png"))

    df = bd.build_dataset(raw_dir, out_dir, demo_dir, manifest_path,
                          workers=4, docs_dir=docs_dir)

    # The function's return value and the written parquet must agree.
    on_disk = read_manifest(manifest_path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), on_disk.reset_index(drop=True))

    with open(os.path.join(docs_dir, "splits.json")) as f:
        splits_meta = json.load(f)
    held = splits_meta["heldout_generators"]
    assert len(held) == 2
    # C1/I2: neither a COCO pseudo-generator nor a dataset-level one.
    assert set(held) <= set(GENS)
    assert splits_meta["seed"] == DEFAULT_SEED

    # Exactly one leak (the planted duplicate) was dropped, and it came from
    # the training side: sid_set/real had N_REAL images, one is gone.
    assert splits_meta["leaked_dropped"] == 1
    assert ((df["source"] == "sid_set") & (df["label"] == 0)).sum() == N_REAL - 1

    # --- C1: COCO val2017 is authentic, and excluded from training entirely
    # (spec §4.1(2)) -- not silently ingested as AI-generated. ---
    assert "coco_val2017" not in set(df["source"])
    assert not df["source"].str.contains("coco").any()

    # --- I2: the SID_Set fakes carry the dataset-level pseudo-generator,
    # they clear MIN_HELDOUT_IMAGES on count, and they are still never
    # chosen as a held-out "generator family". ---
    sid_fakes = df[(df["source"] == "sid_set") & (df["label"] == 1)]
    assert set(sid_fakes["generator"]) == {"sid_set"}
    assert len(sid_fakes) >= MIN_HELDOUT_IMAGES
    assert "sid_set" not in held
    assert set(sid_fakes["split"]) <= {"train", "val_internal"}

    # Fake generator totals: all of wildfake survives (no leaks planted
    # there), so every held generator's full count landed in heldout_generator.
    assert (df["generator"].isin(GENS)).sum() == N_PER_GEN * len(GENS)

    # --- I3: manifest paths are absolute (manifest.py's stated contract;
    # Plans 2 and 3 open row["path"] from other working directories). ---
    assert df["path"].map(os.path.isabs).all()
    assert df["path"].map(os.path.exists).all()

    # --- No split overlap: every path appears exactly once, and each split
    # is a disjoint set of paths. ---
    assert df["path"].is_unique
    by_split = {s: set(rows["path"]) for s, rows in df.groupby("split")}
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            assert by_split[splits[i]].isdisjoint(by_split[splits[j]])
    assert set(df["split"]) <= {"train", "val_internal", "heldout_generator"}

    # --- The leaked demo image never appears in the final manifest. ---
    # Total real rows dropped by exactly one already confirms a row was
    # removed; check by pixel content that it is specifically the one that
    # duplicates the planted demo image, not some other authentic row.
    with Image.open(os.path.join(demo_dir, "demo_planted_dup.png")) as demo_im:
        demo_arr = np.asarray(demo_im.convert("RGB"))
    for p in df[df["label"] == 0]["path"]:
        with Image.open(p) as im:
            assert not np.array_equal(np.asarray(im.convert("RGB")), demo_arr)

    # --- Held-out generators are absent from train (and val_internal). ---
    non_heldout_splits = df[df["split"] != "heldout_generator"]
    assert not set(non_heldout_splits["generator"]) & set(held)
    heldout_rows = df[df["split"] == "heldout_generator"]
    assert set(heldout_rows["generator"]) == set(held)
    # Every held-out generator's images are ALL in heldout_generator, none
    # leaked into train or val_internal.
    for g in held:
        assert (df[df["generator"] == g]["split"] == "heldout_generator").all()

    # Non-held-out generators are absent from heldout_generator.
    train_pool_gens = set(df[df["split"] != "heldout_generator"]["generator"]) - {""}
    assert train_pool_gens == (set(GENS) - set(held)) | {"sid_set"}

    # --- Per-row licence provenance matches LICENCES.json exactly. ---
    for source, licence in LICENCES.items():
        rows = df[df["source"] == source]
        if rows.empty:
            continue  # coco_val2017: excluded entirely, nothing to check
        assert set(rows["licence"]) == {licence}
    assert (df["licence"] != "").all()
    assert (df["licence"] != "UNRECORDED").all()

    # --- Both classes are present, and train/val actually got fake images
    # from the non-held-out generators. ---
    train = df[df["split"] == "train"]
    assert set(train["label"]) == {0, 1}

    # --- The audit and split-report side effects exist under docs_dir. ---
    assert os.path.exists(os.path.join(docs_dir, "data_audit.md"))


@pytest.mark.parametrize(
    "licences_json",
    [
        {"wildfake": "CC0"},          # missing key entirely
        {"sid_set": None},            # JSON null
        {"sid_set": ""},              # empty string
        {"sid_set": "   "},           # whitespace-only
    ],
    ids=["missing-key", "null", "empty-string", "whitespace-only"],
)
def test_every_shape_of_blank_licence_raises_loudly_and_names_the_source(tmp_path, licences_json):
    # One test pinning the whole class of blank-provenance values, rather
    # than one narrow test per shape: a missing key, JSON null, "", and
    # whitespace-only must all be rejected by the SAME check, naming the
    # offending source in every case. (Two earlier fix rounds each patched
    # one shape of this and reopened another -- this test is what stops
    # that from happening again.)
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _write_images(raw_dir, "sid_set", "real", 5, rng)
    _write_licences(raw_dir, licences_json)
    with pytest.raises(ValueError, match="sid_set"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        )


def _small_raw_tree(raw_dir, rng):
    # Needs >=2 generators clearing MIN_HELDOUT_IMAGES so the full pipeline
    # (including choose_heldout_generators) actually succeeds on the first
    # build -- the overwrite guard is tested on the *second* call.
    for g in GENS[:2]:
        _write_images(raw_dir, "wildfake", g, N_PER_GEN, rng)
    _write_images(raw_dir, "sid_set", "real", 5, rng)
    _write_licences(raw_dir, {"sid_set": "CC0", "wildfake": "CC0"})


def test_heldout_generators_can_be_pinned_by_a_human(tmp_path):
    """The --heldout-generators override: a human pins the choice instead of
    reseeding until the automatic draw obliges."""
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _small_raw_tree(raw_dir, rng)
    os.makedirs(tmp_path / "demo", exist_ok=True)
    df = bd.build_dataset(
        str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
        str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        heldout_generators=[GENS[1]],
    )
    with open(os.path.join(str(tmp_path / "docs"), "splits.json")) as f:
        assert json.load(f)["heldout_generators"] == [GENS[1]]
    assert set(df[df["split"] == "heldout_generator"]["generator"]) == {GENS[1]}
    assert GENS[1] not in set(df[df["split"] != "heldout_generator"]["generator"])


def test_one_unreadable_image_does_not_kill_the_run(tmp_path):
    """I5: `list(ex.map(...))` aborted the whole pipeline on the first
    truncated file, and the ~20-minute audit pass had to re-run behind it.
    The bad file must be dropped, counted and written down instead."""
    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    rng = np.random.default_rng(0)
    _small_raw_tree(raw_dir, rng)
    os.makedirs(tmp_path / "demo", exist_ok=True)

    real_dir = pathlib.Path(str(raw_dir), "sid_set", "real")
    data = (real_dir / "00000.png").read_bytes()
    truncated = real_dir / "truncated.png"
    truncated.write_bytes(data[: len(data) // 2])

    df = bd.build_dataset(
        str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
        str(tmp_path / "manifest.parquet"), docs_dir=str(docs_dir),
    )
    # Every good image survives; only the unreadable one is missing.
    assert len(df) == N_PER_GEN * 2 + 5

    with open(docs_dir / "normalize_skipped.json") as f:
        skipped = json.load(f)
    assert [row["src"] for row in skipped] == [str(truncated)]
    assert skipped[0]["reason"]  # the reason is recorded, not swallowed
    with open(docs_dir / "splits.json") as f:
        assert json.load(f)["normalize_skipped"] == 1


def test_refuses_to_overwrite_an_existing_manifest_without_force(tmp_path):
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    rng = np.random.default_rng(0)
    _small_raw_tree(raw_dir, rng)
    os.makedirs(tmp_path / "demo", exist_ok=True)

    bd.build_dataset(
        str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
        str(manifest_path), docs_dir=str(tmp_path / "docs"),
    )
    assert manifest_path.exists()
    first_mtime = manifest_path.stat().st_mtime_ns

    with pytest.raises(FileExistsError, match="already exists"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(manifest_path), docs_dir=str(tmp_path / "docs"),
        )
    # The refusal happened before any work was redone: the file on disk is
    # untouched.
    assert manifest_path.stat().st_mtime_ns == first_mtime


def test_force_overwrites_an_existing_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    rng = np.random.default_rng(0)
    _small_raw_tree(raw_dir, rng)
    os.makedirs(tmp_path / "demo", exist_ok=True)

    bd.build_dataset(
        str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
        str(manifest_path), docs_dir=str(tmp_path / "docs"),
    )
    df = bd.build_dataset(
        str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
        str(manifest_path), docs_dir=str(tmp_path / "docs"), force=True,
    )
    on_disk = read_manifest(str(manifest_path))
    expected_n = N_PER_GEN * 2 + 5
    assert len(df) == len(on_disk) == expected_n


# --- Licence-restricted buckets, and the balance cap ----------------------
# The 28 Aug webinar Q&A: "Non-commercial datasets cannot be used." WildFake's
# authentic bucket is re-published FFHQ/CelebA-HQ/AFHQ/ImageNet/LSUN, so it is
# barred while its generated buckets stay. See aigcdet.data.sources.


def _build_raw_tree_with_wildfake_real(raw_dir, rng, n_wf_real=30):
    real_paths = _build_raw_tree(raw_dir, rng)
    _write_images(raw_dir, "wildfake", raw_subdir("wildfake", 0), n_wf_real, rng)
    return real_paths


@pytest.fixture
def restrict_wildfake_real(monkeypatch):
    """Register WildFake with its authentic bucket barred, for the tests of
    the restriction mechanism and of the cap that rebalances after one.

    The real registry restricts nothing: the organisers' 29 Aug rules slide
    lists WildFake as an approved dataset, and the frozen manifest includes
    its authentic bucket. The mechanism is kept for a source that needs it,
    and is exercised here against one registered for the test only."""
    from aigcdet.data import sources
    base = sources.SOURCES["wildfake"]
    monkeypatch.setitem(sources.SOURCES, "wildfake", sources.SourceSpec(
        name=base.name, licence=base.licence, real_buckets=base.real_buckets,
        generator_buckets=base.generator_buckets,
        restricted_buckets=frozenset({"real"}),
        restriction="upstream terms are non-commercial (test)"))


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_a_restricted_bucket_never_reaches_the_manifest(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(1)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"))

    assert ((df["source"] == "wildfake") & (df["label"] == 0)).sum() == 0
    # ... and only the authentic bucket went. WildFake is barred in part, not
    # excluded wholesale: its generated images are the authors' own work and
    # are the entire reason the corpus has generator diversity at all.
    assert set(df[df["source"] == "wildfake"]["generator"]) == set(GENS)
    # The authentic images that remain are SID_Set's, which are CC BY 4.0.
    assert set(df[df["label"] == 0]["source"]) == {"sid_set"}


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_a_restricted_bucket_is_dropped_before_it_is_normalised(tmp_path):
    # Not merely filtered out of the manifest afterwards. Normalising 55,000
    # images we may not use costs an hour of wall-clock and ~17 GB, and a
    # copy of a non-commercial image on our disk is the thing the rule is
    # about -- the manifest row is only its shadow.
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    out_dir = str(tmp_path / "norm")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(2)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    bd.build_dataset(raw_dir, out_dir, demo_dir, str(tmp_path / "m.parquet"),
                     workers=4, docs_dir=str(tmp_path / "docs"))

    # `dst` is out/<source>/<generator or "real">/, so a normalised WildFake
    # authentic image could only land here.
    assert not os.path.exists(os.path.join(out_dir, "wildfake", "real"))


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_the_restriction_is_recorded_with_the_reason_it_fired(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    docs_dir = str(tmp_path / "docs")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(3)
    _build_raw_tree_with_wildfake_real(raw_dir, rng, n_wf_real=30)

    bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                     str(tmp_path / "m.parquet"), workers=4, docs_dir=docs_dir)

    with open(os.path.join(docs_dir, "splits.json")) as f:
        meta = json.load(f)
    assert meta["restricted_dropped"] == {"wildfake/real": 30}
    # A count with no reason is a number, not an audit trail.
    assert "commercial" in meta["restriction_reasons"]["wildfake"].lower()


def test_nothing_is_recorded_as_restricted_when_nothing_was(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    docs_dir = str(tmp_path / "docs")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(4)
    _build_raw_tree(raw_dir, rng)          # no wildfake/real at all

    bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                     str(tmp_path / "m.parquet"), workers=4, docs_dir=docs_dir)

    with open(os.path.join(docs_dir, "splits.json")) as f:
        meta = json.load(f)
    assert meta["restricted_dropped"] == {}
    assert meta["restriction_reasons"] == {}


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_max_per_generator_caps_generated_families_and_leaves_authentic_alone(tmp_path):
    # Dropping WildFake's authentic half leaves the corpus lopsided: every
    # real image now comes from SID_Set while the generated side keeps
    # ~19 WildFake families. The cap is how that is rebalanced without
    # touching the raw tree or losing a family.
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(5)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    cap = N_PER_GEN - 1                    # still clears MIN_HELDOUT_IMAGES
    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"),
                          max_per_generator=cap)

    counts = df[df["label"] == 1]["generator"].value_counts()
    assert counts.max() <= cap
    # Every family survives -- the cap thins families, it does not delete
    # them, which is what makes heldout_generator and LOTO still possible.
    assert set(counts.index) == set(GENS) | {"sid_set"}
    # Authentic images are untouched: they are the scarce side.
    assert (df["label"] == 0).sum() == N_REAL


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_the_cap_is_deterministic_given_the_seed(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(6)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    kw = dict(workers=4, max_per_generator=N_PER_GEN - 1,
              docs_dir=str(tmp_path / "docs"))
    a = bd.build_dataset(raw_dir, str(tmp_path / "n1"), demo_dir,
                         str(tmp_path / "m1.parquet"), **kw)
    b = bd.build_dataset(raw_dir, str(tmp_path / "n2"), demo_dir,
                         str(tmp_path / "m2.parquet"), **kw)
    # Compared on content_sha256, which identifies the IMAGE. Neither `path`
    # nor `rel_path` can be used here: the normalised filename is the row's
    # POSITION (`{i:07d}.png`), so two runs that kept entirely different
    # images still produce byte-identical path lists. That is exactly the
    # assertion this test would have made vacuously.
    assert a["content_sha256"].tolist() == b["content_sha256"].tolist()

    c = bd.build_dataset(raw_dir, str(tmp_path / "n3"), demo_dir,
                         str(tmp_path / "m3.parquet"),
                         seed=DEFAULT_SEED + 1, **kw)
    assert c["content_sha256"].tolist() != a["content_sha256"].tolist()


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_a_cap_larger_than_every_family_changes_nothing(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    kw = dict(workers=4, docs_dir=str(tmp_path / "docs"))
    uncapped = bd.build_dataset(raw_dir, str(tmp_path / "n1"), demo_dir,
                                str(tmp_path / "m1.parquet"), **kw)
    capped = bd.build_dataset(raw_dir, str(tmp_path / "n2"), demo_dir,
                              str(tmp_path / "m2.parquet"),
                              max_per_generator=10_000, **kw)
    assert capped["content_sha256"].tolist() == uncapped["content_sha256"].tolist()


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_the_cap_is_recorded_so_a_rebuild_can_be_reproduced(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    docs_dir = str(tmp_path / "docs")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(8)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    cap = N_PER_GEN - 1
    bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                     str(tmp_path / "m.parquet"), workers=4, docs_dir=docs_dir,
                     max_per_generator=cap)

    with open(os.path.join(docs_dir, "splits.json")) as f:
        meta = json.load(f)
    assert meta["max_per_generator"] == cap
    # SID_Set's unattributed fakes carry the dataset-level pseudo-generator
    # and are capped like any other family: the knob is about how many
    # generated images of each KIND the corpus holds, and "unattributed" is
    # a kind.
    assert meta["capped_dropped"] == {g: 1 for g in (*GENS, "sid_set")}


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_capped_images_are_not_normalised_either(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    out_dir = str(tmp_path / "norm")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(9)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    cap = N_PER_GEN - 2                    # still clears MIN_HELDOUT_IMAGES
    bd.build_dataset(raw_dir, out_dir, demo_dir, str(tmp_path / "m.parquet"),
                     workers=4, docs_dir=str(tmp_path / "docs"),
                     max_per_generator=cap)

    for g in GENS:
        n = len(os.listdir(os.path.join(out_dir, "wildfake", g)))
        assert n == cap, f"{g}: normalised {n} images for a cap of {cap}"


@pytest.mark.usefixtures("restrict_wildfake_real")
def test_the_cap_never_thins_the_authentic_side(tmp_path):
    # The cap is expressed over generator FAMILIES, and authentic rows carry
    # generator "". Treating "" as a family would thin the scarce side --
    # precisely inverting what the knob is for.
    #
    # The test above cannot see that: its cap (201) sits above the authentic
    # count (60), so the branch never fires and the assertion passes either
    # way. Here the cap is well below it. The held-out families are pinned
    # because a cap this small leaves none of them clearing
    # MIN_HELDOUT_IMAGES for the automatic draw.
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(10)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    cap = N_REAL // 2
    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"),
                          max_per_generator=cap,
                          heldout_generators=list(GENS[:2]))

    assert (df["label"] == 0).sum() == N_REAL
    assert df[df["label"] == 1]["generator"].value_counts().max() <= cap


# ---------------------------------------------------------------------------
# corpus presets (aigcdet.data.presets)
#
# The knobs a preset adds -- an authentic-side cap, a sub-band floor, and a
# per-family cap mapping -- are each a way of DELETING rows, so every test
# below is really the same question: did it delete exactly the rows it named,
# and did it refuse when it could not?
# ---------------------------------------------------------------------------

from aigcdet.data.presets import DatasetPreset  # noqa: E402


def _preset(**kw) -> DatasetPreset:
    return DatasetPreset(name=kw.pop("name", "t"),
                         note=kw.pop("note", "a test composition"), **kw)


def test_a_real_cap_thins_the_named_source_and_leaves_its_fakes_alone(tmp_path):
    """`max_per_generator` deliberately never touches authentic rows, so
    balancing a dominant authentic source needed its own knob. This is the
    source-balancing lever `augment.canonical` asks for by name."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(11)
    _build_raw_tree_with_wildfake_real(raw_dir, rng, n_wf_real=30)

    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"),
                          preset=_preset(max_real_per_source={"wildfake": 10}))

    wf_real = df[(df["source"] == "wildfake") & (df["label"] == 0)]
    assert len(wf_real) == 10
    # The OTHER source's authentic rows are untouched: the cap is per source,
    # which is the entire point of it being a mapping.
    assert (df[(df["source"] == "sid_set") & (df["label"] == 0)].shape[0]
            == N_REAL)
    # And WildFake's generated families are untouched by a REAL cap.
    assert (df[(df["source"] == "wildfake")
               & (df["label"] == 1)].shape[0] == len(GENS) * N_PER_GEN)


def test_the_cap_mapping_can_exempt_the_pseudo_generator(tmp_path):
    """A single number cannot say what P1 needs to say. `sid_set` names a
    SOURCE, not a family, so a per-family cap on it thins a whole dataset --
    the same category error as holding it out."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(12)
    _build_raw_tree(raw_dir, rng)

    cap = N_PER_GEN - 5
    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"),
                          preset=_preset(max_per_generator={"*": cap,
                                                            "sid_set": 0},
                                         heldout_generators=["flux"]))
    counts = df[df["label"] == 1]["generator"].value_counts()
    for g in GENS:
        assert counts[g] == cap
    assert counts["sid_set"] == N_SID_FAKE           # exempt, not capped


def test_the_sub_band_floor_drops_only_images_below_it(tmp_path):
    """The residue `augment.canonical` calls irreducible: it band-limits to
    CANON_BAND_SIDE and upscales, which equalises everything AT or ABOVE that
    ceiling and can do nothing below it."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(13)
    # Two families that differ only in native size, straddling the floor.
    _write_images(raw_dir, "wildfake", raw_subdir("wildfake", 1, "tiny"),
                  N_PER_GEN, rng, size=64)
    _write_images(raw_dir, "wildfake", raw_subdir("wildfake", 1, "big"),
                  N_PER_GEN, rng, size=300)
    _write_images(raw_dir, "sid_set", raw_subdir("sid_set", 0),
                  N_REAL, rng, size=300)
    _write_images(raw_dir, "sid_set", raw_subdir("sid_set", 1),
                  N_SID_FAKE, rng, size=300)
    _write_licences(raw_dir)

    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"),
                          preset=_preset(min_short_side=200,
                                         heldout_generators=["big"]))
    counts = df["generator"].value_counts()
    assert "tiny" not in counts                    # 64px: entirely below
    assert counts["big"] == N_PER_GEN              # 300px: entirely above
    assert np.minimum(df["width"], df["height"]).min() >= 200

    with open(os.path.join(str(tmp_path / "docs"), "splits.json")) as f:
        meta = json.load(f)
    assert meta["min_short_side"] == 200
    assert meta["below_short_side_dropped"] == {"wildfake/tiny": N_PER_GEN}


def test_a_preset_naming_a_family_the_corpus_does_not_have_raises(tmp_path):
    """A cap on an absent family caps nothing and a hold-out on one holds
    nothing out -- both silently. The source registry is static so
    `DatasetPreset` checks it; generator names come from the data, so they can
    only be checked against the scan."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(14)
    _build_raw_tree(raw_dir, rng)

    with pytest.raises(ValueError, match="dalle4"):
        bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                         str(tmp_path / "m.parquet"), workers=4,
                         docs_dir=str(tmp_path / "docs"),
                         preset=_preset(max_per_generator={"dalle4": 10}))


def test_a_preset_plus_an_overlapping_argument_raises(tmp_path):
    """Two sources of truth for one knob is the thing presets exist to
    remove: whichever won, the record of what built the corpus would be
    ambiguous."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(15)
    _build_raw_tree(raw_dir, rng)

    with pytest.raises(ValueError, match="max_per_generator"):
        bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                         str(tmp_path / "m.parquet"), workers=4,
                         docs_dir=str(tmp_path / "docs"),
                         max_per_generator=5, preset=_preset())


def test_the_preset_identity_reaches_splits_json(tmp_path):
    """A manifest records the numbers it was built with but not which
    decision they were. Without the name and note in the receipt, a bank
    found on disk cannot be traced back to a composition."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(16)
    _build_raw_tree(raw_dir, rng)

    docs = str(tmp_path / "docs")
    bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                     str(tmp_path / "m.parquet"), workers=4, docs_dir=docs,
                     preset=_preset(name="p1", note="why  this\ncorpus",
                                    max_real_per_source={"sid_set": 20}))
    with open(os.path.join(docs, "splits.json")) as f:
        meta = json.load(f)
    assert meta["preset"]["name"] == "p1"
    assert meta["preset"]["note"] == "why this corpus"
    assert meta["max_real_per_source"] == {"sid_set": 20}
    assert meta["capped_real_dropped"] == {"sid_set": N_REAL - 20}


def test_a_preset_holdout_is_used_instead_of_the_seeded_draw(tmp_path):
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(17)
    _build_raw_tree(raw_dir, rng)

    df = bd.build_dataset(raw_dir, str(tmp_path / "norm"), demo_dir,
                          str(tmp_path / "m.parquet"), workers=4,
                          docs_dir=str(tmp_path / "docs"),
                          preset=_preset(heldout_generators=["flux", "sdxl"]))
    held = set(df[df["split"] == "heldout_generator"]["generator"])
    assert held == {"flux", "sdxl"}


def test_a_preset_build_is_deterministic_given_the_seed(tmp_path):
    """Both new caps draw from one seeded generator in a fixed order, so two
    runs of the same preset must keep the same IMAGES -- compared on
    content_sha256, because the normalised filename is the row's position and
    two runs that kept different images still produce identical path lists."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(18)
    _build_raw_tree_with_wildfake_real(raw_dir, rng, n_wf_real=30)

    kw = dict(workers=4, docs_dir=str(tmp_path / "docs"),
              preset=_preset(max_real_per_source={"wildfake": 10},
                             max_per_generator={"*": N_PER_GEN - 3,
                                                "sid_set": 0},
                             heldout_generators=["flux"]))
    a = bd.build_dataset(raw_dir, str(tmp_path / "n1"), demo_dir,
                         str(tmp_path / "m1.parquet"), **kw)
    b = bd.build_dataset(raw_dir, str(tmp_path / "n2"), demo_dir,
                         str(tmp_path / "m2.parquet"), **kw)
    assert a["content_sha256"].tolist() == b["content_sha256"].tolist()


def test_a_preset_subpath_exclusion_drops_that_directory_only(tmp_path):
    """The WildFake case, in miniature.

    WildFake's authentic images are nested one level BELOW the bucket
    (`wildfake/real/<subset>/`) so that `classify` still reads bucket "real"
    and label 0 while the directory records which upstream they came from.
    `restricted_buckets` therefore cannot bar five of the six subsets and keep
    the sixth: `is_restricted_bucket` is asked with `bucket == "real"` for all
    of them.
    """
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(21)
    _build_raw_tree(raw_dir, rng)
    real_bucket = os.path.join("wildfake", raw_subdir("wildfake", 0))
    _write_images(raw_dir, real_bucket, "real_ffhq", 12, rng)
    _write_images(raw_dir, real_bucket, "real_laion5b", 9, rng)

    docs = str(tmp_path / "docs")
    df = bd.build_dataset(
        raw_dir, str(tmp_path / "norm"), demo_dir, str(tmp_path / "m.parquet"),
        workers=4, docs_dir=docs,
        preset=_preset(exclude_subpaths=["wildfake/real/real_ffhq"]))

    # The kept subset survives; the excluded one is gone; nothing else moved.
    wf_real = df[(df["source"] == "wildfake") & (df["label"] == 0)]
    assert len(wf_real) == 9
    assert (df["source"] == "sid_set").sum() == N_REAL + N_SID_FAKE

    with open(os.path.join(docs, "splits.json")) as f:
        meta = json.load(f)
    assert meta["excluded_subpath_dropped"] == {"wildfake/real/real_ffhq/": 12}
    assert meta["preset"]["exclude_subpaths"] == ["wildfake/real/real_ffhq"]


def test_a_subpath_exclusion_does_not_match_a_sibling_by_prefix(tmp_path):
    """`real_ffhq` must not also take `real_ffhq_v2`. This is what the
    trailing separator in `excluded_prefixes` buys, and a bare `startswith`
    would silently delete the sibling."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(22)
    _build_raw_tree(raw_dir, rng)
    real_bucket = os.path.join("wildfake", raw_subdir("wildfake", 0))
    _write_images(raw_dir, real_bucket, "real_ffhq", 5, rng)
    _write_images(raw_dir, real_bucket, "real_ffhq_v2", 7, rng)

    df = bd.build_dataset(
        raw_dir, str(tmp_path / "norm"), demo_dir, str(tmp_path / "m.parquet"),
        workers=4, docs_dir=str(tmp_path / "docs"),
        preset=_preset(exclude_subpaths=["wildfake/real/real_ffhq"]))
    assert ((df["source"] == "wildfake") & (df["label"] == 0)).sum() == 7


def test_a_subpath_exclusion_that_matches_nothing_raises(tmp_path):
    """An exclusion that excludes nothing leaves a corpus disagreeing with the
    preset describing it. The path names a directory in someone's raw tree, so
    it cannot be checked against the static registry -- the scan is the only
    place it can be checked at all."""
    raw_dir, demo_dir = str(tmp_path / "raw"), str(tmp_path / "demo")
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(23)
    _build_raw_tree_with_wildfake_real(raw_dir, rng)

    with pytest.raises(ValueError, match="matched no images"):
        bd.build_dataset(
            raw_dir, str(tmp_path / "norm"), demo_dir,
            str(tmp_path / "m.parquet"), workers=4,
            docs_dir=str(tmp_path / "docs"),
            preset=_preset(exclude_subpaths=["wildfake/real/real_typo"]))
