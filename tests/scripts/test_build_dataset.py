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
