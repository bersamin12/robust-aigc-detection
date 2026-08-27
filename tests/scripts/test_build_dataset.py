"""End-to-end test of scripts/build_dataset.py: the seam where every Plan 1
module (audit, normalize, dedupe, splits, manifest) meets. Runs entirely
against synthetic fixtures under tmp_path -- never against real data (a real
run is a human decision).
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import read_manifest
from aigcdet.data.splits import DEFAULT_SEED, MIN_HELDOUT_IMAGES

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "build_dataset.py")


def _load_build_dataset_module():
    spec = importlib.util.spec_from_file_location("build_dataset_script", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load_build_dataset_module()


def _write_images(raw_root, source, bucket, n, rng, size=32):
    d = os.path.join(raw_root, source, bucket)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        p = os.path.join(d, f"{i:05d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


N_PER_GEN = MIN_HELDOUT_IMAGES + 2  # clear the threshold with a small margin
GENS = ("g1", "g2", "g3", "g4")
N_REAL = 60
N_COCO = 12

LICENCES = {
    "wildfake": "see https://modelscope.cn/datasets/hy2628982280/WildFake — confirm before use",
    "real_src": "CC0 — https://example.org/real_src",
    "coco_val2017": "CC BY 4.0 — https://cocodataset.org/#termsofuse",
}


def _build_raw_tree(raw_dir, rng, licences=LICENCES):
    for g in GENS:
        _write_images(raw_dir, "wildfake", g, N_PER_GEN, rng)
    real_paths = _write_images(raw_dir, "real_src", "real", N_REAL, rng)
    _write_images(raw_dir, "coco_val2017", "real", N_COCO, rng)
    with open(os.path.join(raw_dir, "LICENCES.json"), "w") as f:
        for dataset, licence in licences.items():
            f.write(json.dumps({dataset: licence}) + "\n")
    return real_paths


def test_missing_licences_file_raises_loudly(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _write_images(str(raw_dir), "real_src", "real", 5, rng)
    with pytest.raises(FileNotFoundError, match="LICENCES.json"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        )


def test_source_missing_from_licences_raises_loudly(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(0)
    _write_images(str(raw_dir), "real_src", "real", 5, rng)
    # LICENCES.json exists but has no entry for "real_src".
    with open(raw_dir / "LICENCES.json", "w") as f:
        f.write(json.dumps({"some_other_source": "CC0"}) + "\n")
    with pytest.raises(ValueError, match="real_src"):
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
    assert set(held) <= set(GENS)
    assert splits_meta["seed"] == DEFAULT_SEED

    # Exactly one leak (the planted duplicate) was dropped, and it came from
    # the training side: real_src had N_REAL images, one is gone.
    assert splits_meta["leaked_dropped"] == 1
    assert (df["source"] == "real_src").sum() == N_REAL - 1

    # coco_val2017 (a COCO-derived authentic source) is excluded from
    # training entirely, per spec §4.1(2) -- not merely deduped.
    assert "coco_val2017" not in set(df["source"])

    # Fake generator totals: all of wildfake survives (no leaks planted
    # there), so every held generator's full count landed in heldout_generator.
    assert (df["generator"].isin(GENS)).sum() == N_PER_GEN * len(GENS)

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
    # Total real_src rows dropped by exactly one already confirms a row was
    # removed; check by pixel content that it is specifically the one that
    # duplicates the planted demo image, not some other real_src row.
    with Image.open(os.path.join(demo_dir, "demo_planted_dup.png")) as demo_im:
        demo_arr = np.asarray(demo_im.convert("RGB"))
    for p in df[df["source"] == "real_src"]["path"]:
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
    assert train_pool_gens == set(GENS) - set(held)

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
        {"some_other_source": "CC0"},  # missing key entirely
        {"real_src": None},            # JSON null
        {"real_src": ""},              # empty string
        {"real_src": "   "},           # whitespace-only
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
    _write_images(str(raw_dir), "real_src", "real", 5, rng)
    with open(raw_dir / "LICENCES.json", "w") as f:
        f.write(json.dumps(licences_json) + "\n")
    with pytest.raises(ValueError, match="real_src"):
        bd.build_dataset(
            str(raw_dir), str(tmp_path / "out"), str(tmp_path / "demo"),
            str(tmp_path / "manifest.parquet"), docs_dir=str(tmp_path / "docs"),
        )


def _small_raw_tree(raw_dir, rng):
    # Needs >=2 generators clearing MIN_HELDOUT_IMAGES so the full pipeline
    # (including choose_heldout_generators) actually succeeds on the first
    # build -- the overwrite guard is tested on the *second* call.
    for g in GENS[:2]:
        _write_images(str(raw_dir), "wildfake", g, N_PER_GEN, rng)
    _write_images(str(raw_dir), "real_src", "real", 5, rng)
    with open(raw_dir / "LICENCES.json", "w") as f:
        f.write(json.dumps({"real_src": "CC0"}) + "\n")
        f.write(json.dumps({"wildfake": "CC0"}) + "\n")


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
