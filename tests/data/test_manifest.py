import os

import numpy as np
import pandas as pd
import pytest

from aigcdet.data.manifest import (
    DUMMY_GENERATORS, MANIFEST_COLUMNS, make_dummy_manifest, read_manifest,
    validate_manifest, write_manifest,
)


def _valid_rows(n=4):
    return pd.DataFrame([{
        "path": os.path.abspath(f"/tmp/img_{i}.png"), "label": i % 2,
        "generator": "sdxl" if i % 2 else "", "source": "wildfake",
        "licence": "CC0", "width": 512, "height": 512,
        "split": "train" if i else "val_internal",
    } for i in range(n)], columns=MANIFEST_COLUMNS)


def test_validate_manifest_accepts_a_well_formed_manifest():
    validate_manifest(_valid_rows())  # must not raise


def test_validate_manifest_rejects_a_label_outside_zero_one():
    df = _valid_rows()
    df.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        validate_manifest(df)


def test_validate_manifest_rejects_an_unknown_split():
    df = _valid_rows()
    df.loc[0, "split"] = "holdout"
    with pytest.raises(ValueError, match="split must be one of"):
        validate_manifest(df)


def test_validate_manifest_rejects_duplicate_paths():
    df = _valid_rows()
    df.loc[1, "path"] = df.loc[0, "path"]
    with pytest.raises(ValueError, match="duplicated path"):
        validate_manifest(df)


def test_validate_manifest_rejects_relative_paths():
    # The defect this exists to catch: build_dataset.py wrote
    # "seam/norm/<source>/..." under a column documented as absolute, which
    # Plans 2 and 3 open from a different working directory.
    df = _valid_rows()
    df.loc[2, "path"] = "data/normalized/wildfake/sdxl/0000002.png"
    with pytest.raises(ValueError, match="relative path"):
        validate_manifest(df)


def test_validate_manifest_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        validate_manifest(pd.DataFrame({"path": ["/a.png"]}))

def test_dummy_manifest_has_schema_and_real_files(tmp_path):
    rng = np.random.default_rng(0)
    df = make_dummy_manifest(20, str(tmp_path / "img"), rng)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 20
    assert set(df["label"].unique()) <= {0, 1}
    # every path must exist and be readable
    from PIL import Image
    for p in df["path"]:
        assert Image.open(p).size[0] > 0

def test_manifest_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    df = make_dummy_manifest(8, str(tmp_path / "img"), rng)
    out = tmp_path / "m.parquet"
    write_manifest(df, str(out))
    back = read_manifest(str(out))
    assert list(back.columns) == MANIFEST_COLUMNS
    assert len(back) == 8
    assert back["path"].tolist() == df["path"].tolist()

def test_write_manifest_rejects_missing_column(tmp_path):
    import pandas as pd, pytest
    bad = pd.DataFrame({"path": ["a.png"]})
    with pytest.raises(ValueError, match="missing columns"):
        write_manifest(bad, str(tmp_path / "bad.parquet"))

def test_dummy_manifest_cross_call_reproducibility(tmp_path):
    """Verify that two calls with the same seed produce byte-identical PNGs."""
    from PIL import Image
    import numpy as np

    # Generate two manifests with the same seed to different directories
    rng1 = np.random.default_rng(42)
    df1 = make_dummy_manifest(8, str(tmp_path / "dir_a"), rng1)

    rng2 = np.random.default_rng(42)
    df2 = make_dummy_manifest(8, str(tmp_path / "dir_b"), rng2)

    # Manifest rows should match (except path column)
    assert len(df1) == len(df2)
    cols_to_check = [c for c in MANIFEST_COLUMNS if c != "path"]
    for col in cols_to_check:
        assert df1[col].tolist() == df2[col].tolist(), f"Column {col} differs"

    # PNG pixel data should be byte-identical
    for i in range(len(df1)):
        img1 = Image.open(df1.iloc[i]["path"])
        img2 = Image.open(df2.iloc[i]["path"])
        assert np.array_equal(np.array(img1), np.array(img2)), \
            f"PNG pixels differ at index {i}"

def test_dummy_manifest_dimensions_match_metadata(tmp_path):
    """Verify that image dimensions on disk match width/height in manifest."""
    from PIL import Image

    rng = np.random.default_rng(0)
    df = make_dummy_manifest(20, str(tmp_path / "img"), rng)

    for idx, row in df.iterrows():
        img = Image.open(row["path"])
        actual_width, actual_height = img.size
        assert actual_width == row["width"], \
            f"Row {idx}: width mismatch {actual_width} != {row['width']}"
        assert actual_height == row["height"], \
            f"Row {idx}: height mismatch {actual_height} != {row['height']}"

def test_dummy_manifest_paths_are_absolute(tmp_path, monkeypatch):
    """Verify that paths recorded in manifest are absolute even from relative out_dir.

    Uses monkeypatch.chdir to ensure relative paths resolve within tmp_path,
    keeping the test hermetic and preventing repo littering.
    """
    import os

    # Change to tmp_path so relative paths resolve there
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(0)
    # Use a relative path for out_dir (resolves under tmp_path due to chdir)
    rel_dir = "relative_dummy_dir"
    df = make_dummy_manifest(5, rel_dir, rng)

    # All paths should be absolute
    for path in df["path"]:
        assert os.path.isabs(path), f"Path is not absolute: {path}"
        # And files should actually exist at those absolute paths
        assert os.path.exists(path), f"File does not exist at absolute path: {path}"
        # Verify the absolute path is within tmp_path
        assert str(path).startswith(str(tmp_path)), \
            f"Path {path} is not within tmp_path {tmp_path}"


def test_dummy_manifest_exercises_all_three_splits_and_several_generators(tmp_path):
    """The fixture exists so Plans 2 and 3 can be built without real data.
    All 500 rows being split="train" with one generator meant a bank built
    from it raised "bank has no val_internal rows", and
    choose_heldout_generators had nothing to choose between.
    """
    df = make_dummy_manifest(200, str(tmp_path / "img"), np.random.default_rng(0))

    assert set(df["split"]) == {"train", "val_internal", "heldout_generator"}
    fake_gens = set(df[df["label"] == 1]["generator"])
    assert len(fake_gens) >= 3 and fake_gens <= set(DUMMY_GENERATORS)
    assert set(df[df["label"] == 0]["generator"]) == {""}

    # The held-out generator is genuinely held out, exactly as a real
    # manifest's would be.
    held = set(df[df["split"] == "heldout_generator"]["generator"])
    assert len(held) == 1
    assert not held & set(df[df["split"] != "heldout_generator"]["generator"])

    validate_manifest(df)


def test_dummy_manifest_split_assignment_is_deterministic_under_its_rng(tmp_path):
    a = make_dummy_manifest(60, str(tmp_path / "a"), np.random.default_rng(3))
    b = make_dummy_manifest(60, str(tmp_path / "b"), np.random.default_rng(3))
    assert a["split"].tolist() == b["split"].tolist()
    assert a["generator"].tolist() == b["generator"].tolist()


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 20, 60])
def test_dummy_manifest_is_usable_at_the_small_n_later_plans_call_it_with(n):
    """Plans 2 and 3 call make_dummy_manifest with n = 3, 4, 5 and 6.

    Holding out the LAST present generator relocated I4's own failure to
    small n: at n=3 there is a single fake generator, so holding it out left
    train with no fakes and no val_internal rows at all -- the "bank has no
    val_internal rows" failure the fixture exists to prevent. Both splits
    must be non-empty for every n a caller actually uses.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        df = make_dummy_manifest(n, d, np.random.default_rng(0))

    assert (df["split"] == "train").sum() >= 1
    assert (df["split"] == "val_internal").sum() >= 1
    train_fakes = df[(df["split"] == "train") & (df["label"] == 1)]
    heldout = set(df[df["split"] == "heldout_generator"]["generator"])
    if n >= 4:
        # Two or more fake generators exist, so one can be held out while
        # train keeps fakes to learn from.
        assert len(heldout) == 1
        assert len(train_fakes) >= 1
        assert not heldout & set(df[df["split"] != "heldout_generator"]["generator"])
    else:
        # One fake generator only: holding it out would empty train of fakes.
        assert heldout == set()
    validate_manifest(df)
