import numpy as np
from aigcdet.data.manifest import (
    MANIFEST_COLUMNS, make_dummy_manifest, read_manifest, write_manifest,
)

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

def test_dummy_manifest_paths_are_absolute(tmp_path):
    """Verify that paths recorded in manifest are absolute."""
    import os

    rng = np.random.default_rng(0)
    # Use a relative path for out_dir
    rel_dir = "relative_dummy_dir"
    df = make_dummy_manifest(5, rel_dir, rng)

    # All paths should be absolute
    for path in df["path"]:
        assert os.path.isabs(path), f"Path is not absolute: {path}"
        # And files should actually exist at those absolute paths
        assert os.path.exists(path), f"File does not exist at absolute path: {path}"
