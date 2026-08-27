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
