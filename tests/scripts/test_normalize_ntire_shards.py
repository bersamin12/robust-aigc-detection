import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import normalize_ntire_shards as nns  # noqa: E402


def _shard(root, k, n=6, bad=False):
    d = root / f"shard_{k}"
    (d / "images").mkdir(parents=True)
    rng = np.random.default_rng(k)
    names, labels = [], []
    for i in range(n):
        name = f"img{k}_{i}.jpg"
        # 700px short side so normalisation actually resizes (SHORT_SIDE 512).
        arr = rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / "images" / name, quality=90)
        names.append(name)
        labels.append(i % 2)
    if bad:
        (d / "images" / "broken.jpg").write_bytes(b"not an image")
        names.append("broken.jpg")
        labels.append(1)
    pd.DataFrame({"image_name": names, "label": labels}).to_csv(d / "labels.csv", index=False)
    return d


def test_normalises_into_prefixed_names_with_the_union_buckets(tmp_path):
    src, dest = tmp_path / "raw", tmp_path / "norm"
    _shard(src, 4)
    df, failures = nns.normalize_shard(str(src / "shard_4"), 4, str(dest), workers=2)
    assert failures == []
    assert len(df) == 6
    assert set(df.columns) == set(nns.SIDECAR_COLUMNS)
    # Bucket by label, shard-prefixed running index, never the union's bare index.
    for r in df.itertuples():
        bucket = "real" if r.label == 0 else "ntire"
        assert r.path == str(dest / "ntire" / bucket / f"s4_{int(r.src_name.split('_')[1][:-4]):07d}.png")
        assert os.path.exists(r.path)
        assert r.source == "ntire" and r.shard == 4
        assert r.generator == ("" if r.label == 0 else "ntire")
        assert min(r.width, r.height) == 512
        assert len(r.content_sha256) == 64


def test_label_polarity_follows_the_csv(tmp_path):
    src, dest = tmp_path / "raw", tmp_path / "norm"
    _shard(src, 3)
    df, _ = nns.normalize_shard(str(src / "shard_3"), 3, str(dest), workers=2)
    csv = pd.read_csv(src / "shard_3" / "labels.csv").set_index("image_name")
    for r in df.itertuples():
        assert r.label == int(csv.loc[r.src_name, "label"])


def test_rerun_is_idempotent_and_measures_rather_than_rewrites(tmp_path):
    src, dest = tmp_path / "raw", tmp_path / "norm"
    _shard(src, 5, n=4)
    a, _ = nns.normalize_shard(str(src / "shard_5"), 5, str(dest), workers=2)
    mtimes = {p: os.path.getmtime(p) for p in a.path}
    b, _ = nns.normalize_shard(str(src / "shard_5"), 5, str(dest), workers=2)
    pd.testing.assert_frame_equal(a, b)
    assert all(os.path.getmtime(p) == m for p, m in mtimes.items())


def test_unreadable_file_is_recorded_not_fatal(tmp_path):
    src, dest = tmp_path / "raw", tmp_path / "norm"
    _shard(src, 3, n=4, bad=True)
    df, failures = nns.normalize_shard(str(src / "shard_3"), 3, str(dest), workers=2)
    assert len(df) == 4
    assert len(failures) == 1 and failures[0][0].endswith("broken.jpg")


def test_cli_writes_sidecar_and_skipped_json(tmp_path):
    src, dest = tmp_path / "raw", tmp_path / "norm"
    _shard(src, 3, n=4, bad=True)
    side, skip = tmp_path / "side", tmp_path / "skip"
    assert nns.main(["--src", str(src), "--shard", "3", "--dest", str(dest),
                     "--sidecar-dir", str(side), "--skipped-dir", str(skip),
                     "--workers", "2"]) == 0
    df = pd.read_parquet(side / "ntire_shard3_rows.parquet")
    assert len(df) == 4
    rec = json.load(open(skip / "normalize_skipped_shard3.json"))
    assert len(rec) == 1 and "broken.jpg" in rec[0]["src"]


def test_unknown_label_value_is_refused(tmp_path):
    src = tmp_path / "raw"
    d = _shard(src, 3, n=2)
    pd.DataFrame({"image_name": ["img3_0.jpg"], "label": [2]}).to_csv(d / "labels.csv", index=False)
    with pytest.raises(SystemExit, match="label"):
        nns.normalize_shard(str(d), 3, str(tmp_path / "norm"), workers=1)
