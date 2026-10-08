import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "build_external_manifest", os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "build_external_manifest.py"))
bem = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bem)


def _write(path, w, h, fmt, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(a).save(path, format=fmt)


def _tree(root):
    for i in range(4):
        _write(f"{root}/aigibench/SocialRF/0_real/r{i}.png", 64, 48, "PNG", i)
        _write(f"{root}/aigibench/SocialRF/1_fake/f{i}.png", 64, 48, "PNG", 10 + i)
        _write(f"{root}/aigibench/CommunityAI/0_real/r{i}.jpg", 900, 700, "JPEG", 20 + i)
        _write(f"{root}/aigibench/CommunityAI/1_fake/f{i}.png", 600, 600, "PNG", 30 + i)
        _write(f"{root}/synthbuster/synthbuster/RAISE/r{i}.png", 120, 80, "PNG", 40 + i)
        _write(f"{root}/synthbuster/synthbuster/dalle3/f{i}.png", 96, 96, "PNG", 50 + i)
        _write(f"{root}/synthbuster/synthbuster/midjourney-v5/f{i}.png", 96, 96, "PNG", 60 + i)


def test_builds_every_set_in_three_readings_with_leak_diagnostics(tmp_path):
    _tree(tmp_path)
    out = tmp_path / "ext.parquet"
    prov = tmp_path / "prov.json"
    bem.main(["--ext-root", str(tmp_path), "--train-manifest", "",
              "--out", str(out), "--provenance", str(prov), "--workers", "2"])
    df = pd.read_parquet(out)
    assert set(df.split) == {f"{b}{v}" for b in ("ext_aigi_socialrf",
                                                 "ext_aigi_community",
                                                 "ext_synthbuster")
                             for v in ("", "_deg", "_jpeg95")}
    sb = df[df.split == "ext_synthbuster"]
    assert (sb[sb.label == 0].generator == "").all()
    assert set(sb[sb.label == 1].generator) == {"dalle3", "midjourney-v5"}
    p = json.loads(prov.read_text())["sets"]
    # CommunityAI's confound is caught: format alone separates the classes
    assert p["ext_aigi_community"]["format_vs_label_auc"] == 1.0
    assert p["ext_aigi_socialrf"]["format_vs_label_auc"] == 0.5
    assert p["ext_aigi_community"]["short_side"]["p50"] > 0


def test_a_file_the_model_trained_on_is_dropped_and_counted(tmp_path):
    _tree(tmp_path)
    first = bem.build_set("aigibench_socialrf",
                          str(tmp_path / "aigibench/SocialRF"), "binary", 2)
    train = tmp_path / "train.parquet"
    first.iloc[[0]][["content_sha256"]].to_parquet(train)
    df, prov = bem.build(str(tmp_path), ["aigibench_socialrf"], str(train), 2)
    assert prov["ext_aigi_socialrf"]["dropped_overlap_with_ntire_manifest"] == 1
    assert prov["ext_aigi_socialrf"]["n"] == 7


def test_an_absent_set_is_skipped_unless_required(tmp_path):
    _tree(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "synthbuster")
    bem.main(["--ext-root", str(tmp_path), "--train-manifest", "",
              "--out", str(tmp_path / "o.parquet"),
              "--provenance", str(tmp_path / "p.json"), "--workers", "2"])
    assert json.loads((tmp_path / "p.json").read_text())["missing"] == ["synthbuster"]
    with pytest.raises(SystemExit, match="missing"):
        bem.main(["--ext-root", str(tmp_path), "--train-manifest", "",
                  "--require-all", "--out", str(tmp_path / "o.parquet"),
                  "--provenance", str(tmp_path / "p.json")])


def test_a_by_generator_tree_without_one_raise_folder_is_refused(tmp_path):
    _write(f"{tmp_path}/sb/dalle3/f.png", 32, 32, "PNG", 0)
    with pytest.raises(SystemExit, match="RAISE"):
        bem.list_set(str(tmp_path / "sb"), "by_generator")
