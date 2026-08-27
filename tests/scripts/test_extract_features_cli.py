"""Stage A CLI: the --split contract, which the documented command got wrong.

`scripts/extract_features.py` documented `--split train`, but Stage B's
`train_rung` evaluates on the bank's own `val_internal` rows, so that bank is
rejected -- on Kaggle, after 8-13 hours of extraction. These tests pin the
comma-separated form and the error that names what a bank actually holds.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from aigcdet.features.bank import N_VIEWS, BankWriter
from aigcdet.train.train_head import RungConfig, train_rung

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ef = _load_script("extract_features")


def _manifest(n=12):
    splits = ["train", "val_internal", "heldout_generator", "benchmark"]
    return pd.DataFrame({
        "path": [f"/p{i}.png" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "generator": ["" if i % 2 == 0 else "g" for i in range(n)],
        "source": ["s"] * n,
        "split": [splits[i % len(splits)] for i in range(n)],
    })


def test_split_accepts_a_comma_separated_list():
    df = _manifest()
    out = ef.select_splits(df, "train,val_internal")
    assert sorted(out["split"].unique()) == ["train", "val_internal"]
    assert len(out) == 6


def test_split_accepts_a_single_name_and_tolerates_whitespace():
    df = _manifest()
    assert set(ef.select_splits(df, "train")["split"]) == {"train"}
    assert sorted(ef.select_splits(df, " train , val_internal ")["split"].unique()) == \
        ["train", "val_internal"]


def test_empty_split_keeps_every_row():
    df = _manifest()
    assert ef.select_splits(df, "") is df


def test_split_preserves_the_frozen_manifest_index_labels():
    """extract_bank keys each view's RNG on the index label, so the filter
    must never reset it (a reset would change which views get drawn and break
    shard consistency)."""
    df = _manifest()
    out = ef.select_splits(df, "train,val_internal")
    assert out.index.tolist() == [0, 1, 4, 5, 8, 9]


def test_unknown_split_name_fails_before_any_extraction():
    df = _manifest()
    with pytest.raises(ValueError, match="does not contain"):
        ef.select_splits(df, "train,val")


def test_documented_command_in_the_module_docstring_names_both_splits():
    """The docstring is the command a human copies onto Kaggle. It must name
    the combination that actually produces a trainable bank."""
    assert "--split train,val_internal" in ef.__doc__
    assert "--split train " not in ef.__doc__


def _bank_without_val(tmp_path, n=8, dim=4):
    w = BankWriter(str(tmp_path / "b"), n, N_VIEWS, dim, "t", 0)
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        w.write_image(i, {"path": f"/p{i}", "label": i % 2, "generator": "",
                          "source": "s", "split": "train"},
                      feats=np.zeros((N_VIEWS, dim), np.float32),
                      presence=pres, severity=np.zeros((N_VIEWS, 6), np.float32),
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return str(tmp_path / "b")


def test_train_rung_names_the_splits_the_bank_actually_contains(tmp_path):
    cfg = RungConfig(name="a0", bank_dir=_bank_without_val(tmp_path), epochs=1,
                     out_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError) as exc:
        train_rung(cfg)
    msg = str(exc.value)
    assert "val_internal" in msg
    assert "'train': 8" in msg            # names what the bank DOES contain
    assert "--split train,val_internal" in msg
