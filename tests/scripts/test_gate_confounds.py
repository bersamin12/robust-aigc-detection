"""`scripts/gate_confounds.py` -- the measurement that can cancel a GPU night.

`scripts/audit_confounds.py` reads a bank's cached proxies, which is the right
instrument once a bank exists and the wrong one before, because the question
this answers is whether to spend the night that would produce it. So the
properties that matter are: does it measure the pixels extraction WOULD cache,
does it refuse when it should, and does its AUC read a confound in either
direction.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.augment.canonical import MODE_BAND, MODE_CROP, CanonPolicy

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gc = _load("gate_confounds")


# --------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------

def test_auc_is_orientation_corrected():
    """A confound that predicts the label BACKWARDS is exactly as usable to a
    head as one that predicts it forwards, so the separability is the quantity
    and the direction is not. This is the same convention
    `docs/low_level_confounds.md` reports in."""
    y = np.array([0, 0, 1, 1])
    assert gc.auc(np.array([1.0, 2, 3, 4]), y) == 1.0
    assert gc.auc(np.array([4.0, 3, 2, 1]), y) == 1.0


def test_auc_is_a_half_for_a_signal_that_carries_nothing():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    assert gc.auc(rng.normal(size=4000), y) == pytest.approx(0.5, abs=0.03)


def test_auc_handles_ties_without_inflating():
    """A constant feature separates nothing. Rank-averaging is what keeps it at
    0.5 rather than reporting whatever the sort order happened to be."""
    y = np.array([0, 0, 1, 1])
    assert gc.auc(np.ones(4), y) == pytest.approx(0.5)


def test_auc_is_nan_when_one_class_is_absent():
    """`coco_train2017` contributes only authentic rows, so its within-source
    AUC does not exist. Returning nan rather than raising is what lets the
    caller report the group and move on."""
    assert np.isnan(gc.auc(np.arange(5.0), np.zeros(5, int)))


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _frame(n_per_group=50):
    rows = []
    for src in ("wildfake", "sid_set"):
        for lab in (0, 1):
            for _ in range(n_per_group):
                rows.append({"source": src, "label": lab})
    return pd.DataFrame(rows)


def test_sampling_preserves_the_corpus_source_mix():
    """Not stratified, and the difference is the gate's whole validity.
    `--max-auc` is read against a figure measured on the corpus as it IS --
    0.6721 on a pool that is 85% WildFake. Measured against the frozen
    manifest, equal-per-source sampling put pooled laplacian_var at 0.6118 and
    noise_floor at 0.6683 where the corpus's true figures are 0.6721 and
    0.6374: right answers to a question nobody asked, and the threshold would
    have been compared to them."""
    df = pd.concat([_frame(400), _frame(20).assign(source="sid_set")])
    df = pd.DataFrame({"source": ["wildfake"] * 8500 + ["sid_set"] * 1500,
                       "label": [0, 1] * 5000})
    got = gc.sample_rows(df, n=2000, seed=0)
    assert len(got) == 2000
    share = (got["source"] == "sid_set").mean()
    assert share == pytest.approx(0.15, abs=0.03)


def test_sampling_keeps_a_frame_smaller_than_the_quota_whole():
    df = _frame(5)
    assert len(gc.sample_rows(df, n=10_000, seed=0)) == len(df)


def test_sampling_preserves_manifest_order():
    """Row order is scan order, and `read_manifest` index labels are the RNG
    keys everything downstream is derived from. Reordering here would not
    break this script, but it would make its output impossible to line up
    against anything else."""
    df = _frame(50).reset_index(drop=True)
    got = gc.sample_rows(df, n=20, seed=0)
    assert got.index.tolist() == sorted(got.index.tolist())


def test_sampling_is_deterministic_given_the_seed():
    a = gc.sample_rows(_frame(), 10, seed=7).index.tolist()
    b = gc.sample_rows(_frame(), 10, seed=7).index.tolist()
    assert a == b


# --------------------------------------------------------------------------
# it must measure the pixels extraction would actually cache
# --------------------------------------------------------------------------

def test_crop_mode_uses_the_same_window_extraction_would_cache(tmp_path):
    """Not a centre crop that merely resembles it. The window is drawn from
    the per-view key, so this measures view 0's real pixels -- otherwise the
    gate would pass or fail on a different image than the one that gets
    extracted."""
    from aigcdet.augment.canonical import canonical_rng, canonicalise

    img = np.random.default_rng(0).integers(
        0, 256, (400, 600, 3), dtype=np.uint8)
    policy = CanonPolicy(mode=MODE_CROP, crop_side=200)
    got = gc.canonicalise_for(img, policy, seed=7, row_id=42)
    want = canonicalise(img, policy=policy, rng=canonical_rng(7, 42, 0))
    assert np.array_equal(got, want)
    # ...and NOT the centre window, which is what a lazier implementation
    # would have produced.
    assert not np.array_equal(got, canonicalise(img, policy=policy))


def test_band_mode_passes_no_rng_so_the_band_is_unjittered(tmp_path):
    """Extraction's band path passes no rng either. Jittering here and not
    there would make the gate measure a corpus that is never built."""
    from aigcdet.augment.canonical import canonicalise

    img = np.random.default_rng(1).integers(
        0, 256, (400, 600, 3), dtype=np.uint8)
    policy = CanonPolicy(mode=MODE_BAND)
    assert np.array_equal(gc.canonicalise_for(img, policy, seed=7, row_id=42),
                          canonicalise(img, policy=policy))


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------

def _corpus(tmp_path, real_bright, fake_bright, n=14):
    """A manifest whose two classes differ ONLY in a low-level statistic.

    Brightness drives `laplacian_var` here through the noise amplitude, so the
    gap between the two classes is a dial the test can turn -- which is what
    makes "refuses a bad corpus, accepts a clean one" testable without needing
    real images.
    """
    rng = np.random.default_rng(0)
    rows = []
    for lab, amp in ((0, real_bright), (1, fake_bright)):
        for i in range(n):
            p = tmp_path / f"{lab}_{i}.png"
            arr = (rng.random((300, 300, 3)) * amp).astype(np.uint8)
            Image.fromarray(arr).save(p)
            rows.append({"path": str(p), "label": lab, "generator":
                         "" if lab == 0 else "g", "source": "wildfake",
                         "split": "train", "width": 300, "height": 300,
                         "rel_path": p.name, "licence": "x",
                         "content_sha256": f"{lab}{i}", "pixel_sha256": f"{lab}{i}"})
    df = pd.DataFrame(rows)
    mf = tmp_path / "m.parquet"
    df.to_parquet(mf, index=False)
    return str(mf)


def test_a_corpus_whose_classes_differ_in_a_proxy_is_refused(tmp_path, monkeypatch):
    """The whole point: a corpus that hands the head a stronger
    one-dimensional shortcut than the one we already have makes a headline AUC
    mean less, not more."""
    monkeypatch.setenv("AIGCDET_DATA_ROOT", str(tmp_path))
    mf = _corpus(tmp_path, real_bright=255, fake_bright=40)
    with pytest.raises(SystemExit, match="REFUSED"):
        gc.main(["--manifest", mf, "--n", "40", "--workers", "1",
                 "--max-auc", "0.70"])


def test_a_corpus_within_the_bound_passes_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIGCDET_DATA_ROOT", str(tmp_path))
    mf = _corpus(tmp_path, real_bright=200, fake_bright=200)
    out = gc.main(["--manifest", mf, "--n", "40", "--workers", "1",
                   "--max-auc", "0.95"])
    assert out["worst"] <= 0.95
    assert out["worst_name"] in ("jpeg_quality", "laplacian_var", "noise_floor")
    printed = capsys.readouterr().out
    assert "POOLED" in printed
    # The frozen figure is printed next to it, because a number with nothing
    # to read it against is not a gate.
    assert "0.6721" in printed


def test_without_max_auc_it_reports_and_does_not_refuse(tmp_path, monkeypatch):
    """A gate you cannot run as a report is a gate nobody runs."""
    monkeypatch.setenv("AIGCDET_DATA_ROOT", str(tmp_path))
    mf = _corpus(tmp_path, real_bright=255, fake_bright=40)
    out = gc.main(["--manifest", mf, "--n", "40", "--workers", "1"])
    assert out["worst"] > 0.7
