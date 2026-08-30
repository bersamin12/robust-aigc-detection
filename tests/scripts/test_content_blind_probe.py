"""The content-blind control, run after standardisation rather than before it.

The confound this exists to catch is CREATED by crop standardisation and is
absent from the files on disk: a 200x200 window is a whole frame for a 200px
image and a detail for an 800px one, so field of view tracks native
resolution, which tracks source. Two of the union's sources are
authentic-only, which turns that into a route to the right answer with nothing
to do with generation. Losing any property pinned here would return a
reassuring number on exactly the corpus this is meant to indict.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cbp = _load_script("content_blind_probe")


def _img(path, w, h, rng):
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


def test_the_thumbnail_is_of_the_canonicalised_view_not_the_file(tmp_path):
    """The whole point. A 600px image under CROP contributes a 200px window;
    under BAND it contributes the whole frame box-filtered. If this thumbnailed
    the file, the two policies would return identical features and the control
    could never distinguish them."""
    from aigcdet.augment.canonical import CanonPolicy, MODE_BAND, MODE_CROP

    rng = np.random.default_rng(0)
    p = tmp_path / "a.png"
    _img(p, 600, 600, rng)
    crop = cbp.canonicalised_thumbnail(
        str(p), CanonPolicy(mode=MODE_CROP, crop_side=200), 7, 3)
    band = cbp.canonicalised_thumbnail(
        str(p), CanonPolicy(mode=MODE_BAND), 7, 3)
    assert crop.shape == band.shape == (cbp.THUMB * cbp.THUMB * 3,)
    assert not np.allclose(crop, band)


def test_the_crop_window_is_keyed_on_the_row_id_the_extraction_uses(tmp_path):
    """`canonical_rng(seed, row_id, view)` is the extraction's key. A control
    that keyed on a positional index would thumbnail a DIFFERENT window than
    the model sees, and would then be measuring a corpus nobody trained on."""
    from aigcdet.augment.canonical import CanonPolicy, MODE_CROP

    rng = np.random.default_rng(1)
    p = tmp_path / "a.png"
    _img(p, 600, 600, rng)
    pol = CanonPolicy(mode=MODE_CROP, crop_side=200)
    a = cbp.canonicalised_thumbnail(str(p), pol, 7, 3)
    same = cbp.canonicalised_thumbnail(str(p), pol, 7, 3)
    other_row = cbp.canonicalised_thumbnail(str(p), pol, 7, 4)
    other_seed = cbp.canonicalised_thumbnail(str(p), pol, 8, 3)
    assert np.array_equal(a, same)
    assert not np.allclose(a, other_row)
    assert not np.allclose(a, other_seed)


def test_features_use_the_manifest_index_label_not_the_position(tmp_path):
    """A probe manifest is a subset, so its index labels are not 0..N-1 unless
    something reset them. `features_for` must follow the labels."""
    from aigcdet.augment.canonical import CanonPolicy, MODE_CROP

    rng = np.random.default_rng(2)
    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.png"
        _img(p, 600, 600, rng)
        paths.append(str(p))
    pol = CanonPolicy(mode=MODE_CROP, crop_side=200)
    df = pd.DataFrame({"path": paths, "label": [0, 1, 0, 1]},
                      index=[100, 101, 102, 103])
    got = cbp.features_for(df, pol, seed=7, workers=2)
    want = np.stack([cbp.canonicalised_thumbnail(paths[i], pol, 7, 100 + i)
                     for i in range(4)])
    assert np.allclose(got, want)


def test_it_sees_a_field_of_view_shortcut_that_band_limiting_erases(tmp_path):
    """The end-to-end claim, on a corpus built to isolate the mechanism.

    Every image is the same smooth radial gradient. The classes differ ONLY in
    native size: 210 px, or that same image upscaled to 800 px. Band-limiting
    takes both to a 200 px ceiling and back up, so they arrive looking alike
    and there is nothing left to separate them by. Crop takes a 200 px window
    -- the whole frame for the small class, a quarter-width slice for the
    large one -- so a centred gradient becomes an off-centre one and
    composition alone says which is which.

    Radial and not a sinusoid: a periodic pattern with random phase is not
    separable by a pixel-space classifier at all, so that fixture returned
    chance under BOTH policies (measured: 0.51 crop, 0.45 band) and proved
    nothing. What field of view actually changes is SYMMETRY.

    This is the union's shape exactly -- 200 px WildFake against 800 px NTIRE,
    with two authentic-only sources. A control that reads the FILE cannot see
    it, because the confound does not exist until standardisation creates it.
    """
    from aigcdet.augment.canonical import CanonPolicy, MODE_BAND, MODE_CROP
    from aigcdet.eval.controls import NO_QUALITY_COLUMN, content_blind_auc

    rng = np.random.default_rng(3)
    base = 210
    yy, xx = np.mgrid[0:base, 0:base] / (base - 1)
    radial = 1.0 - np.hypot(xx - 0.5, yy - 0.5) / 0.7071

    paths, labels = [], []
    for i in range(120):
        big = i % 2 == 0
        pat = radial * rng.uniform(200, 240)      # per-image brightness jitter
        im = Image.fromarray(
            np.repeat(pat[:, :, None], 3, axis=2).astype(np.uint8))
        if big:
            im = im.resize((800, 800), Image.BICUBIC)
        p = tmp_path / f"{i}.png"
        im.save(p)
        paths.append(str(p))
        labels.append(0 if big else 1)
    labels = np.asarray(labels)

    df = pd.DataFrame({"path": paths, "label": labels}, index=range(len(paths)))
    crop = content_blind_auc(
        cbp.features_for(df, CanonPolicy(mode=MODE_CROP, crop_side=200), 7, 4),
        labels, quality_branches=NO_QUALITY_COLUMN)
    band = content_blind_auc(
        cbp.features_for(df, CanonPolicy(mode=MODE_BAND), 7, 4),
        labels, quality_branches=NO_QUALITY_COLUMN)
    assert band["auc"] < 0.75, band["auc"]
    assert crop["auc"] > band["auc"] + 0.15, (crop["auc"], band["auc"])


def test_a_single_class_source_is_reported_not_skipped():
    """COCO and Open Images are authentic-only in the union. A per-source loop
    that skipped them silently would hide the very asymmetry under
    investigation."""
    rng = np.random.default_rng(4)
    feats = rng.normal(size=(1200, 12)).astype(np.float32)
    labels = np.r_[np.zeros(600), np.ones(600)].astype(int)
    sources = np.array(["coco"] * 600 + ["wildfake"] * 600)
    out = cbp.within_source(feats, labels, sources, seed=1, min_rows=100)
    assert "one class only" in out["coco"]["note"]
    assert "one class only" in out["wildfake"]["note"]


def test_within_source_reports_an_auc_when_both_classes_are_present():
    rng = np.random.default_rng(5)
    n = 600
    feats = np.r_[rng.normal(0, 1, (n, 8)),
                  rng.normal(3, 1, (n, 8))].astype(np.float32)
    labels = np.r_[np.zeros(n), np.ones(n)].astype(int)
    sources = np.array(["ntire"] * (2 * n))
    out = cbp.within_source(feats, labels, sources, seed=1, min_rows=100)
    assert out["ntire"]["auc"] > 0.9
    assert out["ntire"]["n"] == 2 * n


def test_the_clean_view_is_the_one_measured():
    """View 0 carries no degradation recipe, so its thumbnail isolates what
    STANDARDISATION did to the frame. Any other view would fold the recipe's
    own resize and crop into the number."""
    assert cbp.CLEAN_VIEW == 0
