"""Tests for the content-blind control (spec §4.2 defence 2, §6.5).

The control's whole job is to be believed when it says "your dataset is fine",
so these tests are written against the ways it could lie:

* it fires when nothing is wrong (leakage in the cross-validation, or the
  JPEG-quality estimator's two branches acting as a class label);
* it stays quiet when something is wrong (the verdict thresholds drifting, or
  the estimator's pixel fallback smearing a real difference away).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image
from sklearn.preprocessing import StandardScaler

from aigcdet.eval import controls
from aigcdet.eval.controls import (
    VERDICT_THRESHOLDS,
    content_blind_auc,
    metadata_control,
    metadata_features,
    quality_estimator_branches,
    thumbnail_features,
)
from aigcdet.features.proxies import estimate_jpeg_quality


def _write(p, size, fmt="PNG", quality=None, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(p, format=fmt, **({"quality": quality} if quality else {}))
    return str(p)


def _photo(p, size, fmt="PNG", quality=None, seed=0):
    """A smooth, compressible image -- unlike white noise, JPEG can damage it."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (max(2, size[1] // 32), max(2, size[0] // 32), 3), dtype=np.uint8)
    im = Image.fromarray(small).resize(size, Image.BICUBIC)
    im.save(p, format=fmt, **({"quality": quality} if quality else {}))
    return str(p)


# --------------------------------------------------------------------------
# feature extractors
# --------------------------------------------------------------------------

def test_thumbnail_features_have_the_declared_width(tmp_path):
    paths = [_write(tmp_path / f"{i}.png", (200, 200)) for i in range(3)]
    f = thumbnail_features(paths, size=16)
    assert f.shape == (3, 16 * 16 * 3) and f.dtype == np.float32


def test_thumbnail_features_are_scaled_to_unit_range_and_track_brightness(tmp_path):
    """Kills a stub that returns zeros/ones, and pins the /255 normalisation."""
    dark = str(tmp_path / "dark.png")
    bright = str(tmp_path / "bright.png")
    Image.fromarray(np.full((64, 64, 3), 10, np.uint8)).save(dark)
    Image.fromarray(np.full((64, 64, 3), 240, np.uint8)).save(bright)
    f = thumbnail_features([dark, bright], size=8)
    assert f.shape == (2, 8 * 8 * 3)
    assert 0.0 <= f.min() and f.max() <= 1.0
    assert f[0].mean() == pytest.approx(10 / 255, abs=1e-3)
    assert f[1].mean() == pytest.approx(240 / 255, abs=1e-3)


def _half_and_half(path, left_first, rng):
    """64x64: one half near-white, the other near-black, plus faint noise.

    The two orientations have the SAME per-channel mean by construction, so
    only something that keeps the spatial grid can tell them apart.
    """
    arr = np.zeros((64, 64, 3), np.uint8)
    arr[:, :32] = 235 if left_first else 20
    arr[:, 32:] = 20 if left_first else 235
    jitter = rng.integers(-8, 9, (64, 64, 3))
    Image.fromarray(np.clip(arr + jitter, 0, 255).astype(np.uint8)).save(path)
    return str(path)


def test_thumbnail_features_keep_spatial_layout_not_just_average_colour(tmp_path):
    """Left-white/right-black versus its mirror: identical per-channel means.

    Both classes have the same mean colour to within the jitter, so a
    thumbnail that collapsed to a per-channel average -- or resized to 1x1 and
    tiled back up -- would score near chance here while still satisfying every
    other test in this file. That failure matters in one direction only: it
    would report "content-blind control: clean" on a dataset whose classes
    differ in composition, which is the false reassurance this whole module
    exists to prevent.
    """
    rng = np.random.default_rng(11)
    paths, labels = [], []
    for i in range(20):
        paths.append(_half_and_half(tmp_path / f"L{i}.png", True, rng)); labels.append(0)
    for i in range(20):
        paths.append(_half_and_half(tmp_path / f"R{i}.png", False, rng)); labels.append(1)

    features = thumbnail_features(paths, size=16)
    per_class_mean = [features[:20].mean(), features[20:].mean()]
    assert per_class_mean[0] == pytest.approx(per_class_mean[1], abs=0.01), (
        "fixture is broken: the classes differ in mean colour, so this test "
        "would pass without any spatial information being kept"
    )

    res = content_blind_auc(features, np.array(labels))
    assert res["auc"] > VERDICT_THRESHOLDS["broken"]


def test_thumbnail_resampler_averages_rather_than_point_samples(tmp_path):
    """One-pixel alternating columns, downscaled 4x.

    An averaging filter blends them to mid-grey (~0.5); `Image.NEAREST` samples
    every other column and returns pure white (1.0), aliasing the fine detail
    into the thumbnail instead of destroying it. The whole content-blindness
    argument is that this band does not survive the downscale, so a
    point-sampling resampler would quietly invalidate the docstring's claim
    and move every number the control reports.
    """
    arr = np.zeros((64, 64, 3), np.uint8)
    arr[:, ::2] = 255
    path = str(tmp_path / "stripes.png")
    Image.fromarray(arr).save(path)

    f = thumbnail_features([path], size=16)
    assert f.mean() == pytest.approx(0.5, abs=0.05)
    assert f.max() < 0.9, "fine detail was point-sampled into the thumbnail"


def test_metadata_features_capture_size_and_quality(tmp_path):
    """The brief's original assertion, with the branch confound removed.

    Comparing a JPEG's *exact* quality against a PNG's *pixel-estimated* one
    compares two different estimators, so it would pass even if the estimator
    were blind to quality. Both files here are JPEGs, so both take the exact
    quantisation-table branch and the comparison is a real one.
    """
    a = _photo(tmp_path / "a.jpg", (640, 480), "JPEG", 20)
    b = _photo(tmp_path / "b.jpg", (1024, 1024), "JPEG", 95)
    f = metadata_features([a, b])
    assert f.shape == (2, 4) and f.dtype == np.float32
    assert f[1, 0] > f[0, 0]                              # width
    assert f[1, 1] > f[0, 1]                              # height
    assert f[0, 2] == pytest.approx(640 / 480)            # aspect
    assert f[1, 2] == pytest.approx(1.0)
    assert f[0, 3] < f[1, 3]                              # q20 below q95


def test_metadata_features_warn_when_the_two_estimator_branches_are_mixed(tmp_path):
    """A JPEG and a PNG take different branches of estimate_jpeg_quality, so
    column 3 is on two different scales and is partly format-identifying."""
    a = _photo(tmp_path / "a.jpg", (64, 64), "JPEG", 40)
    b = _photo(tmp_path / "b.png", (64, 64))
    with pytest.warns(UserWarning, match="two different branches"):
        metadata_features([a, b])


def test_metadata_features_do_not_warn_when_every_file_takes_one_branch(tmp_path):
    paths = [_photo(tmp_path / f"{i}.png", (64, 64), seed=i) for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metadata_features(paths)


def test_branch_labels_match_which_branch_estimate_jpeg_quality_actually_took(tmp_path):
    """Guards against drift: the branch labels are computed by controls.py, but
    they must describe what proxies.estimate_jpeg_quality really did.  The
    observable difference is that passing the path changes the answer only when
    the quantisation-table branch fires.
    """
    jpg = _photo(tmp_path / "a.jpg", (128, 128), "JPEG", 30)
    png = _photo(tmp_path / "b.png", (128, 128))
    branches = quality_estimator_branches([jpg, png])
    assert list(branches) == ["exact", "estimated"]

    for path, branch in zip([jpg, png], branches):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with_path = estimate_jpeg_quality(arr, path)
        pixels_only = estimate_jpeg_quality(arr, None)
        if branch == "exact":
            assert with_path != pixels_only
        else:
            assert with_path == pixels_only


def test_metadata_quality_column_is_the_exact_table_value_for_a_jpeg(tmp_path):
    """Column 3 must be the estimator's output, not a constant or a placeholder."""
    jpg = _photo(tmp_path / "a.jpg", (128, 128), "JPEG", 30)
    with Image.open(jpg) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    expected = estimate_jpeg_quality(arr, jpg)
    assert metadata_features([jpg])[0, 3] == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------
# the brief's headline behaviours
# --------------------------------------------------------------------------

def test_control_detects_a_deliberately_broken_dataset(tmp_path):
    """Reals 640x480 JPEG, fakes 1024x1024 PNG: metadata alone must separate them."""
    paths, labels = [], []
    for i in range(30):
        paths.append(_photo(tmp_path / f"r{i}.jpg", (320, 240), "JPEG", 40, seed=i)); labels.append(0)
    for i in range(30):
        paths.append(_photo(tmp_path / f"f{i}.png", (512, 512), seed=100 + i)); labels.append(1)
    with pytest.warns(UserWarning, match="two different branches"):
        feats = metadata_features(paths)
    res = content_blind_auc(feats, np.array(labels))
    assert res["auc"] > VERDICT_THRESHOLDS["broken"]
    assert res["verdict"] == "broken"


def test_control_reports_clean_when_classes_are_indistinguishable(tmp_path):
    paths, labels = [], []
    for i in range(40):
        paths.append(_photo(tmp_path / f"x{i}.png", (256, 256), seed=i))
        labels.append(i % 2)          # label uncorrelated with anything visible
    res = content_blind_auc(metadata_features(paths), np.array(labels))
    assert res["verdict"] == "clean"
    assert res["auc"] < VERDICT_THRESHOLDS["suspect"]


def test_result_includes_a_confidence_interval(tmp_path):
    paths = [_photo(tmp_path / f"y{i}.png", (256, 256), seed=i) for i in range(40)]
    labels = np.array([i % 2 for i in range(40)])
    res = content_blind_auc(metadata_features(paths), labels)
    lo, hi = res["auc_ci"]
    assert lo <= res["auc"] <= hi


def test_thumbnails_separate_classes_that_differ_only_in_gross_appearance(tmp_path):
    """16x16 thumbnails keep colour and composition; that is what they detect."""
    paths, labels = [], []
    for i in range(20):
        p = str(tmp_path / f"red{i}.png")
        arr = np.zeros((64, 64, 3), np.uint8); arr[..., 0] = 200 + i
        Image.fromarray(arr).save(p); paths.append(p); labels.append(0)
    for i in range(20):
        p = str(tmp_path / f"blue{i}.png")
        arr = np.zeros((64, 64, 3), np.uint8); arr[..., 2] = 200 + i
        Image.fromarray(arr).save(p); paths.append(p); labels.append(1)
    res = content_blind_auc(thumbnail_features(paths), np.array(labels))
    assert res["auc"] > VERDICT_THRESHOLDS["broken"]


# --------------------------------------------------------------------------
# the verdict thresholds, pinned behaviourally
# --------------------------------------------------------------------------

def _graded_features(separation: float, n_per_class: int = 200):
    """One feature that is `separation` standard deviations apart per class.

    The mapping separation -> cross-validated AUC is smooth, so the four values
    below straddle each published threshold from both sides.
    """
    rng = np.random.default_rng(7)
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    x = labels * separation + rng.normal(0.0, 1.0, 2 * n_per_class)
    return x.reshape(-1, 1), labels


@pytest.mark.parametrize(
    "separation, auc_lo, auc_hi, expected",
    [
        (0.62, 0.68, 0.70, "clean"),     # just below the suspect threshold
        (0.70, 0.70, 0.73, "suspect"),   # just above it
        (1.30, 0.84, 0.85, "suspect"),   # just below the broken threshold
        (1.40, 0.85, 0.88, "broken"),    # just above it
    ],
)
def test_verdict_changes_where_the_published_thresholds_say_it_does(
    separation, auc_lo, auc_hi, expected
):
    """Behavioural pin for VERDICT_THRESHOLDS.

    Asserting the dict's literal values would survive any mutation of the
    comparison; these cases sit within 0.02 AUC of each published threshold, so
    moving 0.85 or 0.70 by more than that flips a verdict here. The AUC bands
    are asserted too, so if the classifier ever changes and the cases stop
    straddling the thresholds, this fails loudly instead of silently testing
    nothing.
    """
    features, labels = _graded_features(separation)
    res = content_blind_auc(features, labels)
    assert auc_lo < res["auc"] < auc_hi, f"case no longer straddles: auc={res['auc']}"
    assert res["verdict"] == expected


def test_verdict_thresholds_are_ordered_and_are_the_documented_values():
    assert VERDICT_THRESHOLDS == {"broken": 0.85, "suspect": 0.70}
    assert VERDICT_THRESHOLDS["broken"] > VERDICT_THRESHOLDS["suspect"]


# --------------------------------------------------------------------------
# cross-validation hygiene: a leak here is a false "your dataset is broken"
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_splits, n_per_class", [(5, 50), (3, 45)])
def test_preprocessing_is_fitted_inside_each_fold_not_on_the_full_array(
    monkeypatch, n_splits, n_per_class
):
    """Any scaler fitted on the whole array before splitting leaks the test
    fold's distribution into training and inflates the AUC.

    Fitting is invisible in the returned number for a scaler -- standardising
    is so nearly harmless to logistic regression that the leak moves the AUC by
    0.0000 -- so this checks the mechanism: the scaler must be fitted once per
    fold, on a fold-sized slice, and never on all `n` rows.

    Parametrised over `n_splits` because that argument is otherwise unpinned:
    it is documented, echoed in the result, and every other test leaves it at
    its default, so hard-coding `cv=5` would go unnoticed while the returned
    dict advertised 3.
    """
    seen: list[int] = []

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            seen.append(len(X))
            return super().fit(X, y, sample_weight)

    monkeypatch.setattr(controls, "StandardScaler", RecordingScaler)
    features, labels = _graded_features(1.0, n_per_class=n_per_class)
    n = 2 * n_per_class
    res = content_blind_auc(features, labels, n_splits=n_splits)

    assert res["n_splits"] == n_splits
    assert seen, "the scaler was never fitted -- is it still in the pipeline?"
    assert n not in seen, f"scaler saw all {n} rows: leak"
    assert len(seen) == n_splits
    assert set(seen) == {n - n // n_splits}


def test_the_classifier_receives_the_callers_array_untouched(monkeypatch):
    """The general form of the leak guard: nothing may be fitted on the full
    array, not just the scaler this module happens to use today.

    A `SelectKBest(f_classif).fit_transform(features, labels)` inserted before
    `cross_val_predict` sees every label while choosing columns and inflates
    the AUC by enough to turn `clean` into `suspect` -- a false "your dataset
    is broken". Any such step necessarily replaces the array that reaches
    `cross_val_predict`, so identity is the thing to assert.
    """
    seen: dict = {}
    real = controls.cross_val_predict

    def spy(estimator, X, y, **kwargs):
        seen["X"], seen["y"] = X, y
        return real(estimator, X, y, **kwargs)

    monkeypatch.setattr(controls, "cross_val_predict", spy)
    features, labels = _graded_features(1.0, n_per_class=50)
    content_blind_auc(features, labels)

    assert seen["X"] is features, "features were preprocessed before the split"
    assert seen["y"] is labels


def test_a_classifier_scored_on_its_own_training_data_would_be_caught(tmp_path):
    """20 pure-noise features over 100 rows with alternating labels: nothing is
    learnable, but a model evaluated in-sample scores ~0.95. Held-out scoring
    must stay near chance, so this fails if cross-validation is dropped."""
    rng = np.random.default_rng(3)
    features = rng.normal(0.0, 1.0, (100, 20))
    labels = np.array([i % 2 for i in range(100)])
    res = content_blind_auc(features, labels)
    assert res["auc"] < VERDICT_THRESHOLDS["suspect"]
    assert res["verdict"] == "clean"


def test_the_seed_is_reproducible_and_actually_reaches_the_bootstrap():
    """Same seed -> identical result; different seed -> same AUC, different CI.

    The first half alone would pass with the seed dropped at both call sites,
    since the cross-validated AUC does not vary with it at all (the folds are
    deterministic and lbfgs is deterministic on this data). The CI is the only
    place the seed has an observable effect, so it is the only place that can
    prove the seed was threaded through rather than ignored.
    """
    features, labels = _graded_features(1.0)
    a = content_blind_auc(features, labels, seed=1234)
    b = content_blind_auc(features, labels, seed=1234)
    c = content_blind_auc(features, labels, seed=999)

    assert a["auc"] == b["auc"] and a["auc_ci"] == b["auc_ci"]
    assert c["auc"] == a["auc"]
    assert c["auc_ci"] != a["auc_ci"], "the seed never reached bootstrap_ci"


# --------------------------------------------------------------------------
# the estimator-branch confound
# --------------------------------------------------------------------------

def test_a_verdict_is_withheld_when_the_estimator_branch_tracks_the_label():
    """Reals JPEG (exact branch), fakes PNG (pixel branch) -- exactly the
    organisers' benchmark. Column 3 is then partly a format label, so the
    headline verdict must not be reported as if it measured the images."""
    features, labels, branches = _confound_case(20, 0)
    res = content_blind_auc(features, labels, quality_branches=branches)

    assert res["verdict"] == "confounded"
    assert res["verdict_ignoring_branch_confound"] == "broken"
    check = res["quality_branch_check"]
    assert check["confounded"] is True
    assert check["status"] == "mixed"
    assert check["branch_label_auc"] == pytest.approx(1.0)
    assert check["exact_fraction_by_class"] == {0: 1.0, 1: 0.0}


def test_a_verdict_survives_when_the_branch_split_is_balanced_across_classes():
    features, labels, branches = _confound_case(10, 10)
    res = content_blind_auc(features, labels, quality_branches=branches)

    assert res["verdict"] == "broken"          # a real, non-format separation
    assert res["quality_branch_check"]["confounded"] is False
    assert res["quality_branch_check"]["branch_label_auc"] == pytest.approx(0.5)


def test_the_confound_threshold_is_a_boundary_not_a_decoration():
    """Behavioural pin for BRANCH_CONFOUND_THRESHOLD (0.55, i.e. a 10-point gap
    in exact-branch share between the classes). 8/20 vs 12/20 is a 0.20 gap
    (branch AUC 0.60) and must confound; 9/20 vs 10/20 is a 0.05 gap (0.525)
    and must not."""
    def run(n_exact_real, n_exact_fake):
        features, labels, branches = _confound_case(n_exact_real, n_exact_fake)
        return content_blind_auc(features, labels, quality_branches=branches)

    over = run(12, 8)
    under = run(10, 9)
    assert over["quality_branch_check"]["branch_label_auc"] == pytest.approx(0.60)
    assert under["quality_branch_check"]["branch_label_auc"] == pytest.approx(0.525)
    assert over["verdict"] == "confounded"
    assert under["verdict"] == "broken"


def test_an_unchecked_result_says_so_rather_than_looking_checked():
    features, labels = _graded_features(1.4)
    res = content_blind_auc(features, labels)
    assert res["quality_branch_check"] == "not performed: no branch provenance supplied"


# --------------------------------------------------------------------------
# metadata_control: the end-to-end path the team should actually run
# --------------------------------------------------------------------------

def test_metadata_control_flags_the_organisers_benchmark_shape(tmp_path):
    """JPEG reals vs PNG fakes: confounded verdict, and a geometry-only verdict
    that is still trustworthy because it cannot involve the estimator."""
    paths, labels = [], []
    for i in range(20):
        paths.append(_photo(tmp_path / f"r{i}.jpg", (320, 240), "JPEG", 40, seed=i)); labels.append(0)
    for i in range(20):
        paths.append(_photo(tmp_path / f"f{i}.png", (512, 512), seed=100 + i)); labels.append(1)
    with pytest.warns(UserWarning, match="two different branches"):
        res = metadata_control(paths, np.array(labels))

    assert res["verdict"] == "confounded"
    assert res["quality_branch_check"]["status"] == "mixed"
    assert res["quality_branch_check"]["counts"] == {"exact": 20, "estimated": 20}
    assert res["geometry"]["verdict"] == "broken"      # resolution alone separates them
    assert res["geometry"]["auc"] > VERDICT_THRESHOLDS["broken"]
    assert any("branch" in c for c in res["caveats"])


def test_metadata_control_still_caveats_a_clean_verdict_on_all_png_input(tmp_path):
    """Formats matched -> no branch confound -> but column 3 is now a pixel
    statistic with a 14-31 point error, so a quiet control is weak evidence and
    must say so."""
    paths, labels = [], []
    for i in range(40):
        paths.append(_photo(tmp_path / f"x{i}.png", (256, 256), seed=i))
        labels.append(i % 2)
    res = metadata_control(paths, np.array(labels))

    assert res["verdict"] == "clean"
    assert res["quality_branch_check"]["status"] == "pixel_derived"
    assert res["quality_branch_check"]["counts"] == {"exact": 0, "estimated": 40}
    assert any("pixel" in c for c in res["caveats"]), res["caveats"]


def test_metadata_control_has_no_caveats_when_every_file_is_a_real_jpeg(tmp_path):
    paths, labels = [], []
    for i in range(40):
        paths.append(_photo(tmp_path / f"j{i}.jpg", (256, 256), "JPEG", 80, seed=i))
        labels.append(i % 2)
    res = metadata_control(paths, np.array(labels))

    assert res["quality_branch_check"]["status"] == "file_metadata"
    assert res["caveats"] == []
    assert res["verdict"] in {"clean", "suspect", "broken"}


def test_geometry_control_excludes_the_estimator_derived_column(tmp_path):
    """The geometry sub-result exists to be trustworthy when the quality column
    is not, so it must be computed from width/height/aspect ONLY.

    Every file here is 256x256, so geometry cannot separate the classes at all;
    only the estimator branch can (JPEG reals, PNG fakes). If column 3 leaked
    into the geometry control it would report "broken" instead of "clean".
    """
    paths, labels = [], []
    for i in range(20):
        paths.append(_photo(tmp_path / f"r{i}.jpg", (256, 256), "JPEG", 30, seed=i)); labels.append(0)
    for i in range(20):
        paths.append(_photo(tmp_path / f"f{i}.png", (256, 256), seed=100 + i)); labels.append(1)
    with pytest.warns(UserWarning, match="two different branches"):
        res = metadata_control(paths, np.array(labels))

    assert res["verdict"] == "confounded"
    assert res["geometry"]["verdict"] == "clean"
    assert res["geometry"]["auc"] < VERDICT_THRESHOLDS["suspect"]
    assert res["geometry"]["quality_branch_check"] == "not applicable: geometry columns only"


def _confound_case(n_exact_real, n_exact_fake):
    """40 rows whose quality column separates the classes perfectly, with the
    estimator branch split as asked. Returns (features, labels, branches)."""
    labels = np.array([0] * 20 + [1] * 20)
    features = np.column_stack([
        np.full(40, 512.0), np.full(40, 512.0), np.ones(40),
        np.concatenate([np.full(20, 40.0), np.full(20, 92.0)]),
    ])
    branches = np.array(
        ["exact"] * n_exact_real + ["estimated"] * (20 - n_exact_real)
        + ["exact"] * n_exact_fake + ["estimated"] * (20 - n_exact_fake)
    )
    return features, labels, branches


def test_a_confounded_result_withholds_the_number_not_just_the_verdict():
    """`verdict: "confounded"` beside a plain `auc: 1.0` is not a withheld
    result: the number is what gets lifted into a results table under time
    pressure, and it carries no trace of the caveat. It must move to a name
    that cannot be quoted without incriminating itself, exactly as the verdict
    does, and `result["auc"]` must fail loudly rather than answer.
    """
    features, labels, branches = _confound_case(20, 0)
    res = content_blind_auc(features, labels, quality_branches=branches)

    assert "auc" not in res and "auc_ci" not in res
    with pytest.raises(KeyError):
        res["auc"]
    assert res["auc_ignoring_branch_confound"] == pytest.approx(1.0)
    lo, hi = res["auc_ci_ignoring_branch_confound"]
    assert lo <= res["auc_ignoring_branch_confound"] <= hi


def test_an_unconfounded_result_still_reports_auc_under_its_plain_name():
    features, labels, branches = _confound_case(10, 10)
    res = content_blind_auc(features, labels, quality_branches=branches)
    assert res["verdict"] == "broken"
    assert res["auc"] == pytest.approx(1.0)
    assert "auc_ignoring_branch_confound" not in res


def test_the_confound_threshold_includes_its_boundary_value():
    """The docstring says "at or above" 0.55. 10/20 exact against 12/20 is a
    0.10 difference in exact-branch share, i.e. a branch AUC of exactly 0.55
    (exactly representable, verified), so `>=` and `>` disagree here and
    nowhere else."""
    features, labels, branches = _confound_case(10, 12)
    res = content_blind_auc(features, labels, quality_branches=branches)
    assert res["quality_branch_check"]["branch_label_auc"] == 0.55
    assert res["verdict"] == "confounded"


@pytest.mark.parametrize("bad", [["Exact"] * 20 + ["estimated"] * 20,
                                 ["exact"] * 20 + ["fallback"] * 20])
def test_unknown_branch_labels_are_rejected_rather_than_read_as_pixel_derived(bad):
    """`== BRANCH_EXACT` maps anything unrecognised to "not exact", which reads
    out as pixel_derived / confounded: False / "no file carried a quantisation
    table" -- three false statements and no error, on the reassuring side."""
    features, labels, _ = _confound_case(20, 0)
    with pytest.raises(ValueError, match="unknown estimator branch label"):
        content_blind_auc(features, labels, quality_branches=np.array(bad))


def test_metadata_control_caveats_a_mixed_column_even_when_it_is_not_confounded(tmp_path):
    """Both formats present in both classes: the branch split does not follow
    the label, so no verdict is withheld -- but column 3 still spans two
    scales, which is not a like-for-like comparison. This is the state a team
    lands in after half-fixing the formats, so the caveat has to be there.
    """
    paths, labels = [], []
    for i in range(20):
        fmt = ("JPEG", "jpg") if i % 2 else ("PNG", "png")
        paths.append(_photo(tmp_path / f"r{i}.{fmt[1]}", (256, 256), fmt[0],
                            80 if fmt[0] == "JPEG" else None, seed=i))
        labels.append(0)
    for i in range(20):
        fmt = ("JPEG", "jpg") if i % 2 else ("PNG", "png")
        paths.append(_photo(tmp_path / f"f{i}.{fmt[1]}", (256, 256), fmt[0],
                            80 if fmt[0] == "JPEG" else None, seed=100 + i))
        labels.append(1)
    with pytest.warns(UserWarning, match="two different branches"):
        res = metadata_control(paths, np.array(labels))

    check = res["quality_branch_check"]
    assert check["status"] == "mixed" and check["confounded"] is False
    assert res["verdict"] != "confounded"
    assert any("two scales" in c for c in res["caveats"]), res["caveats"]
