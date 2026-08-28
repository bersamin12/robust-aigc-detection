"""Tests for the resolution-only control.

Three fixtures, and WHICH ONE a test uses is the whole design. The recurring
failure on this project is a fixture that makes the property under test
unreachable, and this module has two properties whose fixtures are mutually
exclusive:

* `separable_frame` -- resolution separates the classes PERFECTLY. It is the
  only fixture that can detect a broken classifier (a working one scores
  exactly 1.0, so any regression moves the number). It CANNOT detect a metric
  bug: when the classes are perfectly separated, TPR is 1.0 at every FPR, so
  computing it at 5% instead of 1% is invisible.
* `overlapping_frame` -- resolution separates the classes PARTIALLY, tuned so
  that `0 < TPR@1% < TPR@5% < 1`. It is the only fixture that can detect the
  operating point being wrong. It is a poor detector of a broken classifier,
  because a degraded model still lands somewhere in the same open interval.
* `uninformative_frame` -- resolution carries NOTHING (both classes share one
  resolution pool) while a `pixel_mean` column separates the classes
  perfectly. It is the only fixture that can detect a pixel-derived feature
  reaching the classifier: on it the honest answer is chance and the
  pixel-leaking answer is 1.0, so the two are maximally far apart. On either
  of the other two fixtures a pixel leak is invisible, because dimensions
  already carry the answer.

`test_reports_a_low_floor_when_resolution_carries_nothing` and
`test_detects_a_real_resolution_leak` are a matched pair on purpose: the first
alone passes for a classifier that always returns 0, the second alone passes
for one that always returns 1.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from aigcdet.baselines.resolution import (
    BASELINE_ROW_FOOTNOTE,
    BASELINE_ROW_LABEL,
    CANONICAL_SHORT_SIDES,
    DIMENSION_COLUMNS,
    LABEL_COLUMN,
    RESOLUTION_FEATURE_NAMES,
    SPLIT_COLUMN,
    SUBSTANTIAL_MARGIN,
    ResolutionBaseline,
    describe,
    resolution_features,
    resolution_leak_report,
)
from aigcdet.operating_point import TARGET_FPR, tpr_column_name

TPR_KEY = tpr_column_name(TARGET_FPR)

#: The two split labels the fixtures use. Deliberately NOT the real split
#: names: nothing in this module may depend on a hard-coded "train".
FIT, SCORE = "fitrows", "scorerows"

#: The fit and score halves get DIFFERENT row counts throughout, so a swap of
#: the two masks is visible in `n_fit`/`n_score` rather than cancelling out.
#: Equal counts would make `test_fit_and_score_rows_are_the_ones_named` pass
#: against exactly the bug it exists to catch.
N_FIT, N_SCORE = 600, 400


def _frame(width, height, label, split, **extra) -> pd.DataFrame:
    """A manifest-shaped frame. `extra` columns are decoys: nothing in the
    module under test may read them."""
    df = pd.DataFrame({
        DIMENSION_COLUMNS[0]: np.asarray(width, dtype=np.int64),
        DIMENSION_COLUMNS[1]: np.asarray(height, dtype=np.int64),
        LABEL_COLUMN: np.asarray(label, dtype=np.int64),
        SPLIT_COLUMN: np.asarray(split),
    })
    for k, v in extra.items():
        df[k] = v
    return df


def _split_labels(n_fit: int, n_score: int) -> np.ndarray:
    return np.array([FIT] * n_fit + [SCORE] * n_score)


@pytest.fixture
def separable_frame() -> pd.DataFrame:
    """Resolution separates the classes PERFECTLY: reals are 200-short, fakes
    are 1024-short, exactly the shape of the organisers' benchmark.

    Carries a decoy column `y` holding the INVERTED label. A metric computed
    against the wrong label column then reports ~0.0 instead of ~1.0 rather
    than merely raising KeyError, so the failure names the bug.
    """
    n = N_FIT + N_SCORE
    rng = np.random.default_rng(0)
    label = np.tile([0, 1], n // 2)
    short = np.where(label == 0, 200, 1024)
    long = short + rng.integers(0, 200, n)
    return _frame(short, long, label, _split_labels(N_FIT, N_SCORE),
                  y=1 - label)


@pytest.fixture
def overlapping_frame() -> pd.DataFrame:
    """Resolution separates the classes PARTIALLY.

    The short side is a CONTINUUM and the fake rate ramps linearly across it,
    with a pure-real floor below 350 and a pure-fake ceiling above 750. Both
    ends are load-bearing:

    * the ramp is what makes the ROC finely graded. With only a handful of
      discrete resolutions the curve has a handful of vertices, and TPR@1% and
      TPR@5% land on the same one -- which would silently un-test the operating
      point while every assertion still passed.
    * the pure-fake ceiling is what keeps TPR@1% off the floor. Without a
      region that is unambiguously generated, the tree's top leaf carries
      negatives, the 1% budget is spent before any true positive is admitted,
      and TPR@1% is 0.0 -- equal to nothing and less than everything, so the
      comparison against TPR@5% would pass for the wrong reason.

    Seed and shape are pinned to a draw measured to give TPR@1% 0.3828 against
    TPR@5% 0.7464, the widest margin of sixteen candidates.
    `test_the_fixture_actually_separates_the_operating_points` asserts that
    property directly, so this tuning is checked rather than assumed.
    """
    n = N_FIT + N_SCORE
    rng = np.random.default_rng(0)
    short = rng.integers(200, 901, n)
    p = np.clip((short - 350) / 400.0, 0.0, 1.0)
    label = (rng.random(n) < p).astype(np.int64)
    long = short + rng.integers(0, 120, n)
    return _frame(short, long, label, _split_labels(N_FIT, N_SCORE))


@pytest.fixture
def uninformative_frame() -> pd.DataFrame:
    """Resolution carries NOTHING; a pixel statistic carries EVERYTHING.

    Both classes draw their dimensions from one pool, so the honest answer is
    chance. `pixel_mean` and `pixel_std` are perfect separators. Any code path
    by which a pixel-derived column reaches the classifier turns the honest
    ~0.5 into ~1.0, which is the largest gap this suite can construct.
    """
    n = N_FIT + N_SCORE
    rng = np.random.default_rng(2)
    label = np.tile([0, 1], n // 2)
    short = rng.choice([200, 256, 384, 512, 768, 1024], n)
    long = short + rng.integers(0, 200, n)
    return _frame(short, long, label, _split_labels(N_FIT, N_SCORE),
                  pixel_mean=label.astype(float) * 100.0 + rng.normal(0, 1, n),
                  pixel_std=label.astype(float) * -50.0 + rng.normal(0, 1, n))


# --- the features are dimensions only --------------------------------------

def test_feature_names_match_the_matrix_width(separable_frame):
    X = resolution_features(separable_frame)
    assert X.shape == (len(separable_frame), len(RESOLUTION_FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_features_ignore_every_column_that_is_not_a_dimension(uninformative_frame):
    """Bit-identical output for two frames that differ everywhere EXCEPT
    `DIMENSION_COLUMNS`. This is the structural claim, tested structurally
    rather than by reading the source."""
    other = uninformative_frame.copy()
    rng = np.random.default_rng(3)
    other["pixel_mean"] = rng.normal(0, 1, len(other))
    other["pixel_std"] = rng.normal(0, 1, len(other))
    other[LABEL_COLUMN] = 1 - other[LABEL_COLUMN]
    other["generator"] = "sdxl"
    assert np.array_equal(resolution_features(uninformative_frame),
                          resolution_features(other))


def test_a_perfectly_separating_pixel_column_cannot_reach_the_score(
        uninformative_frame):
    """THE pixel-leak test. Dimensions carry nothing here and `pixel_mean`
    carries everything, so a classifier that can see pixels scores ~1.0 and one
    that cannot scores ~chance."""
    r = resolution_leak_report(uninformative_frame, fit_split=FIT,
                               score_split=SCORE)
    assert r["auc"] < 0.75, (
        "the control scored far above chance on a frame whose DIMENSIONS carry "
        "no signal at all -- the only separating columns are pixel statistics, "
        "so something pixel-derived reached the classifier")
    assert r[TPR_KEY] < 0.25


def test_a_feature_matrix_is_refused(uninformative_frame):
    """The interface itself is the guarantee: there is no way to hand this
    baseline an array a caller could have put pixel statistics into."""
    X = resolution_features(uninformative_frame)
    with pytest.raises(TypeError, match="DataFrame"):
        resolution_features(X)
    with pytest.raises(TypeError, match="DataFrame"):
        ResolutionBaseline().fit(X)


def test_image_paths_are_refused():
    with pytest.raises(TypeError, match="DataFrame"):
        resolution_features(["/some/image.png", "/another.png"])


def test_declared_dimension_columns_are_width_and_height():
    assert DIMENSION_COLUMNS == ("width", "height")


def test_non_positive_dimensions_are_refused():
    df = _frame([200, 0], [300, 300], [0, 1], [FIT, FIT])
    with pytest.raises(ValueError, match="non-positive"):
        resolution_features(df)


def test_canonical_short_side_flag_is_a_declared_prior_not_a_fitted_one():
    """Both of the manifest's giveaway sizes are checked, in both directions:
    224/256 are canonical AND 100% fake, 450 is 100% fake and NOT canonical,
    200 is the benchmark's entire real half and NOT canonical. A tuple quietly
    tuned to the observed leak would have added 450 and 200."""
    assert 224 in CANONICAL_SHORT_SIDES and 256 in CANONICAL_SHORT_SIDES
    assert 450 not in CANONICAL_SHORT_SIDES
    assert 200 not in CANONICAL_SHORT_SIDES
    col = RESOLUTION_FEATURE_NAMES.index("short_side_is_canonical")
    X = resolution_features(_frame([256, 450, 200, 512],
                                   [300, 500, 250, 900],
                                   [0, 1, 0, 1], [FIT] * 4))
    assert X[:, col].tolist() == [1.0, 0.0, 0.0, 1.0]


# --- the classifier actually works, and actually reports a floor -----------

def test_detects_a_real_resolution_leak(separable_frame):
    """Perfectly separable by resolution, so a working control scores exactly
    1.0. Any regression in the classifier moves this number."""
    r = resolution_leak_report(separable_frame, fit_split=FIT, score_split=SCORE)
    assert r[TPR_KEY] == 1.0
    assert r["auc"] == 1.0


def test_reports_a_low_floor_when_resolution_carries_nothing(uninformative_frame):
    """The other half of the pair: without this, a control that always returns
    1.0 passes `test_detects_a_real_resolution_leak`."""
    r = resolution_leak_report(uninformative_frame, fit_split=FIT,
                               score_split=SCORE)
    assert r[TPR_KEY] < 0.25
    assert 0.25 < r["auc"] < 0.75


def test_scores_are_probabilities_and_higher_means_generated(separable_frame):
    fit = separable_frame[separable_frame[SPLIT_COLUMN] == FIT]
    scored = separable_frame[separable_frame[SPLIT_COLUMN] == SCORE]
    s = ResolutionBaseline().fit(fit).score(scored)
    assert ((s >= 0.0) & (s <= 1.0)).all()
    y = scored[LABEL_COLUMN].to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_the_fitted_tree_is_printable(separable_frame):
    fit = separable_frame[separable_frame[SPLIT_COLUMN] == FIT]
    text = ResolutionBaseline().fit(fit).format_tree()
    assert any(name in text for name in RESOLUTION_FEATURE_NAMES)


def test_the_result_is_reproducible(separable_frame, overlapping_frame):
    for frame in (separable_frame, overlapping_frame):
        a = resolution_leak_report(frame, fit_split=FIT, score_split=SCORE)
        b = resolution_leak_report(frame, fit_split=FIT, score_split=SCORE)
        assert a[TPR_KEY] == b[TPR_KEY] and a["auc"] == b["auc"]


# --- the operating point ---------------------------------------------------

def test_the_fixture_actually_separates_the_operating_points(overlapping_frame):
    """Guards the guard. If a later edit makes `overlapping_frame` separable
    (or hopeless), the two tests below stop testing anything and pass anyway;
    this one fails instead and says why."""
    r1 = resolution_leak_report(overlapping_frame, fit_split=FIT,
                                score_split=SCORE, target_fpr=0.01)
    r5 = resolution_leak_report(overlapping_frame, fit_split=FIT,
                                score_split=SCORE, target_fpr=0.05)
    assert 0.0 < r1[tpr_column_name(0.01)] < r5[tpr_column_name(0.05)] < 1.0, (
        "overlapping_frame no longer places TPR@1% strictly below TPR@5% "
        "strictly below 1.0, so it can no longer detect the TPR being computed "
        "at the wrong FPR")


def test_tpr_is_reported_at_the_project_operating_point(overlapping_frame):
    """THE wrong-FPR test. The default must be the project operating point and
    nothing else; on this fixture 1% and 5% give different answers, so a
    hard-coded 5% (or any other point) shows up as a value mismatch."""
    default = resolution_leak_report(overlapping_frame, fit_split=FIT,
                                     score_split=SCORE)
    explicit = resolution_leak_report(overlapping_frame, fit_split=FIT,
                                      score_split=SCORE, target_fpr=TARGET_FPR)
    assert default[TPR_KEY] == explicit[TPR_KEY]
    assert default["target_fpr"] == TARGET_FPR

    wrong = resolution_leak_report(overlapping_frame, fit_split=FIT,
                                   score_split=SCORE, target_fpr=0.05)
    assert default[TPR_KEY] != wrong[tpr_column_name(0.05)], (
        "TPR at 1% equals TPR at 5% on a fixture built so they differ -- the "
        "reported TPR is not being computed at the requested FPR")


def test_the_tpr_key_is_derived_from_the_operating_point(overlapping_frame):
    """A moved operating point must rename the column rather than leave a
    `tpr_at_1pct` holding 5% values -- `operating_point`'s whole reason for
    existing."""
    r = resolution_leak_report(overlapping_frame, fit_split=FIT,
                               score_split=SCORE, target_fpr=0.05)
    assert tpr_column_name(0.05) in r
    assert tpr_column_name(0.01) not in r
    assert r["operating_point"] == "5%"


# --- fit and score rows -----------------------------------------------------

def test_fit_and_score_rows_are_the_ones_named(separable_frame):
    """The two halves have DIFFERENT counts, so swapping the masks is visible
    here rather than cancelling out."""
    r = resolution_leak_report(separable_frame, fit_split=FIT, score_split=SCORE)
    assert (r["n_fit"], r["n_score"]) == (N_FIT, N_SCORE)
    assert r["fit_split"] == FIT and r["score_split"] == SCORE
    assert N_FIT != N_SCORE, "fixture can no longer detect a swap"


def test_score_refuses_rows_from_the_split_it_was_fitted_on(separable_frame):
    """THE train-on-test test, at the class level where the guard lives."""
    fit = separable_frame[separable_frame[SPLIT_COLUMN] == FIT]
    model = ResolutionBaseline().fit(fit)
    with pytest.raises(ValueError, match="fitted on"):
        model.score(fit)
    with pytest.raises(ValueError, match="fitted on"):
        model.score(separable_frame)          # score rows PLUS one fit row
    scored = separable_frame[separable_frame[SPLIT_COLUMN] == SCORE]
    assert len(model.score(scored)) == N_SCORE


def test_one_split_may_not_be_both_the_fit_and_the_score_split(separable_frame):
    with pytest.raises(ValueError, match="fitting and scoring the same rows"):
        resolution_leak_report(separable_frame, fit_split=FIT, score_split=FIT)


def test_a_fit_spanning_two_splits_is_refused(separable_frame):
    """Without one split label there is nothing for `score` to refuse, so the
    train-on-test guard would silently become a no-op."""
    with pytest.raises(ValueError, match="ONE split label"):
        ResolutionBaseline().fit(separable_frame)


def test_an_absent_split_is_named(separable_frame):
    with pytest.raises(ValueError, match="matches no rows"):
        resolution_leak_report(separable_frame, fit_split=FIT,
                               score_split="nosuchsplit")


def test_score_before_fit_raises(separable_frame):
    with pytest.raises(RuntimeError, match="before fit"):
        ResolutionBaseline().score(separable_frame)


def test_a_constant_score_is_refused_rather_than_reported_as_no_leak():
    """A tree that found no split scores every row identically: AUC exactly
    0.5, TPR exactly 0.0. In a results table that is indistinguishable from the
    honest finding 'resolution carries nothing here' -- and it is the dangerous
    reading, because it publishes a floor of zero that every model clears.

    Same defence, and the same reasoning, as `aeroblade_scores`' all-zero
    column check.
    """
    n = N_FIT + N_SCORE
    rng = np.random.default_rng(4)
    label = np.tile([0, 1], n // 2)
    # Every FIT row is one single resolution, so the tree cannot split at all.
    width = np.where(np.arange(n) < N_FIT, 512, rng.choice([256, 1024], n))
    df = _frame(width, width, label, _split_labels(N_FIT, N_SCORE))
    with pytest.raises(ValueError, match="no usable split"):
        resolution_leak_report(df, fit_split=FIT, score_split=SCORE)


def test_a_single_class_score_split_is_refused(separable_frame):
    """`heldout_generator` is 100% generated in the frozen manifest, and a TPR
    at a false-POSITIVE rate needs negatives."""
    df = separable_frame.copy()
    df.loc[(df[SPLIT_COLUMN] == SCORE) & (df[LABEL_COLUMN] == 0),
           SPLIT_COLUMN] = FIT
    with pytest.raises(ValueError, match="single class"):
        resolution_leak_report(df, fit_split=FIT, score_split=SCORE)


def test_a_single_class_fit_split_is_refused(separable_frame):
    df = separable_frame.copy()
    df.loc[(df[SPLIT_COLUMN] == FIT) & (df[LABEL_COLUMN] == 0),
           SPLIT_COLUMN] = SCORE
    with pytest.raises(ValueError, match="single class"):
        resolution_leak_report(df, fit_split=FIT, score_split=SCORE)


# --- the label column -------------------------------------------------------

def test_the_label_column_is_the_manifest_one():
    from aigcdet.data.manifest import MANIFEST_COLUMNS
    assert LABEL_COLUMN == "label" and LABEL_COLUMN in MANIFEST_COLUMNS


def test_the_metric_is_computed_against_label_and_not_a_decoy(separable_frame):
    """THE wrong-label-column test. `separable_frame` carries `y`, the inverted
    label; scoring against it turns 1.0 into 0.0, so the failure is a value
    mismatch that names the bug rather than a bare KeyError."""
    assert (separable_frame["y"] != separable_frame[LABEL_COLUMN]).all()
    r = resolution_leak_report(separable_frame, fit_split=FIT, score_split=SCORE)
    assert r["auc"] == 1.0
    assert r["positive_rate_score_split"] == pytest.approx(
        separable_frame.loc[separable_frame[SPLIT_COLUMN] == SCORE,
                            LABEL_COLUMN].mean())


def test_a_frame_with_no_label_column_is_refused(separable_frame):
    df = separable_frame.drop(columns=[LABEL_COLUMN])
    with pytest.raises(ValueError, match="label"):
        resolution_leak_report(df, fit_split=FIT, score_split=SCORE)


# --- what the number means --------------------------------------------------

def test_describe_states_the_number_and_what_it_is(overlapping_frame):
    r = resolution_leak_report(overlapping_frame, fit_split=FIT, score_split=SCORE)
    text = describe(r)
    assert f"{r[TPR_KEY]:.4f}" in text
    assert "1% FPR" in text
    assert "DIMENSIONS ALONE" in text
    assert "never decoded a pixel" in text
    assert "generation artefacts" in text
    assert f"{SUBSTANTIAL_MARGIN:.2f}" in text


def test_describe_warns_only_when_there_is_no_headroom(separable_frame,
                                                       overlapping_frame):
    """A perfectly separable population makes EVERY model score uninterpretable,
    which is the situation on the organisers' benchmark. Saying so is the
    deliverable."""
    perfect = describe(resolution_leak_report(separable_frame, fit_split=FIT,
                                              score_split=SCORE))
    assert "WARNING" in perfect and "indistinguishable" in perfect

    partial = describe(resolution_leak_report(overlapping_frame, fit_split=FIT,
                                              score_split=SCORE))
    assert "WARNING" not in partial


def test_describe_follows_the_operating_point(overlapping_frame):
    r = resolution_leak_report(overlapping_frame, fit_split=FIT,
                               score_split=SCORE, target_fpr=0.05)
    assert "5% FPR" in describe(r)


def test_the_row_label_says_it_is_not_a_detector():
    assert "dimensions" in BASELINE_ROW_LABEL.lower()
    assert "not a detector" in BASELINE_ROW_FOOTNOTE.lower()


# --- against the real, frozen data -----------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_MANIFEST = os.path.join(_REPO_ROOT, "data", "manifest.parquet")
_BENCHMARK = os.path.join(_REPO_ROOT, "data", "benchmark_manifest.parquet")


@pytest.mark.skipif(not os.path.isfile(_MANIFEST),
                    reason="the frozen manifest is not materialised here")
def test_the_resolution_leak_in_the_frozen_manifest_is_real_and_large():
    """Read-only pin on the ACTUAL frozen manifest.

    Every bound is well below what was measured on 2026-08-29 over all 138,116
    rows -- TPR@1%FPR 0.5654, AUC 0.9220, 29.6% of rows at a 100%-fake short
    side -- so this fails on a real regression rather than on noise, and the
    measured values are recorded in the assertion messages so a future reader
    can see how much slack there is.

    The 29.6% counts every 100%-fake short side in the manifest. The 29.1% /
    40,150 quoted in the module docstring is the narrower WildFake-only figure;
    the two are different counts of the same leak, not a disagreement.
    """
    df = pd.read_parquet(_MANIFEST)
    r = resolution_leak_report(df, fit_split="train", score_split="val_internal")

    assert r[TPR_KEY] > 0.40, (
        f"measured 0.5654 on 2026-08-29; got {r[TPR_KEY]:.4f}")
    assert r["auc"] > 0.85, f"measured 0.9220; got {r['auc']:.4f}"

    short = np.minimum(df["width"], df["height"])
    by_size = df.groupby(short)[LABEL_COLUMN].agg(["size", "mean"])
    all_fake = by_size[by_size["mean"] == 1.0]["size"].sum()
    assert all_fake / len(df) > 0.25, (
        f"measured 29.6% of rows at a 100%-fake short side; got "
        f"{all_fake / len(df):.1%}")


@pytest.mark.skipif(not os.path.isfile(_BENCHMARK),
                    reason="the benchmark manifest is not materialised here")
def test_the_organisers_benchmark_is_perfectly_separable_by_resolution():
    """Read-only pin on the SCORED benchmark. This is the finding.

    Measured over all 13,841 rows of `data/benchmark_manifest.parquet` on
    2026-08-29: every one of the 4,998 COCO reals has a short side of exactly
    200, and all 8,843 DALL-E 3 fakes have one between 346 and 1,746. The
    ranges do not touch, so fitting on half and scoring on the disjoint other
    half gives TPR 1.0000 at 1% FPR and AUC 1.0000 -- from a tree that is a
    SINGLE threshold on width -- without decoding a pixel.

    The consequence is asserted, not just the number: on this population a
    perfect detector and a ruler are indistinguishable, so `describe` must emit
    its no-headroom warning. Any headline we report on this benchmark has to be
    published beside this row.
    """
    df = pd.read_parquet(_BENCHMARK)
    assert set(df[SPLIT_COLUMN]) == {"benchmark"}

    short = np.minimum(df["width"], df["height"])
    real, fake = short[df[LABEL_COLUMN] == 0], short[df[LABEL_COLUMN] == 1]
    assert len(real) and len(fake)
    assert real.max() < fake.min(), (
        f"the benchmark halves' short sides no longer separate: real max "
        f"{real.max()}, fake min {fake.min()} (measured 200 and 346)")

    # Fit and score halves of the benchmark itself: the question is whether
    # THIS population is separable, so both halves must come from it.
    rng = np.random.default_rng(0)
    scored = df.copy()
    scored[SPLIT_COLUMN] = np.where(rng.random(len(df)) < 0.5, FIT, SCORE)
    r = resolution_leak_report(scored, fit_split=FIT, score_split=SCORE)

    assert r[TPR_KEY] == 1.0, f"measured 1.0000; got {r[TPR_KEY]:.4f}"
    assert r["auc"] == 1.0
    assert "WARNING" in describe(r)
