"""The package-level contract in `aigcdet.calibrate.__init__`.

Both fits and the decision policy lean on these two helpers, so they are tested
where they live rather than three times over through their callers.
"""
import numpy as np
import pytest

from aigcdet.calibrate import (
    INTERNAL_VAL_SPLIT,
    check_fit_split,
    fit_standardiser,
)


def test_an_all_validation_column_passes():
    check_fit_split(np.full(5, INTERNAL_VAL_SPLIT), 5)


def test_a_pandas_split_column_passes():
    """`bank.meta["split"]` is a pandas Series, which is what callers hand it."""
    pd = pytest.importorskip("pandas")
    check_fit_split(pd.Series([INTERNAL_VAL_SPLIT] * 5), 5)


@pytest.mark.parametrize("wrong", ["train", "test", "heldout_generator", ""])
def test_a_column_of_another_split_is_refused(wrong):
    with pytest.raises(ValueError, match="val_internal"):
        check_fit_split(np.full(5, wrong), 5)


def test_the_offending_splits_are_named():
    s = np.array([INTERNAL_VAL_SPLIT, "train", "test", INTERNAL_VAL_SPLIT, "train"])
    with pytest.raises(ValueError, match=r"3 of 5 rows carry \['test', 'train'\]"):
        check_fit_split(s, 5)


def test_a_scalar_promise_is_refused():
    """The guard this replaced compared exactly this string to a constant."""
    with pytest.raises(ValueError, match="one label per row"):
        check_fit_split(INTERNAL_VAL_SPLIT, 5)


def test_a_two_dimensional_split_is_refused():
    with pytest.raises(ValueError, match="one label per row"):
        check_fit_split(np.full((5, 1), INTERNAL_VAL_SPLIT), 5)


@pytest.mark.parametrize("n_labels", [4, 6])
def test_a_column_of_the_wrong_length_is_refused(n_labels):
    with pytest.raises(ValueError, match="expected 5 labels"):
        check_fit_split(np.full(n_labels, INTERNAL_VAL_SPLIT), 5)


# --- fit_standardiser ------------------------------------------------------

def test_an_informative_column_is_standardised_to_unit_variance():
    rng = np.random.default_rng(0)
    c = rng.normal(3.0, 2.0, size=(500, 1))
    mu, sd, const = fit_standardiser(c)
    assert const == ()
    z = (c - mu) / sd
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0, abs=1e-12)


def test_an_exactly_constant_column_is_scaled_by_one():
    mu, sd, const = fit_standardiser(np.full((10, 1), 4.0))
    assert const == (0,)
    assert sd[0, 0] == 1.0


@pytest.mark.parametrize("sd_true", [1e-9, 1e-8, 1e-7])
def test_a_near_constant_column_is_scaled_by_one_too(sd_true):
    """The tolerance is RELATIVE to the column's own scale. An absolute one
    (the 1e-12 this replaced, or the brief's `sd + 1e-6`) leaves a column with
    sd 1e-9 about a mean of 4.0 divided by ~1e-9, which multiplies any shifted
    value at inference by a billion."""
    rng = np.random.default_rng(1)
    c = 4.0 + sd_true * rng.normal(size=(500, 1))
    _, sd, const = fit_standardiser(c)
    assert const == (0,)
    assert sd[0, 0] == 1.0


def test_a_genuinely_informative_small_column_is_not_neutralised():
    """The relative tolerance must not swallow a real column just because its
    numbers are small: severity in [0, 1] has sd ~0.3 of a mean ~0.5."""
    rng = np.random.default_rng(2)
    c = rng.uniform(0.0, 1.0, size=(500, 1))
    _, sd, const = fit_standardiser(c)
    assert const == ()
    assert sd[0, 0] > 0.2


def test_the_tolerance_is_relative_and_the_boundary_is_where_it_is_documented():
    """CONST_COLUMN_TOL is 1e-6 of the column's own scale (~10 float32 ulps).
    A column whose relative spread is above that is real variation and is kept,
    however small its absolute numbers -- so this pins the line rather than
    leaving it to be moved by accident."""
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(500, 1))
    kept = 4.0 + 1e-4 * noise            # relative spread 2.5e-5, above the bar
    dropped = 4.0 + 1e-8 * noise         # relative spread 2.5e-9, below it
    assert fit_standardiser(kept)[2] == ()
    assert fit_standardiser(dropped)[2] == (0,)


def test_columns_are_judged_independently():
    rng = np.random.default_rng(3)
    c = np.column_stack([rng.normal(size=500), np.full(500, 7.0),
                         rng.uniform(size=500), 4.0 + 1e-9 * rng.normal(size=500)])
    _, sd, const = fit_standardiser(c)
    assert const == (1, 3)
    assert sd[0, 1] == 1.0 and sd[0, 3] == 1.0
    assert sd[0, 0] > 0.5 and sd[0, 2] > 0.2
