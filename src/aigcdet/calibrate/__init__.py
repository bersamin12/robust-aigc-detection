"""Calibration and abstention (spec §3.6-3.7).

Hygiene, spec §6.7: every fit in this package consumes INTERNAL VALIDATION
rows and nothing else. A temperature, an EQI or a decision policy fitted on the
external benchmark, on test rows, or on the training rows the classifier
already saw does not measure calibration -- it launders the answer into the
number the report quotes, and the leak is invisible downstream because the
probabilities still look like probabilities.

So every `fit` takes a REQUIRED keyword-only `split=`, holding one split label
PER ROW -- the column `bank.meta["split"]` already carries -- and refuses the
whole fit unless every row is `INTERNAL_VAL_SPLIT`.

The per-row array and the absence of a default are both load-bearing, and this
guard was rewritten because its first version had neither. A scalar `split`
defaulting to `"val_internal"` compares a string the caller types against a
constant: it cannot see the rows it was handed, `fit(test_logits, test_y)`
sails straight through it, and because the default is already correct it fires
only for a caller who was threading a real split variable, i.e. one already
being careful. A caller who genuinely has no split column must now construct
one, which is the point -- that construction is where they have to look at
which rows they are about to fit on.
"""
from __future__ import annotations

import numpy as np

#: The only split any fit in this package may consume. Matches the label
#: `aigcdet.data.splits.assign_splits` writes for internal validation.
INTERNAL_VAL_SPLIT = "val_internal"

#: RELATIVE tolerance below which a conditioning column counts as constant on
#: the fit rows: `sd < CONST_COLUMN_TOL * max(1, |mean|)`. Relative, because an
#: absolute threshold is meaningless against a column's own scale -- a
#: degradation proxy with mean 4.0 and sd 1e-9 varies by 2.5e-10 of itself and
#: is constant in every sense that matters, yet clears any small absolute bar.
#: 1e-6 is ~10 float32 ulps: below it the variation is numerical noise from an
#: upstream float32 feature, and dividing by it amplifies that noise a
#: million-fold.
CONST_COLUMN_TOL = 1e-6


class CalibrationError(RuntimeError):
    """A fit that did not converge, or converged somewhere unusable.

    Raised instead of returning whatever the optimiser's last iterate happened
    to be: a temperature of ~0 (infinite confidence) or a huge one (every
    prediction collapsed to 0.5) would otherwise poison every downstream
    probability, ECE, risk-coverage curve and decision in the report.
    """


def check_fit_split(split, n: int) -> None:
    """Refuse the fit unless all `n` rows carry `INTERNAL_VAL_SPLIT` (spec §6.7).

    `split` is one label per row, not a promise the caller types: pass the
    split column itself (`bank.meta["split"]` for the rows being fitted). A
    scalar, a wrong length, or a single contaminating row all fail here.
    """
    s = np.asarray(split)
    if s.ndim != 1:
        shape = "a scalar" if s.ndim == 0 else f"a {s.ndim}-D array"
        raise ValueError(
            f"split must be one label per row -- a 1-D array of length {n}, "
            f"such as bank.meta['split'] for the rows being fitted -- got "
            f"{shape}. A single string is a promise, not evidence")
    if s.shape[0] != n:
        raise ValueError(
            f"split must have one label per row: expected {n} labels, got {s.shape[0]}")
    bad = s != INTERNAL_VAL_SPLIT
    if bad.any():
        raise ValueError(
            f"calibration may only be fitted on the internal validation split "
            f"{INTERNAL_VAL_SPLIT!r}; {int(bad.sum())} of {n} rows carry "
            f"{np.unique(s[bad]).tolist()}. Fitting on training, test or "
            f"benchmark rows invalidates every calibration number in the report")


def fit_standardiser(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Per-column centre and scale for a conditioning matrix.

    Returns `(mu, sd, constant_columns)`. A column whose spread is negligible
    *relative to its own scale* carries no information, so it is centred and
    then scaled by 1 rather than by its ~0 standard deviation. Without that, a
    column with sd 1e-9 about a mean of 4.0 is divided by 1e-9, and a single
    row whose value differs at inference lands 1e9 scale-units from the origin
    and saturates whatever consumes it.

    Shared by both fits in this package so that "a degradation family
    validation never exercised" means the same thing in each.
    """
    mu = cond.mean(axis=0, keepdims=True)
    sd = cond.std(axis=0, keepdims=True)
    const = sd[0] < CONST_COLUMN_TOL * np.maximum(1.0, np.abs(mu[0]))
    return mu, np.where(const, 1.0, sd), tuple(int(i) for i in np.flatnonzero(const))
