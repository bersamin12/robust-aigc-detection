"""Calibration and abstention (spec §3.6-3.7).

Hygiene, spec §6.7: every fit in this package consumes INTERNAL VALIDATION
rows and nothing else. A temperature or an EQI fitted on the external
benchmark, on test rows, or on the training rows the classifier already saw
does not measure calibration -- it launders the answer into the number the
report quotes, and the leak is invisible downstream because the probabilities
still look like probabilities.

So each `fit` takes an explicit `split=`, naming the split the caller is
handing it, and refuses anything but `INTERNAL_VAL_SPLIT`. A caller wiring up
the pipeline threads its `split` column straight through; getting it wrong is
then an exception rather than a silently better-looking table.
"""
from __future__ import annotations

import numpy as np

#: The only split any fit in this package may consume. Matches the label
#: `aigcdet.data.splits.assign_splits` writes for internal validation.
INTERNAL_VAL_SPLIT = "val_internal"

#: Below this per-column standard deviation a conditioning column is constant
#: on the fit rows and carries no information.
CONST_COLUMN_TOL = 1e-12


class CalibrationError(RuntimeError):
    """A fit that did not converge, or converged somewhere unusable.

    Raised instead of returning whatever the optimiser's last iterate happened
    to be: a temperature of ~0 (infinite confidence) or a huge one (every
    prediction collapsed to 0.5) would otherwise poison every downstream
    probability, ECE, risk-coverage curve and decision in the report.
    """


def check_fit_split(split: str) -> None:
    """Refuse to fit on anything but internal validation (spec §6.7)."""
    if split != INTERNAL_VAL_SPLIT:
        raise ValueError(
            f"calibration may only be fitted on the internal validation split "
            f"{INTERNAL_VAL_SPLIT!r}, got {split!r}; fitting on training, test "
            f"or benchmark rows invalidates every calibration number in the report")


def fit_standardiser(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Per-column centre and scale for a conditioning matrix.

    Returns `(mu, sd, constant_columns)`. A column that is constant on the fit
    rows carries no information, so it is centred to exactly zero and scaled by
    1 rather than by its ~0 standard deviation: at inference a column holding a
    different constant then contributes nothing, instead of a value divided by
    ~0 that saturates the temperature (or the EQI) into nonsense.

    Shared by both fits in this package so that "a degradation family
    validation never exercised" means the same thing in each.
    """
    mu = cond.mean(axis=0, keepdims=True)
    sd = cond.std(axis=0, keepdims=True)
    const = sd[0] < CONST_COLUMN_TOL
    return mu, np.where(const, 1.0, sd), tuple(int(i) for i in np.flatnonzero(const))
