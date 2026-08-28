"""AEROBLADE baseline (spec §6.3): training-free, from branch `r` alone.

Latent-diffusion images round-trip through their own autoencoder with low
error, so the AIGC score is simply the negated round-trip distance. No
training, which makes it the cheapest baseline in the set. It is also free of
compute here: the round-trip already happened when
`features.recon.recon_features` cached the `r` vector, so this module reads a
number out of that vector and never loads a VAE.

**The distance defaults to LPIPS, not L1.** Ricker et al.'s AEROBLADE reports
a deep perceptual distance between image and reconstruction, and the paper's
own finding is that it substantially beats pixel-space L1/L2. Both are computed
in the same pass by `recon_features` and sit in the same cached vector, so the
published variant costs one string and zero extra compute -- and shipping the
weaker one unlabelled in a §6.3 comparison table would flatter our own model in
the one place where that is an integrity problem rather than a preference. L1
stays available via `distance="l1"` so both rows can be reported.
"""
from __future__ import annotations

import numpy as np

from aigcdet.features.recon import RECON_FEATURE_NAMES

#: The published headline distance. Changing this changes what the results
#: table's AEROBLADE row means, so it is named rather than inlined.
DEFAULT_DISTANCE = "lpips"

#: The `r` vector's entries that are round-trip DISTANCES, i.e. the ones for
#: which "small means the autoencoder has seen the likes of this image" holds.
#: The other ten entries are descriptive statistics and spectral bands, and
#: negating one of those would not be AEROBLADE.
DISTANCES: tuple[str, ...] = ("lpips", "l1")


def _slot(distance: str) -> int:
    """Resolve a distance name to its index in `RECON_FEATURE_NAMES`.

    Looked up on every call, not frozen into a module constant, so that the
    index tracks the tuple. `RECON_FEATURE_NAMES` is the positional contract
    the `r` bank is stored against; a hard-coded literal would keep pointing at
    the old slot if that tuple were ever reordered, and today `l1` happens to
    be index 0, so a literal `0` looks correct until the day it is not.
    """
    if distance not in DISTANCES:
        raise ValueError(
            f"distance must be one of {DISTANCES}; got {distance!r}. The other "
            f"entries of the r vector are descriptive statistics, not "
            f"round-trip distances, and negating one is not AEROBLADE")
    return RECON_FEATURE_NAMES.index(distance)


def aeroblade_score(recon_vec: np.ndarray, distance: str = DEFAULT_DISTANCE) -> float:
    """Negated round-trip distance: HIGHER means more likely AI-generated.

    The negation is the whole sign convention and it is easy to lose. The
    distance is an ERROR -- small for an image the autoencoder has seen the
    likes of, i.e. an AI-generated one -- while `aigcdet.eval.metrics` expects a
    SCORE that increases with P(AI-generated). Return `+distance` and every AUC
    in the report lands below 0.5, which reads as "the baseline fails" when it
    means "we inverted it".

    Use `aeroblade_scores` for a whole bank: a single vector cannot tell an
    honest zero from a column that is zero because `lpips` was never installed
    when the features were extracted.
    """
    value = float(recon_vec[_slot(distance)])
    if not np.isfinite(value):
        raise ValueError(
            f"the {distance!r} slot of this r vector is {value}; a non-finite "
            f"distance would propagate silently into every metric downstream")
    return -value


def aeroblade_scores(recon_bank: np.ndarray,
                     distance: str = DEFAULT_DISTANCE) -> np.ndarray:
    """Scores for a whole `r` bank, `(n, len(RECON_FEATURE_NAMES))` in.

    Fails loudly when the chosen column is unusable, which is the failure this
    function exists for. `lpips` is an optional extra (`pip install
    aigcdet[recon]`) and is deliberately absent from this project's environment;
    a recon extraction done without it, or one that errored past the LPIPS call,
    leaves that column all-zero. Scoring on it silently returns a constant 0 for
    every row, which is an AUC of exactly 0.5 -- indistinguishable in the
    results table from "AEROBLADE does not work on our data".
    """
    bank = np.asarray(recon_bank, dtype=np.float64)
    n_names = len(RECON_FEATURE_NAMES)
    if bank.ndim != 2 or bank.shape[1] != n_names:
        raise ValueError(
            f"recon_bank must be (n_rows, {n_names}) matching "
            f"RECON_FEATURE_NAMES; got shape {bank.shape}")
    if bank.shape[0] == 0:
        raise ValueError("recon_bank is empty; nothing to score")
    column = bank[:, _slot(distance)]
    if not np.isfinite(column).all():
        raise ValueError(
            f"the {distance!r} column holds "
            f"{int((~np.isfinite(column)).sum())} non-finite values of "
            f"{len(column)}; refusing to score a bank with a poisoned column")
    if np.all(column == 0.0):
        raise ValueError(
            f"the {distance!r} column is all-zero across {len(column)} rows, so "
            f"this bank carries no {distance} signal at all. The usual cause is "
            f"a recon extraction run without the optional `lpips` package "
            f"installed; scoring it would report a constant 0 and an AUC of "
            f"exactly 0.5, which is indistinguishable from a real null result")
    return -column
