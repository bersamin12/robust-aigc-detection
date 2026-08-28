"""AEROBLADE baseline (spec §6.3): training-free, from branch `r` alone.

Latent-diffusion images round-trip through their own autoencoder with low
error, so the AIGC score is simply the negated L1 reconstruction error.
No training, which makes it the cheapest baseline in the set.

It is also free of compute here: the round-trip already happened when
`features.recon.recon_features` cached the `r` vector, so this module reads a
number out of that vector and never loads a VAE.
"""
from __future__ import annotations

import numpy as np

from aigcdet.features.recon import RECON_FEATURE_NAMES

#: Resolved by NAME, not written as a literal. `RECON_FEATURE_NAMES` is the
#: contract the bank stores positionally; if that tuple is ever reordered this
#: index follows it, where a hard-coded 0 would quietly start reading `lpips`
#: or `err_std` and invert the baseline.
_L1_INDEX = RECON_FEATURE_NAMES.index("l1")


def aeroblade_score(recon_vec: np.ndarray) -> float:
    """Negated L1 round-trip error: HIGHER means more likely AI-generated.

    The negation is the whole sign convention and it is easy to lose. `l1` is
    an ERROR -- small for an image the autoencoder has seen the likes of, i.e.
    an AI-generated one -- while `aigcdet.eval.metrics` expects a SCORE that
    increases with P(AI-generated). Return `+l1` and every AUC in the report
    lands below 0.5, which reads as "the baseline fails" when it means "we
    inverted it".
    """
    return float(-recon_vec[_L1_INDEX])
