"""The frequency block: NPR's periodic-upsampling descriptor, per cached view.

`baselines.npr.npr_feature` is a 4-d descriptor (two magnitudes plus
`contrast_h`/`contrast_v`) of the periodic cell structure a transposed
convolution leaves behind. This attaches it to a bank the same way the
reconstruction branch is attached, sharing `features.replay` so the two cannot
disagree about which pixels a view is.

CROP ONLY, and that is enforced rather than documented. Band standardisation
resamples every image to a nominal side, which destroys the generator's native
pixel grid and substitutes the RESAMPLER's own -- the descriptor would then be
measuring the interpolation kernel, identically for both classes, except where
class-correlated source resolution makes it a confound. The content-blind probe
measures exactly that: band reaches 0.6105 pooled and 0.9976 on SID_Set from
low-level statistics alone, where crop is near chance (0.5081 / 0.6316). Under
band this block would destroy the real signal and supply a fake one.
"""
from __future__ import annotations

import numpy as np

#: The stride the cell structure is measured at. 2 is one transposed-conv
#: doubling, which is what every generator in the corpus upsamples by.
DEFAULT_STRIDE = 2


def _require_crop(bank, override: bool) -> None:
    from aigcdet.augment.canonical import MODE_CROP

    mode = (bank.config.get("canon_policy") or {}).get("mode", "band")
    if mode == MODE_CROP or override:
        return
    raise ValueError(
        f"the bank at {bank.path} is canonicalised with mode={mode!r}, and the "
        "frequency block is only measurable under crop: band resampling "
        "replaces the generator's native pixel grid with the resampler's, so "
        "this descriptor would measure the interpolation kernel and, where "
        "source resolution is class-correlated, leak instead. Pass "
        "allow_band=True only to reproduce that negative result deliberately.")


def attach_freq_to_bank(bank, manifest_df, seed: int = 20260827, *,
                        stride: int = DEFAULT_STRIDE, start: int = 0,
                        stop: int | None = None, attach: bool = True,
                        allow_band: bool = False) -> np.ndarray:
    """`(rows, n_views, FREQ_DIM)` of NPR descriptors over the cached views.

    Pure numpy on the already-decoded view, so it needs no GPU and no model --
    the whole cost is the decode and the recipe replay `features.replay` does
    anyway.
    """
    from aigcdet.baselines.npr import npr_feature
    from aigcdet.features.bank import FREQ_DIM
    from aigcdet.features.replay import replay_views

    _require_crop(bank, allow_band)
    out = replay_views(
        bank, manifest_df, lambda view: npr_feature(view, stride=stride),
        FREQ_DIM, seed=seed, start=start, stop=stop,
        allow_partial=not attach, desc="freq")
    if attach:
        bank.attach_block(out, "freq")
    return out


def merge_freq_shards(bank, parts) -> np.ndarray:
    """Assemble contiguous frequency shards and attach them."""
    from aigcdet.features.bank import FREQ_DIM
    from aigcdet.features.replay import merge_blocks

    return merge_blocks(bank, parts, "freq", FREQ_DIM)
