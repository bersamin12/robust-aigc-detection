"""Rung A6: degradation-aware test-time augmentation (spec §6.4).

Eight views mixing geometric and degradation transforms, following the top
NTIRE 2026 entries. Logits are averaged, not probabilities: the mean of a
saturated and an unsaturated probability is dominated by the saturated one,
whereas in logit space each view contributes on the same scale, and the head's
temperature calibration is fitted on logits too.

**The two degradation views and the held-out severity bands.** `jpeg_95` and
`blur_0.3` are degradations, and this project reserves JPEG q in [65, 75] and
blur sigma in [0.85, 1.15] as severities the training sampler never draws
(`augment.recipes.HELDOUT_JPEG_Q`, `HELDOUT_BLUR_SIGMA`), so that the eval
grid's `jpeg_q70` and `blur_s1.0` rows measure unseen severities. Both TTA
views sit OUTSIDE those bands (95 is above 75; 0.3 is below 0.85), so nothing
here narrows the held-out claim. It would not do so even if they were inside:
TTA is applied at inference, to whatever image is being scored, and never
during training, so it cannot make a held-out severity a seen one. What it
WOULD do is re-encode every scored image at the very severity the report
presents as never-seen before measuring it there, which is a footnote no reader
should have to reconstruct -- hence `VIEW_PARAMS` below, and the test that
fails if a future view moves into a band.

Cost note: TTA multiplies inference by len(views). It is therefore evaluated
from images rather than from the cached eval bank, and only on the ablation
tier's 5k+5k subsample (spec §4.4a). `scripts/run_ablation.py --tta` records
that multiplier and that tier in `selection.json`; there is no silent cap.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from aigcdet.augment import ops

TTA_VIEWS: tuple[str, ...] = (
    "identity", "hflip", "scale_0.75", "scale_1.25",
    "jpeg_95", "blur_0.3", "hflip_scale_0.75", "hflip_jpeg_95",
)

#: The degradation parameter each view applies, as (family, value), so the
#: held-out-band check is a fact about the views rather than a comment.
VIEW_PARAMS: dict[str, tuple[str, float]] = {
    "jpeg_95": ("jpeg", 95),
    "blur_0.3": ("blur", 0.3),
    "hflip_jpeg_95": ("jpeg", 95),
}


def _hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1])


def _scale(img: np.ndarray, factor: float) -> np.ndarray:
    """Rescale then restore the original size, so views stay shape-compatible.

    `ops.resize_roundtrip` already is that operation (INTER_AREA down,
    INTER_CUBIC back), and reusing it keeps the interpolation choice in one
    place; it is equally correct for factor > 1, where the round trip goes up
    and then back down.
    """
    return ops.resize_roundtrip(img, factor)


_VIEW_FUNCS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda i: i.copy(),
    "hflip": _hflip,
    "scale_0.75": lambda i: _scale(i, 0.75),
    "scale_1.25": lambda i: _scale(i, 1.25),
    "jpeg_95": lambda i: ops.jpeg(i, quality=95),
    "blur_0.3": lambda i: ops.blur(i, sigma=0.3),
    "hflip_scale_0.75": lambda i: _scale(_hflip(i), 0.75),
    "hflip_jpeg_95": lambda i: ops.jpeg(_hflip(i), quality=95),
}


def apply_tta_view(img: np.ndarray, view: str) -> np.ndarray:
    if view not in _VIEW_FUNCS:
        raise KeyError(f"unknown TTA view {view!r}; expected one of {TTA_VIEWS}")
    return _VIEW_FUNCS[view](img).astype(np.uint8)


@torch.no_grad()
def tta_logit(backbone, spec, model, img: np.ndarray, device: str = "cuda",
              views: Sequence[str] = TTA_VIEWS, recon_fn=None,
              embed_fn=None) -> float:
    """Mean LOGIT across `views` for one image.

    `backbone` is an already-loaded tower: A6 is inference-only and is called
    once per image, so loading weights here would download and start a GPU
    process per call. `embed_fn` is injectable so the averaging can be tested
    without a backbone at all.

    `recon_fn`, when the head uses the recon branch (A4+), is applied to each
    DEGRADED view rather than once to the original: the recon feature describes
    the image the head is being asked about.
    """
    embed = embed_fn
    if embed is None:
        # Resolved through the module, not bound at import time, so a test that
        # monkeypatches `features.backbones.embed` is what actually runs.
        from aigcdet.features import backbones as _backbones
        embed = _backbones.embed

    logits = []
    for v in views:
        view = apply_tta_view(img, v)
        f = embed(backbone, spec, [view], device=device, batch_size=1)
        r = None
        if getattr(model, "use_recon", False):
            if recon_fn is None:
                raise ValueError(
                    "this head uses the recon branch, so TTA needs recon "
                    "features for each view; pass recon_fn")
            r = torch.from_numpy(
                np.asarray(recon_fn(view), dtype=np.float32)[None]).to(device)
        out = model(torch.from_numpy(np.asarray(f, dtype=np.float32)).to(device), r)
        logits.append(float(out["logit"].reshape(-1)[0]))
    if not logits:
        raise ValueError("tta_logit was given no views to average")
    return float(np.mean(logits))
