"""Where the evidence sits (spec section 3.8): a per-patch AIGC map.

The head consumes the global average of the backbone's patch tokens. So
applying that same head to each token individually gives a spatial map for
free -- no Grad-CAM, no second forward pass, no extra training. The map and
the score are literally the same computation, one pooled and one not.

**It is a heuristic, and it is labelled as one.** The head was fitted on the
pooled vector; a single token is off-distribution for it, and a bright patch
means "evidence concentrates here", never "this region is 80% likely to be
generated". `PATCH_HEATMAP_CAVEAT` is the sentence the dashboard must show
next to it, and it says both halves -- naming the heuristic without denying
the probability reading still lets a viewer read the colours as calibrated.

**This is a decode site**, the fifth. It canonicalises before the backbone
sees anything, exactly as `features/extract.py`, `eval/grid.py`,
`features/recon.py` and `infer.py` do. Resolution separates the training pool
almost perfectly and transfers *inverted* to the benchmark
(`docs/resolution_shortcut.md`), so a map computed at native resolution would
explain a version of the image that the score never saw -- and it would do so
next to that score, on the same screen.

`to_overlay` serves both maps. The reconstruction branch's per-pixel error map
(`features.recon.error_map`, 256x256 whatever the image) renders through the
same function, so the two heatmaps in the demo cannot end up with different
colour scales or different resampling.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from aigcdet.augment.canonical import canonicalise
from aigcdet.features.backbones import model_inputs

PATCH_HEATMAP_CAVEAT = (
    "Heuristic: the classifier was trained on pooled features, so per-patch "
    "scores show where evidence concentrates -- they are not calibrated "
    "per-region probabilities."
)


@torch.inference_mode()
def patch_scores(backbone, spec, model, img: np.ndarray,
                 device: str = "cuda") -> np.ndarray:
    """A `(g, g)` map of per-patch AIGC logits, in the image's own layout.

    Raw logits, not probabilities: `to_overlay` normalises per image for
    display, and anything stronger than a relative reading is what the caveat
    exists to refuse.
    """
    if getattr(model, "use_recon", False):
        raise ValueError(
            "the patch heatmap is only defined for a model without the recon "
            "branch: that Detector's input is the embedding concatenated with "
            "12 VAE features, which exist per image and not per patch. Show "
            "`features.recon.error_map` instead.")

    try:
        dtype = next(backbone.parameters()).dtype
    except StopIteration:                     # a parameterless stand-in
        dtype = torch.float32
    inputs = model_inputs(spec, [canonicalise(img)], device, dtype)
    h = backbone(**inputs).last_hidden_state          # (1, T, D)
    # Prefix tokens (CLS, registers) carry no position; a cell for them would
    # correspond to nowhere in the picture.
    tokens = h[0, spec.num_prefix_tokens:, :].float()
    logits = model(tokens)["logit"].float().cpu().numpy()

    n = int(logits.shape[0])
    g = int(round(math.sqrt(n)))
    if g * g != n:
        raise ValueError(
            f"{n} patch tokens is not a square grid, so they cannot be laid "
            f"out over the image. Truncating to {g}x{g} would keep raster "
            "order for most of the map and silently misplace the tail of it. "
            f"Check spec.num_prefix_tokens ({spec.num_prefix_tokens}) against "
            f"{spec.name}'s published config.")
    return logits.reshape(g, g).astype(np.float32)


def to_overlay(img: np.ndarray, heat: np.ndarray,
               alpha: float = 0.45) -> np.ndarray:
    """`heat` colour-mapped and blended over `img`, at the image's own size.

    Normalised to its own min/max, so the colours are relative WITHIN one
    image and say nothing across images -- a uniformly suspicious picture and
    a uniformly clean one both come out flat. That is the honest rendering of
    a quantity with no calibrated scale; the number beside it is the
    calibrated one.
    """
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(
            f"to_overlay expects an HxWx3 uint8 RGB image, got shape "
            f"{img.shape!r} dtype {img.dtype!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")

    h = np.asarray(heat, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"to_overlay expects a 2-D map, got shape {h.shape!r}")
    span = float(h.max() - h.min())
    h = np.zeros_like(h) if span < 1e-8 else (h - h.min()) / span
    # INTER_CUBIC either way: the patch map upsamples and the 256x256 error
    # map usually downsamples, and one kernel for both keeps the two heatmaps
    # in the demo visually comparable.
    h = cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    colour = cv2.applyColorMap(
        np.clip(h * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    out = (1.0 - alpha) * img.astype(np.float32) + alpha * colour.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
