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
should have to reconstruct -- hence the band check over `VIEW_PARAMS`.

For that check to be worth running, `VIEW_PARAMS` must be a fact about the
views rather than a list kept in step by hand. It is therefore DERIVED, along
with `TTA_VIEWS` and the view functions themselves, from one declaration,
`_VIEW_SPECS`: a view is a sequence of `(op, value)` steps in application
order, and the degradation steps of every view are what `VIEW_PARAMS` reports.
A new degradation view cannot exist without appearing there, because there is
no other way to write one -- the previous hand-maintained dict let a `jpeg`
view at q=70, the dead centre of a held-out band, be added and pass the whole
suite including the test named for exactly that.

**A6 and the fitted temperature (not live yet -- read this before wiring it).**
`tta_logit` returns the MEAN of eight per-view logits. A mean of eight
correlated-but-not-identical logits has materially smaller spread than the
single-view logits that `calibrate.temperature.ConditionalTemperature.fit` was
fitted on, so a `T` fitted on single-view logits is being applied to a
differently-scaled quantity: A6's probabilities come out systematically
under-confident (the same `T` now divides a narrower distribution). The
inference clamp in `calibrate/temperature.py` will NOT catch it, because that
clamp is keyed on the CONDITION and the condition has not changed -- only the
logit's spread moved. Nothing in the repo produces A6 scores today
(`scripts/run_ablation.py --tta` records the cost multiplier and the tier and
emits no `a6` row), so this is not a live defect; it is a trap for whoever
wires A6 up, which is the rung most likely to be added last and least likely
to be recalibrated. Whoever does that must REFIT the temperature on TTA-averaged
logits over `val_internal` and carry it as a separate `T`, not reuse the
single-view one.

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

#: The primitive each `(op, value)` step of a view applies, split by whether it
#: is a DEGRADATION -- something with a severity that could land inside a
#: held-out band -- or a geometric transform, which has no severity to check.
#: A step's op must be in one of the two; there is no third, unclassified kind.
_DEGRADATION_OPS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "jpeg": lambda img, value: ops.jpeg(img, quality=value),
    "blur": lambda img, value: ops.blur(img, sigma=value),
}

_GEOMETRIC_OPS: dict[str, Callable[[np.ndarray, float | None], np.ndarray]] = {
    "hflip": lambda img, _: np.ascontiguousarray(img[:, ::-1]),
    # `ops.resize_roundtrip` already is "rescale then restore the original
    # size" (INTER_AREA down, INTER_CUBIC back), and reusing it keeps the
    # interpolation choice in one place; it is equally correct for a factor
    # above 1, where the round trip goes up and then back down. Views stay
    # shape-compatible either way.
    "scale": lambda img, value: ops.resize_roundtrip(img, value),
}

#: The single declaration every other name in this module is derived from:
#: view name -> the (op, value) steps it applies, in application order.
#: `value` is None for an op that takes none.
_VIEW_SPECS: dict[str, tuple[tuple[str, float | None], ...]] = {
    "identity": (),
    "hflip": (("hflip", None),),
    "scale_0.75": (("scale", 0.75),),
    "scale_1.25": (("scale", 1.25),),
    "jpeg_95": (("jpeg", 95),),
    "blur_0.3": (("blur", 0.3),),
    "hflip_scale_0.75": (("hflip", None), ("scale", 0.75)),
    "hflip_jpeg_95": (("hflip", None), ("jpeg", 95)),
}


def _compose(steps: Sequence[tuple[str, float | None]]
             ) -> Callable[[np.ndarray], np.ndarray]:
    """Build one view's function from its steps, checking every op is known."""
    for op, value in steps:
        if op in _DEGRADATION_OPS:
            if value is None:
                raise ValueError(f"degradation op {op!r} needs a severity value")
        elif op not in _GEOMETRIC_OPS:
            raise ValueError(
                f"unknown TTA op {op!r}; it must be one of "
                f"{sorted(set(_DEGRADATION_OPS) | set(_GEOMETRIC_OPS))}. An op "
                "outside those two tables has no declared kind, so the "
                "held-out-severity check could not say whether it is a "
                "degradation.")

    def run(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        for op, value in steps:
            fn = _DEGRADATION_OPS.get(op) or _GEOMETRIC_OPS[op]
            out = fn(out, value)
        return out

    return run


#: The view names, in order. Derived, so a view cannot be implemented without
#: being declared or declared without being implemented.
TTA_VIEWS: tuple[str, ...] = tuple(_VIEW_SPECS)

_VIEW_FUNCS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    name: _compose(steps) for name, steps in _VIEW_SPECS.items()
}

def degradation_params(specs: dict[str, tuple[tuple[str, float | None], ...]]
                       ) -> dict[str, tuple[tuple[str, float], ...]]:
    """The degradation steps of every view in `specs` that has any.

    A function rather than a comprehension so the held-out-band check can be
    run against a HYPOTHETICAL set of views -- which is how a test can prove
    the check fires on the view somebody is about to add.
    """
    return {name: tuple((op, value) for op, value in steps
                        if op in _DEGRADATION_OPS)
            for name, steps in specs.items()
            if any(op in _DEGRADATION_OPS for op, _ in steps)}


#: The degradation parameters each view applies, as (family, value) pairs, so
#: the held-out-band check is a fact about the views rather than a comment.
#: DERIVED from `_VIEW_SPECS`: every degradation step of every view is here,
#: because there is nowhere else a view's degradations can be written.
VIEW_PARAMS: dict[str, tuple[tuple[str, float], ...]] = degradation_params(_VIEW_SPECS)


def check_views_avoid_heldout_bands(
        params: dict[str, tuple[tuple[str, float], ...]] | None = None) -> None:
    """Refuse a TTA view whose severity sits inside a held-out band.

    Called at import, on `VIEW_PARAMS`, so the claim in the module docstring is
    an invariant of the module rather than something a test happens to check.
    A view at, say, jpeg q=70 would re-encode every scored image at the exact
    severity the report presents as never-seen before measuring it there.
    """
    from aigcdet.augment.recipes import HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q
    bands = {"jpeg": HELDOUT_JPEG_Q, "blur": HELDOUT_BLUR_SIGMA}
    for name, steps in (VIEW_PARAMS if params is None else params).items():
        for family, value in steps:
            lo, hi = bands[family]
            if lo <= value <= hi:
                raise ValueError(
                    f"TTA view {name!r} applies {family}={value}, inside the "
                    f"held-out severity band {(lo, hi)} that the training "
                    "sampler never draws. TTA runs at inference so it cannot "
                    "make a held-out severity a trained-on one, but it would "
                    "re-encode every scored image at the severity the report "
                    "presents as unseen, before measuring it there.")


check_views_avoid_heldout_bands()


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

    The returned value is a mean of `len(views)` logits and therefore has a
    NARROWER distribution than the single-view logits any fitted temperature
    was calibrated on. It must not be pushed through a `T` fitted on
    single-view logits -- see the module docstring's A6/temperature note.
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
