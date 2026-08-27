"""The evaluation grid: the brief's exact conditions plus five named chains.

Kept separate from `recipes.sample_training_recipe` so evaluation conditions
can never be silently drawn from the training distribution (spec §5).
"""
from __future__ import annotations

from aigcdet.augment.recipes import (
    HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q, Op, Recipe,
)

CORE_CONDITIONS: dict[str, Recipe] = {
    "clean": Recipe(()),
    **{f"jpeg_q{q}": Recipe((Op("jpeg", {"quality": q}),)) for q in (90, 70, 50, 30)},
    **{f"blur_s{s}": Recipe((Op("blur", {"sigma": s}),)) for s in (0.5, 1.0, 2.0)},
    **{f"resize_{sc}": Recipe((Op("resize", {"scale": sc}),)) for sc in (0.5, 0.25)},
    **{f"noise_s{s}": Recipe((Op("noise", {"sigma": s}),)) for s in (0.02, 0.05, 0.10)},
    "jitter_20": Recipe((Op("jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}),)),
    "crop_80": Recipe((Op("crop", {"frac": 0.8}),)),
}

COMPOSITE_SCENARIOS: dict[str, Recipe] = {
    "social_repost": Recipe((Op("resize", {"scale": 0.5}), Op("jpeg", {"quality": 70}))),
    "messaging_app": Recipe((Op("resize", {"scale": 0.25}), Op("jpeg", {"quality": 30}))),
    "screenshot": Recipe((Op("crop", {"frac": 0.8}), Op("resize", {"scale": 0.5}),
                          Op("jpeg", {"quality": 50}))),
    "filtered_upload": Recipe((Op("jitter", {"brightness": 0.2, "contrast": 0.2,
                                             "saturation": 0.2}),
                               Op("jpeg", {"quality": 70}))),
    "low_light_share": Recipe((Op("noise", {"sigma": 0.05}), Op("jpeg", {"quality": 50}))),
}

EVAL_GRID: dict[str, Recipe] = {**CORE_CONDITIONS, **COMPOSITE_SCENARIOS}


def _touches_heldout_band(recipe: Recipe) -> bool:
    for op in recipe.ops:
        if op.name == "jpeg" and HELDOUT_JPEG_Q[0] <= op.params["quality"] <= HELDOUT_JPEG_Q[1]:
            return True
        if op.name == "blur" and HELDOUT_BLUR_SIGMA[0] <= op.params["sigma"] <= HELDOUT_BLUR_SIGMA[1]:
            return True
    return False


#: Conditions the training sampler can never have seen at this severity.
#: The robustness table marks these rows so the distinction is visible.
HELDOUT_SEVERITY_CONDITIONS = frozenset(
    name for name, r in EVAL_GRID.items() if _touches_heldout_band(r)
)
