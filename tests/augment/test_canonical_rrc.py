"""The rrc policy (paper_prereg.md amendment 5, Test 7): random resized crop
squished to the nominal side.

What must hold for the test to mean what it registers:
- the window's area fraction and aspect stay inside the registered ranges;
- the draw is a pure function of (seed, row_id, view), so training is
  reproducible and the two classes are treated alike;
- with no rng it is exactly squish, and squish itself did not change when its
  resize was moved into the shared `_squish_to` helper;
- the training view builder draws a window per view under rrc, as it does
  under crop, instead of hoisting one canonicalisation out of the view loop.
"""
from __future__ import annotations

import cv2
import numpy as np

from aigcdet.augment import canonical as canon
from aigcdet.augment.canonical import (
    MODE_RRC, MODE_SQUISH, RRC_AREA_RANGE, RRC_ASPECT_RANGE, CanonPolicy,
    _random_resized_window, canonical_rng, canonicalise,
)

RRC_224 = CanonPolicy(mode=MODE_RRC, nominal_side=224)
SQUISH_224 = CanonPolicy(mode=MODE_SQUISH, nominal_side=224)


def _image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _squish_before_refactor(img: np.ndarray, n: int) -> np.ndarray:
    """The squish branch of `canonicalise` as it was written before rrc."""
    out = img
    if min(out.shape[:2]) > n:
        out = canon._resize_short_side(out, n, canon._DOWNSCALE_INTERP)
    if out.shape[:2] != (n, n):
        out = cv2.resize(out, (n, n), interpolation=canon._DOWNSCALE_INTERP)
    return out.copy() if out is img else out


def test_rrc_windows_stay_inside_the_registered_area_and_aspect_ranges():
    img = _image(600, 800)
    # Rounding the window to whole pixels moves area and aspect slightly, so
    # allow 1% on each bound.
    for view in range(500):
        window = _random_resized_window(img, canonical_rng(20260827, 7, view))
        wh, ww = window.shape[:2]
        area_fraction = wh * ww / (600 * 800)
        aspect = ww / wh
        assert RRC_AREA_RANGE[0] * 0.99 <= area_fraction <= RRC_AREA_RANGE[1] * 1.01
        assert RRC_ASPECT_RANGE[0] * 0.99 <= aspect <= RRC_ASPECT_RANGE[1] * 1.01


def test_rrc_is_reproducible_from_its_key_and_differs_across_views():
    img = _image(500, 700)
    first = canonicalise(img, policy=RRC_224, rng=canonical_rng(1, 2, 3))
    again = canonicalise(img, policy=RRC_224, rng=canonical_rng(1, 2, 3))
    other_view = canonicalise(img, policy=RRC_224, rng=canonical_rng(1, 2, 4))
    assert first.shape == (224, 224, 3) and first.dtype == np.uint8
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_view)


def test_rrc_without_rng_is_squish_and_squish_is_unchanged():
    for h, w in ((600, 800), (224, 224), (150, 400), (1200, 300)):
        img = _image(h, w)
        squished = canonicalise(img, policy=SQUISH_224)
        assert np.array_equal(squished, _squish_before_refactor(img, 224))
        assert np.array_equal(canonicalise(img, policy=RRC_224), squished)
        assert squished.base is None or squished.base is not img


def test_rrc_falls_back_to_a_clamped_whole_image_for_extreme_aspects():
    # A 50 x 1000 strip: no window of area >= 20% fits at aspect <= 4/3,
    # so every attempt fails and the fallback trims the long axis to 4/3.
    img = _image(50, 1000)
    window = _random_resized_window(img, canonical_rng(0, 0, 0))
    assert window.shape[:2] == (50, int(round(50 * RRC_ASPECT_RANGE[1])))


def test_the_training_view_builder_draws_a_window_per_view_under_rrc():
    from aigcdet.augment.recipes import Recipe
    from aigcdet.features.extract import build_view

    img = _image(480, 640)
    identity = Recipe(ops=())
    # A caller-supplied base (one deterministic canonicalisation) must be
    # ignored under rrc, as it is under crop, or all views share one window.
    hoisted_base = canonicalise(img, policy=RRC_224)
    view_1, _ = build_view(img, 1, 5, 20260827, policy=RRC_224, recipe=identity,
                           base=hoisted_base)
    view_2, _ = build_view(img, 2, 5, 20260827, policy=RRC_224, recipe=identity,
                           base=hoisted_base)
    assert view_1.shape == (224, 224, 3)
    assert not np.array_equal(view_1, view_2)
