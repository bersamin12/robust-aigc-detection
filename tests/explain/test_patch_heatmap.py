"""The explainability maps.

Two things are worth pinning here, and they are not the ones a heatmap test
usually reaches for. First, `patch_scores` is a DECODE SITE: it must
canonicalise like the other four, or the picture explains a version of the
image the score never saw. Second, an overlay is a claim about WHERE, so the
tests check position, not only shape and range -- a transposed or flipped map
passes every shape assertion while pointing at the wrong half of the image.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aigcdet.augment.canonical import canonicalise
from aigcdet.explain.patch_heatmap import (
    PATCH_HEATMAP_CAVEAT, patch_scores, to_overlay)
from aigcdet.features.backbones import BackboneSpec
from aigcdet.models.heads import Detector

_DIM = 16
_PREFIX = 1
_GRID = 4


class _Tower(torch.nn.Module):
    """A stand-in vision tower that returns `_PREFIX + _GRID**2` tokens and
    records the pixel tensor it was handed."""

    def __init__(self, n_tokens=_PREFIX + _GRID * _GRID, dim=_DIM, tokens=None):
        super().__init__()
        self.n_tokens, self.dim, self.tokens = n_tokens, dim, tokens
        self.seen = None       # the pixel tensor it was handed
        self.emitted = None    # the hidden state it returned
        self.proj = torch.nn.Linear(1, 1)          # so .parameters() has a dtype

    def forward(self, pixel_values=None, **kw):
        self.seen = pixel_values
        b = pixel_values.shape[0]
        if self.tokens is not None:
            h = self.tokens[None].to(pixel_values.device)
        else:
            g = torch.Generator().manual_seed(0)
            h = torch.rand(b, self.n_tokens, self.dim, generator=g).to(pixel_values.device)
        self.emitted = h
        return type("Out", (), {"last_hidden_state": h})()


class _FirstElement:
    """A stand-in head whose logit for a token IS that token's first element,
    so the map's layout can be read straight off the output. `Detector` is
    randomly initialised and non-linear -- with it, every orientation of the
    grid looks equally plausible."""

    use_recon = False

    def __call__(self, f):
        return {"logit": f[:, 0]}


def _spec(image_size=64):
    return BackboneSpec("fake", "none", image_size, _DIM, _PREFIX, 0)


def _img(seed=0, shape=(120, 200, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


# --------------------------------------------------------------- the caveat

def test_caveat_names_the_heuristic_and_says_what_it_is_not():
    """The dashboard displays this string verbatim. It has to say both halves:
    that the map is a heuristic, and that it is not a per-region probability.
    Half the sentence is worse than none -- 'heuristic' alone still lets a
    viewer read the colours as calibrated."""
    lower = PATCH_HEATMAP_CAVEAT.lower()
    assert "heuristic" in lower
    assert "pooled" in lower
    assert "probabilit" in lower


# ------------------------------------------------------------- patch_scores

def test_patch_scores_canonicalises_before_it_embeds():
    """The fifth decode site. Resolution leaks the label and transfers
    inverted (docs/resolution_shortcut.md); a map computed at native
    resolution explains an image the head was never shown."""
    tower, spec = _Tower(), _spec()
    model = Detector(dim_feat=_DIM)

    patch_scores(tower, spec, model, _img(shape=(120, 200, 3)), device="cpu")

    # `model_inputs` squishes to spec.image_size, so the tensor size cannot
    # reveal what happened before it. Compare pixels instead.
    got = tower.seen[0].permute(1, 2, 0).numpy()
    from aigcdet.features.backbones import model_inputs
    want = model_inputs(
        spec, [canonicalise(_img(shape=(120, 200, 3)))], "cpu",
        torch.float32)["pixel_values"][0].permute(1, 2, 0).numpy()
    assert np.allclose(got, want, atol=1e-5)
    # ... and that is NOT what the raw image produces.
    raw = model_inputs(spec, [_img(shape=(120, 200, 3))], "cpu",
                       torch.float32)["pixel_values"][0].permute(1, 2, 0).numpy()
    assert not np.allclose(got, raw, atol=1e-3)


def test_the_map_is_square_and_matches_the_token_grid():
    tower, spec = _Tower(), _spec()
    heat = patch_scores(tower, spec, Detector(dim_feat=_DIM), _img(), device="cpu")
    assert heat.shape == (_GRID, _GRID)
    assert heat.dtype == np.float32


def test_prefix_tokens_are_dropped_before_the_head_sees_them():
    """CLS and register tokens carry no position. Scoring them would put a
    cell in the grid that corresponds to nowhere in the image -- and with
    `num_prefix_tokens` counted in, the token count is no longer square, so
    the failure is a wrong-shaped map rather than a wrong-looking one."""
    tower = _Tower(n_tokens=_PREFIX + _GRID * _GRID)
    heat = patch_scores(tower, _spec(), Detector(dim_feat=_DIM), _img(), device="cpu")
    assert heat.size == _GRID * _GRID


def test_token_k_lands_at_row_k_over_g_column_k_mod_g():
    """Raster order, the order every ViT emits patches in. A transpose or a
    vertical flip is invisible to every shape, range and dtype assertion --
    the map still looks like a heatmap, it just points at the wrong part of
    the picture, which is the only thing a heatmap is for."""
    n = _GRID * _GRID
    tok = torch.zeros(_PREFIX + n, _DIM)
    tok[_PREFIX:, 0] = torch.arange(n, dtype=torch.float32)

    heat = patch_scores(_Tower(tokens=tok), _spec(), _FirstElement(),
                        _img(), device="cpu")

    assert np.array_equal(heat, np.arange(n, dtype=np.float32).reshape(_GRID, _GRID))


def test_a_token_count_that_is_not_a_square_is_refused():
    """Truncating to the nearest square keeps raster order for most of the map
    and silently rotates the tail of it. A picture that is wrong in its last
    row is worse than no picture."""
    tower = _Tower(n_tokens=_PREFIX + 18)
    with pytest.raises(ValueError, match="square"):
        patch_scores(tower, _spec(), Detector(dim_feat=_DIM), _img(), device="cpu")


def test_the_head_scores_the_tokens_the_pooled_path_would_have_averaged():
    """The whole claim of this module is that the map costs nothing: the head
    already runs on the mean of these tokens, so running it per token is the
    same computation left unpooled. That only holds if the tokens reaching the
    head here are the same ones `embed` averages -- final layer, prefix
    stripped, nothing re-normalised on the way. A map built from a different
    tensor would still look like a plausible heatmap."""
    tower, spec = _Tower(), _spec()
    model = Detector(dim_feat=_DIM).eval()
    seen = {}
    inner = model.forward
    model.forward = lambda f, r=None: (seen.setdefault("f", f), inner(f, r))[1]

    patch_scores(tower, spec, model, _img(), device="cpu")

    expected = tower.emitted[0, _PREFIX:, :]
    assert torch.allclose(seen["f"].float(), expected.float(), atol=1e-6)


def test_a_recon_model_is_refused_by_name():
    """A recon Detector's input is the embedding concatenated with 12 VAE
    features, which exist per IMAGE and not per patch. There is no honest way
    to feed it a token, and `features.recon.error_map` is the map to show
    instead."""
    model = Detector(dim_feat=_DIM, use_recon=True)
    # Matched on `error_map`, not on "recon": a recon Detector reached with no
    # `r` raises "this Detector expects recon features `r`" all by itself, so
    # matching the looser word passes whether or not this guard exists at all.
    with pytest.raises(ValueError, match="error_map"):
        patch_scores(_Tower(), _spec(), model, _img(), device="cpu")


# --------------------------------------------------------------- to_overlay

def test_overlay_matches_the_source_image_shape():
    out = to_overlay(_img(shape=(120, 200, 3)),
                     np.random.default_rng(1).random((8, 8)).astype(np.float32))
    assert out.shape == (120, 200, 3) and out.dtype == np.uint8


def test_overlay_is_bounded_and_actually_blends():
    img = np.zeros((64, 64, 3), np.uint8)
    out = to_overlay(img, np.ones((4, 4), np.float32), alpha=1.0)
    assert out.max() <= 255 and out.min() >= 0
    assert not np.array_equal(out, img)


def test_full_alpha_replaces_the_image_rather_than_adding_to_it():
    """A blend is a weighted average, not a sum. Adding instead of averaging
    is invisible on the black test images every other case here uses -- it
    only shows on a bright one, where the sum saturates to flat white and the
    heatmap disappears at exactly the moment it is turned up to full."""
    heat = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = to_overlay(np.full((32, 32, 3), 255, np.uint8), heat, alpha=1.0)
    assert out.min() < 200


def test_uniform_heat_produces_a_uniform_overlay():
    out = to_overlay(np.full((32, 32, 3), 100, np.uint8),
                     np.full((4, 4), 0.5, np.float32))
    assert out.std(axis=(0, 1)).max() < 5.0


def test_a_flat_map_does_not_divide_by_its_own_zero_range():
    """`patch_scores` returns raw logits, and a genuinely flat map is a
    legitimate input. Without the guard this is 0/0: numpy warns, the NaNs
    cast to whatever uint8 the platform picks, and the overlay looks fine --
    a wrong picture produced quietly is the failure mode this whole module is
    written against."""
    with np.errstate(all="raise"):
        out = to_overlay(np.full((16, 16, 3), 100, np.uint8),
                         np.full((4, 4), 7.0, np.float32))
    assert np.isfinite(out.astype(np.float32)).all()


def test_the_map_is_normalised_to_its_own_range_not_assumed_to_be_zero_to_one():
    """Logits are not in [0, 1] and nothing constrains their scale. Feeding
    them to the colour map unnormalised clips everything above 1.0 to the same
    colour, which turns the busiest maps into flat ones."""
    heat = np.linspace(100.0, 200.0, 16, dtype=np.float32).reshape(4, 4)
    out = to_overlay(np.zeros((64, 64, 3), np.uint8), heat, alpha=1.0)
    assert out.max() > 200 and out.min() < 60


def test_the_hot_corner_lands_in_the_hot_corner():
    """Shape and range assertions all pass on a transposed or flipped map.
    Position is the only thing an overlay actually asserts, so it is the only
    thing worth testing. Top-RIGHT, deliberately: a corner off the diagonal is
    the one a transpose moves."""
    heat = np.zeros((4, 4), np.float32)
    heat[0, -1] = 1.0
    out = to_overlay(np.zeros((64, 64, 3), np.uint8), heat, alpha=1.0)

    q = {"tl": out[:16, :16], "tr": out[:16, -16:],
         "bl": out[-16:, :16], "br": out[-16:, -16:]}
    hottest = max(q, key=lambda k: q[k].mean())
    assert hottest == "tr", {k: round(float(v.mean()), 1) for k, v in q.items()}


def test_an_error_map_at_its_own_resolution_overlays_onto_any_image():
    """`features.recon.error_map` returns 256x256 regardless of the image.
    One overlay function serves both maps, so it must not assume the map is
    smaller than the picture."""
    out = to_overlay(_img(shape=(80, 300, 3)),
                     np.random.default_rng(2).random((256, 256)).astype(np.float32))
    assert out.shape == (80, 300, 3)


def test_alpha_zero_returns_the_image_untouched():
    img = _img(shape=(40, 40, 3))
    assert np.array_equal(to_overlay(img, np.ones((4, 4), np.float32), alpha=0.0), img)


@pytest.mark.gpu
def test_patch_scores_grid_matches_the_real_backbone_geometry():
    from aigcdet.features.backbones import load_backbone
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    bb, spec = load_backbone("clipl", device="cuda")
    model = Detector(dim_feat=spec.dim).to("cuda")
    heat = patch_scores(bb, spec, model, _img(shape=(512, 512, 3)), device="cuda")
    assert heat.shape == (spec.image_size // 14, spec.image_size // 14)


def test_patch_scores_refuses_a_spatially_pooled_backbone():
    """Per-patch scoring is undefined for POOL_SPATIAL_MS, not merely
    unimplemented: the head's input is the mean and std over ALL spatial
    positions, so no single position carries a feature of that width. Without
    this guard the conv tower's (1, C, H, W) last_hidden_state is sliced as if
    it were (1, T, D) and the failure surfaces as a shape error inside the
    head, several frames from the cause."""
    import numpy as np
    import pytest

    from aigcdet.explain.patch_heatmap import patch_scores
    from aigcdet.features.backbones import BACKBONES

    spec = BACKBONES["convnextt"]
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="only defined for a token-pooled"):
        patch_scores(backbone=None, spec=spec, model=None, img=img, device="cpu")
