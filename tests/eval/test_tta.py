"""Rung A6: degradation-aware test-time augmentation.

A6 is inference-only, so `tta_logit` takes a real backbone in production. No
test here loads one: `embed` is either injected through `embed_fn` or
monkeypatched on `aigcdet.features.backbones`, so no weights are downloaded and
no GPU process starts.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aigcdet.eval.tta import TTA_VIEWS, apply_tta_view, tta_logit


def _img():
    return np.random.default_rng(0).integers(0, 256, (128, 160, 3), dtype=np.uint8)


class _Spec:
    dim = 4
    image_size = 64


def _stub_embed(model, spec, imgs, device="cpu", batch_size=16):
    return np.zeros((len(imgs), 4), np.float32)


# --- the views -------------------------------------------------------------

def test_identity_view_is_the_identity():
    img = _img()
    assert np.array_equal(apply_tta_view(img, "identity"), img)


def test_every_declared_view_applies_and_preserves_shape():
    img = _img()
    for v in TTA_VIEWS:
        out = apply_tta_view(img, v)
        assert out.shape == img.shape, v
        assert out.dtype == np.uint8, v


def test_hflip_is_its_own_inverse():
    img = _img()
    assert np.array_equal(apply_tta_view(apply_tta_view(img, "hflip"), "hflip"), img)


def test_views_are_distinct_from_the_original_except_identity():
    img = _img()
    for v in TTA_VIEWS:
        if v == "identity":
            continue
        assert not np.array_equal(apply_tta_view(img, v), img), v


def test_the_views_are_distinct_from_each_other():
    """Eight views that collapse to fewer distinct images would multiply the
    inference cost by eight while averaging duplicates."""
    img = _img()
    seen = {apply_tta_view(img, v).tobytes() for v in TTA_VIEWS}
    assert len(seen) == len(TTA_VIEWS)


def test_the_composite_views_really_compose_both_halves():
    """`hflip_scale_0.75` must be the flip AND the rescale, not either alone."""
    img = _img()
    both = apply_tta_view(img, "hflip_scale_0.75")
    assert not np.array_equal(both, apply_tta_view(img, "hflip"))
    assert not np.array_equal(both, apply_tta_view(img, "scale_0.75"))
    np.testing.assert_array_equal(
        both, apply_tta_view(apply_tta_view(img, "hflip"), "scale_0.75"))

    both = apply_tta_view(img, "hflip_jpeg_95")
    assert not np.array_equal(both, apply_tta_view(img, "hflip"))
    assert not np.array_equal(both, apply_tta_view(img, "jpeg_95"))
    np.testing.assert_array_equal(
        both, apply_tta_view(apply_tta_view(img, "hflip"), "jpeg_95"))


def test_the_declared_views_are_exactly_the_implemented_ones():
    from aigcdet.eval.tta import _VIEW_FUNCS, _VIEW_SPECS
    assert tuple(_VIEW_FUNCS) == TTA_VIEWS
    # Both must come from the one declaration. A view implemented straight into
    # `_VIEW_FUNCS`, or listed straight into `TTA_VIEWS`, would have no declared
    # ops -- and `VIEW_PARAMS`, which the held-out-band check reads, is derived
    # from those ops. That is exactly how a jpeg q=70 view survived the suite.
    assert tuple(_VIEW_SPECS) == TTA_VIEWS
    assert set(_VIEW_FUNCS) == set(_VIEW_SPECS)


def test_every_view_is_built_from_ops_of_a_declared_kind():
    """There is no third, unclassified kind of op.

    `VIEW_PARAMS` is "the degradation steps of every view". That is only a
    complete statement if every step is classified, so an op in neither table
    is refused at composition time rather than silently counting as geometric.
    """
    from aigcdet.eval.tta import (
        _DEGRADATION_OPS, _GEOMETRIC_OPS, _VIEW_SPECS, _compose,
    )
    known = set(_DEGRADATION_OPS) | set(_GEOMETRIC_OPS)
    assert not set(_DEGRADATION_OPS) & set(_GEOMETRIC_OPS)
    for name, steps in _VIEW_SPECS.items():
        for op, _ in steps:
            assert op in known, (name, op)
    with pytest.raises(ValueError, match="unknown TTA op"):
        _compose((("posterize", 4),))
    with pytest.raises(ValueError, match="needs a severity value"):
        _compose((("jpeg", None),))


def test_unknown_view_raises():
    with pytest.raises(KeyError):
        apply_tta_view(_img(), "not_a_view")


def test_no_tta_view_lands_inside_a_heldout_severity_band():
    """The held-out severity claim must stay checkable by a machine.

    TTA runs at inference, so a view inside a held-out band would not be
    training exposure and would not contaminate the unseen-severity claim. It
    would, though, mean the ONE severity the report presents as never-seen is
    also the severity every image is silently re-encoded at before scoring,
    which is the kind of footnote a reader should never have to reconstruct.
    `jpeg_95` and `blur_0.3` both sit outside; this fails if either moves in.

    The set checked is DERIVED from the view declarations, not hand-listed
    beside them. When it was hand-listed, adding a jpeg q=70 view -- the dead
    centre of `HELDOUT_JPEG_Q` -- and leaving `VIEW_PARAMS` alone passed the
    whole suite, this test included.
    """
    from aigcdet.augment.recipes import HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q
    from aigcdet.eval.tta import VIEW_PARAMS, check_views_avoid_heldout_bands

    check_views_avoid_heldout_bands()          # the shipped views, re-checked
    bands = {"jpeg": HELDOUT_JPEG_Q, "blur": HELDOUT_BLUR_SIGMA}
    for view, steps in VIEW_PARAMS.items():
        assert view in TTA_VIEWS
        assert steps, view
        for family, value in steps:
            band = bands[family]
            assert not band[0] <= value <= band[1], (
                f"TTA view {view!r} uses {family}={value}, inside the held-out "
                f"band {band}")


def test_the_checked_degradations_are_derived_from_the_views_themselves():
    """Kills the mutant that adds a degradation view without declaring it.

    `VIEW_PARAMS` must be exactly the degradation steps of `_VIEW_SPECS` -- for
    every view, not just the ones somebody remembered. A view with a
    degradation step and no `VIEW_PARAMS` entry is a severity the held-out-band
    check never sees.
    """
    from aigcdet.eval.tta import (
        _DEGRADATION_OPS, _VIEW_SPECS, VIEW_PARAMS, degradation_params,
    )
    assert VIEW_PARAMS == degradation_params(_VIEW_SPECS)
    assert set(VIEW_PARAMS) == {"jpeg_95", "blur_0.3", "hflip_jpeg_95"}
    for name, steps in _VIEW_SPECS.items():
        has_degradation = any(op in _DEGRADATION_OPS for op, _ in steps)
        assert has_degradation == (name in VIEW_PARAMS), name


def test_a_degradation_view_cannot_be_added_without_reaching_the_band_check():
    """The mutation the reviewer ran, run here against the real check.

    Adding a jpeg q=70 view means adding it to `_VIEW_SPECS`, because that is
    the only place a view can be written -- `TTA_VIEWS`, `_VIEW_FUNCS` and
    `VIEW_PARAMS` are all derived from it. The band check is then handed a
    severity in the dead centre of `HELDOUT_JPEG_Q` and refuses. Before this
    restructuring the same mutant left the hand-maintained `VIEW_PARAMS`
    untouched and the whole suite passed, this test's ancestor included.
    """
    from aigcdet.augment.recipes import HELDOUT_JPEG_Q
    from aigcdet.eval.tta import (
        _VIEW_SPECS, check_views_avoid_heldout_bands, degradation_params,
    )
    assert HELDOUT_JPEG_Q[0] <= 70 <= HELDOUT_JPEG_Q[1], \
        "fixture: q=70 must be inside the held-out band for this to mean anything"
    mutant = dict(_VIEW_SPECS) | {"jpeg_70": (("jpeg", 70),)}
    assert degradation_params(mutant)["jpeg_70"] == (("jpeg", 70),)
    with pytest.raises(ValueError, match="inside the held-out severity band"):
        check_views_avoid_heldout_bands(degradation_params(mutant))
    # ... and a geometric view added the same way is still fine.
    check_views_avoid_heldout_bands(
        degradation_params(dict(_VIEW_SPECS) | {"scale_2": (("scale", 2.0),)}))


# --- the averaged logit ----------------------------------------------------

class _StubModel:
    use_recon = False

    def __init__(self, logits=(2.0,)):
        self.logits = list(logits)
        self.calls = 0
        self.recons = []

    def __call__(self, f, r=None):
        self.recons.append(r)
        value = self.logits[self.calls % len(self.logits)]
        self.calls += 1
        return {"logit": torch.tensor([value])}


def test_tta_logit_averages_over_views():
    model = _StubModel()
    out = tta_logit(None, _Spec(), model, _img(), device="cpu",
                    views=("identity", "hflip"), embed_fn=_stub_embed)
    assert out == pytest.approx(2.0)
    assert model.calls == 2


def test_tta_logit_averages_logits_and_not_probabilities():
    """Pins WHICH space the mean is taken in.

    Averaging probabilities and averaging logits give different answers, and
    the difference is invisible on a stub that returns one constant. With
    logits 3 and 1 the logit mean is exactly 2.0, while the probability mean is
    sigmoid(3)/2 + sigmoid(1)/2 = 0.8418 (whose own logit is 1.669) -- so this
    kills both `mean(sigmoid(logits))` and `logit(mean(sigmoid(logits)))`.
    """
    model = _StubModel(logits=(3.0, 1.0))
    out = tta_logit(None, _Spec(), model, _img(), device="cpu",
                    views=("identity", "hflip"), embed_fn=_stub_embed)
    assert out == pytest.approx(2.0)

    probs = torch.sigmoid(torch.tensor([3.0, 1.0]))
    assert out != pytest.approx(float(probs.mean()))
    assert out != pytest.approx(float(torch.logit(probs.mean())))


def test_tta_logit_scores_every_view_and_not_the_original_eight_times():
    """Kills the loop that embeds `img` instead of `apply_tta_view(img, v)`.

    The stub records the pixels it was handed, so the eight arrays must be
    eight distinct images rather than eight copies of the input.
    """
    seen = []

    def recording_embed(model, spec, imgs, device="cpu", batch_size=16):
        seen.extend(i.tobytes() for i in imgs)
        return np.zeros((len(imgs), 4), np.float32)

    img = _img()
    tta_logit(None, _Spec(), _StubModel(), img, device="cpu",
              embed_fn=recording_embed)
    assert len(seen) == len(TTA_VIEWS)
    assert len(set(seen)) == len(TTA_VIEWS)
    assert seen[0] == img.tobytes()          # "identity" is view 0


def test_tta_logit_uses_the_real_embed_when_none_is_injected(monkeypatch):
    """The injection point must not be the only wiring that works: with no
    `embed_fn` the function has to reach `features.backbones.embed`."""
    import aigcdet.features.backbones as backbones

    calls = {"n": 0}

    def fake_embed(model, spec, imgs, device="cpu", batch_size=16):
        calls["n"] += 1
        return np.zeros((len(imgs), 4), np.float32)

    monkeypatch.setattr(backbones, "embed", fake_embed)
    out = tta_logit(None, _Spec(), _StubModel(), _img(), device="cpu",
                    views=("identity", "hflip"))
    assert out == pytest.approx(2.0)
    assert calls["n"] == 2


def test_a_recon_model_without_a_recon_fn_is_refused():
    model = _StubModel()
    model.use_recon = True
    with pytest.raises(ValueError, match="recon_fn"):
        tta_logit(None, _Spec(), model, _img(), device="cpu",
                  views=("identity",), embed_fn=_stub_embed)


def test_the_recon_branch_is_recomputed_per_view():
    """A6 degrades the image before scoring it, so the recon feature has to be
    computed from the DEGRADED view; reusing the clean image's would describe a
    picture the head never saw."""
    model = _StubModel()
    model.use_recon = True
    seen = []

    def recon_fn(view):
        seen.append(view.tobytes())
        return np.zeros(3, np.float32)

    out = tta_logit(None, _Spec(), model, _img(), device="cpu",
                    views=("identity", "hflip"), embed_fn=_stub_embed,
                    recon_fn=recon_fn)
    assert out == pytest.approx(2.0)
    assert len(seen) == 2 and len(set(seen)) == 2
    assert all(r is not None for r in model.recons)


def test_an_unknown_view_in_the_views_argument_is_refused():
    with pytest.raises(KeyError):
        tta_logit(None, _Spec(), _StubModel(), _img(), device="cpu",
                  views=("identity", "sepia"), embed_fn=_stub_embed)


def test_tta_logit_never_loads_a_backbone():
    """A6 takes an already-loaded backbone. A `load_backbone` call inside it
    would download weights and start a GPU process on every image."""
    import aigcdet.features.backbones as backbones

    def explode(*args, **kwargs):
        raise AssertionError("tta_logit must not load a backbone")

    original = backbones.load_backbone
    backbones.load_backbone = explode
    try:
        tta_logit(None, _Spec(), _StubModel(), _img(), device="cpu",
                  views=("identity",), embed_fn=_stub_embed)
    finally:
        backbones.load_backbone = original


def test_an_empty_view_list_is_refused():
    """`np.mean([])` is nan with a RuntimeWarning, which would travel silently
    into a score column."""
    with pytest.raises(ValueError, match="no views"):
        tta_logit(None, _Spec(), _StubModel(), _img(), device="cpu",
                  views=(), embed_fn=_stub_embed)
