import numpy as np
import pytest

from aigcdet.features.backbones import BACKBONES, squish

_MIN_FREE_BYTES = 4 * 1024**3  # 4 GB; a ViT-L backbone needs headroom beyond its weights


def _skip_unless_gpu_has_headroom():
    """Real @pytest.mark.gpu tests must skip honestly on a near-full GPU rather
    than attempting a load that OOMs. torch.cuda.is_available() is True even at
    a few hundred MiB free, so the guard checks free VRAM via mem_get_info(),
    not just device availability."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if free_bytes < _MIN_FREE_BYTES:
        pytest.skip(
            f"only {free_bytes / 1024**2:.0f} MiB free on GPU, "
            f"need at least {_MIN_FREE_BYTES / 1024**2:.0f} MiB to load a ViT-L backbone"
        )


def test_registry_has_the_three_planned_backbones():
    assert set(BACKBONES) == {"dinov3l", "siglip2l", "clipl"}
    for spec in BACKBONES.values():
        assert spec.dim > 0 and spec.image_size in (224, 384)
        assert spec.num_prefix_tokens >= 1 or spec.name == "siglip2l"  # at least a CLS token to strip, except SigLIP2 which has none


def test_two_shipped_backbones_stay_under_1b_working_ceiling():
    # A tighter working budget for the two backbones this project trains against
    # day to day (dinov3l + siglip2l). clipl is a third registry entry not summed
    # here -- see test_full_registry_stays_under_the_2b_hackathon_hard_limit for
    # the constraint that actually gates the hackathon submission.
    assert BACKBONES["dinov3l"].params + BACKBONES["siglip2l"].params < 1_000_000_000


def test_full_registry_stays_under_the_2b_hackathon_hard_limit():
    # The hackathon's real, hard constraint (project-constraints.md): every model
    # load_backbone can load, summed, must stay under 2B parameters total. This
    # sums every BACKBONES entry -- including clipl, which the tighter working
    # ceiling above does not cover -- and leaves documented margin for the
    # auxiliary models Task 4 adds (SD 1.5 VAE ~84M params, LPIPS AlexNet ~2.5M
    # params; neither is registered in BACKBONES).
    total = sum(spec.params for spec in BACKBONES.values())
    assert total < 2_000_000_000


def test_backbone_params_are_positive_and_real_looking():
    # Guards against a placeholder value that happens to satisfy the budget
    # inequality: every recorded count must be a plausible ViT-L parameter count.
    for spec in BACKBONES.values():
        assert 200_000_000 < spec.params < 500_000_000


def test_squish_ignores_aspect_ratio():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    out = squish(img, 384)
    assert out.shape == (384, 384, 3) and out.dtype == np.uint8


def test_squish_preserves_pixel_content():
    # A regression guard against a stub that returns a black/constant image:
    # squish must actually resample the input, not fabricate output.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
    out = squish(img, 64)
    assert out.std() > 0
    assert not np.array_equal(out, np.zeros_like(out))


def test_squish_distorts_aspect_ratio_by_known_factor():
    # squish must stretch x and y independently, not letterbox/pad-and-resize.
    # A resize that preserved content aspect ratio would pass
    # test_squish_ignores_aspect_ratio (it only checks output shape) while
    # quietly changing what every backbone sees -- random resized cropping and
    # aspect-preserving resizes were both rejected for this project because
    # they can destroy the localised forensic cues detection depends on. Pin
    # the distortion itself: a square marker must come out as a rectangle whose
    # dimensions match squish's independent x/y scale factors.
    h, w, size = 100, 300, 384
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255  # a 10x10 white square, aspect ratio 1:1

    out = squish(img, size)

    mask = out[..., 0] > 128
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    out_height = int(rows.max() - rows.min() + 1)
    out_width = int(cols.max() - cols.min() + 1)

    expected_height = round(10 * size / h)  # independent y scale: 384/100 -> ~38
    expected_width = round(10 * size / w)   # independent x scale: 384/300 -> ~13
    assert abs(out_height - expected_height) <= 3
    assert abs(out_width - expected_width) <= 3
    # The square must become a visibly non-square rectangle: a letterboxed or
    # otherwise aspect-preserving resize would keep it square here.
    assert out_height != out_width


@pytest.mark.gpu
def test_embed_returns_pooled_vectors_of_the_right_width():
    _skip_unless_gpu_has_headroom()
    from aigcdet.features.backbones import embed, load_backbone

    model, spec = load_backbone("clipl", device="cuda")
    imgs = [np.random.default_rng(i).integers(0, 256, (512, 640, 3), dtype=np.uint8)
            for i in range(3)]
    out = embed(model, spec, imgs, device="cuda", batch_size=2)
    assert out.shape == (3, spec.dim) and out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.gpu
def test_embedding_is_deterministic():
    _skip_unless_gpu_has_headroom()
    from aigcdet.features.backbones import embed, load_backbone

    model, spec = load_backbone("clipl", device="cuda")
    img = [np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)]
    a = embed(model, spec, img, device="cuda")
    b = embed(model, spec, img, device="cuda")
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# embed()'s tensor contract, per backbone, on CPU.
#
# These build the SAME architecture each registry entry names, at toy width,
# from a locally-constructed config: no download, no GPU, no weights. That is
# what would have caught C1 on day 1 -- `embed` asserted one input contract
# (`model(pixel_values=(B, 3, H, W))`) across three backbones that do not share
# one, and SigLIP2 could not be embedded at all.
# --------------------------------------------------------------------------

_TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
             num_attention_heads=4)
_TINY_PATCH = 16
_TINY_IMAGE = 64          # 4 x 4 = 16 patches


def _tiny_tower(name: str):
    """(vision tower, spec) for `name`'s real architecture at toy width.

    The returned spec keeps every contract-bearing field of the registry entry
    -- `input_format`, `patch_size`, `num_prefix_tokens` -- and only shrinks
    `image_size`/`dim`/`params`, so a wrong input_format or prefix-token count
    in `BACKBONES` fails here.
    """
    from dataclasses import replace

    real = BACKBONES[name]
    if name == "dinov3l":
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        # num_register_tokens=4 plus DINOv3's CLS token == the registry's
        # num_prefix_tokens=5, so the real value is exercised, not a stand-in.
        model = DINOv3ViTModel(DINOv3ViTConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_register_tokens=real.num_prefix_tokens - 1, **_TINY))
    elif name == "siglip2l":
        from transformers import Siglip2VisionConfig, Siglip2VisionModel
        model = Siglip2VisionModel(Siglip2VisionConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_patches=(_TINY_IMAGE // _TINY_PATCH) ** 2, num_channels=3, **_TINY))
    elif name == "clipl":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        model = CLIPVisionModel(CLIPVisionConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE, **_TINY))
    else:
        raise AssertionError(f"no tiny tower recipe for {name!r}")

    # The same unwrapping load_backbone does.
    model = getattr(model, "vision_model", model).eval()
    spec = replace(real, image_size=_TINY_IMAGE, dim=_TINY["hidden_size"], params=0)
    return model, spec


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_embed_tensor_contract_holds_for_every_backbone_on_cpu(name):
    from aigcdet.features.backbones import embed

    model, spec = _tiny_tower(name)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (120, 200, 3), dtype=np.uint8) for _ in range(3)]

    out = embed(model, spec, imgs, device="cpu", batch_size=2)

    assert out.shape == (3, spec.dim)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Not a constant/stub: three different images must give three different
    # pooled vectors.
    assert len({tuple(np.round(r, 5)) for r in out}) == 3


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_num_prefix_tokens_matches_the_real_architecture(name):
    """`embed` strips `spec.num_prefix_tokens` before pooling. If that count is
    wrong for an architecture, pooling silently averages a CLS/register token
    into the patch mean (or drops a real patch) and nothing else notices."""
    import torch

    from aigcdet.features.backbones import model_inputs

    model, spec = _tiny_tower(name)
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 256, (90, 90, 3), dtype=np.uint8) for _ in range(2)]

    with torch.inference_mode():
        h = model(**model_inputs(spec, imgs, "cpu", torch.float32)).last_hidden_state

    expected_patches = (spec.image_size // _TINY_PATCH) ** 2
    assert h.shape[1] - spec.num_prefix_tokens == expected_patches
    assert h.shape[-1] == spec.dim


def test_every_registry_entry_declares_a_supported_input_format():
    from aigcdet.features.backbones import INPUT_FORMATS, INPUT_SIGLIP2_PATCHES

    for spec in BACKBONES.values():
        assert spec.input_format in INPUT_FORMATS
        if spec.input_format == INPUT_SIGLIP2_PATCHES:
            assert spec.patch_size > 0
            assert spec.image_size % spec.patch_size == 0


def test_spec_rejects_an_unusable_input_contract():
    from aigcdet.features.backbones import INPUT_SIGLIP2_PATCHES, BackboneSpec

    with pytest.raises(ValueError, match="input_format"):
        BackboneSpec("bad", "none", 384, 1024, 0, 1, input_format="conv2d")
    with pytest.raises(ValueError, match="patch_size"):
        BackboneSpec("bad", "none", 384, 1024, 0, 1,
                     input_format=INPUT_SIGLIP2_PATCHES)
    with pytest.raises(ValueError, match="divisible"):
        BackboneSpec("bad", "none", 100, 1024, 0, 1,
                     input_format=INPUT_SIGLIP2_PATCHES, patch_size=16)


def test_patchify_matches_the_transformers_reference_implementation():
    """SigLIP2's patch_embedding is a Linear over flattened patches, so the
    element ORDER inside each patch is part of the weight contract. Pin it
    against the transformers implementation the real processor uses."""
    from transformers.models.siglip2.image_processing_siglip2 import (
        convert_image_to_patches,
    )

    from aigcdet.features.backbones import _patchify

    rng = np.random.default_rng(3)
    arr = rng.normal(size=(2, 64, 48, 3)).astype(np.float32)
    ours, n_h, n_w = _patchify(arr, 16)
    assert (n_h, n_w) == (4, 3)
    for b in range(2):
        np.testing.assert_array_equal(ours[b], convert_image_to_patches(arr[b], 16))
