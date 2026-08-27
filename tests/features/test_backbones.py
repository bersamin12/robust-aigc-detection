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
