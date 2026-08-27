import numpy as np
import pytest

from aigcdet.augment import ops


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(128, 96, 3), dtype=np.uint8)


@pytest.mark.parametrize("q", [90, 70, 50, 30])
def test_jpeg_preserves_shape_and_degrades_monotonically(img, q):
    out = ops.jpeg(img, quality=q)
    assert out.shape == img.shape and out.dtype == np.uint8

def test_jpeg_lower_quality_is_further_from_original(img):
    d90 = np.abs(ops.jpeg(img, 90).astype(int) - img.astype(int)).mean()
    d30 = np.abs(ops.jpeg(img, 30).astype(int) - img.astype(int)).mean()
    assert d30 > d90

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_blur_reduces_high_frequency_energy(img, sigma):
    out = ops.blur(img, sigma=sigma)
    assert out.shape == img.shape
    # Laplacian variance is a standard sharpness proxy; blur must reduce it
    import cv2
    sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    soft = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    assert soft < sharp

def test_blur_sigma_zero_is_identity(img):
    assert np.array_equal(ops.blur(img, sigma=0.0), img)

@pytest.mark.parametrize("scale", [0.5, 0.25])
def test_resize_roundtrip_returns_original_shape(img, scale):
    out = ops.resize_roundtrip(img, scale=scale)
    assert out.shape == img.shape

@pytest.mark.parametrize("sigma", [0.02, 0.05, 0.10])
def test_noise_is_deterministic_given_rng_and_scales_with_sigma(img, sigma):
    a = ops.noise(img, sigma=sigma, rng=np.random.default_rng(7))
    b = ops.noise(img, sigma=sigma, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)
    small = np.abs(ops.noise(img, 0.02, np.random.default_rng(1)).astype(int) - img.astype(int)).mean()
    big = np.abs(ops.noise(img, 0.10, np.random.default_rng(1)).astype(int) - img.astype(int)).mean()
    assert big > small

def test_jitter_identity_at_zero(img):
    assert np.array_equal(ops.jitter(img, 0.0, 0.0, 0.0), img)

def test_jitter_brightness_raises_mean(img):
    assert ops.jitter(img, 0.2, 0.0, 0.0).mean() > img.mean()

def test_center_crop_80_preserves_shape_and_drops_border(img):
    out = ops.center_crop(img, frac=0.8)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)

def test_op_funcs_covers_all_six_families():
    assert set(ops.OP_FUNCS) == {"jpeg", "blur", "resize", "noise", "jitter", "crop"}
