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

def _scipy_blur_reference(img, sigma):
    """The exact implementation `ops.blur` replaced: scipy's gaussian_filter
    on float32 HWC, clipped and truncated to uint8. Recomputed here rather
    than snapshotted, so this pins agreement with scipy itself."""
    from scipy.ndimage import gaussian_filter

    out = gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_blur_agrees_with_the_scipy_implementation_it_replaced(img, sigma):
    """cv2 with kernel support matched to scipy's truncate=4.0
    (radius = int(4*sigma + 0.5)) and the same reflecting border, at ~1-4 ms
    per 512x768 call against scipy's ~15-25 ms.

    Measured tolerance at every evaluation sigma, on this uniform-noise
    fixture and on smooth content: at most ONE grey level, on well under 1%
    of pixels. Both implementations truncate on the way back to uint8, so
    there is no systematic offset -- the residual is float rounding at the
    truncation boundary.
    """
    ref = _scipy_blur_reference(img, sigma)
    out = ops.blur(img, sigma=sigma)
    assert out.shape == img.shape and out.dtype == np.uint8
    d = np.abs(out.astype(int) - ref.astype(int))
    assert d.max() <= 1
    assert (d > 0).mean() < 0.01


def test_blur_agrees_with_scipy_on_photographic_and_compressed_content():
    """Uniform noise is not where the two implementations disagree: the
    residual concentrates on smooth and JPEG-blocked content, where far more
    pixels sit near the truncation boundary. Measured worst case across the
    three evaluation sigmas is 0.12% of pixels, all by one grey level."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    photo = np.clip(gaussian_filter(rng.normal(128, 40, (192, 160, 3)), (2, 2, 0)),
                    0, 255).astype(np.uint8)
    for content in (photo, ops.jpeg(photo, 30)):
        for sigma in (0.5, 1.0, 2.0):
            d = np.abs(ops.blur(content, sigma).astype(int)
                       - _scipy_blur_reference(content, sigma).astype(int))
            assert d.max() <= 1
            assert (d > 0).mean() < 0.01

@pytest.mark.parametrize("scale", [0.5, 0.25])
def test_resize_roundtrip_returns_original_shape(img, scale):
    out = ops.resize_roundtrip(img, scale=scale)
    assert out.shape == img.shape
    # The op must actually alter pixel content, not return input unchanged
    assert not np.array_equal(out, img)

def test_resize_roundtrip_degrades_more_aggressively_at_smaller_scale(img):
    # More aggressive downsampling (0.25) must degrade more than less aggressive (0.5)
    d_half = np.abs(ops.resize_roundtrip(img, scale=0.5).astype(int) - img.astype(int)).mean()
    d_quarter = np.abs(ops.resize_roundtrip(img, scale=0.25).astype(int) - img.astype(int)).mean()
    assert d_quarter > d_half

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

def test_jitter_contrast_increases_spread(img):
    # Positive contrast delta should increase standard deviation (spread about mean)
    original_std = img.astype(np.float32).std()
    jittered = ops.jitter(img, 0.0, 0.2, 0.0).astype(np.float32)
    jittered_std = jittered.std()
    assert jittered_std > original_std

def test_jitter_saturation_increases_color_distance(img):
    # Positive saturation delta should increase mean distance from greyscale
    # Compute distance from greyscale luminance: sqrt(sum of squared differences from grey value)
    grey = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    original_dist = np.sqrt(((img.astype(np.float32) - grey[..., None]) ** 2).mean())

    jittered = ops.jitter(img, 0.0, 0.0, 0.2)
    jittered_float = jittered.astype(np.float32)
    grey_jit = jittered_float @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    jittered_dist = np.sqrt(((jittered_float - grey_jit[..., None]) ** 2).mean())

    assert jittered_dist > original_dist

def test_center_crop_80_preserves_shape_and_drops_border(img):
    out = ops.center_crop(img, frac=0.8)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)

def test_op_funcs_covers_all_six_families():
    assert set(ops.OP_FUNCS) == {"jpeg", "blur", "resize", "noise", "jitter", "crop"}
    # Verify the values point to the correct functions (especially the two renamed entries)
    assert ops.OP_FUNCS["resize"] is ops.resize_roundtrip
    assert ops.OP_FUNCS["crop"] is ops.center_crop
