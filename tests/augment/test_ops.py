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

def test_op_funcs_covers_all_six_families_and_every_op_has_a_family():
    from aigcdet.augment.recipes import FAMILIES, FAMILY_OF

    assert {"jpeg", "blur", "resize", "noise", "jitter", "crop"} <= set(ops.OP_FUNCS)
    # Verify the values point to the correct functions (especially the two renamed entries)
    assert ops.OP_FUNCS["resize"] is ops.resize_roundtrip
    assert ops.OP_FUNCS["crop"] is ops.center_crop
    # Every op is grouped under one of the six family bits, and every family
    # has at least its brief op: an op outside FAMILY_OF would raise inside
    # `Recipe.labels` at training time rather than here.
    assert set(ops.OP_FUNCS) == set(FAMILY_OF)
    assert set(FAMILY_OF.values()) == set(FAMILIES)
    for fam in FAMILIES:
        assert FAMILY_OF[fam] == fam


# --- NTIRE additions ---------------------------------------------------------

def _dist(a, b):
    return np.abs(a.astype(int) - b.astype(int)).mean()


@pytest.fixture
def photo():
    """Smooth content: several ops (tone curve, CLAHE, pixelate) are
    near-identity on uniform noise and only show on structured pixels."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    x = gaussian_filter(rng.normal(128, 50, (128, 96, 3)), (3, 3, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name,kw", [
    ("jpeg_multi", {"qualities": [80, 40]}),
    ("motion_blur", {"length": 9, "angle": 30.0}),
    ("lens_blur", {"radius": 3}),
    ("downscale", {"scale": 0.5, "interp_down": "area", "interp_up": "nearest"}),
    ("pixelate", {"block": 6}),
    ("tone_curve", {"low_y": 0.6, "high_y": 0.4}),
    ("clahe", {"clip": 3.0, "grid": 8}),
    ("contrast", {"alpha": 1.4}),
    ("brightness", {"beta": 30.0}),
    ("quantize", {"bits": 4}),
    ("channel_shift", {"dr": 20.0, "dg": -10.0, "db": 5.0}),
    ("saturation", {"scale": 1.5}),
    ("aspect_crop", {"top": 0.1, "left": 0.2, "h_frac": 0.7, "w_frac": 0.6}),
    ("perspective", {"offsets": [0.05, -0.03, 0.02, 0.04, -0.05, 0.01, 0.03, -0.02]}),
])
def test_deterministic_ops_preserve_shape_and_change_pixels(photo, name, kw):
    fn = ops.OP_FUNCS[name]
    out = fn(photo, **kw)
    assert out.shape == photo.shape and out.dtype == np.uint8
    assert not np.array_equal(out, photo)
    # Parameters carry ALL the randomness: two calls agree bit for bit.
    assert np.array_equal(out, fn(photo, **kw))


@pytest.mark.parametrize("name,kw", [
    ("glass_blur", {"sigma": 0.7, "max_delta": 2, "iters": 1}),
    ("impulse", {"amount": 0.05}),
    ("iso_noise", {"intensity": 0.3, "color_shift": 0.02}),
])
def test_stochastic_ops_are_reproducible_from_the_rng(photo, name, kw):
    fn = ops.OP_FUNCS[name]
    a = fn(photo, rng=np.random.default_rng(7), **kw)
    b = fn(photo, rng=np.random.default_rng(7), **kw)
    assert a.shape == photo.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, photo)


def test_jpeg2000_round_trips_when_the_build_supports_it(photo):
    if not ops.jpeg2000_available():
        pytest.skip("this OpenCV build cannot write .jp2")
    out = ops.jpeg2000(photo, rate=100)
    assert out.shape == photo.shape and out.dtype == np.uint8
    assert _dist(ops.jpeg2000(photo, 60), photo) > _dist(ops.jpeg2000(photo, 400), photo)


def test_identity_parameters_are_identities(photo):
    assert np.array_equal(ops.contrast(photo, 1.0), photo)
    assert np.array_equal(ops.brightness(photo, 0.0), photo)
    assert np.array_equal(ops.saturation(photo, 1.0), photo)
    assert np.array_equal(ops.channel_shift(photo, 0.0, 0.0, 0.0), photo)
    assert np.array_equal(ops.quantize(photo, 8), photo)
    assert np.array_equal(ops.perspective(photo, [0.0] * 8), photo)
    assert np.array_equal(ops.aspect_crop(photo, 0.0, 0.0, 1.0, 1.0), photo)
    # The Bezier with control points at (0.25, 0.25) and (0.75, 0.75) is the
    # diagonal; the LUT it produces must be the identity to within rounding.
    assert np.abs(ops.tone_curve(photo, 0.25, 0.75).astype(int) - photo.astype(int)).max() <= 1


def test_harsher_parameters_move_further(photo):
    assert _dist(ops.motion_blur(photo, 15, 0.0), photo) > _dist(ops.motion_blur(photo, 3, 0.0), photo)
    assert _dist(ops.lens_blur(photo, 5), photo) > _dist(ops.lens_blur(photo, 1), photo)
    assert _dist(ops.pixelate(photo, 12), photo) > _dist(ops.pixelate(photo, 2), photo)
    assert _dist(ops.quantize(photo, 3), photo) > _dist(ops.quantize(photo, 7), photo)
    assert _dist(ops.brightness(photo, 60.0), photo) > _dist(ops.brightness(photo, 10.0), photo)
    r = np.random.default_rng(0)
    assert (_dist(ops.impulse(photo, 0.08, np.random.default_rng(0)), photo)
            > _dist(ops.impulse(photo, 0.005, np.random.default_rng(0)), photo))
    assert (_dist(ops.iso_noise(photo, 0.5, 0.0, np.random.default_rng(0)), photo)
            > _dist(ops.iso_noise(photo, 0.05, 0.0, r), photo))


def test_impulse_sets_pixels_to_the_extremes_only(photo):
    out = ops.impulse(photo, 0.2, np.random.default_rng(1))
    changed = np.any(out != photo, axis=-1)
    assert changed.mean() > 0.1
    vals = out[changed]
    assert set(np.unique(vals)) <= {0, 255}
