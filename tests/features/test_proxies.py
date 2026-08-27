import numpy as np
import pytest
from PIL import Image

from aigcdet.augment import ops
from aigcdet.features.proxies import (
    PROXY_NAMES, laplacian_variance, noise_floor, proxy_vector,
    estimate_jpeg_quality,
)


def _photo(seed=0):
    """Smooth-ish synthetic image; pure noise has no blockiness structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (256, 256, 3))
    return np.clip(ops.blur(np.clip(base, 0, 255).astype(np.uint8), 2.0), 0, 255)


def test_proxy_vector_shape_and_names():
    v = proxy_vector(_photo())
    assert v.shape == (3,) and v.dtype == np.float32
    assert PROXY_NAMES == ("jpeg_quality", "laplacian_var", "noise_floor")


def test_laplacian_variance_drops_with_blur():
    img = _photo()
    assert laplacian_variance(ops.blur(img, 2.0)) < laplacian_variance(img)


def test_noise_floor_rises_with_added_noise():
    img = _photo()
    clean = noise_floor(img)
    noisy = noise_floor(ops.noise(img, 0.10, np.random.default_rng(0)))
    assert noisy > clean


def test_estimated_jpeg_quality_tracks_true_quality_from_file(tmp_path):
    img = _photo()
    est = {}
    for q in (30, 90):
        p = tmp_path / f"q{q}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=q)
        est[q] = estimate_jpeg_quality(np.asarray(Image.open(p).convert("RGB")), str(p))
    assert est[90] > est[30]


def test_estimated_jpeg_quality_without_path_still_ranks_pixels(tmp_path):
    img = _photo()
    low = estimate_jpeg_quality(ops.jpeg(img, 30))
    high = estimate_jpeg_quality(ops.jpeg(img, 95))
    assert high > low


def test_proxies_are_finite_on_flat_image():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert np.all(np.isfinite(proxy_vector(flat)))


def _textured(seed=1, size=256):
    """High-detail synthetic image: higher-variance noise with only mild
    blur, so it retains real local structure at many scales (unlike the
    near-flat `_photo()` above). Blur sigma is kept well above `_photo()`'s
    own warning threshold ("pure noise has no blockiness structure") — a
    lighter blur (tried during calibration) put this too close to literal
    iid pixel noise, which JPEG can't meaningfully compress and so carries
    no usable blockiness signal at any quality."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 90, (size, size, 3))
    return np.clip(ops.blur(np.clip(base, 0, 255).astype(np.uint8), 1.0), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("content_fn", [_photo, _textured], ids=["smooth", "textured"])
def test_estimated_jpeg_quality_fallback_is_non_inverting(content_fn):
    """The pixel-only fallback must never rank a worse-compressed image as
    higher quality than a better-compressed one, on smooth/low-texture
    content (`_photo`) and on high-texture content (`_textured`) alike.

    Ties are fine (the fallback's blockiness->quality lookup table has
    plateaus); a sign reversal is not. See the calibration notes in
    aigcdet.features.proxies for the offline sweep this guards and the
    honest residual error it does not try to hide.
    """
    img = content_fn().astype(np.uint8)
    qs = [5, 10, 20, 30, 50, 70, 90]
    estimates = [estimate_jpeg_quality(ops.jpeg(img, q)) for q in qs]
    for lo, hi in zip(estimates, estimates[1:]):
        assert hi >= lo - 1e-6, f"non-inverting fallback violated: {estimates} for q={qs}"


class _JpegWithNonZeroDqtIndex:
    """Stands in for a JPEG whose only DQT segment is not index 0.

    `im.quantization` is a DICT keyed by table index, not a list, so
    `tables[0]` raises KeyError on such a file. PIL will not write one, so
    the dict is supplied directly rather than through a real encode; the code
    under test only ever touches `.quantization`.
    """

    quantization = {1: [16] * 64}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_odd_quantisation_table_index_falls_back_instead_of_aborting(tmp_path, monkeypatch):
    """estimate_jpeg_quality runs inside audit_table over every image, so an
    exception here aborts the whole ~20-minute audit pass. Normalisation
    proving a file decodable does not prove its DQT is well-formed, so the
    skip list does not cover this one -- it has to fall through to the
    pixel-based estimate, as the pre-narrowing bare `except` did.
    """
    import aigcdet.features.proxies as proxies_module

    img = _photo()
    path = tmp_path / "odd_dqt.jpg"
    Image.fromarray(img).save(path, format="JPEG", quality=80)
    monkeypatch.setattr(proxies_module.Image, "open",
                        lambda _p: _JpegWithNonZeroDqtIndex())

    q = estimate_jpeg_quality(img, str(path))
    assert 1.0 <= q <= 100.0
    # It fell back to the pixel estimate rather than reading the table.
    assert q == estimate_jpeg_quality(img)
