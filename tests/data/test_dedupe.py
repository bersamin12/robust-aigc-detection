import numpy as np
from PIL import Image

from aigcdet.augment import ops
from aigcdet.data.dedupe import build_hash_index, find_leaks, hamming, phash


def _photo(seed):
    rng = np.random.default_rng(seed)
    return ops.blur(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), 3.0)


def _flat_photo(seed):
    """Near-uniform, low-texture image: a narrow-band, blurred backdrop
    (sky, wall, studio backdrop) rather than raw per-pixel noise. Real
    low-texture regions vary smoothly (lighting, vignetting), which is what
    the blur reproduces; unblurred per-pixel noise degrades to pure
    quantisation noise once resized down for hashing and is not
    representative of anything phash is expected to handle. This is the
    low-AC-energy regime where a DC leak into the hash bits would matter
    most, since the DC/AC magnitude gap is at its narrowest here."""
    rng = np.random.default_rng(seed)
    base = rng.integers(118, 139, (256, 256, 3), dtype=np.uint8)
    return ops.blur(base, 10.0)


def test_identical_images_hash_identically():
    img = _photo(0)
    assert phash(img) == phash(img.copy())


def test_recompressed_image_stays_within_threshold():
    img = _photo(1)
    assert hamming(phash(img), phash(ops.jpeg(img, 40))) <= 4


def test_resized_image_stays_within_threshold():
    img = _photo(2)
    assert hamming(phash(img), phash(ops.resize_roundtrip(img, 0.5))) <= 4


def test_brightness_shift_stays_within_threshold():
    # Colour jitter (+/-20% brightness) is one of the six degradation
    # families the whole detector is evaluated against; a demo image that
    # reached a training pool via an auto-enhance filter is exactly the
    # leakage case this guard must still catch.
    busy = _photo(5)
    assert hamming(phash(busy), phash(ops.jitter(busy, 0.2, 0.0, 0.0))) <= 4

    flat = _flat_photo(6)
    assert hamming(phash(flat), phash(ops.jitter(flat, 0.2, 0.0, 0.0))) <= 4


def test_different_images_are_far_apart():
    assert hamming(phash(_photo(3)), phash(_photo(99))) > 10


def test_find_leaks_flags_a_recompressed_duplicate(tmp_path):
    img = _photo(4)
    demo_p = tmp_path / "demo.png"
    cand_p = tmp_path / "cand.png"
    Image.fromarray(img).save(demo_p)
    Image.fromarray(ops.jpeg(img, 50)).save(cand_p)
    other_p = tmp_path / "other.png"
    Image.fromarray(_photo(77)).save(other_p)

    demo = build_hash_index([str(demo_p)])
    cand = build_hash_index([str(cand_p), str(other_p)])
    leaks = find_leaks(cand, demo, max_distance=4)
    assert str(cand_p) in leaks
    assert str(other_p) not in leaks
