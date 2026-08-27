import numpy as np
from PIL import Image

from aigcdet.augment import ops
from aigcdet.data.dedupe import build_hash_index, find_leaks, hamming, phash


def _photo(seed):
    rng = np.random.default_rng(seed)
    return ops.blur(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), 3.0)


def test_identical_images_hash_identically():
    img = _photo(0)
    assert phash(img) == phash(img.copy())


def test_recompressed_image_stays_within_threshold():
    img = _photo(1)
    assert hamming(phash(img), phash(ops.jpeg(img, 40))) <= 4


def test_resized_image_stays_within_threshold():
    img = _photo(2)
    assert hamming(phash(img), phash(ops.resize_roundtrip(img, 0.5))) <= 4


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
