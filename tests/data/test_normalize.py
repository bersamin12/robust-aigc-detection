import numpy as np
from PIL import Image

from aigcdet.data.normalize import SHORT_SIDE, normalize_image, normalize_many


def _src(tmp_path, name, size, fmt="JPEG"):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(p, format=fmt)
    return str(p)


def test_short_side_is_512_and_exceeds_model_input():
    assert SHORT_SIDE == 512 and SHORT_SIDE > 384


def test_downscales_large_image_to_short_side_and_writes_png(tmp_path):
    src = _src(tmp_path, "big.jpg", (2048, 1024))
    dst = str(tmp_path / "out.png")
    w, h = normalize_image(src, dst)
    assert min(w, h) == 512
    assert (w, h) == (1024, 512)          # aspect ratio preserved
    with Image.open(dst) as im:
        assert im.format == "PNG"


def test_does_not_upscale_a_small_image(tmp_path):
    src = _src(tmp_path, "small.jpg", (300, 200))
    dst = str(tmp_path / "small.png")
    w, h = normalize_image(src, dst)
    assert (w, h) == (300, 200)


def test_downscales_portrait_image_by_shorter_side(tmp_path):
    src = _src(tmp_path, "portrait.jpg", (1024, 2048))
    dst = str(tmp_path / "portrait_out.png")
    w, h = normalize_image(src, dst)
    assert (w, h) == (512, 1024)


def test_leaves_image_unchanged_when_short_side_is_exactly_512(tmp_path):
    src = _src(tmp_path, "exact.jpg", (512, 900))
    dst = str(tmp_path / "exact_out.png")
    w, h = normalize_image(src, dst)
    assert (w, h) == (512, 900)


def test_converts_greyscale_and_rgba_to_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.zeros((600, 600), dtype=np.uint8), mode="L").save(p)
    dst = str(tmp_path / "g_out.png")
    normalize_image(str(p), dst)
    with Image.open(dst) as im:
        assert im.mode == "RGB"


def test_normalize_many_processes_all_pairs(tmp_path):
    pairs = [(_src(tmp_path, f"i{i}.jpg", (800, 600)), str(tmp_path / f"o{i}.png"))
             for i in range(5)]
    sizes = normalize_many(pairs, workers=2)
    assert len(sizes) == 5
    assert all(min(w, h) == 512 for w, h in sizes)
