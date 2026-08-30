import numpy as np
import pytest
from PIL import Image

import os

from aigcdet.data.normalize import (
    PNG_MODE_OVERRIDES, PNG_NATIVE_MODES, SHORT_SIDE, normalize_image,
    normalize_many, png_target_mode, save_png,
)


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


# Deliberately all different, and NOT in ascending order: the returned list
# becomes the manifest's row order, which Plan 2's feature banks index
# positionally, so a reordering must fail this test. Five same-size inputs
# (the previous fixture) could not detect one.
_ORDERING_SIZES = [(2048, 1024), (900, 1500), (1024, 2048), (300, 200), (1600, 1200)]
_EXPECTED = [(1024, 512), (512, 853), (512, 1024), (300, 200), (683, 512)]


def test_normalize_many_preserves_input_order(tmp_path):
    pairs = [(_src(tmp_path, f"i{i}.jpg", wh), str(tmp_path / f"o{i}.png"))
             for i, wh in enumerate(_ORDERING_SIZES)]
    sizes, failures = normalize_many(pairs, workers=4)
    assert failures == []
    assert sizes == _EXPECTED


def test_normalize_many_skips_unreadable_files_and_reports_them(tmp_path):
    """One truncated file in 100k streamed images is near-certain, and
    `list(ex.map(...))` killed the whole ~20-minute run on the first one."""
    good = _src(tmp_path, "good0.jpg", (800, 600))
    truncated = tmp_path / "truncated.png"
    with open(_src(tmp_path, "whole.png", (800, 600), fmt="PNG"), "rb") as f:
        whole = f.read()
    truncated.write_bytes(whole[: len(whole) // 2])
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    good2 = _src(tmp_path, "good1.jpg", (1024, 2048))

    pairs = [(good, str(tmp_path / "a.png")), (str(truncated), str(tmp_path / "b.png")),
             (str(empty), str(tmp_path / "c.png")), (good2, str(tmp_path / "d.png"))]
    sizes, failures = normalize_many(pairs, workers=2)

    # Successful files keep their input position; failures hold their slot.
    assert sizes == [(683, 512), None, None, (512, 1024)]
    assert [src for src, _ in failures] == [str(truncated), str(empty)]
    assert all(msg for _, msg in failures)  # the reason is recorded, not swallowed


def _jpeg_with_orientation(path, size, orientation):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[0x0112] = orientation  # 274, Orientation
    Image.fromarray(arr).save(path, format="JPEG", exif=exif)
    return str(path)


@pytest.mark.parametrize("orientation", [6, 8])
def test_exif_orientation_is_applied_before_measuring(tmp_path, orientation):
    """Authentic photographs carry EXIF and generator output does not, so
    ignoring orientation lands on ONE CLASS ONLY -- inside the step whose
    purpose is removing class-distinguishing container properties. PNG drops
    the tag, so the wrong rotation would be baked in permanently.
    """
    src = _jpeg_with_orientation(tmp_path / f"rot{orientation}.jpg", (800, 600), orientation)
    dst = str(tmp_path / f"rot{orientation}.png")
    w, h = normalize_image(src, dst)
    # 800x600 rotated a quarter turn is 600x800; short side 600 -> 512.
    assert (w, h) == (512, 683)
    with Image.open(dst) as im:
        assert im.size == (w, h)


def test_upright_image_is_unaffected_by_exif_handling(tmp_path):
    src = _jpeg_with_orientation(tmp_path / "up.jpg", (800, 600), 1)
    w, h = normalize_image(src, str(tmp_path / "up.png"))
    assert (w, h) == (683, 512)


# ---------------------------------------------------------------------------
# save_png: the raw-tree writer's mode policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", sorted(PNG_NATIVE_MODES))
def test_every_declared_native_mode_really_survives_the_png_encoder(tmp_path, mode):
    """PNG_NATIVE_MODES is a claim about Pillow, whose own table is private
    and has changed between releases. This makes the claim, not the list, the
    thing under test: each mode must encode AND come back with its samples
    intact, or `save_png` would be quietly corrupting a class of image."""
    im = Image.new(mode, (4, 4))
    dst = str(tmp_path / f"{mode.replace(';', '_')}.png")
    assert save_png(im, dst) == mode
    with Image.open(dst) as back:
        assert back.mode == mode
        assert back.tobytes() == im.tobytes()


@pytest.mark.parametrize(("mode", "expected"), [
    ("RGB", "RGB"), ("RGBA", "RGBA"), ("L", "L"), ("P", "P"), ("LA", "LA"),
    ("CMYK", "RGB"),      # PNG cannot hold CMYK; no alpha to lose
    ("YCbCr", "RGB"),
    ("F", "RGB"),
    ("RGBa", "RGBA"),     # premultiplied alpha: still alpha
    ("La", "RGBA"),
    ("I", PNG_MODE_OVERRIDES["I"]),
])
def test_png_target_mode_converts_only_what_png_cannot_hold(mode, expected):
    assert png_target_mode(mode, Image.new(mode, (2, 2)).getbands()) == expected


def test_a_cmyk_image_saves_as_rgb_instead_of_raising(tmp_path):
    """`OSError: cannot write mode CMYK as PNG` ended a real SID_Set run."""
    dst = str(tmp_path / "cmyk.png")
    assert save_png(Image.new("CMYK", (4, 4), (0, 0, 0, 0)), dst) == "RGB"
    with Image.open(dst) as back:
        assert back.mode == "RGB"


def test_alpha_is_never_flattened_against_black(tmp_path):
    """The deliberate choice: a non-native mode WITH alpha goes to RGBA, not
    RGB. `convert("RGB")` composites against black, inventing a colour for
    every transparent pixel."""
    im = Image.new("RGBa", (4, 4), (10, 20, 30, 0))
    dst = str(tmp_path / "a.png")
    assert save_png(im, dst) == "RGBA"
    with Image.open(dst) as back:
        assert back.mode == "RGBA"
        assert back.getpixel((0, 0))[3] == 0


def test_save_png_leaves_no_partial_file_when_the_encoder_fails(tmp_path):
    """Resume tests `os.path.exists(dst)`, so a truncated file left at `dst`
    by an interrupted write is worse than no file at all."""
    class Exploding:
        mode = "RGB"

        def getbands(self):
            return ("R", "G", "B")

        def save(self, path, **kw):
            with open(path, "wb") as f:
                f.write(b"partial")
            raise OSError("disk full")

    dst = str(tmp_path / "boom.png")
    with pytest.raises(OSError, match="disk full"):
        save_png(Exploding(), dst)
    assert not os.path.exists(dst)
    assert not os.path.exists(dst + ".part")


def test_save_png_creates_the_destination_directory(tmp_path):
    dst = str(tmp_path / "a" / "b" / "x.png")
    assert save_png(Image.new("RGB", (2, 2)), dst) == "RGB"
    assert os.path.exists(dst)


# ---------------------------------------------------------------------------
# Metadata that survives the container change
# ---------------------------------------------------------------------------

def test_normalisation_writes_a_png_it_can_read_back(tmp_path):
    """The round-trip property, which was not holding.

    Pillow will WRITE an arbitrarily large iCCP chunk and then refuse to READ
    it back (`_safe_zlib_decompress` caps it at `MAX_TEXT_CHUNK`). One COCO
    photograph among 183,543 carried a profile that big, so normalisation
    produced a file normalisation itself could not reopen. It passed the audit
    -- which reads the RAW tree -- and killed the perceptual-hash pass 95
    minutes into a corpus build, at the step whose comment says normalisation
    has "already proved the candidate files decodable". It had proved the
    INPUTS decodable.
    """
    from aigcdet.data.normalize import _without_stripped_metadata

    # A JPEG source specifically, because that is the only way the bug can
    # occur: JPEG chunks its ICC across APP2 segments and imposes no read cap,
    # so Pillow reads a 2 MB profile back happily -- and then writes it into
    # the PNG as one iCCP chunk it will not read. A PNG source with the same
    # profile is unreadable at the SOURCE and is simply skipped, which is a
    # different (already handled) case.
    src = tmp_path / "src.jpg"
    arr = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    Image.fromarray(arr).save(src, format="JPEG", quality=90,
                              icc_profile=b"\0" * (2 * 1024 * 1024))
    with Image.open(src) as probe:
        probe.load()
        assert len(probe.info["icc_profile"]) == 2 * 1024 * 1024, (
            "fixture must reproduce a source Pillow reads but cannot "
            "round-trip through PNG")

    dst = tmp_path / "dst.png"
    normalize_image(str(src), str(dst))
    with Image.open(dst) as out:      # must not raise
        out.load()
        assert out.size == (400, 300)
    assert b"iCCP" not in dst.read_bytes()


def test_save_png_would_have_written_an_unreadable_file_without_the_strip(tmp_path):
    """The mutation: keep the profile, and the output stops being readable.
    Without this the test above passes for any implementation that happens not
    to write a profile, including one that never could."""
    arr = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    im.info["icc_profile"] = b"\0" * (2 * 1024 * 1024)
    kept = tmp_path / "kept.png"
    im.save(kept, format="PNG", optimize=False)      # no strip
    assert b"iCCP" in kept.read_bytes()
    with pytest.raises(ValueError, match="MAX_TEXT_CHUNK"):
        with Image.open(kept) as x:
            x.load()


def test_stripping_the_profile_changes_no_pixel(tmp_path):
    """An ICC profile is metadata, not samples. Pillow does not apply one on
    decode and `convert("RGB")` does not either, so removing it cannot move a
    value -- which is what makes this safe to do to a forensics corpus."""
    from aigcdet.data.normalize import _without_stripped_metadata

    arr = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    im.info["icc_profile"] = b"\0" * 4096
    kept, stripped = tmp_path / "kept.png", tmp_path / "stripped.png"
    im.save(kept, format="PNG", optimize=False)
    im.save(stripped, format="PNG", optimize=False, **_without_stripped_metadata(im))

    assert np.array_equal(np.asarray(Image.open(kept).convert("RGB")),
                          np.asarray(Image.open(stripped).convert("RGB")))
    assert np.array_equal(np.asarray(Image.open(stripped).convert("RGB")), arr)


def test_save_png_strips_the_profile_too(tmp_path):
    """`save_png` is the acquisition-side writer. A second, differently
    opinionated copy of this policy is how the classes start differing by
    container -- which is what the module docstring says it exists to
    prevent."""
    arr = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    im.info["icc_profile"] = b"\0" * (16 * 1024 * 1024)
    dst = tmp_path / "out.png"
    save_png(im, str(dst))
    with Image.open(dst) as out:      # must not raise
        out.load()
    assert b"iCCP" not in dst.read_bytes()
