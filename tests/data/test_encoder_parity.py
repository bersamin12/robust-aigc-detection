import io

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from aigcdet.data.encoder_parity import (
    EncodingProfile, ParityError, conform, crop_to_aspect, read_profile,
    save_matched, save_matched_to_real,
)
from aigcdet.features.proxies import estimate_jpeg_quality


def _photo(w, h, seed=0):
    """Pink-ish noise: `proxies` documents that near-white noise carries no
    usable blockiness signal, so a flat random array would make the parity
    assertions below pass for the wrong reason."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (max(1, h // 8), max(1, w // 8), 3), dtype=np.uint8)
    return Image.fromarray(small).resize((w, h), Image.BICUBIC)


def _real(tmp_path, name="real.jpg", size=(400, 600), quality=71, subsampling=2,
          progressive=False, mode="RGB"):
    im = _photo(*size)
    if mode == "L":
        im = im.convert("L")
    p = tmp_path / name
    im.save(p, format="JPEG", quality=quality, subsampling=subsampling,
            progressive=progressive)
    return str(p)


# --- read_profile -----------------------------------------------------------

def test_reads_geometry_tables_and_subsampling_off_a_real(tmp_path):
    prof = read_profile(_real(tmp_path, size=(400, 600), quality=71, subsampling=2))
    assert (prof.width, prof.height) == (400, 600)
    assert prof.short_side == 400
    assert prof.qtables and all(len(t) == 64 for t in prof.qtables.values())
    assert prof.subsampling == 2
    assert prof.progressive is False


def test_refuses_a_png_rather_than_defaulting(tmp_path):
    """A silent fallback would encode fakes to a house default while reals kept
    their own -- the exact confound this module removes, via the error path."""
    p = tmp_path / "not_a_jpeg.png"
    _photo(64, 64).save(p, format="PNG")
    with pytest.raises(ParityError, match="no quantisation tables"):
        read_profile(str(p))


def test_refuses_a_missing_file(tmp_path):
    with pytest.raises(ParityError, match="unreadable real"):
        read_profile(str(tmp_path / "absent.jpg"))


def test_progressive_flag_is_carried(tmp_path):
    prof = read_profile(_real(tmp_path, "prog.jpg", progressive=True))
    assert prof.progressive is True


# --- crop_to_aspect ---------------------------------------------------------

def test_crop_to_aspect_trims_the_wider_axis_only():
    im = _photo(1024, 1024)
    out = crop_to_aspect(im, 400 / 600)
    assert out.height == 1024                      # tall enough already
    assert out.width == round(1024 * 400 / 600)


def test_crop_to_aspect_trims_height_for_a_tall_target():
    im = _photo(1024, 1024)
    out = crop_to_aspect(im, 3 / 2)
    assert out.width == 1024
    assert out.height == round(1024 / (3 / 2))


def test_crop_to_aspect_is_a_no_op_at_the_target_aspect():
    im = _photo(600, 400)
    assert crop_to_aspect(im, 1.5) is im


def test_crop_is_centred():
    """A corner crop would systematically favour one region of every generated
    frame, which is a content bias rather than an encoding one."""
    im = _photo(1000, 1000)
    out = crop_to_aspect(im, 0.5)
    expected = im.crop((250, 0, 750, 1000))
    assert np.array_equal(np.asarray(out), np.asarray(expected))


# --- conform ----------------------------------------------------------------

def test_conform_lands_on_the_profiles_exact_size(tmp_path):
    prof = read_profile(_real(tmp_path, size=(400, 600)))
    out = conform(_photo(1024, 1024), prof)
    assert out.size == (400, 600)


def test_conform_refuses_to_upscale(tmp_path):
    """An upscale means the pairing is wrong; interpolating would fabricate the
    forensic evidence the corpus exists to measure."""
    prof = read_profile(_real(tmp_path, size=(400, 600)))
    with pytest.raises(ParityError, match="would upscale"):
        conform(_photo(200, 300), prof)


def test_conform_matches_a_greyscale_real(tmp_path):
    prof = read_profile(_real(tmp_path, "grey.jpg", size=(400, 600), mode="L"))
    assert prof.mode == "L"
    assert conform(_photo(1024, 1024), prof).mode == "L"


# --- save_matched: the property the gate depends on -------------------------

def test_saved_fake_carries_the_reals_exact_quantisation_tables(tmp_path):
    """The whole point. Copying the tables rather than re-deriving a quality
    integer is what makes the two classes identical in `jpeg_quality` by
    construction instead of approximately."""
    real_path = _real(tmp_path, size=(400, 600), quality=71)
    prof = read_profile(real_path)
    dst = str(tmp_path / "fake.jpg")
    save_matched(_photo(1024, 1024, seed=9), dst, prof)

    with Image.open(real_path) as r, Image.open(dst) as f:
        assert {k: list(v) for k, v in f.quantization.items()} == prof.qtables
        assert f.quantization == r.quantization
        assert f.size == r.size
        assert f.format == "JPEG"


def test_exact_jpeg_quality_branch_returns_the_same_value_for_both(tmp_path):
    """`gate_confounds.py` scores `jpeg_quality` via
    `proxies.estimate_jpeg_quality`, whose exact branch reads the quantisation
    table. Identical tables mean an identical reading, so the proxy carries
    exactly zero signal about the label."""
    real_path = _real(tmp_path, size=(400, 600), quality=63)
    dst = str(tmp_path / "fake.jpg")
    save_matched_to_real(_photo(1024, 1024, seed=3), dst, real_path)

    with Image.open(real_path) as r:
        q_real = estimate_jpeg_quality(np.asarray(r.convert("RGB")), real_path)
    with Image.open(dst) as f:
        q_fake = estimate_jpeg_quality(np.asarray(f.convert("RGB")), dst)
    assert q_real == q_fake


@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_subsampling_is_reproduced(tmp_path, subsampling):
    real_path = _real(tmp_path, f"s{subsampling}.jpg", subsampling=subsampling)
    dst = str(tmp_path / f"fake{subsampling}.jpg")
    save_matched_to_real(_photo(1024, 1024), dst, real_path)
    with Image.open(dst) as f:
        assert JpegImagePlugin.get_sampling(f) == subsampling


def test_progressive_is_reproduced(tmp_path):
    real_path = _real(tmp_path, "prog.jpg", progressive=True)
    dst = str(tmp_path / "fake_prog.jpg")
    save_matched_to_real(_photo(1024, 1024), dst, real_path)
    with Image.open(dst) as f:
        assert bool(f.info.get("progression", False)) is True


def test_greyscale_real_yields_a_single_channel_fake(tmp_path):
    real_path = _real(tmp_path, "grey.jpg", mode="L")
    dst = str(tmp_path / "fake_grey.jpg")
    save_matched_to_real(_photo(1024, 1024), dst, real_path)
    with Image.open(dst) as f:
        assert f.mode == "L"


def test_no_part_file_survives_a_successful_write(tmp_path):
    real_path = _real(tmp_path)
    dst = str(tmp_path / "fake.jpg")
    save_matched_to_real(_photo(1024, 1024), dst, real_path)
    assert not (tmp_path / "fake.jpg.part").exists()


def test_failed_write_leaves_no_part_file(tmp_path, monkeypatch):
    """Acquisition resumes on `os.path.exists(dst)`, so a truncated leftover
    that resume treats as done is worse than no file at all."""
    prof = read_profile(_real(tmp_path))
    dst = str(tmp_path / "boom.jpg")

    def explode(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", explode)
    with pytest.raises(OSError):
        save_matched(_photo(1024, 1024), dst, prof)
    assert not (tmp_path / "boom.jpg.part").exists()
    assert not (tmp_path / "boom.jpg").exists()


def test_save_matched_creates_missing_directories(tmp_path):
    real_path = _real(tmp_path)
    dst = str(tmp_path / "nested" / "deeper" / "fake.jpg")
    save_matched_to_real(_photo(1024, 1024), dst, real_path)
    assert Image.open(dst).size == (400, 600)


def test_returned_profile_records_what_was_applied(tmp_path):
    """The parity claim is only auditable if the settings are written down."""
    real_path = _real(tmp_path, size=(400, 600), quality=55, subsampling=1)
    prof = save_matched_to_real(_photo(1024, 1024), str(tmp_path / "f.jpg"), real_path)
    assert isinstance(prof, EncodingProfile)
    assert (prof.width, prof.height, prof.subsampling) == (400, 600, 1)


# --- the end-to-end property the brief actually gates on --------------------

def test_pixel_fallback_readings_converge_after_parity(tmp_path):
    """Beyond the exact branch: a PNG-stored fake reads far from its real on
    the pixel-only blockiness estimate, and parity closes most of that gap.

    This is the residual the module docstring refuses to guess at -- a real is
    a re-encode of an already-compressed original, so some double-quantisation
    difference survives. The assertion is that parity shrinks the gap by a
    wide margin, not that it reaches zero.
    """
    real_path = _real(tmp_path, size=(400, 600), quality=45)
    with Image.open(real_path) as r:
        q_real = estimate_jpeg_quality(np.asarray(r.convert("RGB")))

    fake = _photo(1024, 1024, seed=11)

    # As a PNG would reach the corpus today: no JPEG history at all.
    buf = io.BytesIO()
    fake.resize((400, 600), Image.LANCZOS).save(buf, format="PNG")
    buf.seek(0)
    with Image.open(buf) as p:
        q_png = estimate_jpeg_quality(np.asarray(p.convert("RGB")))

    dst = str(tmp_path / "parity.jpg")
    save_matched_to_real(fake, dst, real_path)
    with Image.open(dst) as f:
        q_parity = estimate_jpeg_quality(np.asarray(f.convert("RGB")))

    assert abs(q_parity - q_real) < abs(q_png - q_real)
