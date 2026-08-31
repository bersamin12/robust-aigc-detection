"""Encoder parity: the mitigation `docs/02` §1 calls mandatory.

The Open Images reals are thumbnails -- already twice-compressed JPEGs -- and
`docs/low_level_confounds.md` measured `jpeg_quality` alone separating the
frozen corpus at AUC 0.5532. A fake saved at a fixed quality hands a detector
the compression history as a free label, so the fake is saved through its
real's own 64 quantisation integers and its real's subsampling.
"""
import io

import numpy as np
import pytest
from PIL import Image

from aigcdet.generate.encode import (assert_parity, quality_of,
                                     reproducible_encoder, save_matched,
                                     source_encoder)


def _noise(w=64, h=96, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(a).convert(mode)


@pytest.mark.parametrize("q", [30, 50, 75, 90, 95])
def test_quality_estimate_round_trips_through_pil(tmp_path, q):
    """Also pins the table ORDER. A JPEG file stores quantisation tables in
    zigzag order but PIL de-zigzags them; comparing against a zigzag baseline
    scrambled the estimate silently -- it read q=50 as 42.9 -- rather than
    raising."""
    p = tmp_path / f"q{q}.jpg"
    _noise().save(p, "JPEG", quality=q)
    assert abs(source_encoder(p)["jpeg_quality"] - q) <= 1.5


def test_quality_of_rejects_a_wrong_length_table():
    with pytest.raises(ValueError):
        quality_of([16] * 63)


def test_fake_inherits_the_reals_quantisation_tables(tmp_path):
    real = tmp_path / "real.jpg"
    _noise(seed=1).save(real, "JPEG", quality=83, subsampling=2)
    enc = source_encoder(real)
    fake = tmp_path / "fake.jpg"
    save_matched(_noise(seed=2), fake, enc)
    got = source_encoder(fake)
    assert got["qtables"] == enc["qtables"]
    assert got["subsampling"] == enc["subsampling"]
    assert got["size"] == enc["size"]


@pytest.mark.parametrize("sub", [0, 1, 2])
def test_parity_holds_across_every_subsampling_we_admit(tmp_path, sub):
    real = tmp_path / f"r{sub}.jpg"
    _noise(seed=3).save(real, "JPEG", quality=90, subsampling=sub)
    fake = tmp_path / f"f{sub}.jpg"
    save_matched(_noise(seed=4), fake, source_encoder(real))
    assert_parity(real, fake)


def test_assert_parity_catches_a_dimension_mismatch(tmp_path):
    """The exact leak `docs/03` §1 recorded: 6 of 6 fakes were a multiple of 8
    while their reals kept their native size."""
    real = tmp_path / "real.jpg"
    _noise(w=63, h=97, seed=5).save(real, "JPEG", quality=90)
    enc = source_encoder(real)
    fake = tmp_path / "fake.jpg"
    save_matched(_noise(w=64, h=96, seed=6), fake, enc)
    with pytest.raises(RuntimeError, match="size"):
        assert_parity(real, fake)


def test_assert_parity_catches_a_fixed_quality_save(tmp_path):
    real = tmp_path / "real.jpg"
    _noise(seed=7).save(real, "JPEG", quality=72)
    fake = tmp_path / "fake.jpg"
    _noise(seed=8).save(fake, "JPEG", quality=95)      # the naive save path
    with pytest.raises(RuntimeError, match="qtables"):
        assert_parity(real, fake)


def test_grayscale_and_odd_sampling_reals_are_not_reproducible(tmp_path):
    """0.6% of the pool is grayscale and 7.8% carries a chroma layout PIL
    cannot name; both would put a difference between the classes that has
    nothing to do with the generator."""
    gray = tmp_path / "gray.jpg"
    _noise(seed=9, mode="L").save(gray, "JPEG", quality=90)
    assert not reproducible_encoder(source_encoder(gray))

    rgb = tmp_path / "rgb.jpg"
    _noise(seed=10).save(rgb, "JPEG", quality=90, subsampling=0)
    assert reproducible_encoder(source_encoder(rgb))


def test_source_encoder_refuses_a_non_jpeg(tmp_path):
    p = tmp_path / "x.png"
    _noise().save(p, "PNG")
    with pytest.raises(ValueError, match="JPEG"):
        source_encoder(p)
