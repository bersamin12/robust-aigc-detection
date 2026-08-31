"""The two bugs that killed the first attempt at this corpus were both here.

`docs/03` §1: 6 of 6 smoke pairs leaked the label because every fake came out a
multiple of 8 while its real kept its native size, and the confound gate read
`jpeg_quality` AUC 0.0000 -- perfect inverse separation -- because the fakes'
DCT grids sat at a different phase from their reals'. Neither is subtle once
written down; both survived because the code was in a notebook cell.
"""
import pytest
from PIL import Image

from aigcdet.generate.geometry import (MCU_ALIGN, box_mask, crop_box, order_key,
                                       seed_for, shard_block)


@pytest.mark.parametrize("w,h", [(409, 640), (426, 640), (427, 639), (1200, 1800),
                                 (640, 480), (321, 321), (2400, 3600), (500, 333)])
def test_box_and_offset_are_mcu_aligned(w, h):
    """Size AND offset. An aligned size with an unaligned offset still puts
    each fake's DCT grid at its own phase, which is the AUC 0.0000 bug."""
    l, t, r, b = crop_box(w, h)
    assert l % MCU_ALIGN == 0 and t % MCU_ALIGN == 0
    assert (r - l) % MCU_ALIGN == 0 and (b - t) % MCU_ALIGN == 0
    assert 0 <= l and 0 <= t and r <= w and b <= h


@pytest.mark.parametrize("w,h", [(409, 640), (1200, 1800), (640, 480), (500, 333)])
def test_aspect_is_preserved_within_one_mcu(w, h):
    """The cap must scale both sides by one factor. Clamping each side
    independently turns a portrait into a square and moves the aspect
    distribution of the generated class away from its reals."""
    l, t, r, b = crop_box(w, h)
    # The error is multiplicative in the ratio, not additive, and it is set by
    # the OUTPUT side: trimming <=15px off a side leaves a relative error of at
    # most mcu/side, and after the max_side cap that side can be far shorter
    # than the input's.
    assert abs(((r - l) / (b - t)) / (w / h) - 1.0) < MCU_ALIGN / min(r - l, b - t)


def test_crop_never_grows_the_image():
    for w, h in [(320, 400), (4000, 6000), (1024, 1024)]:
        l, t, r, b = crop_box(w, h)
        assert r - l <= w and b - t <= h


def test_max_side_caps_by_cropping_not_resizing():
    l, t, r, b = crop_box(2400, 3600, max_side=1024)
    assert max(r - l, b - t) <= 1024


def test_degenerate_sizes_raise_rather_than_return_empty():
    with pytest.raises(ValueError):
        crop_box(8, 8)
    with pytest.raises(ValueError):
        crop_box(0, 100)


def test_seed_is_content_addressed_not_positional():
    """A counter makes the pixels a function of where the image fell in the
    run, so a rerun with one dropped file regenerates a different corpus."""
    assert seed_for("abc", 7) == seed_for("abc", 7)
    assert seed_for("abc", 7) != seed_for("abd", 7)
    assert seed_for("abc", 7) != seed_for("abc", 8)
    assert 0 <= seed_for("abc", 7) < 2 ** 56


def test_order_key_is_stable_and_spreads():
    keys = [order_key(f"img{i:05d}", 20260830) for i in range(500)]
    assert len(set(keys)) == 500
    assert keys == [order_key(f"img{i:05d}", 20260830) for i in range(500)]


@pytest.mark.parametrize("n,k", [(10, 4), (100, 7), (54624, 8), (3, 3), (5, 1)])
def test_shards_partition_exactly(n, k):
    """Disjoint and complete: two workers must never draw the same real, and
    no real may fall between two shards."""
    blocks = [shard_block(n, i, k) for i in range(k)]
    covered = [i for s, e in blocks for i in range(s, e)]
    assert covered == list(range(n))
    assert max(e - s for s, e in blocks) - min(e - s for s, e in blocks) <= 1


def test_shard_out_of_range_raises():
    with pytest.raises(ValueError):
        shard_block(10, 4, 4)


def test_box_mask_is_deterministic_and_covers_a_middling_area():
    import numpy as np
    m1 = np.asarray(box_mask("abc", 416, 640, 7))
    assert np.array_equal(m1, np.asarray(box_mask("abc", 416, 640, 7)))
    assert m1.shape == (640, 416)
    assert 0.2 < (m1 > 0).mean() < 0.5
    assert not np.array_equal(m1, np.asarray(box_mask("abd", 416, 640, 7)))
