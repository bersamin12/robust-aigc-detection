import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import ntire_crop_probe as probe  # noqa: E402

POLICY = SimpleNamespace(nominal_side=224)


def _gradient(h, w):
    """Smooth, position-coded image: row index in red, column index in green."""
    r = np.linspace(0, 255, h)[:, None].repeat(w, 1)
    g = np.linspace(0, 255, w)[None, :].repeat(h, 0)
    return np.stack([r, g, np.full((h, w), 128.0)], -1).astype(np.uint8)


@pytest.mark.parametrize("arm", ["short_center", "letterbox"])
@pytest.mark.parametrize("hw", [(100, 150), (224, 224), (480, 640), (3000, 2000)])
def test_aspect_preserving_arms_give_one_nominal_square(arm, hw):
    (v,) = probe.ARMS[arm](_gradient(*hw), POLICY)
    assert v.shape == (224, 224, 3) and v.dtype == np.uint8


def test_short_center_keeps_the_whole_short_axis_and_the_middle_of_the_long():
    (v,) = probe.views_short_center(_gradient(400, 800), POLICY)
    # Rows: the full red ramp. Columns: the middle half of the green ramp.
    assert v[0, :, 0].mean() < 5 and v[-1, :, 0].mean() > 250
    assert 55 < v[:, 0, 1].mean() < 75 and 180 < v[:, -1, 1].mean() < 200


def test_letterbox_keeps_the_whole_frame_and_pads_the_short_axis_black():
    (v,) = probe.views_letterbox(_gradient(400, 800), POLICY)
    # 400x800 -> 112x224, centred: 56 black rows above and below.
    assert (v[:56] == 0).all() and (v[-56:] == 0).all()
    content = v[56:168]
    assert content[:, 0, 1].mean() < 5 and content[:, -1, 1].mean() > 250


def test_short_center_on_a_square_image_is_squish():
    img = _gradient(600, 600)
    (a,) = probe.views_short_center(img, POLICY)
    (b,) = probe.views_squish(img, POLICY)
    assert np.array_equal(a, b)
