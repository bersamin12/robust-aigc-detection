import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import public_detector_probe as pdp  # noqa: E402


def _gradient(h, w):
    r = np.linspace(0, 255, h)[:, None].repeat(w, 1)
    g = np.linspace(0, 255, w)[None, :].repeat(h, 0)
    return np.stack([r, g, np.full((h, w), 128.0)], -1).astype(np.uint8)


@pytest.mark.parametrize("hw", [(500, 700), (100, 150), (224, 224), (223, 300)])
def test_center_crop_pad_is_torchvision_center_crop_on_pil(hw):
    from PIL import Image
    from torchvision import transforms
    img = _gradient(*hw)
    want = np.asarray(transforms.CenterCrop(224)(Image.fromarray(img)))
    assert np.array_equal(pdp.center_crop_pad(img, 224), want)


@pytest.mark.parametrize("arm", pdp.ARMS)
@pytest.mark.parametrize("hw", [(100, 150), (480, 640), (3000, 2000)])
def test_univfd_views_are_224_squares(arm, hw):
    assert pdp.univfd_view(_gradient(*hw), arm).shape == (224, 224, 3)


def test_univfd_train_fov_covers_its_training_share_of_the_short_side():
    # 1024-row image: short side to 256, then 224 rows = 87.5% of the ramp.
    v = pdp.univfd_view(_gradient(1024, 1024), "train_fov")
    span = float(v[-1, :, 0].mean()) - float(v[0, :, 0].mean())
    assert 0.85 * 255 < span < 0.90 * 255


def test_univfd_native_is_a_native_window():
    # 2000-row image: the 224-row native window spans ~11% of the ramp.
    v = pdp.univfd_view(_gradient(2000, 2000), "native")
    span = float(v[-1, :, 0].mean()) - float(v[0, :, 0].mean())
    assert span < 0.13 * 255


def test_bfree_views():
    img = _gradient(1500, 2000)
    assert pdp.bfree_view(img, "native") is img
    assert min(pdp.bfree_view(img, "train_fov").shape[:2]) == 512
    assert pdp.bfree_view(img, "squish").shape == (504, 504, 3)


def test_replicate_wrap_tiles_a_small_grid():
    x = torch.arange(6.).view(1, 1, 2, 3)
    y = pdp.replicate_wrap(x, (5, 4))
    assert y.shape == (1, 1, 5, 4)
    assert torch.equal(y[0, 0, 2:4, :3], x[0, 0])


def test_to_tensor_normalises_per_channel():
    t = pdp.to_tensor(np.full((2, 2, 3), 255, np.uint8), pdp.IMAGENET_NORM)
    want = (1 - 0.485) / 0.229
    assert t.shape == (3, 2, 2) and abs(float(t[0, 0, 0]) - want) < 1e-5


def test_ntire_test_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--detector", "univfd", "--splits", "ntire_test"])
    with pytest.raises(SystemExit, match="sealed"):
        pdp.main()


def _metrics(native, train_fov, squish, small_gain, large_gain):
    def arm(r, gs, gl):
        return {"robust_auc": r, "by_bucket": {"80-336": {"robust_auc": 0.7 + gs},
                                               "752-4016": {"robust_auc": 0.7 + gl}}}
    return {"native": arm(native, 0, 0), "train_fov": arm(train_fov, small_gain, large_gain),
            "squish": arm(squish, small_gain, large_gain)}


def _boot(diff, lo, hi):
    d = {"robust_auc": {"diff": diff, "ci": [lo, hi]}}
    return {"vs_baseline": {"squish": d, "train_fov": d}}


def test_verdict_passes_a_clear_gain_that_grows_with_size():
    v = pdp.verdict(_boot(0.05, 0.02, 0.08), _metrics(0.70, 0.75, 0.75, 0.01, 0.08))
    assert v["interpretable"] and v["primary"]["passed"] and v["secondary"]["passed"]
    assert v["mechanism_squish"]["passed"] and v["mechanism_train_fov"]["passed"]


def test_verdict_fails_an_interval_touching_zero_and_a_flat_gain():
    v = pdp.verdict(_boot(0.02, -0.001, 0.04), _metrics(0.70, 0.72, 0.72, 0.03, 0.02))
    assert not v["primary"]["passed"] and not v["mechanism_squish"]["passed"]


def test_verdict_applies_the_floor():
    v = pdp.verdict(_boot(0.05, 0.02, 0.08), _metrics(0.52, 0.58, 0.59, 0.0, 0.1))
    assert not v["interpretable"] and v["best_robust_auc"] == 0.59
