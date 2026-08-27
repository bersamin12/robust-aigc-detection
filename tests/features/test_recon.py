import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.features.bank import RECON_DIM, BankWriter, FeatureBank
from aigcdet.features.recon import (
    RECON_FEATURE_NAMES,
    _radial_bands,
    attach_recon_to_bank,
    error_map,
    load_recon_models,
    native_center_crop,
    recon_features,
)


# --------------------------------------------------------------------------
# Stub VAE / LPIPS -- same call interface as diffusers.AutoencoderKL and
# lpips.LPIPS, so recon_features/error_map/attach_recon_to_bank get exercised
# without ever importing diffusers or lpips or touching a GPU.
# --------------------------------------------------------------------------

class _LatentDist:
    def __init__(self, x):
        self._x = x

    def mode(self):
        return self._x


class _EncoderOutput:
    def __init__(self, x):
        self.latent_dist = _LatentDist(x)


class _DecoderOutput:
    def __init__(self, sample):
        self.sample = sample


class _StubVAE:
    """Fake "reconstruction": subtracts a fixed, known offset. Lets tests
    predict the exact error map analytically instead of trusting a real
    autoencoder's output."""

    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def encode(self, x):
        return _EncoderOutput(x)

    def decode(self, latent):
        return _DecoderOutput(latent - self.offset)


def _stub_lpips(x, rec):
    return (x - rec).abs().mean()


# --------------------------------------------------------------------------
# Contract: RECON_FEATURE_NAMES
# --------------------------------------------------------------------------

def test_feature_names_match_the_declared_width():
    assert len(RECON_FEATURE_NAMES) == RECON_DIM


def test_feature_names_are_the_exact_fixed_order():
    # Plan 3's AEROBLADE baseline reads RECON_FEATURE_NAMES.index("l1") and
    # the bank stores this vector positionally -- pin the whole order, not
    # just its length, so a reorder is caught here rather than downstream.
    assert RECON_FEATURE_NAMES == (
        "l1", "lpips",
        "err_mean", "err_std", "err_p90", "err_max",
        "spec_b0", "spec_b1", "spec_b2", "spec_b3",
        "spec_mid_ratio", "spec_high_ratio",
    )
    assert len(set(RECON_FEATURE_NAMES)) == len(RECON_FEATURE_NAMES)  # no dupes


# --------------------------------------------------------------------------
# native_center_crop
# --------------------------------------------------------------------------

def test_native_center_crop_does_not_resize():
    img = np.random.default_rng(0).integers(0, 256, (512, 700, 3), dtype=np.uint8)
    out = native_center_crop(img, 256)
    assert out.shape == (256, 256, 3)
    # must be an exact slice of the original, never an interpolation
    top, left = (512 - 256) // 2, (700 - 256) // 2
    np.testing.assert_array_equal(out, img[top:top + 256, left:left + 256])


def test_native_center_crop_pads_a_small_image_instead_of_upscaling():
    img = np.random.default_rng(0).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    out = native_center_crop(img, 256)
    assert out.shape == (256, 256, 3)
    top, left = (256 - 100) // 2, (256 - 120) // 2
    # the source pixels must appear untouched (padded, not resampled) at the
    # centre of the output
    np.testing.assert_array_equal(out[top:top + 100, left:left + 120], img)
    # reflect-padding only ever reuses source pixel values -- it never
    # invents a new one (e.g. zero-padding would introduce black borders not
    # present in the source)
    assert np.isin(out, img).all()


def test_native_center_crop_pads_only_the_deficient_dimension():
    # height is smaller than the crop, width is larger: only height should
    # be padded, width must still be an exact (unresized) slice.
    img = np.random.default_rng(1).integers(0, 256, (100, 700, 3), dtype=np.uint8)
    out = native_center_crop(img, 256)
    assert out.shape == (256, 256, 3)
    top = (256 - 100) // 2
    left = (700 - 256) // 2
    np.testing.assert_array_equal(out[top:top + 100], img[:, left:left + 256])


# --------------------------------------------------------------------------
# _radial_bands
# --------------------------------------------------------------------------

def test_radial_bands_returns_four_nonnegative_finite_values():
    err = np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32)
    bands = _radial_bands(err)
    assert bands.shape == (4,)
    assert np.isfinite(bands).all()
    assert (bands >= 0).all()


def test_radial_bands_puts_more_relative_energy_in_the_top_band_for_high_frequency_content():
    h = w = 256
    smooth = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))  # low-frequency gradient
    yy, xx = np.mgrid[:h, :w]
    checker = ((yy + xx) % 2).astype(np.float32)  # highest possible spatial frequency at this res

    smooth_bands = _radial_bands(smooth)
    checker_bands = _radial_bands(checker)

    # compare distribution shape (each map's own top-band fraction), not
    # raw magnitude, since the two synthetic maps have different total power
    smooth_top_frac = smooth_bands[3] / smooth_bands.sum()
    checker_top_frac = checker_bands[3] / checker_bands.sum()
    assert checker_top_frac > smooth_top_frac


# --------------------------------------------------------------------------
# recon_features / error_map -- construction, ordering, and values
# --------------------------------------------------------------------------

def test_recon_features_are_all_zero_for_a_perfect_roundtrip():
    img = np.random.default_rng(2).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    vae = _StubVAE(offset=0.0)
    v = recon_features(img, vae, _stub_lpips, device="cpu")
    assert v.shape == (RECON_DIM,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()
    np.testing.assert_allclose(v, np.zeros(RECON_DIM, dtype=np.float32), atol=1e-3)


def test_recon_features_values_match_a_known_constant_offset():
    # A flat grey crop reconstructed with a fixed offset gives an error map
    # that is the same constant everywhere, so every entry of the feature
    # vector is analytically predictable -- this pins both the *values* and
    # the *order* the brief's RECON_FEATURE_NAMES promises.
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    offset = 0.05
    vae = _StubVAE(offset=offset)
    v = recon_features(img, vae, _stub_lpips, device="cpu")

    by_name = dict(zip(RECON_FEATURE_NAMES, v))
    for name in ("l1", "lpips", "err_mean", "err_p90", "err_max"):
        assert by_name[name] == pytest.approx(offset, abs=5e-3), name
    assert by_name["err_std"] == pytest.approx(0.0, abs=5e-3)
    # a spatially constant error map carries no AC power in any band, and
    # the log1p(0) floor -> 0 flows through to both ratios
    for name in ("spec_b0", "spec_b1", "spec_b2", "spec_b3",
                 "spec_mid_ratio", "spec_high_ratio"):
        assert by_name[name] == pytest.approx(0.0, abs=1e-3), name


def test_error_map_matches_the_per_pixel_error_recon_features_derives_stats_from():
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    offset = 0.05
    vae = _StubVAE(offset=offset)
    err = error_map(img, vae, device="cpu")
    assert err.shape == (256, 256)
    np.testing.assert_allclose(err, offset, atol=5e-3)


def test_recon_features_distinguish_different_images():
    # A regression guard against a stub-like implementation that ignores its
    # input: two different crops through a non-identity VAE must not collapse
    # to the same feature vector.
    rng = np.random.default_rng(3)
    img_a = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    img_b = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    vae = _StubVAE(offset=0.05)
    va = recon_features(img_a, vae, _stub_lpips, device="cpu")
    vb = recon_features(img_b, vae, _stub_lpips, device="cpu")
    assert not np.allclose(va, vb)


# --------------------------------------------------------------------------
# attach_recon_to_bank
# --------------------------------------------------------------------------

def _make_bank(tmp_path, monkeypatch, images):
    """Build a real, tiny FeatureBank (via BankWriter, exactly as Stage A
    does) with one clean view and one jitter-augmented view per image, then
    open it. `images` is a list of (h, w) uint8 RGB arrays already on disk."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths = []
    for i, arr in enumerate(images):
        p = img_dir / f"{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))

    n_views, dim = 2, 4  # dim is arbitrary -- recon.py never reads bank.feats
    out_dir = tmp_path / "bank"
    writer = BankWriter(str(out_dir), len(images), n_views, dim, "stub", seed=0)
    clean_recipe = json.dumps([])
    jitter_recipe = json.dumps(
        [{"name": "jitter", "params": {"brightness": 0.2, "contrast": 0.0, "saturation": 0.0}}])
    for i in range(len(images)):
        writer.write_image(
            i,
            {"path": paths[i], "label": i % 2, "generator": "none",
             "source": "test", "split": "train"},
            feats=np.zeros((n_views, dim), dtype=np.float32),
            presence=np.zeros((n_views, 6), dtype=np.float32),
            severity=np.zeros((n_views, 6), dtype=np.float32),
            proxies=np.zeros((n_views, 3), dtype=np.float32),
            recipes=[clean_recipe, jitter_recipe],
        )
    writer.close()

    monkeypatch.setattr(
        "aigcdet.features.recon.load_recon_models",
        lambda device: (_StubVAE(offset=0.05), _stub_lpips))

    return FeatureBank.open(str(out_dir))


def test_attach_recon_to_bank_writes_the_right_shape_for_every_view(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    images = [rng.integers(0, 256, (300, 300, 3), dtype=np.uint8) for _ in range(2)]
    bank = _make_bank(tmp_path, monkeypatch, images)

    attach_recon_to_bank(bank, bank.meta, device="cpu")

    assert bank.recon is not None
    assert bank.recon.shape == (2, 2, RECON_DIM)
    assert np.isfinite(np.asarray(bank.recon)).all()


def test_attach_recon_to_bank_reads_each_images_own_file_and_own_recipe(tmp_path, monkeypatch):
    # Image 0 and image 1 must produce different features (different source
    # pixels), and within one image, view 0 (clean) and view 1 (jittered)
    # must also differ -- both guard against attach_recon_to_bank silently
    # reusing the wrong image or the wrong recipe for a row.
    rng = np.random.default_rng(5)
    images = [rng.integers(0, 256, (300, 300, 3), dtype=np.uint8) for _ in range(2)]
    bank = _make_bank(tmp_path, monkeypatch, images)

    attach_recon_to_bank(bank, bank.meta, device="cpu")
    recon = np.asarray(bank.recon)

    assert not np.allclose(recon[0], recon[1])
    assert not np.allclose(recon[0, 0], recon[0, 1])
    assert not np.allclose(recon[1, 0], recon[1, 1])


def test_attach_recon_to_bank_is_deterministic(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    images = [rng.integers(0, 256, (300, 300, 3), dtype=np.uint8) for _ in range(2)]
    bank = _make_bank(tmp_path, monkeypatch, images)

    attach_recon_to_bank(bank, bank.meta, device="cpu")
    first = np.asarray(bank.recon).copy()

    attach_recon_to_bank(bank, bank.meta, device="cpu")
    second = np.asarray(bank.recon)

    np.testing.assert_array_equal(first, second)


def test_attach_recon_to_bank_rejects_a_misaligned_manifest(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (300, 300, 3), dtype=np.uint8) for _ in range(2)]
    bank = _make_bank(tmp_path, monkeypatch, images)

    wrong_manifest = pd.DataFrame({"path": ["not/a/real/path.png"]})
    with pytest.raises(ValueError):
        attach_recon_to_bank(bank, wrong_manifest, device="cpu")
    assert bank.recon is None  # rejected before anything was written


# --------------------------------------------------------------------------
# load_recon_models -- import isolation, no GPU/download involved
# --------------------------------------------------------------------------

def test_load_recon_models_defers_the_diffusers_and_lpips_imports():
    # Neither diffusers nor lpips is installed in this environment (and must
    # not be -- see project-constraints.md). recon.py must therefore still be
    # importable (every test above already proves that); only *calling*
    # load_recon_models should require them, and fail cleanly when they are
    # missing rather than doing a partial, silent load.
    with pytest.raises(ModuleNotFoundError):
        load_recon_models("cpu")


# --------------------------------------------------------------------------
# The real, GPU-marked test. Guards on free VRAM (not availability): at the
# ~885 MiB free this project's shared A4500 typically has,
# torch.cuda.is_available() returns True, and attempting to load a real SD
# 1.5 VAE + LPIPS would download multiple GB and then OOM rather than skip.
# --------------------------------------------------------------------------

_MIN_FREE_BYTES = 4 * 1024**3  # 4 GB headroom for the VAE + LPIPS(AlexNet)


def _skip_unless_gpu_has_headroom():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if free_bytes < _MIN_FREE_BYTES:
        pytest.skip(
            f"only {free_bytes / 1024**2:.0f} MiB free on GPU, need at least "
            f"{_MIN_FREE_BYTES / 1024**2:.0f} MiB to load the SD 1.5 VAE + LPIPS")


@pytest.mark.gpu
def test_recon_features_are_finite_and_lower_error_for_a_vae_roundtrip():
    _skip_unless_gpu_has_headroom()

    vae, lp = load_recon_models("cuda")
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    v = recon_features(photo, vae, lp, "cuda")
    assert v.shape == (RECON_DIM,) and np.isfinite(v).all()
