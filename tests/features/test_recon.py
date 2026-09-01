import json

import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.features.bank import RECON_DIM, BankWriter, FeatureBank
from aigcdet.augment.canonical import canonicalise
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
        lambda device, kind='kl': (_StubVAE(offset=0.05), _stub_lpips))

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


def test_attach_recon_to_bank_replays_extract_banks_own_pixels_bit_exactly(tmp_path, monkeypatch):
    """The regression test for the RNG-derivation bug a reviewer caught:
    attach_recon_to_bank must reproduce the *exact* pixels extract_bank
    cached for every view -- not merely something self-consistent across two
    replay calls (test_attach_recon_to_bank_is_deterministic above already
    covered that, which is exactly why it didn't catch this).

    Two things are deliberately exercised together, because the bug was
    actually two bugs of the same kind:

    - The manifest is filtered to a non-contiguous set of index labels (like
      the CLI's own documented `--split train` usage), so a row's bank
      position and its true manifest row_id differ. Recovering row_id from
      `manifest_df.index` (not the bank's own positional `image_idx`) is
      what this exercises.
    - At least one replayed view's recipe contains a `noise` op -- the only
      op that reads from the per-view generator, so it's the only op whose
      pixels depend on how that generator is derived. A view with no noise
      op would pass even with the old, broken RNG derivation.
    """
    from aigcdet.augment.recipes import Recipe
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                         lambda m, s, imgs, device, batch_size=16:
                             np.zeros((len(imgs), s.dim), np.float32))

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    n_full = 16
    paths = [str(img_dir / f"{i}.png") for i in range(n_full)]
    for p in paths:
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(p)

    full_manifest = pd.DataFrame({
        "path": paths, "label": [0] * n_full, "generator": [""] * n_full,
        "source": ["test"] * n_full,
        # a non-contiguous split, like a real --split filter would produce
        "split": ["train" if i % 3 else "val_internal" for i in range(n_full)],
    })
    train_df = full_manifest[full_manifest["split"] == "train"]
    assert not np.array_equal(train_df.index.to_numpy(),
                               np.arange(len(train_df)))  # genuinely non-contiguous

    out_dir = tmp_path / "bank"
    extract.extract_bank(train_df, "fake", str(out_dir), seed=42, device="cpu")
    bank = FeatureBank.open(str(out_dir))

    # Confirm the fixture actually has a noise-op view to replay -- fail
    # loudly here (not "vacuously pass") if it ever doesn't.
    has_noise = any(
        any(o.name == "noise" for o in Recipe.from_json(bank.recipe_json(i, j)).ops)
        for i in range(len(bank.meta)) for j in range(bank.config["n_views"]))
    assert has_noise, "fixture must include a view with a noise op"

    # Ground truth: recompute exactly what extract_bank itself produced for
    # every (image, view), independently of attach_recon_to_bank's code path,
    # keyed the same way extract_bank documents it derives each view's
    # generator: (seed, row_id, view_idx).
    row_ids = train_df.index.to_numpy()
    expected = {}
    for i in range(len(bank.meta)):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
            # This block is extract_bank's preprocessing, recomputed by hand as
            # ground truth independent of attach_recon_to_bank. It has to
            # mirror EVERY step extract_bank takes before applying a recipe,
            # canonicalisation included -- otherwise it compares recon's
            # canonicalised pixels against an uncanonicalised expectation and
            # reports a divergence that is the test's, not the code's.
            base = canonicalise(base)
        rid = int(row_ids[i])
        for j in range(bank.config["n_views"]):
            apply_rng = np.random.default_rng([42, rid, j])
            recipe = Recipe.from_json(bank.recipe_json(i, j))
            expected[(i, j)] = recipe.apply(base, apply_rng)

    # Capture what attach_recon_to_bank actually feeds recon_features, in
    # call order (i outer, j inner -- see the loop in attach_recon_to_bank).
    captured: list[np.ndarray] = []

    def _spy_recon_features(img, vae, lp, device):
        captured.append(img.copy())
        return np.zeros(RECON_DIM, dtype=np.float32)

    monkeypatch.setattr("aigcdet.features.recon.recon_features", _spy_recon_features)
    monkeypatch.setattr("aigcdet.features.recon.load_recon_models",
                         lambda device, kind='kl': (None, None))

    attach_recon_to_bank(bank, train_df, device="cpu", seed=42)

    n_views = bank.config["n_views"]
    assert len(captured) == len(bank.meta) * n_views
    for i in range(len(bank.meta)):
        for j in range(n_views):
            np.testing.assert_array_equal(
                captured[i * n_views + j], expected[(i, j)],
                err_msg=f"image {i}, view {j} diverged from extract_bank's own pixels")


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
    if os.environ.get("AIGCDET_ALLOW_GPU_TESTS") != "1":
        pytest.skip(
            "GPU tests are opt-in: set AIGCDET_ALLOW_GPU_TESTS=1 to run them. "
            "They load real backbone weights and will download them if absent."
        )

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


def test_attach_recon_replays_bit_exactly_from_a_reset_index_manifest(tmp_path, monkeypatch):
    """H1: the caller's manifest index must NOT be load-bearing.

    Before row_id was stored in meta.parquet, attach_recon_to_bank recovered
    the replay key from `manifest_df.index`. A caller who passed a
    `reset_index(drop=True)`ed frame still got past `verify_against_manifest`
    (it compares paths positionally, and reset_index changes no path), and
    then every noise-containing view was replayed against different pixels --
    measured at 133/133 corrupt views, max delta 164/255. The key now comes
    from `bank.row_ids`, so this must be bit-exact.
    """
    from aigcdet.augment.recipes import Recipe
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                         lambda m, s, imgs, device, batch_size=16:
                             np.zeros((len(imgs), s.dim), np.float32))

    img_dir = tmp_path / "imgs_h1"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    n_full = 16
    paths = [str(img_dir / f"{i}.png") for i in range(n_full)]
    for p in paths:
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(p)
    full = pd.DataFrame({
        "path": paths, "label": [0] * n_full, "generator": [""] * n_full,
        "source": ["test"] * n_full,
        "split": ["train" if i % 3 else "val_internal" for i in range(n_full)],
    })
    train_df = full[full["split"] == "train"]
    assert not np.array_equal(train_df.index.to_numpy(), np.arange(len(train_df)))

    out_dir = tmp_path / "bank_h1"
    extract.extract_bank(train_df, "fake", str(out_dir), seed=42, device="cpu")
    bank = FeatureBank.open(str(out_dir))

    # The bank stores the manifest labels, not its own positions.
    np.testing.assert_array_equal(bank.row_ids, train_df.index.to_numpy())

    n_views = bank.config["n_views"]
    has_noise = any(
        any(o.name == "noise" for o in Recipe.from_json(bank.recipe_json(i, j)).ops)
        for i in range(len(bank.meta)) for j in range(n_views))
    assert has_noise, "fixture must include a view with a noise op"

    expected = {}
    for i in range(len(bank.meta)):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
            # This block is extract_bank's preprocessing, recomputed by hand as
            # ground truth independent of attach_recon_to_bank. It has to
            # mirror EVERY step extract_bank takes before applying a recipe,
            # canonicalisation included -- otherwise it compares recon's
            # canonicalised pixels against an uncanonicalised expectation and
            # reports a divergence that is the test's, not the code's.
            base = canonicalise(base)
        rid = int(train_df.index.to_numpy()[i])
        for j in range(n_views):
            expected[(i, j)] = Recipe.from_json(bank.recipe_json(i, j)).apply(
                base, np.random.default_rng([42, rid, j]))

    captured: list[np.ndarray] = []
    monkeypatch.setattr("aigcdet.features.recon.recon_features",
                         lambda img, vae, lp, device: (captured.append(img.copy())
                                                       or np.zeros(RECON_DIM, np.float32)))
    monkeypatch.setattr("aigcdet.features.recon.load_recon_models",
                         lambda device, kind='kl': (None, None))

    # The whole point: the manifest handed in has had its index thrown away.
    attach_recon_to_bank(bank, train_df.reset_index(drop=True), device="cpu", seed=42)

    assert len(captured) == len(bank.meta) * n_views
    for i in range(len(bank.meta)):
        for j in range(n_views):
            np.testing.assert_array_equal(
                captured[i * n_views + j], expected[(i, j)],
                err_msg=f"image {i}, view {j} replayed against different pixels")


def test_attach_recon_to_bank_replays_crop_and_dihedral_pixels_bit_exactly(
        tmp_path, monkeypatch):
    """The same regression, one policy later.

    `attach_recon_to_bank` is the dangerous decode site: it re-derives pixels
    that are already cached, and a divergence produces reconstruction features
    computed on different images than the embedding was, silently and with no
    shape error anywhere.

    Crop standardisation and dihedral augmentation add two more things it has
    to reproduce, and NEITHER is stored on disk -- the crop offset and the
    orientation index are pure functions of `(seed, row_id, view_idx)`. So the
    replay now depends on the policy being read back off the BANK (which is
    why `attach_recon_to_bank` takes it from `bank.config` rather than from a
    caller) and on three separate generators being derived from the same key
    in the same way at both sites.

    Ground truth is recomputed here by hand, independently of the replay's
    code path, mirroring every step `extract._prepare_image` takes.
    """
    from aigcdet.augment.canonical import (
        MODE_CROP, CanonPolicy, canonical_rng, canonicalise)
    from aigcdet.augment.geometric import dihedral, geometric_rng, sample_dihedral
    from aigcdet.augment.recipes import Recipe
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    n_full = 12
    paths = [str(img_dir / f"{i}.png") for i in range(n_full)]
    for p in paths:
        # Non-square and comfortably larger than the window, so the crop has
        # somewhere to move and a transposing rotation would change the shape
        # if the crop were skipped.
        Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)).save(p)

    full = pd.DataFrame({
        "path": paths, "label": [0] * n_full, "generator": [""] * n_full,
        "source": ["test"] * n_full,
        "split": ["train" if i % 3 else "val_internal" for i in range(n_full)],
    })
    train_df = full[full["split"] == "train"]
    assert not np.array_equal(train_df.index.to_numpy(),
                              np.arange(len(train_df)))

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    out_dir = tmp_path / "bank"
    extract.extract_bank(train_df, "fake", str(out_dir), seed=42, device="cpu",
                         policy=policy, geometric=True)
    bank = FeatureBank.open(str(out_dir))

    # The replay reads these off the bank, not off the caller.
    assert bank.config["canon_policy"] == policy.as_record()
    assert bank.config["geometric"] == "dihedral8"

    has_noise = any(
        any(o.name == "noise" for o in Recipe.from_json(bank.recipe_json(i, j)).ops)
        for i in range(len(bank.meta)) for j in range(bank.config["n_views"]))
    assert has_noise, "fixture must include a view with a noise op"

    row_ids = train_df.index.to_numpy()
    expected = {}
    for i in range(len(bank.meta)):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
        rid = int(row_ids[i])
        for j in range(bank.config["n_views"]):
            std = canonicalise(decoded, policy=policy,
                               rng=canonical_rng(42, rid, j))
            std = dihedral(std, sample_dihedral(geometric_rng(42, rid, j)))
            expected[(i, j)] = Recipe.from_json(bank.recipe_json(i, j)).apply(
                std, np.random.default_rng([42, rid, j]))

    # A crop that never moved, or an orientation that was always the identity,
    # would make this test pass while proving nothing.
    windows = {expected[(i, 0)].tobytes() for i in range(len(bank.meta))}
    assert len(windows) == len(bank.meta), "fixture views are not distinct"

    captured: list[np.ndarray] = []

    def _spy_recon_features(img, vae, lp, device):
        captured.append(img.copy())
        return np.zeros(RECON_DIM, dtype=np.float32)

    monkeypatch.setattr("aigcdet.features.recon.recon_features", _spy_recon_features)
    monkeypatch.setattr("aigcdet.features.recon.load_recon_models",
                        lambda device, kind='kl': (None, None))

    attach_recon_to_bank(bank, train_df, device="cpu", seed=42)

    v = bank.config["n_views"]
    assert len(captured) == len(bank.meta) * v
    for i in range(len(bank.meta)):
        for j in range(v):
            got = captured[i * v + j]
            assert np.array_equal(got, expected[(i, j)]), (
                f"replayed pixels differ at image {i}, view {j}")


def _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42, n_full=9):
    """A real crop eval bank, written by the production `extract_eval_bank`."""
    from aigcdet.augment.recipes import Recipe
    from aigcdet.eval import grid
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(grid, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    img_dir = tmp_path / "eimgs"
    img_dir.mkdir()
    rng = np.random.default_rng(7)
    paths = [str(img_dir / f"e{i}.png") for i in range(n_full)]
    for p in paths:
        Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)).save(p)
    df = pd.DataFrame({
        "path": paths, "label": [i % 2 for i in range(n_full)],
        "generator": [""] * n_full, "source": ["test"] * n_full,
        "split": ["val_internal"] * n_full,
    })
    # A noise condition is the point: its realisation is the only thing that
    # reads the per-view generator, so a wrong rng key shows up in the pixels.
    conditions = {
        "clean": Recipe.from_json("[]"),
        "noise_s0.05": Recipe.from_json(
            '[{"name": "noise", "params": {"sigma": 0.05}}]'),
        "jpeg_q70": Recipe.from_json(
            '[{"name": "jpeg", "params": {"quality": 70}}]'),
    }
    out_dir = tmp_path / "ebank"
    grid.extract_eval_bank(df, "fake", str(out_dir), conditions=conditions,
                           device="cpu", seed=seed, policy=policy)
    return FeatureBank.open(str(out_dir)), df, conditions


def test_attach_recon_replays_a_CROP_EVAL_banks_pixels_bit_exactly(
        tmp_path, monkeypatch):
    """The eval bank is standardised differently from the training bank.

    `eval/grid` canonicalises ONCE per image with NO rng and shares that
    result across every condition, so that a condition's measured effect is
    not confounded with "a different picture". Under crop, `rng is None` means
    the CENTRE window. The training path instead draws a FRESH RANDOM window
    per view from `canonical_rng(seed, row_id, view_idx)`.

    Replaying an eval bank down the training branch therefore computes
    reconstruction features on windows the bank never contained -- with no
    shape error, no warning and no crash, only wrong numbers in a rung that
    would be reported as A4. This is the regression test for that.

    Ground truth is built here by hand from `eval/grid`'s own recipe, not by
    calling the replay under test.
    """
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy, canonicalise
    from aigcdet.augment.recipes import Recipe

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)

    assert "conditions" in bank.config, "fixture is not an eval bank"
    assert bank.config["canon_policy"] == policy.as_record()

    row_ids = df.index.to_numpy()
    expected = {}
    for i in range(len(bank.meta)):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # ONE canonicalisation, no rng, shared by every condition -- grid.py.
        base = canonicalise(decoded, policy=policy)
        rid = int(row_ids[i])
        for j in range(bank.config["n_views"]):
            expected[(i, j)] = Recipe.from_json(bank.recipe_json(i, j)).apply(
                base, np.random.default_rng([42, rid, j]))

    # If the centre window and a jittered window happened to coincide the test
    # would pass while proving nothing. They must differ for at least one row.
    from aigcdet.augment.canonical import canonical_rng
    differs = False
    for i in range(len(bank.meta)):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
        rid = int(row_ids[i])
        train_style = canonicalise(decoded, policy=policy,
                                   rng=canonical_rng(42, rid, 0))
        if not np.array_equal(train_style, canonicalise(decoded, policy=policy)):
            differs = True
            break
    assert differs, "fixture cannot distinguish a centre from a jittered window"

    captured = []

    def _spy(img, vae, lp, device):
        captured.append(img.copy())
        return np.zeros(RECON_DIM, dtype=np.float32)

    monkeypatch.setattr("aigcdet.features.recon.recon_features", _spy)
    monkeypatch.setattr("aigcdet.features.recon.load_recon_models",
                        lambda device, kind='kl': (None, None))
    attach_recon_to_bank(bank, df, device="cpu", seed=42)

    v = bank.config["n_views"]
    assert len(captured) == len(bank.meta) * v
    for i in range(len(bank.meta)):
        for j in range(v):
            assert np.array_equal(captured[i * v + j], expected[(i, j)]), (
                f"eval replay differs at image {i}, condition {j}")


def test_recon_shards_reassemble_into_the_unsharded_block(tmp_path, monkeypatch):
    """Splitting one bank's rows across processes must change nothing.

    The four-GPU replay computes contiguous row blocks independently and
    merges them; the merged block has to equal what one process would have
    produced, or the shard count silently becomes part of the measurement.
    """
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy
    from aigcdet.features.recon import recon_bounds

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)

    def _fake(img, vae, lp, device):
        # A deterministic function OF THE PIXELS, so a block computed under a
        # wrong row offset cannot match by coincidence.
        return np.full(RECON_DIM, float(img.sum() % 9973), dtype=np.float32)

    monkeypatch.setattr("aigcdet.features.recon.recon_features", _fake)
    monkeypatch.setattr("aigcdet.features.recon.load_recon_models",
                        lambda device, kind='kl': (None, None))

    whole = attach_recon_to_bank(bank, df, device="cpu", seed=42, attach=False)

    n = len(bank.meta)
    for n_shards in (2, 3, n):
        parts = []
        for k in range(n_shards):
            start, stop = recon_bounds(n, k, n_shards)
            parts.append((start, stop, attach_recon_to_bank(
                bank, df, device="cpu", seed=42,
                start=start, stop=stop, attach=False)))
        assert sum(p[1] - p[0] for p in parts) == n
        merged = np.concatenate([b for _, _, b in parts], axis=0)
        assert np.array_equal(merged, whole), f"{n_shards} shards diverged"


def test_merge_recon_shards_refuses_a_cover_with_a_hole(tmp_path, monkeypatch):
    """A missing row is twelve zeros that read as a real measurement."""
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy
    from aigcdet.features.recon import merge_recon_shards

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)
    n, v = len(bank.meta), bank.config["n_views"]

    good = np.zeros((n, v, RECON_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="cover|contiguous"):
        merge_recon_shards(bank, [(0, n - 1, good[:n - 1])])
    with pytest.raises(ValueError, match="contiguous"):
        merge_recon_shards(bank, [(0, 2, good[:2]), (3, n, good[3:])])
    with pytest.raises(ValueError, match="non-finite"):
        bad = good.copy()
        bad[0, 0, 0] = np.nan
        merge_recon_shards(bank, [(0, n, bad)])
    # and the whole, finite cover is accepted
    assert merge_recon_shards(bank, [(0, n, good)]).shape == (n, v, RECON_DIM)


def test_attach_refuses_to_attach_a_partial_block(tmp_path, monkeypatch):
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)
    monkeypatch.setattr("aigcdet.features.recon.recon_features",
                        lambda *a, **k: np.zeros(RECON_DIM, dtype=np.float32))
    monkeypatch.setattr("aigcdet.features.recon.load_recon_models",
                        lambda device, kind='kl': (None, None))
    with pytest.raises(ValueError, match="refusing to attach"):
        attach_recon_to_bank(bank, df, device="cpu", seed=42, stop=2)


# --- the second autoencoder's block -----------------------------------------

def _bank_with_blocks(tmp_path, monkeypatch, kl=True, vq=True):
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)
    n, v = len(bank.meta), bank.config["n_views"]
    if kl:
        bank.attach_recon(np.full((n, v, RECON_DIM), 1.0, np.float32), kind="kl")
    if vq:
        bank.attach_recon(np.full((n, v, RECON_DIM), 2.0, np.float32), kind="vq")
    return bank, df


def test_recon_width_counts_one_block_per_enabled_flag():
    from aigcdet.features.bank import recon_width

    assert recon_width(False, False) == 0
    assert recon_width(True, False) == RECON_DIM
    assert recon_width(False, True) == RECON_DIM
    assert recon_width(True, True) == 2 * RECON_DIM


def test_the_vq_block_is_a_separate_file_and_does_not_disturb_the_kl_one(
        tmp_path, monkeypatch):
    """Two named (N, V, 12) arrays, never one wider `recon.npy`.

    `attach_recon` pins the width and `test_rung_ladder` allows one flag per
    rung, so widening RECON_DIM would break the artefact contract and the
    ladder at once.
    """
    import os
    from aigcdet.features.bank import FeatureBank

    bank, _ = _bank_with_blocks(tmp_path, monkeypatch)
    assert os.path.exists(os.path.join(bank.path, "recon.npy"))
    assert os.path.exists(os.path.join(bank.path, "recon_vq.npy"))
    reopened = FeatureBank.open(bank.path)
    assert reopened.recon.shape == reopened.recon_vq.shape
    assert float(np.asarray(reopened.recon).max()) == 1.0
    assert float(np.asarray(reopened.recon_vq).max()) == 2.0


def test_recon_blocks_returns_kl_before_vq_always(tmp_path, monkeypatch):
    """The ORDER is the contract.

    A head trained on [kl | vq] and scored against [vq | kl] raises nothing --
    the shapes match. It is simply wrong. Every consumer takes the order from
    this one function, so this is the test that pins it.
    """
    bank, _ = _bank_with_blocks(tmp_path, monkeypatch)
    blocks = bank.recon_blocks(True, True)
    assert len(blocks) == 2
    assert float(np.asarray(blocks[0]).max()) == 1.0, "kl must come first"
    assert float(np.asarray(blocks[1]).max()) == 2.0, "vq must come second"
    assert bank.recon_blocks(False, False) == []
    assert len(bank.recon_blocks(True, False)) == 1
    assert float(np.asarray(bank.recon_blocks(False, True)[0]).max()) == 2.0


def test_recon_blocks_names_the_flag_when_its_block_is_missing(
        tmp_path, monkeypatch):
    bank, _ = _bank_with_blocks(tmp_path, monkeypatch, kl=True, vq=False)
    with pytest.raises(ValueError, match="use_recon_vq"):
        bank.recon_blocks(True, True)
    second = tmp_path / "b2"
    second.mkdir()
    bank2, _ = _bank_with_blocks(second, monkeypatch, kl=False, vq=True)
    with pytest.raises(ValueError, match="use_recon"):
        bank2.recon_blocks(True, True)


def test_attach_recon_rejects_an_unknown_kind(tmp_path, monkeypatch):
    bank, _ = _bank_with_blocks(tmp_path, monkeypatch, kl=False, vq=False)
    n, v = len(bank.meta), bank.config["n_views"]
    with pytest.raises(ValueError, match="unknown recon kind"):
        bank.attach_recon(np.zeros((n, v, RECON_DIM), np.float32), kind="jpeg")


def test_a_detector_built_for_both_blocks_accepts_the_concatenated_width():
    """`recon_width` and `recon_blocks` must agree, or the head is built for a
    different input than it is handed."""
    import torch
    from aigcdet.features.bank import recon_width
    from aigcdet.models.heads import Detector

    for use_kl, use_vq in ((True, False), (False, True), (True, True)):
        w = recon_width(use_kl, use_vq)
        m = Detector(dim_feat=32, use_recon=True, recon_dim=w)
        out = m(torch.zeros(4, 32), torch.zeros(4, w))
        assert out["logit"].shape == (4,) or out["logit"].shape == (4, 1)


# --- the frequency block ------------------------------------------------------

def test_freq_refuses_a_band_bank(tmp_path, monkeypatch):
    """Band standardisation makes this descriptor measure the resampler.

    Band resampling replaces the generator's native pixel grid with the
    interpolation kernel's own, and the content-blind probe shows band leaking
    from low-level statistics alone (0.6105 pooled, 0.9976 on SID_Set) where
    crop is near chance (0.5081 / 0.6316). A band frequency block would destroy
    the real signal and supply a fake one, so it is refused rather than
    documented.
    """
    from aigcdet.augment.canonical import CanonPolicy, MODE_BAND
    from aigcdet.features.freq import attach_freq_to_bank

    policy = CanonPolicy(mode=MODE_BAND, band_side=64, nominal_side=128)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)
    with pytest.raises(ValueError, match="only measurable under crop"):
        attach_freq_to_bank(bank, df)


def test_freq_block_is_four_dimensional_and_attaches(tmp_path, monkeypatch):
    from aigcdet.augment.canonical import CanonPolicy, MODE_CROP
    from aigcdet.features.bank import FREQ_DIM, FeatureBank
    from aigcdet.features.freq import attach_freq_to_bank

    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    bank, df, _ = _tiny_eval_bank(tmp_path, monkeypatch, policy, seed=42)
    out = attach_freq_to_bank(bank, df)
    n, v = len(bank.meta), bank.config["n_views"]
    assert out.shape == (n, v, FREQ_DIM)
    assert np.isfinite(out).all()
    assert FeatureBank.open(bank.path).freq.shape == (n, v, FREQ_DIM)


def test_aux_blocks_orders_recon_then_vq_then_freq(tmp_path, monkeypatch):
    """Three blocks now, two different widths. The ORDER is still the contract."""
    from aigcdet.features.bank import FREQ_DIM, aux_width

    bank, _ = _bank_with_blocks(tmp_path, monkeypatch)
    n, v = len(bank.meta), bank.config["n_views"]
    bank.attach_block(np.full((n, v, FREQ_DIM), 3.0, np.float32), "freq")

    blocks = bank.aux_blocks(True, True, True)
    assert [float(np.asarray(b).max()) for b in blocks] == [1.0, 2.0, 3.0]
    assert aux_width(True, True, True) == 12 + 12 + 4
    assert aux_width(use_freq=True) == FREQ_DIM
    assert sum(b.shape[-1] for b in blocks) == aux_width(True, True, True)
