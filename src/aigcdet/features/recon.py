"""Low-level reconstruction branch (spec §3.3).

Latent-diffusion outputs round-trip through their own VAE with anomalously low
error. Real photographs, and outputs from generators with a different decoder,
do not. The crop is taken at NATIVE pixel resolution: resizing before the crop
would attenuate exactly the signal being measured (VAE round-trip error lives
in the high-frequency residual, and any resample low-pass filters it away).

Two failure modes are expected and reported rather than hidden: the signal is
specific to the SD 1.5 autoencoder, and reconstruction error falls for any
heavily degraded image because degraded images are easier to reconstruct.

`load_recon_models` is the only place `diffusers` or `lpips` is imported, and
only inside the function body: neither package is installed in this project's
dev/test environment (nor is either meant to be pulled or run here -- see
project-constraints.md's no-GPU/no-download limits), so importing them at
module scope would make this module, and everything that imports it, fail to
import at all. Every other function here operates on plain numpy arrays and a
caller-supplied model object with the right call interface, so it is fully
testable with a stub in place of a real VAE/LPIPS model.
"""
from __future__ import annotations

import numpy as np
import torch

from aigcdet.augment.canonical import canonicalise

# Fixed order -- this is a contract, not a convenience. `bank.RECON_DIM` (12)
# must match this tuple's length, the bank stores this vector positionally,
# and Plan 3's AEROBLADE baseline reads out a specific entry by
# `RECON_FEATURE_NAMES.index("l1")`. Reordering this tuple silently changes
# what every downstream consumer reads.
RECON_FEATURE_NAMES: tuple[str, ...] = (
    "l1", "lpips",
    "err_mean", "err_std", "err_p90", "err_max",
    "spec_b0", "spec_b1", "spec_b2", "spec_b3",
    "spec_mid_ratio", "spec_high_ratio",
)


def native_center_crop(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Exact pixel slice, centred, never an interpolation.

    An image smaller than `size` on either side is reflect-padded up to
    `size` (never upscaled -- this project's normalisation rule is that a
    transform never invents pixels; reflect-padding only reuses pixels that
    are already there) before the same exact-slice crop is taken, so the
    return value is always an untouched, unresampled view of real pixels.
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        top_pad, bottom_pad = pad_h // 2, pad_h - pad_h // 2
        left_pad, right_pad = pad_w // 2, pad_w - pad_w // 2
        img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                     mode="reflect")
        h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


def load_recon_models(device: str = "cuda"):
    """Load the frozen SD 1.5 VAE and LPIPS(AlexNet), both eval-mode, no grad.

    Only ever call this against a real GPU with confirmed free VRAM -- see
    the module docstring and project-constraints.md. Never call this from a
    test that is not marked `@pytest.mark.gpu` and guarded on free VRAM.
    """
    from diffusers import AutoencoderKL
    import lpips

    vae = AutoencoderKL.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae",
        dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lp = lpips.LPIPS(net="alex").to(device).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    return vae, lp


@torch.inference_mode()
def _roundtrip(crop: np.ndarray, vae, device: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Encode/decode `crop` (256, 256, 3) uint8 through `vae`. Returns the
    per-pixel channel-averaged absolute error map (256, 256) float32, plus
    the normalised input and reconstruction tensors (1, 3, 256, 256) float32
    in [-1, 1], for callers (e.g. lpips) that need the tensors themselves."""
    x = torch.from_numpy(crop.astype(np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1)[None].to(device, torch.float16)
    lat = vae.encode(x).latent_dist.mode()
    rec = vae.decode(lat).sample.clamp(-1, 1)
    x, rec = x.float(), rec.float()
    err = (x - rec).abs().mean(dim=1)[0].cpu().numpy()
    return err, x, rec


def _radial_bands(err: np.ndarray, n_bands: int = 4) -> np.ndarray:
    """Azimuthally averaged power spectrum of the error map, split into
    `n_bands` concentric rings from DC (band 0) to the highest spatial
    frequency (band `n_bands - 1`).

    Mid-frequency error survives compression better than the highest
    frequencies, so the bands are kept separate rather than summed into one
    number.
    """
    f = np.abs(np.fft.fftshift(np.fft.fft2(err - err.mean()))) ** 2
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rad.max()
    out = []
    for b in range(n_bands):
        lo, hi = rmax * b / n_bands, rmax * (b + 1) / n_bands
        # The last band's upper edge is inclusive: the highest spatial
        # frequency (rad == rmax, e.g. an exact checkerboard) is exactly the
        # signal this band exists to capture, and a strict "<" would drop it
        # entirely, silently zeroing the one band that matters most.
        m = (rad >= lo) & (rad <= hi if b == n_bands - 1 else rad < hi)
        out.append(float(np.log1p(f[m].mean())) if m.any() else 0.0)
    return np.array(out, dtype=np.float32)


@torch.inference_mode()
def recon_features(img: np.ndarray, vae, lpips_fn, device: str = "cuda") -> np.ndarray:
    """(12,) float32, ordered per `RECON_FEATURE_NAMES`, from a native
    256x256 centre crop of `img`.

    `l1` and `err_mean` are the same quantity by construction (both are the
    mean of the same per-pixel error map): `l1` is kept as its own named,
    stable-index slot because it is the classic single-number AEROBLADE
    statistic downstream consumers read by name, while `err_mean` groups
    alongside `err_std`/`err_p90`/`err_max` as one family of descriptive
    statistics over that same map.
    """
    crop = native_center_crop(img, 256)
    err, x, rec = _roundtrip(crop, vae, device)
    l1 = float(np.abs(err).mean())
    lp = float(lpips_fn(x, rec).item())
    bands = _radial_bands(err)
    total = float(bands.sum()) or 1.0
    stats = np.array(
        [l1, lp, float(err.mean()), float(err.std()),
         float(np.percentile(err, 90)), float(err.max())], dtype=np.float32)
    ratios = np.array([bands[1] / total, bands[3] / total], dtype=np.float32)
    return np.concatenate([stats, bands, ratios]).astype(np.float32)


@torch.inference_mode()
def error_map(img: np.ndarray, vae, device: str = "cuda") -> np.ndarray:
    """Per-pixel reconstruction error (256, 256) float32, for the dashboard's
    second heatmap. The same error map `recon_features` derives its
    err_*/spec_* entries from."""
    err, _, _ = _roundtrip(native_center_crop(img, 256), vae, device)
    return err


def attach_recon_to_bank(bank, manifest_df, device: str = "cuda",
                          seed: int = 20260827) -> None:
    """Recompute every view exactly as Stage A did, then cache `r` for ALL of
    them. Partial view coverage would make A3 vs A4 a comparison across
    different augmentation budgets (spec §3.3).

    Checks the bank is still positionally aligned with `manifest_df` before
    writing anything -- a re-split manifest would otherwise silently attach
    reconstruction features to the wrong rows (project-constraints.md's
    "manifest is frozen once written" rule). `manifest_df` is used ONLY for
    that check: the replay key comes from `bank.row_ids`, which `extract_bank`
    stores in `meta.parquet`. It used to be recovered from
    `manifest_df.index`, which made the caller's index a load-bearing input
    that nothing verified -- a `reset_index()`ed frame passed
    `verify_against_manifest` (it compares paths positionally) and then
    replayed every noise-containing view against different pixels.

    Reproduces each view's exact cached pixels, not just an equally-valid
    resampling of the same recipe: `extract_bank` derives a fresh
    `np.random.default_rng([seed, row_id, view_idx])` to APPLY each view
    (a separate generator from the one used to sample its recipe), so the
    only randomness an already-known recipe's replay needs -- the `noise`
    op's realisation, the one op that reads from the generator -- is
    reproduced by re-deriving that same key here. This holds regardless of
    view order or how many draws `extract_bank`'s own sampling step
    consumed, because that step used a different generator instance
    entirely (see `aigcdet.features.extract`'s module docstring).
    """
    from PIL import Image
    from tqdm import tqdm

    from aigcdet.augment.recipes import Recipe
    from aigcdet.features.bank import RECON_DIM

    bank.verify_against_manifest(manifest_df)

    vae, lp = load_recon_models(device)
    n, v = len(bank.meta), bank.config["n_views"]
    row_ids = bank.row_ids
    out = np.zeros((n, v, RECON_DIM), dtype=np.float32)
    for i in tqdm(range(n), desc="recon"):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # Resolution canonicalisation, BEFORE any stored recipe is replayed
        # (docs/resolution_shortcut.md). This site is the dangerous one: it
        # re-decodes and re-runs recipes to reproduce the EXACT pixels that
        # extract.py cached. If extract.py and grid.py canonicalise and this
        # does not, reconstruction features are computed on different pixels
        # than were cached -- silently, with no shape error anywhere. All
        # three sites or none.
        base = canonicalise(base)
        rid = int(row_ids[i])
        for j in range(v):
            apply_rng = np.random.default_rng([seed, rid, j])
            view = Recipe.from_json(bank.recipe_json(i, j)).apply(base, apply_rng)
            out[i, j] = recon_features(view, vae, lp, device)
    bank.attach_recon(out)
