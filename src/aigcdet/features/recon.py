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

from aigcdet.augment.canonical import (
    DEFAULT_POLICY, MODE_CROP, CanonPolicy, canonical_rng, canonicalise)
from aigcdet.augment.geometric import dihedral, geometric_rng, sample_dihedral

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


#: The autoencoders the reconstruction branch can be computed against, and the
#: bank block each one writes. TWO SEPARATE 12-d BLOCKS, never one wider one:
#: `bank.attach_recon` pins `(N, V, 12)` and `tests/test_rung_ladder.py`
#: enforces one-flag steps, so widening `RECON_DIM` would break the artefact
#: contract and the ladder at once. A second named block breaks neither.
RECON_KINDS: dict[str, str] = {"kl": "recon", "vq": "recon_vq"}

#: Why a second autoencoder exists at all. The held-out families are
#: `SDwithAdaptor_controlnet` (latent diffusion, SD's own KL lineage) and
#: `VQGAN` (vector-quantised). A continuous KL VAE has no structural reason to
#: round-trip a VQ decoder's output anomalously, so the branch as originally
#: built is arguably blind to half the population selection rests on. Measured
#: on 220 images per class at the centre crop, real vs generated:
#:
#:     family                      KL      VQ     both
#:     VQGAN                     0.7919  0.8248  0.8427
#:     SDwithAdaptor_controlnet  0.7823  0.7959  0.8100
#:
#: VQ wins on both and by 2.4x more on VQGAN, and the pair beats either alone
#: -- they are complementary, which is what makes `a4both` a real rung and not
#: a tidier way to spell `a4vq`.
_VQ_REPO = "CompVis/ldm-super-resolution-4x-openimages"


class _VQRoundtrip:
    """`VQModel` wearing `AutoencoderKL`'s encode interface.

    `_roundtrip` calls `encode(x).latent_dist.mode()`; a `VQModel` returns
    `.latents` and quantises inside `decode`. Adapting here rather than
    branching inside `_roundtrip` keeps `recon_features` autoencoder-agnostic,
    which is the property that lets one 12-d extractor serve both blocks.
    """

    def __init__(self, model):
        self.model = model

    def encode(self, x):
        from types import SimpleNamespace
        latents = self.model.encode(x).latents
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mode=lambda: latents))

    def decode(self, z):
        return self.model.decode(z)

    def parameters(self):
        return self.model.parameters()


def load_recon_models(device: str = "cuda", kind: str = "kl"):
    """Load a frozen autoencoder and LPIPS(AlexNet), both eval-mode, no grad.

    `kind` selects which autoencoder, and therefore which bank block the
    features belong in -- see `RECON_KINDS`.

    Only ever call this against a real GPU with confirmed free VRAM -- see
    the module docstring and project-constraints.md. Never call this from a
    test that is not marked `@pytest.mark.gpu` and guarded on free VRAM.
    """
    import lpips

    if kind not in RECON_KINDS:
        raise ValueError(
            f"unknown recon kind {kind!r}; known: {sorted(RECON_KINDS)}")
    if kind == "kl":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae",
            dtype=torch.float16).to(device).eval()
    else:
        from diffusers import VQModel
        vae = VQModel.from_pretrained(
            _VQ_REPO, subfolder="vqvae", dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    if kind == "vq":
        vae = _VQRoundtrip(vae)
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



def recon_bounds(n: int, i: int, n_shards: int) -> tuple[int, int]:
    """Contiguous block `i` of `n_shards` over `n` rows. See `replay.shard_bounds`."""
    from aigcdet.features.replay import shard_bounds

    return shard_bounds(n, i, n_shards)


def merge_recon_shards(bank, parts, kind: str = "kl") -> np.ndarray:
    """Assemble contiguous reconstruction shards and attach them."""
    from aigcdet.features.bank import RECON_DIM
    from aigcdet.features.replay import merge_blocks

    if kind not in RECON_KINDS:
        raise ValueError(
            f"unknown recon kind {kind!r}; known: {sorted(RECON_KINDS)}")
    return merge_blocks(bank, parts, RECON_KINDS[kind], RECON_DIM)


def attach_recon_to_bank(bank, manifest_df, device: str = "cuda",
                          seed: int = 20260827, *, kind: str = "kl",
                          start: int = 0, stop: int | None = None,
                          attach: bool = True) -> np.ndarray:
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
    from aigcdet.features.bank import RECON_DIM
    from aigcdet.features.replay import replay_views

    vae, lp = load_recon_models(device, kind=kind)
    out = replay_views(
        bank, manifest_df,
        # Resolved from this module's globals at CALL time, which is what lets
        # a test substitute a stub extractor for the real VAE round-trip.
        lambda view: recon_features(view, vae, lp, device),
        RECON_DIM, seed=seed, start=start, stop=stop,
        allow_partial=not attach, desc=f"recon:{kind}")
    if attach:
        bank.attach_recon(out, kind=kind)
    return out
