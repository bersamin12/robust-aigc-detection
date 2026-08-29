"""Frozen backbones and pooled embedding (spec §3.2).

Pooling is global average over final-layer patch tokens. The SigLIP2 authors
found this beat CLS-token, attention pooling, and multi-layer concatenation
head-to-head, so it is the default rather than an option (see module docstring
of the task brief for the citation this design follows).

Preprocessing is a "squish" resize to a fixed square, ignoring aspect ratio.
Random resized cropping is deliberately avoided: it can remove or distort the
localised forensic cues (compression grids, resampling artefacts, generator
fingerprints) that this project's detector depends on.

The three registered backbones do NOT share one forward() input contract --
see `INPUT_FORMATS` -- so `BackboneSpec` carries an `input_format` strategy and
`embed` builds each tower's kwargs from it. `embed`'s own signature is
unchanged: callers still hand it a list of HWC uint8 images.

Licence provenance for every weight loaded here is recorded in
`docs/model_licences.md` (spec §4.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

# ImageNet statistics, applied uniformly to all three backbones. CLIP and
# SigLIP2 were each pretrained with their own mean/std, not ImageNet's; using
# one shared normalisation here is an unverified simplification -- it has not
# been measured against per-backbone stats -- not a finding that the
# difference is immaterial. Revisit with a real comparison if a backbone's
# features underperform expectation.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


#: The two input contracts the registry's vision towers actually accept. They
#: are NOT interchangeable. `model(pixel_values=(B, 3, H, W))` is correct for
#: DINOv3, CLIP and the FIXED-RESOLUTION SigLIP2 checkpoints
#: (`siglip2-*-patch16-{224,256,384,512}`, which report `model_type: siglip`
#: and reuse SigLIP v1's `SiglipVisionModel`). It is impossible for the NaFlex
#: checkpoints (`siglip2-*-patch16-naflex`, `model_type: siglip2`), whose
#: transformer signature is `forward(pixel_values, attention_mask,
#: spatial_shapes)` over an ALREADY patchified
#: `(B, num_patches, C * patch * patch)` tensor -- their `patch_embedding` is
#: an `nn.Linear`, not a `nn.Conv2d`.
#:
#: The registry pins the fixed-resolution variant deliberately. Every image
#: reaches a backbone already canonicalised to one size (see
#: `augment.canonical` -- resolution leaks the label, spec 4.4a), so NaFlex's
#: native-resolution handling would buy nothing and cost a second contract.
#: INPUT_SIGLIP2_PATCHES stays implemented and tested against the transformers
#: reference so that switching to NaFlex is a one-line registry change, but no
#: entry uses it today.
INPUT_IMAGE_TENSOR = "image_tensor"       # (B, 3, H, W), pixel_values only
INPUT_SIGLIP2_PATCHES = "siglip2_patches"  # patchified + attention_mask + spatial_shapes
INPUT_FORMATS = (INPUT_IMAGE_TENSOR, INPUT_SIGLIP2_PATCHES)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    image_size: int
    dim: int
    num_prefix_tokens: int   # CLS + register tokens to strip before pooling
    params: int               # real vision-tower parameter count, for the 2B budget check
    #: How `embed` must build this tower's forward() kwargs -- one of
    #: `INPUT_FORMATS`. The registry holds backbones with genuinely different
    #: input contracts, so the strategy is per-spec data rather than an
    #: assumption baked into `embed`.
    input_format: str = INPUT_IMAGE_TENSOR
    #: Patch side length, required by (and only used by) INPUT_SIGLIP2_PATCHES,
    #: which has to patchify the image itself before the model sees it.
    patch_size: int = 0
    #: Whether the HuggingFace repo requires accepting a licence on the
    #: downloading account before the weights can be fetched. A property of the
    #: checkpoint, so it lives here rather than in a notebook's prose: it
    #: decides whether a run needs an HF token at all, and the fleet running an
    #: ungated backbone should not be stopped for one. See
    #: `docs/model_licences.md` for each entry's terms.
    gated: bool = False
    #: The dtype the tower is loaded and run in. float16 halves memory and
    #: roughly triples throughput over float32 on every GPU this project uses,
    #: and SigLIP2 and CLIP are finite in it. DINOv3-L is NOT: its activations
    #: overflow float16 by hidden layer 1, and every pooled vector comes out NaN
    #: -- silently, at full speed, for an entire 5-hour bank (2026-08-29).
    #: bfloat16 keeps float32's exponent range, is finite on DINOv3 at float16's
    #: throughput, and its pooled output (|x| < 2) stores in the bank's float16
    #: without loss. Per backbone, because it is a property of the checkpoint;
    #: see `run_dtype` for the one device-dependent fallback.
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.input_format not in INPUT_FORMATS:
            raise ValueError(
                f"{self.name}: input_format must be one of {INPUT_FORMATS}, "
                f"got {self.input_format!r}")
        if self.input_format == INPUT_SIGLIP2_PATCHES:
            if self.patch_size <= 0:
                raise ValueError(
                    f"{self.name}: {INPUT_SIGLIP2_PATCHES} needs a positive patch_size")
            if self.image_size % self.patch_size != 0:
                raise ValueError(
                    f"{self.name}: image_size {self.image_size} is not divisible by "
                    f"patch_size {self.patch_size}, so the image cannot be patchified")


# Parameter counts below are the vision-tower-only counts actually loaded and
# run by `load_backbone`/`embed` (not the full dual text+image checkpoint size
# reported by some model cards). They were measured, not guessed:
#   - dinov3l: HF Hub safetensors metadata for the (vision-only) checkpoint
#     reports total F32 parameters = 303_129_600.
#   - siglip2l / clipl: instantiated `AutoModel.from_config(...)` (architecture
#     only, no weights downloaded) from each model's published config, then
#     summed `.vision_model.parameters()`. siglip2l -> 316_283_904,
#     clipl -> 303_179_776.
BACKBONES: dict[str, BackboneSpec] = {
    # Gated: Meta's custom DINOv3 licence must be accepted per ACCOUNT.
    "dinov3l": BackboneSpec("dinov3l", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                             384, 1024, 5, 303_129_600, gated=True,
                             dtype=torch.bfloat16),
    "siglip2l": BackboneSpec("siglip2l", "google/siglip2-large-patch16-384",
                              384, 1024, 0, 316_283_904),
    "clipl": BackboneSpec("clipl", "openai/clip-vit-large-patch14",
                           224, 1024, 1, 303_179_776),
}


def squish(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size), ignoring aspect ratio."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _bf16_is_native() -> bool:
    """True when the current CUDA device runs bfloat16 in hardware (Ampere,
    sm_80, and later). Kaggle's T4 (sm_75) and P100 (sm_60) do not."""
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:                       # torch < 2.3 has no such keyword
        return bool(torch.cuda.is_bf16_supported())


def run_dtype(spec: BackboneSpec, device: str) -> torch.dtype:
    """The dtype `spec` actually runs in on `device`.

    `spec.dtype`, except that a bfloat16 spec on a CUDA device without native
    bfloat16 falls back to float32: slower, but finite. It never falls back to
    float16 -- that is the dtype a bfloat16 spec exists to avoid, and a
    fallback into it would reproduce the all-NaN bank at the one site meant to
    prevent it.
    """
    if (spec.dtype is torch.bfloat16 and str(device).startswith("cuda")
            and not _bf16_is_native()):
        return torch.float32
    return spec.dtype


def load_backbone(name: str, device: str = "cuda") -> tuple[torch.nn.Module, BackboneSpec]:
    """Load a frozen backbone's vision tower in eval mode on `device`, in
    `run_dtype(spec, device)`."""
    from transformers import AutoModel

    spec = BACKBONES[name]
    model = AutoModel.from_pretrained(spec.hf_id, dtype=run_dtype(spec, device))
    # CLIP and SigLIP wrap a vision tower alongside a text tower; DINOv3 is
    # already a vision-only model.
    model = getattr(model, "vision_model", model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec


def _normalised_batch(imgs: list[np.ndarray], size: int) -> np.ndarray:
    """Squish, scale to [0, 1] and normalise into (B, size, size, 3) float32,
    channels LAST -- the layout both input formats below start from."""
    arr = np.stack([squish(i, size) for i in imgs]).astype(np.float32) / 255.0
    return (arr - _MEAN) / _STD


def _patchify(arr: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    """(B, H, W, C) -> ((B, n_h * n_w, patch*patch*C), n_h, n_w).

    Element order matches `transformers.models.siglip2.image_processing_siglip2
    .convert_image_to_patches` exactly: raster-scan over patches, then
    (row, col, channel) within a patch. That order is what SigLIP2's
    `patch_embedding` Linear's weights were trained against, so it is a
    contract, not a formatting choice.
    """
    b, h, w, c = arr.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"({h}, {w}) is not divisible by patch_size {patch_size}")
    n_h, n_w = h // patch_size, w // patch_size
    p = arr.reshape(b, n_h, patch_size, n_w, patch_size, c).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(p.reshape(b, n_h * n_w, patch_size * patch_size * c)), n_h, n_w


def model_inputs(spec: BackboneSpec, imgs: list[np.ndarray], device: str,
                 dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Build the forward() kwargs `spec`'s vision tower actually accepts.

    Only `pixel_values` is cast to `dtype`: `attention_mask` and
    `spatial_shapes` are integer index/mask tensors and must stay integral.
    """
    arr = _normalised_batch(imgs, spec.image_size)
    if spec.input_format == INPUT_IMAGE_TENSOR:
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return {"pixel_values": x.to(device, dtype)}
    if spec.input_format == INPUT_SIGLIP2_PATCHES:
        patches, n_h, n_w = _patchify(arr, spec.patch_size)
        b = patches.shape[0]
        return {
            "pixel_values": torch.from_numpy(patches).to(device, dtype),
            # Every image here is squished to the same fixed square, so no
            # patch is padding and the mask is all-ones. It is still passed
            # because Siglip2VisionTransformer.forward requires it positionally.
            "attention_mask": torch.ones((b, n_h * n_w), dtype=torch.long, device=device),
            "spatial_shapes": torch.tensor([[n_h, n_w]] * b, dtype=torch.long, device=device),
        }
    raise ValueError(f"unknown input_format {spec.input_format!r}")


@torch.inference_mode()
def embed(model, spec: BackboneSpec, imgs: list[np.ndarray],
          device: str = "cuda", batch_size: int = 16) -> np.ndarray:
    """Global-average-pool the final-layer patch tokens (CLS/register tokens
    stripped per `spec.num_prefix_tokens`) into an (N, spec.dim) float32 array.

    The forward() call is built per `spec.input_format`, because the registry's
    backbones do not share one input contract (see `INPUT_FORMATS`).
    """
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:                      # a parameterless stand-in
        dtype = torch.float32
    out = []
    for i in range(0, len(imgs), batch_size):
        inputs = model_inputs(spec, imgs[i:i + batch_size], device, dtype)
        h = model(**inputs).last_hidden_state             # (B, T, D)
        patches = h[:, spec.num_prefix_tokens:, :]        # drop CLS + registers
        pooled = patches.mean(dim=1).float()
        # Checked here, on the first batch, rather than discovered by Stage B:
        # a tower running in a dtype it overflows produces NaN for EVERY image
        # at full speed, and nothing downstream of this line -- the bank
        # writer, the row count a chained job checks, `check_invariants` --
        # looked at a single value until 2026-08-29's 5-hour all-NaN bank.
        bad = ~torch.isfinite(pooled).all(dim=1)
        if bool(bad.any()):
            raise ValueError(
                f"{spec.name} produced non-finite features for "
                f"{int(bad.sum())} of {pooled.shape[0]} images in this batch, "
                f"running in {dtype}. This is the backbone overflowing its "
                f"dtype, not a bad image: DINOv3-L overflows float16 at hidden "
                f"layer 1 and needs bfloat16 (or float32). Nothing was "
                f"written; fix BackboneSpec.dtype / run_dtype and re-run.")
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
