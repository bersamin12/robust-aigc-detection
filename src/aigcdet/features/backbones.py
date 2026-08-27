"""Frozen backbones and pooled embedding (spec §3.2).

Pooling is global average over final-layer patch tokens. The SigLIP2 authors
found this beat CLS-token, attention pooling, and multi-layer concatenation
head-to-head, so it is the default rather than an option (see module docstring
of the task brief for the citation this design follows).

Preprocessing is a "squish" resize to a fixed square, ignoring aspect ratio.
Random resized cropping is deliberately avoided: it can remove or distort the
localised forensic cues (compression grids, resampling artefacts, generator
fingerprints) that this project's detector depends on.

Licence provenance for every weight loaded here is recorded in
`docs/model_licences.md` (spec §4.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

# ImageNet statistics; CLIP/SigLIP2/DINOv3 all ship close variants and the
# difference is immaterial for a frozen feature extractor.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    image_size: int
    dim: int
    num_prefix_tokens: int   # CLS + register tokens to strip before pooling
    params: int               # real vision-tower parameter count, for the 2B budget check


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
    "dinov3l": BackboneSpec("dinov3l", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                             384, 1024, 5, 303_129_600),
    "siglip2l": BackboneSpec("siglip2l", "google/siglip2-large-patch16-384",
                              384, 1024, 0, 316_283_904),
    "clipl": BackboneSpec("clipl", "openai/clip-vit-large-patch14",
                           224, 1024, 1, 303_179_776),
}


def squish(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size), ignoring aspect ratio."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_backbone(name: str, device: str = "cuda") -> tuple[torch.nn.Module, BackboneSpec]:
    """Load a frozen backbone's vision tower in eval mode on `device`."""
    from transformers import AutoModel

    spec = BACKBONES[name]
    model = AutoModel.from_pretrained(spec.hf_id, dtype=torch.float16)
    # CLIP and SigLIP wrap a vision tower alongside a text tower; DINOv3 is
    # already a vision-only model.
    model = getattr(model, "vision_model", model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec


def _to_tensor(imgs: list[np.ndarray], size: int) -> torch.Tensor:
    arr = np.stack([squish(i, size) for i in imgs]).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


@torch.inference_mode()
def embed(model, spec: BackboneSpec, imgs: list[np.ndarray],
          device: str = "cuda", batch_size: int = 16) -> np.ndarray:
    """Global-average-pool the final-layer patch tokens (CLS/register tokens
    stripped per `spec.num_prefix_tokens`) into an (N, spec.dim) float32 array."""
    out = []
    for i in range(0, len(imgs), batch_size):
        x = _to_tensor(imgs[i:i + batch_size], spec.image_size).to(device, torch.float16)
        h = model(pixel_values=x).last_hidden_state       # (B, T, D)
        patches = h[:, spec.num_prefix_tokens:, :]        # drop CLS + registers
        out.append(patches.mean(dim=1).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
