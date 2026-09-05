"""Recipes compose ops and carry their own supervision labels.

The severity normalisation below IS the degradation head's target definition
(spec §3.4), so it is fixed here once and imported everywhere else.

FAMILIES vs OPS
---------------
`FAMILIES` is the six-way axis every per-family vector in the project is
indexed by: the bank's presence/severity arrays, the degradation head's two
output layers, the calibrator's conditioning width. It does NOT grow when a
transform is added. The NTIRE 2026 challenge applies 36 transformation types
drawn from a handful of groups (arXiv 2604.11487), and this module mirrors
that structure: every op in `augment.ops` belongs to exactly one family
(`FAMILY_OF`), presence means "some op of this family was applied", and
severity is the harshest op of that family in the chain, each op normalised
to [0, 1] on its own scale.

Widening `N_FAMILIES` instead would change the head, every bank on disk,
`infer.py`'s calibrator width and every existing checkpoint, for a readout
that is auxiliary (it feeds calibration and EQI, never the classifier).

POOLS
-----
Which ops the training sampler may draw from is a named `pool`, recorded in
the bank's config beside `seed`. `"legacy"` is the brief's six ops and is
BIT-IDENTICAL to the sampler before pools existed: for a family with one
candidate op the sampler consumes no extra draw, so every bank extracted
before this module changed still replays from `(seed, row_id, view_idx)`.
`tests/augment/test_pools.py` pins that with a snapshot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from aigcdet.augment.ops import OP_FUNCS

FAMILIES: tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

#: Width of every per-family vector in the project: the bank's presence and
#: severity arrays, and the degradation head's two output layers. Defined here,
#: beside FAMILIES, so those two never drift apart -- heads.py used to hardcode
#: 6 while bank.py used len(FAMILIES), and adding a seventh family would have
#: produced a shape mismatch at training time rather than at edit time.
N_FAMILIES: int = len(FAMILIES)

#: Every op's family. The brief's six ops carry their family's own name; the
#: NTIRE additions are grouped by what they do to the signal.
FAMILY_OF: dict[str, str] = {
    "jpeg": "jpeg", "jpeg_multi": "jpeg", "jpeg2000": "jpeg",
    "blur": "blur", "motion_blur": "blur", "lens_blur": "blur",
    "glass_blur": "blur",
    "resize": "resize", "downscale": "resize", "pixelate": "resize",
    "noise": "noise", "impulse": "noise", "iso_noise": "noise",
    "jitter": "jitter", "tone_curve": "jitter", "clahe": "jitter",
    "contrast": "jitter", "brightness": "jitter", "quantize": "jitter",
    "channel_shift": "jitter", "saturation": "jitter",
    "crop": "crop", "aspect_crop": "crop", "perspective": "crop",
}

#: Ops whose `apply` draws random pixels and therefore takes `rng`. Every
#: other op's randomness lives in its params, so a stored recipe replays
#: byte-for-byte from the recipe json alone.
STOCHASTIC_OPS: frozenset[str] = frozenset(
    {"noise", "glass_blur", "impulse", "iso_noise"})

#: Named op pools the training sampler draws from. ORDER MATTERS: the sampler
#: indexes into a family's candidates by an rng draw, so reordering a pool
#: changes every recipe sampled under it. Append, never insert.
POOLS: dict[str, tuple[str, ...]] = {
    "legacy": FAMILIES,
    "ntire": FAMILIES + (
        "jpeg_multi", "jpeg2000",
        "motion_blur", "lens_blur", "glass_blur",
        "downscale", "pixelate",
        "impulse", "iso_noise",
        "tone_curve", "clahe", "contrast", "brightness", "quantize",
        "channel_shift", "saturation",
        "aspect_crop", "perspective",
    ),
    #: `ntire` without JPEG 2000, for an OpenCV build that cannot encode it.
    #: A separate pool rather than a runtime skip, because the pool name is
    #: what the bank records -- two boxes must not sample different recipes
    #: under one name.
    "ntire_nojp2": FAMILIES + (
        "jpeg_multi",
        "motion_blur", "lens_blur", "glass_blur",
        "downscale", "pixelate",
        "impulse", "iso_noise",
        "tone_curve", "clahe", "contrast", "brightness", "quantize",
        "channel_shift", "saturation",
        "aspect_crop", "perspective",
    ),
}
RECIPE_POOL_DEFAULT: str = "legacy"
MAX_OPS_DEFAULT: int = 3

# Held-out severity bands (spec §4.6): the eval grid's q=70 and sigma=1.0
# conditions must be unseen severities at evaluation time, so the training
# sampler must never draw parameters inside these bands.
HELDOUT_JPEG_Q: tuple[int, int] = (65, 75)
HELDOUT_BLUR_SIGMA: tuple[float, float] = (0.85, 1.15)

# Ranges the training sampler draws from, chosen to cover the eval grid's values.
_JPEG_RANGE = (30, 98)
_BLUR_RANGE = (0.2, 2.2)
_RESIZE_RANGE = (0.25, 0.9)
_NOISE_RANGE = (0.005, 0.11)
_JITTER_RANGE = (0.05, 0.20)
_CROP_RANGE = (0.75, 0.98)
# NTIRE additions.
_JP2_RANGE = (40, 400)            # cv2 compression-x1000; 1000 is lossless
_MOTION_RANGE = (3, 15)           # px
_LENS_RANGE = (1, 5)              # px radius
_GLASS_SIGMA_RANGE = (0.3, 1.2)
_GLASS_DELTA_RANGE = (1, 3)
_PIXELATE_RANGE = (2, 12)         # px block
_IMPULSE_RANGE = (0.002, 0.08)
_ISO_INTENSITY_RANGE = (0.05, 0.5)
_ISO_COLOR_RANGE = (0.0, 0.04)
_TONE_RANGE = (0.10, 0.90)        # control-point y; identity is 0.25 / 0.75
_CLAHE_CLIP_RANGE = (1.0, 4.0)
_CONTRAST_RANGE = (0.5, 1.5)
_BRIGHTNESS_RANGE = (8.0, 64.0)   # magnitude, sign drawn
_QUANTIZE_BITS = (3, 7)
_CHANNEL_SHIFT_RANGE = (5.0, 40.0)
_SATURATION_RANGE = (0.4, 1.6)
_ASPECT_FRAC_RANGE = (0.5, 0.95)
_PERSPECTIVE_MAX = 0.10           # fraction of side per corner


def _severity(name: str, p: dict) -> float:
    """Map raw parameters to a comparable [0, 1] harshness scale."""
    if name == "jpeg":
        return float(np.clip((100.0 - p["quality"]) / 70.0, 0.0, 1.0))
    if name == "blur":
        return float(np.clip(p["sigma"] / 2.0, 0.0, 1.0))
    if name == "resize":
        return float(np.clip((1.0 - p["scale"]) / 0.75, 0.0, 1.0))
    if name == "noise":
        return float(np.clip(p["sigma"] / 0.10, 0.0, 1.0))
    if name == "jitter":
        worst = max(abs(p["brightness"]), abs(p["contrast"]), abs(p["saturation"]))
        return float(np.clip(worst / 0.20, 0.0, 1.0))
    if name == "crop":
        return float(np.clip((1.0 - p["frac"]) / 0.20, 0.0, 1.0))
    # --- NTIRE additions ---
    if name == "jpeg_multi":
        return float(np.clip((100.0 - min(p["qualities"])) / 70.0, 0.0, 1.0))
    if name == "jpeg2000":
        lo, hi = _JP2_RANGE
        return float(np.clip((hi - p["rate"]) / (hi - lo), 0.0, 1.0))
    if name == "motion_blur":
        return float(np.clip(p["length"] / _MOTION_RANGE[1], 0.0, 1.0))
    if name == "lens_blur":
        return float(np.clip(p["radius"] / _LENS_RANGE[1], 0.0, 1.0))
    if name == "glass_blur":
        return float(np.clip((p["max_delta"] * p["iters"]) / 6.0, 0.0, 1.0))
    if name == "downscale":
        return float(np.clip((1.0 - p["scale"]) / 0.75, 0.0, 1.0))
    if name == "pixelate":
        return float(np.clip(p["block"] / _PIXELATE_RANGE[1], 0.0, 1.0))
    if name == "impulse":
        return float(np.clip(p["amount"] / _IMPULSE_RANGE[1], 0.0, 1.0))
    if name == "iso_noise":
        return float(np.clip(p["intensity"] / _ISO_INTENSITY_RANGE[1], 0.0, 1.0))
    if name == "tone_curve":
        dev = max(abs(p["low_y"] - 0.25), abs(p["high_y"] - 0.75))
        return float(np.clip(dev / 0.65, 0.0, 1.0))
    if name == "clahe":
        return float(np.clip(p["clip"] / _CLAHE_CLIP_RANGE[1], 0.0, 1.0))
    if name == "contrast":
        return float(np.clip(abs(p["alpha"] - 1.0) / 0.5, 0.0, 1.0))
    if name == "brightness":
        return float(np.clip(abs(p["beta"]) / _BRIGHTNESS_RANGE[1], 0.0, 1.0))
    if name == "quantize":
        return float(np.clip((8 - p["bits"]) / 5.0, 0.0, 1.0))
    if name == "channel_shift":
        worst = max(abs(p["dr"]), abs(p["dg"]), abs(p["db"]))
        return float(np.clip(worst / _CHANNEL_SHIFT_RANGE[1], 0.0, 1.0))
    if name == "saturation":
        return float(np.clip(abs(p["scale"] - 1.0) / 0.6, 0.0, 1.0))
    if name == "aspect_crop":
        return float(np.clip((1.0 - p["h_frac"] * p["w_frac"]) / 0.75, 0.0, 1.0))
    if name == "perspective":
        return float(np.clip(max(abs(x) for x in p["offsets"]) / _PERSPECTIVE_MAX,
                             0.0, 1.0))
    raise KeyError(name)


@dataclass(frozen=True)
class Op:
    name: str
    params: dict

    def __post_init__(self):
        if self.name not in OP_FUNCS:
            raise KeyError(f"unknown op {self.name!r}")
        if self.name not in FAMILY_OF:
            raise KeyError(f"op {self.name!r} has no family in FAMILY_OF")


@dataclass(frozen=True)
class Recipe:
    ops: tuple[Op, ...] = ()

    def apply(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = img
        for op in self.ops:
            fn = OP_FUNCS[op.name]
            out = (fn(out, rng=rng, **op.params) if op.name in STOCHASTIC_OPS
                   else fn(out, **op.params))
        return out

    def labels(self) -> dict[str, np.ndarray]:
        presence = np.zeros(len(FAMILIES), dtype=np.float32)
        severity = np.zeros(len(FAMILIES), dtype=np.float32)
        for op in self.ops:
            i = FAMILIES.index(FAMILY_OF[op.name])
            presence[i] = 1.0
            # If a family appears twice in a chain, keep the harsher instance.
            severity[i] = max(severity[i], _severity(op.name, op.params))
        return {"presence": presence, "severity": severity}

    def to_json(self) -> str:
        return json.dumps([{"name": o.name, "params": o.params} for o in self.ops])

    @classmethod
    def from_json(cls, s: str) -> "Recipe":
        return cls(tuple(Op(d["name"], d["params"]) for d in json.loads(s)))


def _sample_jpeg_quality(rng: np.random.Generator) -> int:
    lo, hi = HELDOUT_JPEG_Q
    while True:
        q = int(rng.integers(_JPEG_RANGE[0], _JPEG_RANGE[1] + 1))
        if not (lo <= q <= hi):
            return q


def _sample_params(name: str, rng: np.random.Generator) -> dict:
    if name == "jpeg":
        return {"quality": _sample_jpeg_quality(rng)}
    if name == "blur":
        lo, hi = HELDOUT_BLUR_SIGMA
        while True:
            s = float(rng.uniform(*_BLUR_RANGE))
            if not (lo <= s <= hi):
                return {"sigma": s}
    if name == "resize":
        return {"scale": float(rng.uniform(*_RESIZE_RANGE))}
    if name == "noise":
        return {"sigma": float(rng.uniform(*_NOISE_RANGE))}
    if name == "jitter":
        lo, hi = _JITTER_RANGE
        return {k: float(rng.uniform(lo, hi) * rng.choice([-1.0, 1.0]))
                for k in ("brightness", "contrast", "saturation")}
    if name == "crop":
        return {"frac": float(rng.uniform(*_CROP_RANGE))}
    # --- NTIRE additions ---
    if name == "jpeg_multi":
        return {"qualities": [_sample_jpeg_quality(rng), _sample_jpeg_quality(rng)]}
    if name == "jpeg2000":
        return {"rate": int(rng.integers(_JP2_RANGE[0], _JP2_RANGE[1] + 1))}
    if name == "motion_blur":
        return {"length": int(rng.integers(_MOTION_RANGE[0], _MOTION_RANGE[1] + 1)),
                "angle": float(rng.uniform(0.0, 180.0))}
    if name == "lens_blur":
        return {"radius": int(rng.integers(_LENS_RANGE[0], _LENS_RANGE[1] + 1))}
    if name == "glass_blur":
        return {"sigma": float(rng.uniform(*_GLASS_SIGMA_RANGE)),
                "max_delta": int(rng.integers(_GLASS_DELTA_RANGE[0],
                                              _GLASS_DELTA_RANGE[1] + 1)),
                "iters": int(rng.integers(1, 3))}
    if name == "downscale":
        interps = ("nearest", "linear", "area", "cubic")
        return {"scale": float(rng.uniform(*_RESIZE_RANGE)),
                "interp_down": interps[int(rng.integers(len(interps)))],
                "interp_up": interps[int(rng.integers(len(interps)))]}
    if name == "pixelate":
        return {"block": int(rng.integers(_PIXELATE_RANGE[0], _PIXELATE_RANGE[1] + 1))}
    if name == "impulse":
        return {"amount": float(rng.uniform(*_IMPULSE_RANGE))}
    if name == "iso_noise":
        return {"intensity": float(rng.uniform(*_ISO_INTENSITY_RANGE)),
                "color_shift": float(rng.uniform(*_ISO_COLOR_RANGE))}
    if name == "tone_curve":
        return {"low_y": float(rng.uniform(*_TONE_RANGE)),
                "high_y": float(rng.uniform(*_TONE_RANGE))}
    if name == "clahe":
        return {"clip": float(rng.uniform(*_CLAHE_CLIP_RANGE)),
                "grid": int(rng.choice([4, 8]))}
    if name == "contrast":
        return {"alpha": float(rng.uniform(*_CONTRAST_RANGE))}
    if name == "brightness":
        return {"beta": float(rng.uniform(*_BRIGHTNESS_RANGE) * rng.choice([-1.0, 1.0]))}
    if name == "quantize":
        return {"bits": int(rng.integers(_QUANTIZE_BITS[0], _QUANTIZE_BITS[1] + 1))}
    if name == "channel_shift":
        lo, hi = _CHANNEL_SHIFT_RANGE
        return {k: float(rng.uniform(lo, hi) * rng.choice([-1.0, 1.0]))
                for k in ("dr", "dg", "db")}
    if name == "saturation":
        return {"scale": float(rng.uniform(*_SATURATION_RANGE))}
    if name == "aspect_crop":
        h_frac = float(rng.uniform(*_ASPECT_FRAC_RANGE))
        w_frac = float(rng.uniform(*_ASPECT_FRAC_RANGE))
        return {"top": float(rng.uniform(0.0, 1.0 - h_frac)),
                "left": float(rng.uniform(0.0, 1.0 - w_frac)),
                "h_frac": h_frac, "w_frac": w_frac}
    if name == "perspective":
        return {"offsets": [float(x) for x in
                            rng.uniform(-_PERSPECTIVE_MAX, _PERSPECTIVE_MAX, size=8)]}
    raise KeyError(name)


def ops_for_family(family: str, pool: str) -> tuple[str, ...]:
    """The candidate ops of `family` in `pool`, in pool order."""
    try:
        names = POOLS[pool]
    except KeyError:
        raise KeyError(f"unknown recipe pool {pool!r}; known: {sorted(POOLS)}") from None
    return tuple(o for o in names if FAMILY_OF[o] == family)


def sample_training_recipe(rng: np.random.Generator, max_ops: int = MAX_OPS_DEFAULT,
                            families: tuple[str, ...] = FAMILIES,
                            pool: str = RECIPE_POOL_DEFAULT) -> Recipe:
    """1 to `max_ops` chained ops from distinct families (spec §5).

    `families` restricts which families may be drawn, for the
    leave-one-transform-out bank (spec §4.6). It is applied by sampling the
    chain length FIRST and then choosing over the kept families -- never by
    rejection-sampling whole recipes, which biases towards shorter chains
    (an excluded family is likelier to appear in a 3-op recipe than a 1-op
    one, so rejection throws away long recipes disproportionately). The
    project's "identical view coverage across compared rungs" constraint
    binds exactly the A3-vs-A3-LOTO comparison this exists for: the two banks
    must differ by one family and nothing else, not by overall augmentation
    strength.

    `pool` names which ops each family may realise as. The family draws are
    made first and exactly as before; a family with ONE candidate in the pool
    consumes no further draw, which is what keeps `"legacy"` bit-identical to
    the sampler before pools existed.
    """
    if not families:
        raise ValueError("families must not be empty")
    k = int(rng.integers(1, min(max_ops, len(families)) + 1))
    chosen = rng.choice(np.array(families), size=k, replace=False)
    ops = []
    for fam in chosen:
        cands = ops_for_family(str(fam), pool)
        if not cands:
            raise ValueError(f"pool {pool!r} has no op for family {fam!r}")
        name = cands[0] if len(cands) == 1 else cands[int(rng.integers(len(cands)))]
        ops.append(Op(name, _sample_params(name, rng)))
    return Recipe(tuple(ops))
