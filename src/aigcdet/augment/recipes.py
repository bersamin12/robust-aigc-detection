"""Recipes compose ops and carry their own supervision labels.

The severity normalisation below IS the degradation head's target definition
(spec §3.4), so it is fixed here once and imported everywhere else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from aigcdet.augment.ops import OP_FUNCS

FAMILIES: tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

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
    raise KeyError(name)


@dataclass(frozen=True)
class Op:
    name: str
    params: dict

    def __post_init__(self):
        if self.name not in OP_FUNCS:
            raise KeyError(f"unknown op {self.name!r}")


@dataclass(frozen=True)
class Recipe:
    ops: tuple[Op, ...] = ()

    def apply(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = img
        for op in self.ops:
            fn = OP_FUNCS[op.name]
            out = fn(out, rng=rng, **op.params) if op.name == "noise" else fn(out, **op.params)
        return out

    def labels(self) -> dict[str, np.ndarray]:
        presence = np.zeros(len(FAMILIES), dtype=np.float32)
        severity = np.zeros(len(FAMILIES), dtype=np.float32)
        for op in self.ops:
            i = FAMILIES.index(op.name)
            presence[i] = 1.0
            # If a family appears twice in a chain, keep the harsher instance.
            severity[i] = max(severity[i], _severity(op.name, op.params))
        return {"presence": presence, "severity": severity}

    def to_json(self) -> str:
        return json.dumps([{"name": o.name, "params": o.params} for o in self.ops])

    @classmethod
    def from_json(cls, s: str) -> "Recipe":
        return cls(tuple(Op(d["name"], d["params"]) for d in json.loads(s)))


def _sample_params(name: str, rng: np.random.Generator) -> dict:
    if name == "jpeg":
        lo, hi = HELDOUT_JPEG_Q
        while True:
            q = int(rng.integers(_JPEG_RANGE[0], _JPEG_RANGE[1] + 1))
            if not (lo <= q <= hi):
                return {"quality": q}
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
    raise KeyError(name)


def sample_training_recipe(rng: np.random.Generator, max_ops: int = 3,
                            families: tuple[str, ...] = FAMILIES) -> Recipe:
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
    """
    if not families:
        raise ValueError("families must not be empty")
    k = int(rng.integers(1, min(max_ops, len(families)) + 1))
    chosen = rng.choice(np.array(families), size=k, replace=False)
    return Recipe(tuple(Op(str(n), _sample_params(str(n), rng)) for n in chosen))
