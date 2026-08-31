"""Generated counterparts for Open Images V7 reals (docs/02, docs/03).

Every fake is generated **from a specific real** and carries that real's
ImageID. Content, pixel dimensions and JPEG encoder are held fixed across the
pair, so what is left between them is the generator.

The logic lives here rather than in the notebook it was ported from, for the
reason `notebooks/kaggle_bootstrap.py` gives: a notebook cell cannot be
mutation-tested, and the two failures that killed the first attempt --- a
dimension leak and a DCT phase mismatch --- were both in pure functions that a
test would have caught. `geometry` and `encode` are those functions.
"""
from aigcdet.generate.geometry import (MCU_ALIGN, box_mask, crop_box, order_key,
                                       seed_for)
from aigcdet.generate.registry import (METHODS, MODELS, family_of, resolve_suite,
                                       validate_suite)

__all__ = ["MCU_ALIGN", "box_mask", "crop_box", "order_key", "seed_for",
           "METHODS", "MODELS", "family_of", "resolve_suite", "validate_suite"]
