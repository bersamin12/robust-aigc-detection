# Amendment 11, Test 19, CPU check (run: PYTHONPATH=src python scripts/amend5_test19_check_identity.py).
"""Test 19 CPU check: a 196 px centre window at nominal 196 reaches the
dinov2regl196 tower with no pixel changed, while dinov2regl would squish it to 518."""
import numpy as np
from aigcdet.augment.canonical import CanonPolicy, canonicalise
from aigcdet.features.backbones import BACKBONES, squish

rng = np.random.default_rng(0)
image = rng.integers(0, 256, (900, 700, 3), dtype=np.uint8)
policy = CanonPolicy(mode="crop", crop_side=196, nominal_side=196, crop_clamp=True)
canonical_view = canonicalise(image, policy=policy)          # centre window (rng None)
top, left = (900 - 196) // 2, (700 - 196) // 2
native_window = image[top:top + 196, left:left + 196]
assert canonical_view.shape == (196, 196, 3)
assert np.array_equal(canonical_view, native_window), "canonicalise resampled the window"
tower_input = squish(canonical_view, BACKBONES["dinov2regl196"].image_size)
assert np.array_equal(tower_input, native_window), "the tower squish resampled the window"
print("dinov2regl196: window reaches the tower unchanged;",
      "dinov2regl would squish it to", BACKBONES["dinov2regl"].image_size)
