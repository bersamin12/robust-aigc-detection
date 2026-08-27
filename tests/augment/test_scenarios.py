import numpy as np

from aigcdet.augment.recipes import HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q
from aigcdet.augment.scenarios import (
    CORE_CONDITIONS, COMPOSITE_SCENARIOS, EVAL_GRID, HELDOUT_SEVERITY_CONDITIONS,
)


def test_core_has_clean_plus_the_briefs_fourteen():
    assert len(CORE_CONDITIONS) == 15
    assert CORE_CONDITIONS["clean"].ops == ()


def test_brief_parameters_reproduced_exactly():
    for q in (90, 70, 50, 30):
        assert CORE_CONDITIONS[f"jpeg_q{q}"].ops[0].params == {"quality": q}
    for s in (0.5, 1.0, 2.0):
        assert CORE_CONDITIONS[f"blur_s{s}"].ops[0].params == {"sigma": s}
    for sc in (0.5, 0.25):
        assert CORE_CONDITIONS[f"resize_{sc}"].ops[0].params == {"scale": sc}
    for s in (0.02, 0.05, 0.10):
        assert CORE_CONDITIONS[f"noise_s{s}"].ops[0].params == {"sigma": s}
    assert CORE_CONDITIONS["crop_80"].ops[0].params == {"frac": 0.8}
    j = CORE_CONDITIONS["jitter_20"].ops[0].params
    assert j == {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}


def test_five_named_composites_exist_and_chain_two_or_three_ops():
    assert set(COMPOSITE_SCENARIOS) == {
        "social_repost", "messaging_app", "screenshot", "filtered_upload", "low_light_share",
    }
    for r in COMPOSITE_SCENARIOS.values():
        assert 2 <= len(r.ops) <= 3


def test_eval_grid_is_the_union_of_twenty_conditions():
    assert len(EVAL_GRID) == 20
    assert set(EVAL_GRID) == set(CORE_CONDITIONS) | set(COMPOSITE_SCENARIOS)


def test_heldout_severity_conditions_are_flagged():
    # q=70 and sigma=1.0 sit inside the bands the training sampler excludes
    assert "jpeg_q70" in HELDOUT_SEVERITY_CONDITIONS
    assert "blur_s1.0" in HELDOUT_SEVERITY_CONDITIONS
    # and the two composites that use q=70 (spec §5: "which is deliberate")
    assert "social_repost" in HELDOUT_SEVERITY_CONDITIONS
    assert "filtered_upload" in HELDOUT_SEVERITY_CONDITIONS
    assert "jpeg_q30" not in HELDOUT_SEVERITY_CONDITIONS
    lo, hi = HELDOUT_JPEG_Q
    assert lo <= 70 <= hi
    lo, hi = HELDOUT_BLUR_SIGMA
    assert lo <= 1.0 <= hi


def test_every_condition_applies_cleanly():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for name, r in EVAL_GRID.items():
        out = r.apply(img, np.random.default_rng(0))
        assert out.shape == img.shape, name
