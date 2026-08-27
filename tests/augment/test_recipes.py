import numpy as np
import pytest

from aigcdet.augment.recipes import (
    FAMILIES, HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q, Op, Recipe,
    sample_training_recipe,
)


def test_families_order_is_fixed():
    assert FAMILIES == ("jpeg", "blur", "resize", "noise", "jitter", "crop")


def test_empty_recipe_is_identity_and_labels_all_zero():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    r = Recipe(ops=())
    assert np.array_equal(r.apply(img, rng), img)
    lab = r.labels()
    assert lab["presence"].shape == (6,) and lab["severity"].shape == (6,)
    assert lab["presence"].sum() == 0 and lab["severity"].sum() == 0


def test_labels_mark_presence_and_normalised_severity():
    r = Recipe(ops=(Op("jpeg", {"quality": 30}), Op("blur", {"sigma": 2.0})))
    lab = r.labels()
    i_jpeg, i_blur = FAMILIES.index("jpeg"), FAMILIES.index("blur")
    assert lab["presence"][i_jpeg] == 1.0 and lab["presence"][i_blur] == 1.0
    # q=30 is the harshest listed quality -> severity 1.0; sigma=2.0 likewise
    assert lab["severity"][i_jpeg] == pytest.approx(1.0)
    assert lab["severity"][i_blur] == pytest.approx(1.0)
    assert lab["presence"][FAMILIES.index("noise")] == 0.0


def test_severity_is_monotone_in_harshness():
    s90 = Recipe((Op("jpeg", {"quality": 90}),)).labels()["severity"][0]
    s30 = Recipe((Op("jpeg", {"quality": 30}),)).labels()["severity"][0]
    assert s30 > s90


def test_recipe_json_roundtrip():
    r = Recipe((Op("jpeg", {"quality": 50}), Op("noise", {"sigma": 0.05})))
    back = Recipe.from_json(r.to_json())
    assert back == r


def test_apply_is_deterministic_for_a_given_rng_seed():
    img = np.random.default_rng(3).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    r = Recipe((Op("noise", {"sigma": 0.05}),))
    a = r.apply(img, np.random.default_rng(11))
    b = r.apply(img, np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_sampler_never_draws_heldout_bands():
    rng = np.random.default_rng(1234)
    lo_q, hi_q = HELDOUT_JPEG_Q
    lo_s, hi_s = HELDOUT_BLUR_SIGMA
    for _ in range(3000):
        for op in sample_training_recipe(rng).ops:
            if op.name == "jpeg":
                assert not (lo_q <= op.params["quality"] <= hi_q)
            if op.name == "blur":
                assert not (lo_s <= op.params["sigma"] <= hi_s)


def test_sampler_chains_one_to_three_distinct_families():
    rng = np.random.default_rng(5)
    for _ in range(500):
        r = sample_training_recipe(rng)
        names = [o.name for o in r.ops]
        assert 1 <= len(names) <= 3
        assert len(set(names)) == len(names)  # distinct families


def test_sampler_output_applies_cleanly():
    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for _ in range(50):
        out = sample_training_recipe(rng).apply(img, rng)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_sample_training_recipe_can_be_restricted_to_a_family_subset():
    """The leave-one-transform-out bank restricts the family POOL rather than
    rejection-sampling whole recipes (see
    `aigcdet.features.extract._sample_recipe_excluding`), so this parameter is
    what keeps the LOTO comparison unconfounded."""
    rng = np.random.default_rng(0)
    kept = ("jpeg", "blur")
    lengths = set()
    for _ in range(500):
        r = sample_training_recipe(rng, families=kept)
        assert all(o.name in kept for o in r.ops)
        lengths.add(len(r.ops))
    # max_ops is clamped to the number of available families: with 2 kept
    # families a 3-op chain of DISTINCT families is impossible.
    assert lengths == {1, 2}


def test_sample_training_recipe_rejects_an_empty_family_pool():
    with pytest.raises(ValueError, match="families must not be empty"):
        sample_training_recipe(np.random.default_rng(0), families=())
