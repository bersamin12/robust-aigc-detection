"""Recipe pools: the NTIRE op set grouped under the six family bits.

The one guarantee that matters most is the first test: the `"legacy"` pool
must sample EXACTLY what the sampler sampled before pools existed, because
every bank on disk replays its views from `(seed, row_id, view_idx)` alone.
"""
import json

import numpy as np
import pytest

from aigcdet.augment import ops
from aigcdet.augment.recipes import (
    FAMILIES, FAMILY_OF, MAX_OPS_DEFAULT, N_FAMILIES, POOLS,
    RECIPE_POOL_DEFAULT, STOCHASTIC_OPS, Op, Recipe, ops_for_family,
    sample_training_recipe,
)

#: `sample_training_recipe(np.random.default_rng(seed))` x3, captured from the
#: sampler at commit 04c2afe (before pools existed). If this changes, every
#: bank extracted before this file existed no longer replays.
_LEGACY_SNAPSHOT = {
    0: ['[{"name": "jitter", "params": {"brightness": -0.052479145329279366, "contrast": 0.18691333659165826, "saturation": 0.14099536636507698}}, {"name": "blur", "params": {"sigma": 1.2872499829308457}}, {"name": "resize", "params": {"scale": 0.8577970754620494}}]',
        '[{"name": "jitter", "params": {"brightness": 0.05503783629581965, "contrast": 0.15944831696449163, "saturation": -0.17947683835248301}}, {"name": "crop", "params": {"frac": 0.874536080657291}}, {"name": "blur", "params": {"sigma": 0.7994237810747695}}]',
        '[{"name": "crop", "params": {"frac": 0.7785851535948997}}, {"name": "resize", "params": {"scale": 0.6859058695508597}}]'],
    1: ['[{"name": "resize", "params": {"scale": 0.34370374826776195}}, {"name": "jitter", "params": {"brightness": -0.19229741707058662, "contrast": -0.11349896734588635, "saturation": -0.17415538907306627}}]',
        '[{"name": "crop", "params": {"frac": 0.9233080149952055}}, {"name": "noise", "params": {"sigma": 0.06150504788802421}}]',
        '[{"name": "jitter", "params": {"brightness": -0.16826430551426066, "contrast": -0.09547922439374676, "saturation": -0.07010625458707472}}]'],
    2: ['[{"name": "blur", "params": {"sigma": 0.3838318842701938}}, {"name": "crop", "params": {"frac": 0.8880231209721003}}, {"name": "jpeg", "params": {"quality": 86}}]',
        '[{"name": "crop", "params": {"frac": 0.9012095934213863}}, {"name": "jpeg", "params": {"quality": 48}}, {"name": "noise", "params": {"sigma": 0.06403789459194494}}]',
        '[{"name": "jpeg", "params": {"quality": 81}}]'],
    3: ['[{"name": "blur", "params": {"sigma": 1.3643240721287355}}, {"name": "jitter", "params": {"brightness": -0.06411929633605988, "contrast": -0.12185769472112512, "saturation": 0.07396083719556179}}, {"name": "jpeg", "params": {"quality": 80}}]',
        '[{"name": "jpeg", "params": {"quality": 61}}]',
        '[{"name": "noise", "params": {"sigma": 0.06661385000100478}}, {"name": "jitter", "params": {"brightness": -0.16067566809382405, "contrast": 0.1934400882254148, "saturation": -0.14728208106197377}}]'],
    4: ['[{"name": "noise", "params": {"sigma": 0.013487782509038228}}, {"name": "jitter", "params": {"brightness": -0.14110337479925444, "contrast": -0.1702851810478711, "saturation": 0.07617917242160427}}, {"name": "crop", "params": {"frac": 0.8751065221756046}}]',
        '[{"name": "blur", "params": {"sigma": 1.7778934350886588}}, {"name": "jpeg", "params": {"quality": 59}}, {"name": "jitter", "params": {"brightness": 0.19762294998966823, "contrast": -0.19533993039743525, "saturation": 0.18935395816481293}}]',
        '[{"name": "resize", "params": {"scale": 0.7081620846170826}}]'],
}


def test_legacy_pool_is_bit_identical_to_the_sampler_before_pools_existed():
    for seed, want in _LEGACY_SNAPSHOT.items():
        rng = np.random.default_rng(seed)
        got = [sample_training_recipe(rng).to_json() for _ in range(3)]
        assert got == want, f"seed {seed}: the legacy sampler drifted"


def test_the_default_pool_is_legacy_and_three_deep():
    assert RECIPE_POOL_DEFAULT == "legacy"
    assert MAX_OPS_DEFAULT == 3
    assert POOLS["legacy"] == FAMILIES


def test_n_families_is_unchanged_by_the_ntire_pool():
    """The whole point of grouping: the head, the bank arrays and the
    calibrator width stay at six."""
    assert N_FAMILIES == 6
    r = Recipe((Op("motion_blur", {"length": 9, "angle": 0.0}),
                Op("pixelate", {"block": 4})))
    lab = r.labels()
    assert lab["presence"].shape == (6,) and lab["severity"].shape == (6,)
    assert lab["presence"][FAMILIES.index("blur")] == 1.0
    assert lab["presence"][FAMILIES.index("resize")] == 1.0
    assert lab["presence"].sum() == 2.0


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_every_pool_covers_every_family_with_known_ops(pool):
    for fam in FAMILIES:
        cands = ops_for_family(fam, pool)
        assert cands, f"pool {pool!r} has no op for family {fam!r}"
        for name in cands:
            assert name in ops.OP_FUNCS and FAMILY_OF[name] == fam


def test_pool_order_is_append_only():
    """Reordering a pool changes which op an rng draw selects, i.e. every
    recipe sampled under that pool. The brief's six lead every pool."""
    for pool, names in POOLS.items():
        assert names[:6] == FAMILIES, pool
    assert POOLS["ntire_nojp2"] == tuple(o for o in POOLS["ntire"] if o != "jpeg2000")


def test_ntire_pool_chains_one_to_five_distinct_families():
    rng = np.random.default_rng(5)
    lengths = set()
    for _ in range(1000):
        r = sample_training_recipe(rng, max_ops=5, pool="ntire")
        fams = [FAMILY_OF[o.name] for o in r.ops]
        assert 1 <= len(fams) <= 5
        assert len(set(fams)) == len(fams)
        lengths.add(len(fams))
    assert lengths == {1, 2, 3, 4, 5}


def test_ntire_pool_actually_uses_the_new_ops():
    rng = np.random.default_rng(0)
    seen = {o.name for _ in range(2000)
            for o in sample_training_recipe(rng, max_ops=5, pool="ntire").ops}
    assert seen == set(POOLS["ntire"])


def test_severity_stays_in_unit_range_for_every_op():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        lab = sample_training_recipe(rng, max_ops=5, pool="ntire").labels()
        assert (lab["severity"] >= 0).all() and (lab["severity"] <= 1).all()
        # presence and severity agree on which families are present
        assert ((lab["severity"] > 0) <= (lab["presence"] > 0)).all()


def test_sampled_recipes_round_trip_through_json_and_apply_reproducibly():
    """Every op's randomness other than pixel noise lives in its params, so
    a stored recipe replays from the json alone plus the (seed, row, view)
    rng, exactly as `features.replay` does."""
    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    for _ in range(60):
        r = sample_training_recipe(rng, max_ops=5, pool="ntire")
        back = Recipe.from_json(r.to_json())
        assert back == r
        json.loads(r.to_json())          # plain json, no numpy scalars
        a = back.apply(img, np.random.default_rng(3))
        b = back.apply(img, np.random.default_rng(3))
        assert a.shape == img.shape and a.dtype == np.uint8
        assert np.array_equal(a, b)


def test_stochastic_ops_take_rng_and_the_rest_do_not():
    import inspect
    for name, fn in ops.OP_FUNCS.items():
        takes_rng = "rng" in inspect.signature(fn).parameters
        assert takes_rng == (name in STOCHASTIC_OPS), name


def test_unknown_pool_and_orphan_op_are_refused():
    with pytest.raises(KeyError, match="unknown recipe pool"):
        sample_training_recipe(np.random.default_rng(0), pool="nope")
    with pytest.raises(KeyError):
        Op("not_an_op", {})


def test_loto_exclusion_still_restricts_the_family_pool_under_ntire():
    rng = np.random.default_rng(2)
    for _ in range(300):
        r = sample_training_recipe(rng, max_ops=5, families=("jpeg", "blur"),
                                   pool="ntire")
        assert all(FAMILY_OF[o.name] in ("jpeg", "blur") for o in r.ops)
