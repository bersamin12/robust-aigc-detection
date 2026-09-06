"""Recipe pools: the NTIRE op set grouped under the six family bits.

The one guarantee that matters most is the first test: the `"legacy"` pool
must sample EXACTLY what the sampler sampled before pools existed, because
every bank on disk replays its views from `(seed, row_id, view_idx)` alone.
"""
import collections
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


#: 20 `ntire` recipes (seeds 0..19, max_ops 5) from the sampler before
#: `OP_WEIGHTS` existed. The ntire_meta bank was built from these draws.
_NTIRE_SNAPSHOT_20 = [
    "[{\"name\": \"impulse\", \"params\": {\"amount\": 0.04931759050984003}}, {\"name\": \"pixelate\", \"params\": {\"block\": 12}}, {\"name\": \"quantize\", \"params\": {\"bits\": 6}}, {\"name\": \"jpeg_multi\", \"params\": {\"qualities\": [94, 49]}}, {\"name\": \"glass_blur\", \"params\": {\"sigma\": 0.3024646501531333, \"max_delta\": 2, \"iters\": 2}}]",
    "[{\"name\": \"iso_noise\", \"params\": {\"intensity\": 0.19032415340471848, \"color_shift\": 0.016933057958903028}}, {\"name\": \"perspective\", \"params\": {\"offsets\": [0.06554051876408837, -0.01816017272616774, 0.009918737534611893, -0.09448817735138633, 0.05070262173496132, 0.007628662643855644, -0.03405365670018157, 0.05768574068568086]}}, {\"name\": \"resize\", \"params\": {\"scale\": 0.5447736281624235}}]",
    "[{\"name\": \"channel_shift\", \"params\": {\"dr\": 11.57653756783112, \"dg\": 6.9301319566573865, \"db\": -28.01015552064574}}, {\"name\": \"crop\", \"params\": {\"frac\": 0.7845143205602273}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.5312100140231117, \"interp_down\": \"area\", \"interp_up\": \"area\"}}, {\"name\": \"jpeg2000\", \"params\": {\"rate\": 192}}, {\"name\": \"blur\", \"params\": {\"sigma\": 2.1348719049873535}}]",
    "[{\"name\": \"resize\", \"params\": {\"scale\": 0.5613833437915421}}, {\"name\": \"contrast\", \"params\": {\"alpha\": 0.6597389146370786}}, {\"name\": \"jpeg2000\", \"params\": {\"rate\": 305}}, {\"name\": \"noise\", \"params\": {\"sigma\": 0.04607896000204451}}, {\"name\": \"crop\", \"params\": {\"frac\": 0.8688502420029136}}]",
    "[{\"name\": \"impulse\", \"params\": {\"amount\": 0.03136595358142726}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.7712357845407747, \"interp_down\": \"area\", \"interp_up\": \"nearest\"}}, {\"name\": \"quantize\", \"params\": {\"bits\": 7}}, {\"name\": \"crop\", \"params\": {\"frac\": 0.9575094683346773}}]",
    "[{\"name\": \"perspective\", \"params\": {\"offsets\": [-0.023326223842896357, -0.01830535891600027, -0.09094496121951097, -0.0902484578545664, 0.09983522301301428, 0.030473822317597543, -0.05309795966603521, -0.013010489554971597]}}, {\"name\": \"resize\", \"params\": {\"scale\": 0.883221025618516}}, {\"name\": \"jpeg\", \"params\": {\"quality\": 91}}, {\"name\": \"channel_shift\", \"params\": {\"dr\": 18.734163251717234, \"dg\": 22.255805655610992, \"db\": 7.128094953531962}}]",
    "[{\"name\": \"quantize\", \"params\": {\"bits\": 4}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.6612915771946449, \"interp_down\": \"cubic\", \"interp_up\": \"linear\"}}, {\"name\": \"perspective\", \"params\": {\"offsets\": [-0.03400730892290833, 0.035983532232804455, -0.075405525022562, -0.08965415360324497, 0.07003826936175742, -0.09822083889291307, 0.09575351243705935, 0.06540060516381072]}}]",
    "[{\"name\": \"resize\", \"params\": {\"scale\": 0.2534224479676236}}, {\"name\": \"iso_noise\", \"params\": {\"intensity\": 0.41955278827224485, \"color_shift\": 0.03188277715008185}}, {\"name\": \"jitter\", \"params\": {\"brightness\": -0.09545486402289705, \"contrast\": 0.09176384181511602, \"saturation\": -0.116761445882397}}, {\"name\": \"motion_blur\", \"params\": {\"length\": 9, \"angle\": 99.62952337340865}}, {\"name\": \"aspect_crop\", \"params\": {\"top\": 0.1417201019315947, \"left\": 0.04737207913570733, \"h_frac\": 0.8566978636461888, \"w_frac\": 0.7799806532485232}}]",
    "[{\"name\": \"jpeg_multi\", \"params\": {\"qualities\": [90, 33]}}, {\"name\": \"contrast\", \"params\": {\"alpha\": 0.9378818731227988}}, {\"name\": \"motion_blur\", \"params\": {\"length\": 7, \"angle\": 19.251647365099384}}, {\"name\": \"impulse\", \"params\": {\"amount\": 0.020825467316605215}}]",
    "[{\"name\": \"lens_blur\", \"params\": {\"radius\": 4}}, {\"name\": \"quantize\", \"params\": {\"bits\": 6}}, {\"name\": \"iso_noise\", \"params\": {\"intensity\": 0.4371771421322981, \"color_shift\": 0.0367295051619505}}]",
    "[{\"name\": \"channel_shift\", \"params\": {\"dr\": 9.757186140717732, \"dg\": -29.11627679233959, \"db\": 19.89281490917418}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.7864663891501189, \"interp_down\": \"cubic\", \"interp_up\": \"nearest\"}}, {\"name\": \"motion_blur\", \"params\": {\"length\": 12, \"angle\": 135.59433564752584}}, {\"name\": \"aspect_crop\", \"params\": {\"top\": 0.018530790104855404, \"left\": 0.059611142079645255, \"h_frac\": 0.8721967716611579, \"w_frac\": 0.9200473118832115}}]",
    "[{\"name\": \"jpeg2000\", \"params\": {\"rate\": 220}}]",
    "[{\"name\": \"impulse\", \"params\": {\"amount\": 0.01998221723401266}}, {\"name\": \"clahe\", \"params\": {\"clip\": 3.011337228318354, \"grid\": 8}}, {\"name\": \"crop\", \"params\": {\"frac\": 0.9561511559520764}}, {\"name\": \"jpeg\", \"params\": {\"quality\": 89}}]",
    "[{\"name\": \"jpeg2000\", \"params\": {\"rate\": 381}}, {\"name\": \"impulse\", \"params\": {\"amount\": 0.002205198782851378}}, {\"name\": \"lens_blur\", \"params\": {\"radius\": 5}}, {\"name\": \"pixelate\", \"params\": {\"block\": 3}}, {\"name\": \"saturation\", \"params\": {\"scale\": 0.7435559249994046}}]",
    "[{\"name\": \"quantize\", \"params\": {\"bits\": 4}}]",
    "[{\"name\": \"saturation\", \"params\": {\"scale\": 1.2625256522505657}}, {\"name\": \"noise\", \"params\": {\"sigma\": 0.041262432927963166}}, {\"name\": \"perspective\", \"params\": {\"offsets\": [0.09518757747803505, 0.056293205554406145, 0.0687580090527952, 0.011022876814972049, 0.08832427251717864, -0.09633483612497805, 0.07783678117837226, -0.022048303330017283]}}, {\"name\": \"motion_blur\", \"params\": {\"length\": 8, \"angle\": 96.19096090560497}}, {\"name\": \"resize\", \"params\": {\"scale\": 0.8640486811242918}}]",
    "[{\"name\": \"brightness\", \"params\": {\"beta\": -42.804506149318044}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.8185110456645995, \"interp_down\": \"nearest\", \"interp_up\": \"nearest\"}}, {\"name\": \"perspective\", \"params\": {\"offsets\": [-0.09113915684234304, 0.06048419257318083, -0.06304096770263029, 0.03912478297971883, -0.06899994902686667, 0.0383239511966533, 0.09172676621288409, 0.09699699754998956]}}]",
    "[{\"name\": \"jpeg\", \"params\": {\"quality\": 44}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.5283069244108455, \"interp_down\": \"linear\", \"interp_up\": \"area\"}}, {\"name\": \"aspect_crop\", \"params\": {\"top\": 0.04283799414418708, \"left\": 0.2979161950891908, \"h_frac\": 0.8313736417309789, \"w_frac\": 0.5068803454685202}}, {\"name\": \"tone_curve\", \"params\": {\"low_y\": 0.8982109447383813, \"high_y\": 0.7658768996005632}}]",
    "[{\"name\": \"quantize\", \"params\": {\"bits\": 5}}, {\"name\": \"lens_blur\", \"params\": {\"radius\": 4}}, {\"name\": \"iso_noise\", \"params\": {\"intensity\": 0.2639124363784712, \"color_shift\": 0.00489607959391428}}, {\"name\": \"jpeg_multi\", \"params\": {\"qualities\": [85, 51]}}, {\"name\": \"downscale\", \"params\": {\"scale\": 0.8398028714814155, \"interp_down\": \"area\", \"interp_up\": \"area\"}}]",
    "[{\"name\": \"perspective\", \"params\": {\"offsets\": [-0.03789130833737762, 0.043637051574415386, 0.056193116353636635, 0.007739967215918589, -0.0376669164352359, 0.08326994587782993, 0.08561510230393149, -0.012695804858767826]}}, {\"name\": \"blur\", \"params\": {\"sigma\": 1.4264904516558148}}, {\"name\": \"saturation\", \"params\": {\"scale\": 1.144141415879663}}]"
]


# --- ntire2 ------------------------------------------------------------------

def test_ntire2_appends_to_ntire_and_declares_its_pre_ops():
    from aigcdet.augment.recipes import (
        OP_WEIGHTS, PRE_CANONICAL_OPS, pool_has_pre_ops)
    assert POOLS["ntire2"][:len(POOLS["ntire"])] == POOLS["ntire"]
    assert set(POOLS["ntire2"]) - set(POOLS["ntire"]) == {"jpeg_ai", "downscale_keep"}
    assert PRE_CANONICAL_OPS == {"downscale_keep"}
    assert pool_has_pre_ops("ntire2") and not pool_has_pre_ops("ntire")
    assert not pool_has_pre_ops("legacy")
    assert set(OP_WEIGHTS["ntire2"]) <= set(POOLS["ntire2"])
    with pytest.raises(KeyError):
        pool_has_pre_ops("nope")


def test_ntire2_weights_the_new_ops_towards_ntire_frequencies():
    rng = np.random.default_rng(11)
    n = collections.Counter()
    for _ in range(20000):
        for o in sample_training_recipe(rng, max_ops=5, pool="ntire2").ops:
            n[o.name] += 1
    resize = sum(n[o] for o in POOLS["ntire2"] if FAMILY_OF[o] == "resize")
    jpeg = sum(n[o] for o in POOLS["ntire2"] if FAMILY_OF[o] == "jpeg")
    assert abs(n["downscale_keep"] / resize - 0.5) < 0.03      # 3 of 6
    assert abs(n["jpeg_ai"] / jpeg - 0.4) < 0.03               # 2 of 5
    assert abs(n["resize"] / resize - 1 / 6) < 0.03


def test_ntire_is_untouched_by_the_weights():
    """A weighted pool draws with rng.choice; an unweighted one keeps the
    integer draw, so `ntire` recipes are the ones its bank was built from."""
    a = [sample_training_recipe(np.random.default_rng(s), max_ops=5, pool="ntire").to_json()
         for s in range(20)]
    assert a == _NTIRE_SNAPSHOT_20


def test_a_recipe_with_a_pre_op_splits_and_refuses_apply():
    from aigcdet.augment.recipes import Op
    r = Recipe((Op("jpeg", {"quality": 50}),
                Op("downscale_keep", {"scale": 0.5, "interp": "area"}),
                Op("blur", {"sigma": 1.0})))
    pre, post = r.split()
    assert [o.name for o in pre.ops] == ["downscale_keep"]
    assert [o.name for o in post.ops] == ["jpeg", "blur"]
    img = np.random.default_rng(0).integers(0, 256, (64, 48, 3), dtype=np.uint8)
    assert r.apply_pre(img).shape == (32, 24, 3)
    assert r.apply_post(img, np.random.default_rng(0)).shape == img.shape
    with pytest.raises(ValueError, match="pre-canonical"):
        r.apply(img, np.random.default_rng(0))
    # Labels cover the whole chain: resize present, at the pre op's severity.
    lab = r.labels()
    assert lab["presence"][FAMILIES.index("resize")] == 1.0
    assert lab["severity"][FAMILIES.index("resize")] == pytest.approx((1 - 0.5) / 0.7)
    assert lab["severity"][FAMILIES.index("jpeg")] > 0


def test_ntire2_severities_stay_in_range_and_round_trip():
    rng = np.random.default_rng(4)
    for _ in range(500):
        r = sample_training_recipe(rng, max_ops=5, pool="ntire2")
        lab = r.labels()
        assert (lab["severity"] >= 0).all() and (lab["severity"] <= 1).all()
        assert Recipe.from_json(r.to_json()) == r
        for o in r.ops:
            if o.name == "downscale_keep":
                assert 0.3 <= o.params["scale"] <= 0.8
            if o.name == "jpeg_ai":
                assert o.params["quality"] in (1, 2, 3, 4)
