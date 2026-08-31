"""The suite has to be licence-clean and held-out-sane before a model loads.

Every failure caught here is one that a six-hour generation run would otherwise
surface at the end, after the GPU time was spent.
"""
import pytest

from aigcdet.generate import registry

from aigcdet.generate.registry import (HELDOUT_LINEAGE, METHODS, MODELS, SUITE,
                                       FamilySpec, family_of, heldout_groups,
                                       lineage_of, resolve_suite, validate_suite)


def test_the_shipped_suite_is_valid():
    validate_suite()


def test_every_registry_entry_names_a_real_licence():
    for key, spec in MODELS.items():
        assert spec.licence_tag, key
        assert spec.lineage, key
        assert spec.hf_id.count("/") == 1, key


def test_non_commercial_models_are_recorded_not_deleted():
    """`docs/02` §2 lists SDXL-Turbo as OpenRAIL++; its card says
    sai-nc-community. Deleting the entry would let the next person re-derive
    the same wrong conclusion from the same doc."""
    refused = {k for k, s in MODELS.items() if not s.commercial}
    assert {"sdxl_turbo", "flux1_dev", "flux2_klein_9b"} <= refused
    assert MODELS["sdxl_turbo"].licence_tag == "sai-nc-community"


def test_the_suite_uses_only_commercial_weights():
    for name, fam in SUITE.items():
        assert MODELS[fam.model].commercial, name


def test_a_non_commercial_family_is_refused():
    bad = dict(SUITE)
    bad["turbo_t2i"] = FamilySpec("sdxl_turbo", "t2i", 1.0, 4, 0.0)
    with pytest.raises(ValueError, match="refused"):
        validate_suite(bad)


def test_shares_must_sum_to_one():
    bad = {"a": FamilySpec("sdxl_base", "t2i", 0.7, 25, 6.0)}
    with pytest.raises(ValueError, match="sum"):
        validate_suite(bad)


def test_strength_belongs_to_img2img_and_nowhere_else():
    with pytest.raises(ValueError, match="strength"):
        validate_suite({"a": FamilySpec("sdxl_base", "t2i", 1.0, 25, 6.0,
                                        strength=0.7)})
    with pytest.raises(ValueError, match="strength"):
        validate_suite({"a": FamilySpec("sdxl_base", "img2img", 1.0, 25, 6.0)})


def test_holding_out_a_lineage_must_leave_at_least_two_behind():
    """Otherwise the held-out rung measures a jump from a single decoder, not
    the unseen-generator question spec §4.6 defines."""
    thin = {"klein4b_t2i": FamilySpec("flux2_klein_4b", "t2i", 0.5, 20, 1.0),
            "sdxl_t2i": FamilySpec("sdxl_base", "t2i", 0.5, 25, 6.0)}
    with pytest.raises(ValueError, match="lineage"):
        validate_suite(thin)


def test_heldout_group_is_a_whole_lineage():
    groups = heldout_groups()
    assert len(groups) == 1 and groups[0]
    assert all(lineage_of(f) == HELDOUT_LINEAGE for f in groups[0])
    # and nothing outside the group shares its decoder
    outside = set(SUITE) - set(groups[0])
    assert all(lineage_of(f) != HELDOUT_LINEAGE for f in outside)


def test_training_side_keeps_more_than_one_decoder():
    trained = {lineage_of(f) for f in SUITE} - {HELDOUT_LINEAGE}
    assert len(trained) >= 2, trained


def test_every_method_is_documented():
    assert {f.method for f in SUITE.values()} <= set(METHODS)
    assert all(METHODS[m] for m in METHODS)


@pytest.mark.parametrize("total", [7, 100, 2000, 10000, 13337])
def test_resolve_suite_sums_exactly_and_starves_nobody(total):
    counts = resolve_suite(total)
    assert sum(counts.values()) == total
    assert set(counts) == set(SUITE)
    assert min(counts.values()) >= 1


def test_resolve_suite_refuses_a_total_below_the_family_count():
    with pytest.raises(ValueError):
        resolve_suite(len(SUITE) - 1)


def test_heldout_families_clear_the_min_heldout_images_floor():
    """`splits.MIN_HELDOUT_IMAGES` is 200 and a held-out family below it is
    dropped, silently emptying the group."""
    from aigcdet.data.splits import MIN_HELDOUT_IMAGES
    counts = resolve_suite(10000)
    for fam in heldout_groups()[0]:
        assert counts[fam] >= MIN_HELDOUT_IMAGES, (fam, counts[fam])


def test_family_of_reports_the_suite_on_a_typo():
    with pytest.raises(KeyError, match="unknown family"):
        family_of("sdxl_t2img")


# --- the ov7 preset, checked against the registry it describes --------------

def test_ov7_preset_agrees_with_the_registry():
    """Two records of the same held-out design drift apart silently: the
    preset is what `build_dataset` obeys, the registry is what generated the
    images."""
    from aigcdet.data.presets import load_preset
    p = load_preset("configs/datasets/ov7.yaml")
    assert sorted(p.heldout_groups[0]) == heldout_groups()[0]
    assert set(p.heldout_groups[0]) <= set(SUITE)


def test_ov7_preset_splits_by_image_id():
    """Every fake here is generated FROM one real; a per-row draw puts the same
    scene in train under one label and in val under the other."""
    from aigcdet.data.presets import load_preset
    assert load_preset("configs/datasets/ov7.yaml").pair_split_by_stem is True


def test_pair_split_by_stem_is_off_by_default():
    """It changes the RNG stream, and every feature bank on disk fingerprints
    the manifest it was extracted against."""
    from aigcdet.data.presets import load_preset
    assert load_preset("configs/datasets/union.yaml").pair_split_by_stem is False


# --- the additive lineage supplement ---------------------------------------

def test_the_supplement_brings_lineages_the_frozen_suite_does_not_have():
    """The point of the supplement is decoder diversity, not image count. A
    family whose decoder is already in the corpus adds volume and nothing to
    the held-out axis."""
    have = {registry.lineage_of(n, registry.SUITE) for n in registry.SUITE}
    new = {registry.lineage_of(n, registry.LINEAGE_SUPPLEMENT)
           for n in registry.LINEAGE_SUPPLEMENT}
    assert new.isdisjoint(have), f"{new & have} is already in the corpus"
    assert new == {"movq", "dc_ae"}


def test_the_supplement_validates_only_against_the_corpus_it_extends():
    """Standalone it holds out a lineage it does not contain, which is exactly
    the refusal we want -- checked here so nobody 'fixes' it by loosening
    validate_suite."""
    registry.validate_suite(registry.LINEAGE_SUPPLEMENT,
                            corpus=registry.corpus_of("ov7_lineage"))
    with pytest.raises(ValueError, match="held-out lineage"):
        registry.validate_suite(registry.LINEAGE_SUPPLEMENT)


def test_corpus_of_is_the_union_and_leaves_the_frozen_suite_alone():
    c = registry.corpus_of("ov7_lineage")
    assert set(c) == set(registry.SUITE) | set(registry.LINEAGE_SUPPLEMENT)
    assert registry.corpus_of("ov7") == registry.SUITE
    # the frozen corpus is on disk; a mutation here re-cuts it
    assert set(registry.SUITE) == {
        "sdxl_t2i", "sd15_t2i", "sdxl_self_cond", "sdxl_img2img",
        "sd15_img2img", "klein4b_t2i", "klein4b_ref_image"}


def test_five_lineages_make_leave_one_out_a_distribution():
    """Three lineages give one rotation and a point estimate. The supplement
    exists to make that a distribution, and `validate_suite` needs two
    trainable lineages left for each one held out."""
    c = registry.corpus_of("ov7_lineage")
    lineages = {registry.lineage_of(n, c) for n in c}
    assert len(lineages) == 5
    for held in lineages:
        assert len(lineages - {held}) >= 2


def test_every_supplement_model_is_commercial_and_apache():
    for fam in registry.LINEAGE_SUPPLEMENT.values():
        m = registry.MODELS[fam.model]
        assert m.commercial and m.licence_tag == "apache-2.0", m.hf_id


def test_the_supplement_is_t2i_only():
    """An image-conditioned fake shares composition with its real, so it
    carries less of its decoder's fingerprint. For an arm whose only job is to
    represent a decoder, that dilutes the measurement."""
    assert {f.method for f in registry.LINEAGE_SUPPLEMENT.values()} == {"t2i"}


# --- the two hazards the smoke run found ------------------------------------

def test_sana_disables_resolution_binning():
    """Left on, a 432x640 request is generated at 1216x832 and resized back --
    var-Laplacian 2083.8 against 555.9 native. That is a resample on the
    generated class only, in a corpus whose worst confound is sharpness."""
    assert dict(registry.MODELS["sana_1600m"].call_kwargs)[
        "use_resolution_binning"] is False


def test_the_models_that_round_their_own_size_declare_it():
    assert registry.MODELS["sana_1600m"].size_multiple == 32
    assert registry.MODELS["kandinsky22"].size_multiple == 64
    # everything in the frozen suite must stay at 8, or its sizes change
    for fam in registry.SUITE.values():
        assert registry.MODELS[fam.model].size_multiple == 8


def test_kandinsky_declares_the_second_repo_its_pipeline_pulls():
    """The combined pipeline loads its prior from another repo; checking only
    hf_id licence-clears half the weights that made the image."""
    assert registry.MODELS["kandinsky22"].companion_ids == (
        "kandinsky-community/kandinsky-2-2-prior",)
