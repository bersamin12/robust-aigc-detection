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


# --- the second lineage supplement -----------------------------------------

def test_supplement_2_brings_one_more_decoder_class():
    """Was two (`paella_vq` + `cogview_vae`) until cogview4_t2i was removed on
    throughput, 2026-08-31. The removal is what this asserts: the suite still
    contributes a lineage nothing before it had, and it contributes exactly
    one."""
    have = {registry.lineage_of(n, registry.corpus_of("ov7_lineage"))
            for n in registry.corpus_of("ov7_lineage")}
    new = {registry.lineage_of(n, registry.LINEAGE_SUPPLEMENT_2)
           for n in registry.LINEAGE_SUPPLEMENT_2}
    assert new.isdisjoint(have)
    assert new == {"paella_vq"}


def test_cogview4_is_refused_but_still_registered():
    """Removed from the suite, kept in MODELS. The refusal is on THROUGHPUT --
    it is the only decoder in this registry that was dropped while still being
    a genuine unseen lineage, so the entry has to survive to say so."""
    assert "cogview4_t2i" not in registry.LINEAGE_SUPPLEMENT_2
    spec = registry.MODELS["cogview4_6b"]
    assert spec.commercial and spec.licence_tag == "apache-2.0"
    assert spec.note.startswith("REFUSED ON THROUGHPUT")
    assert spec.lineage not in {registry.lineage_of(n, registry.corpus_of(s))
                                for s in registry.SUITES
                                for n in registry.corpus_of(s)}


def test_six_lineages_after_the_second_supplement():
    c = registry.corpus_of("ov7_lineage2")
    assert len({registry.lineage_of(n, c) for n in c}) == 6


def test_supplement_2_validates_against_everything_before_it():
    registry.validate_suite(registry.LINEAGE_SUPPLEMENT_2,
                            corpus=registry.corpus_of("ov7_lineage2"))
    with pytest.raises(ValueError, match="held-out lineage"):
        registry.validate_suite(registry.LINEAGE_SUPPLEMENT_2)


def test_wuerstchen_carries_guidance_under_its_own_kwarg():
    """Its combined pipeline has NO `guidance_scale` -- prior and decoder are
    separate -- and diffusers swallows unknown kwargs instead of raising. The
    usual name would generate the whole family at the pipeline default while
    the manifest recorded ours: a silent lie in every row."""
    m = registry.MODELS["wuerstchen"]
    assert m.guidance_kw == "prior_guidance_scale"
    assert dict(m.call_kwargs)["decoder_guidance_scale"] == 0.0
    assert m.companion_ids == ("warp-ai/wuerstchen-prior",)


def test_every_other_model_uses_the_ordinary_guidance_kwarg():
    for k, m in registry.MODELS.items():
        if k != "wuerstchen":
            assert m.guidance_kw == "guidance_scale", k


def test_cogview4_uses_model_offload_not_sequential():
    """29 GiB across the repo, but the largest single component still fits, so
    sequential would cost far more than it needs to."""
    m = registry.MODELS["cogview4_6b"]
    assert m.offload_mode == "model" and m.vram_gb > 20


def test_offload_mode_is_one_of_the_two_supported():
    for k, m in registry.MODELS.items():
        assert m.offload_mode in ("model", "sequential"), k


def test_the_redundant_lineages_are_recorded_not_deleted():
    """Each is apache-2.0 and usable, and each would add volume to a lineage
    the corpus already has. The reason has to survive, or the next person
    re-derives it."""
    for k, lineage in (("lumina2", "flux1_vae"), ("kolors", "sdxl_vae"),
                       ("shuttle3", "flux1_vae")):
        m = registry.MODELS[k]
        assert m.lineage == lineage
        assert "REFUSED" in m.note
        assert k not in registry.corpus_of("ov7_lineage2")


# --- the scale-up's refusals ------------------------------------------------
# `docs/02` U11/U12. Each of these is the model somebody scaling this corpus
# reaches for first, which is why silence about them is the failure mode.

def test_the_scale_up_candidates_are_recorded_refused_not_deleted():
    for key in ("qwen_image_2512", "wan22_ti2v_5b"):
        assert key in MODELS, key
        assert key not in {f.model for s in registry.SUITES.values()
                           for f in s.values()}, f"{key} is refused"


def test_the_refusals_are_on_hardware_or_lineage_not_licence():
    """All three are apache-2.0. Recording them as non-commercial would be a
    licence claim that is simply false, and the next reader would act on it."""
    for key in ("qwen_image_2512", "wan22_ti2v_5b"):
        assert MODELS[key].licence_tag == "apache-2.0", key
        assert MODELS[key].commercial, key
        assert "REFUSED" in MODELS[key].note, key


def test_zimage_is_labelled_flux1_vae_even_though_it_is_now_generating():
    """Its vae/config.json carries `_name_or_path: "flux-dev"`, so the VAE is
    FLUX.1-dev's by the config's own provenance. Relabelling it to make the
    eighth lineage look independent would be the registry lying about the one
    field `heldout_groups()` reads."""
    assert MODELS["zimage_turbo"].lineage == "flux1_vae"
    assert MODELS["zimage_turbo"].lineage == MODELS["lumina2"].lineage


def test_zimage_declares_the_granularity_its_pipeline_actually_needs():
    """8x VAE downsample and `all_patch_size: [2]` makes 16. The default 8
    would be a size the pipeline silently rounds -- the Kandinsky failure
    (§11) with a different divisor, and the same invisible result: two
    different requests coming back as the same image."""
    assert MODELS["zimage_turbo"].size_multiple == 16
    assert MODELS["zimage_turbo"].arch == "zimage_dit"


def test_a_turbo_family_records_the_guidance_the_model_actually_takes():
    """Turbo models take no CFG; the card says guidance_scale=0.0. A nonzero
    value would be written into every manifest row while the model ignored
    it."""
    fam = registry.LINEAGE_SUPPLEMENT_3["zimage_t2i"]
    assert fam.guidance == 0.0 and fam.steps == 9


def test_training_a_cousin_of_the_held_out_lineage_warns():
    """`zimage_t2i` puts flux1_vae into training while flux2_vae is held out,
    and nobody has measured whether those are the same decoder. Deliberate, so
    it warns rather than raising -- but it must not be silent."""
    import warnings
    corpus = registry.corpus_of("ov7_lineage3")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_suite(registry.SUITES["ov7_lineage3"], corpus=corpus)
    assert any("flux1_vae" in str(w.message) for w in caught)


def test_the_suites_without_zimage_do_not_warn():
    """The warning has to mean something, so it must not fire on the nine
    families already frozen."""
    import warnings
    for name in ("ov7", "ov7_lineage", "ov7_lineage2"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_suite(registry.SUITES[name], corpus=registry.corpus_of(name))
        assert not caught, f"{name} warned: {[str(w.message) for w in caught]}"


def test_qwen_2512_cannot_be_rescued_by_model_level_offload():
    """Its largest single component IS the 40.9 GB transformer, so there is no
    component split that fits a 24 GB card. Declaring "model" here would send
    a run at it that OOMs on the first image."""
    assert MODELS["qwen_image_2512"].offload_mode == "sequential"
    assert MODELS["qwen_image_2512"].vram_gb > 40


def test_no_two_registry_lineages_collide_by_accident():
    """Wan2.1's VAE and Qwen-Image's are the same frozen weights -- identical
    latents_mean across all sixteen channels -- so `wan_vae` names one
    decoder. Wan2.2's is a different autoencoder despite the shared diffusers
    class name, hence a separate key. If these ever merge, holding one out
    stops measuring a lineage jump."""
    assert MODELS["qwen_image_2512"].lineage == "wan_vae"
    assert MODELS["wan22_ti2v_5b"].lineage == "wan22_vae"


# --- the scale-up suite -----------------------------------------------------

def test_ov7_full_holds_every_family_and_lineage():
    c = registry.corpus_of("ov7_full")
    assert set(c) == set(registry.SUITES["ov7_full"])
    assert len(c) == 11
    assert len({registry.lineage_of(n, c) for n in c}) == 7
    assert registry.HELDOUT_LINEAGE in {registry.lineage_of(n, c) for n in c}


def test_ov7_full_shares_sum_to_one():
    assert round(sum(f.share for f in registry.SUITES["ov7_full"].values()), 6) == 1.0


#: Pairs per lineage already on disk when the scale-up was planned, and the
#: size of the free pool it was solved against. `ov7_full`'s shares are only
#: balanced AT THIS TOTAL -- see the note on `registry.LINEAGE_FULL`.
FULL_DONE = {"sdxl_vae": 5400, "sd_vae": 2778, "flux2_vae": 1800,
             "movq": 1000, "dc_ae": 1000, "paella_vq": 0, "flux1_vae": 0}
FULL_TOTAL = 42646


def _ov7_full_lineage_totals(total):
    s = registry.SUITES["ov7_full"]
    out = dict(FULL_DONE)
    for n, f in s.items():
        out[registry.lineage_of(n, s)] += f.share * total
    return out


def test_ov7_full_shares_balance_the_lineages():
    """The shares are solved backwards from a balanced target, so the family
    that is LARGEST in the frozen corpus must be near-smallest here."""
    s = registry.SUITES["ov7_full"]
    assert s["sdxl_t2i"].share < s["wuerstchen_t2i"].share
    # Both lineages start at zero, so they are owed the same number of pairs.
    # The shares are not exactly equal: they are rounded to 4 dp and the slack
    # that makes the suite sum to exactly 1.0 (asserted just above) is absorbed
    # into one of them. One part in ten thousand, or 4 images over the run.
    assert round(abs(s["zimage_t2i"].share - s["wuerstchen_t2i"].share), 6) <= 1e-4
    final = _ov7_full_lineage_totals(FULL_TOTAL).values()
    # 4-dp shares over 42,646 reals cannot land closer than a few pairs.
    assert max(final) - min(final) < 20, sorted(final)


def test_ov7_full_shares_do_not_survive_a_different_total():
    """The failure this pins down is silent: reuse these shares at another
    total and every family still runs, every count still sums, and the corpus
    just comes out lopsided. The suite carried a docstring solved for 32,774
    long after the code had been re-solved for 42,646, and nothing caught it
    because the arithmetic lives in a comment. If a later run legitimately
    changes the total, this test is SUPPOSED to fail: re-solve the shares and
    move FULL_TOTAL, do not widen the bound."""
    off = _ov7_full_lineage_totals(32774).values()
    assert max(off) - min(off) > 1000, sorted(off)


def test_ov7_full_deals_every_family_on_one_shard():
    """Largest-remainder must not round any family to zero at a shard's size."""
    c = registry.corpus_of("ov7_full")
    counts = registry.resolve_suite(10925, registry.SUITES["ov7_full"], corpus=c)
    assert sum(counts.values()) == 10925
    assert min(counts.values()) > 0
