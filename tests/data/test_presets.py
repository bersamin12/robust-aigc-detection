"""`aigcdet.data.presets` -- the corpus composition a manifest was built from.

The theme of every test here is the same as `test_sources.py`'s: a knob that
silently does nothing is worse than one that raises. A cap on a source that is
not registered, a hold-out on a pseudo-generator, a misspelled field name --
each produces a corpus that disagrees with the file describing it, and the
file is the only record of the composition there is.
"""
from __future__ import annotations

import glob
import os

import pytest
import yaml

from aigcdet.data.normalize import SHORT_SIDE
from aigcdet.data.presets import (
    DEFAULT_CAP_KEY, DatasetPreset, load_preset,
)
from aigcdet.data.sources import PSEUDO_GENERATORS, SOURCES

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "datasets")
SHIPPED = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))


def _ok(**kw) -> DatasetPreset:
    return DatasetPreset(name=kw.pop("name", "t"), note=kw.pop("note", "why"), **kw)


# --------------------------------------------------------------------------
# the cap mapping
# --------------------------------------------------------------------------

def test_an_int_cap_is_accepted_and_means_every_family():
    """The CLI passes an int and every existing caller does too; the mapping
    is a widening, not a replacement."""
    p = _ok(max_per_generator=1800)
    assert p.max_per_generator == {DEFAULT_CAP_KEY: 1800}
    assert p.cap_for("BigGAN") == 1800
    assert p.cap_for("anything-at-all") == 1800


def test_int_zero_means_uncapped_and_does_not_become_a_zero_cap():
    """0 has meant "keep everything" since the flag was added. Turning it into
    `{"*": 0}` would read the same through `cap_for`, but it would also make
    `if caps:` true in build_dataset and print a cap line for a cap that is
    not in force."""
    p = _ok(max_per_generator=0)
    assert p.max_per_generator == {}
    assert p.cap_for("BigGAN") == 0


def test_a_named_family_overrides_the_default_and_zero_exempts_it():
    """The whole reason the field is a mapping: `sid_set` is a pseudo-
    generator naming a SOURCE, so a per-family cap on it thins a dataset."""
    p = _ok(max_per_generator={DEFAULT_CAP_KEY: 1500, "sid_set": 0})
    assert p.cap_for("BigGAN") == 1500
    assert p.cap_for("sid_set") == 0


def test_a_mapping_without_a_default_caps_only_what_it_names():
    p = _ok(max_per_generator={"BigGAN": 100})
    assert p.cap_for("BigGAN") == 100
    assert p.cap_for("styleGAN") == 0


def test_named_generators_reports_caps_and_holdouts_but_not_the_wildcard():
    """build_dataset checks this list against the scanned corpus, so the
    wildcard must not appear in it -- there is no family called "*"."""
    p = _ok(max_per_generator={DEFAULT_CAP_KEY: 10, "BigGAN": 5},
            heldout_generators=["VQGAN"])
    assert p.named_generators == ["BigGAN", "VQGAN"]


@pytest.mark.parametrize("bad", [-1, "1500", 1.5, True])
def test_a_cap_that_is_not_a_non_negative_int_raises(bad):
    with pytest.raises(ValueError, match="non-negative int"):
        _ok(max_per_generator={DEFAULT_CAP_KEY: bad})


def test_a_bool_cap_is_rejected_even_though_bool_is_an_int():
    """`max_real_per_source: {wildfake: true}` is a typo, and True == 1 would
    silently thin the source to a single image."""
    with pytest.raises(ValueError, match="non-negative int"):
        _ok(max_real_per_source={"wildfake": True})


# --------------------------------------------------------------------------
# source caps
# --------------------------------------------------------------------------

def test_a_real_cap_on_an_unregistered_source_raises_and_names_it():
    with pytest.raises(ValueError, match="wildfke"):
        _ok(max_real_per_source={"wildfke": 1000})


def test_every_registered_source_is_accepted_as_a_real_cap_key():
    for src in SOURCES:
        _ok(max_real_per_source={src: 10})


# --------------------------------------------------------------------------
# the sub-band floor
# --------------------------------------------------------------------------

def test_a_floor_above_the_normalisation_cap_raises():
    """normalize caps the short side at SHORT_SIDE and never upscales, so a
    floor above it drops the whole corpus rather than its tail."""
    with pytest.raises(ValueError, match="outside"):
        _ok(min_short_side=SHORT_SIDE + 1)


def test_a_floor_exactly_at_the_normalisation_cap_is_allowed():
    assert _ok(min_short_side=SHORT_SIDE).min_short_side == SHORT_SIDE


def test_a_negative_floor_raises():
    with pytest.raises(ValueError, match="outside"):
        _ok(min_short_side=-1)


# --------------------------------------------------------------------------
# held-out families
# --------------------------------------------------------------------------

def test_pinning_a_pseudo_generator_as_heldout_raises():
    """`choose_heldout_generators` already refuses to DRAW one, because
    holding it out removes an entire source and measures dataset shift rather
    than the unseen-generator generalisation of spec 4.6. A pinned list is the
    way round that refusal, so it is refused here too."""
    pseudo = sorted(PSEUDO_GENERATORS)[0]
    with pytest.raises(ValueError, match="pseudo-generator"):
        _ok(heldout_generators=[pseudo])


def test_an_empty_heldout_name_raises_rather_than_holding_out_authentic_rows():
    with pytest.raises(ValueError, match="pseudo-generator"):
        _ok(heldout_generators=[""])


def test_a_repeated_heldout_family_raises():
    with pytest.raises(ValueError, match="repeats"):
        _ok(heldout_generators=["VQGAN", "VQGAN"])


def test_a_genuine_family_is_accepted():
    assert _ok(heldout_generators=["VQGAN", "BigGAN"]).heldout_generators == [
        "VQGAN", "BigGAN"]


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["name", "note"])
def test_a_blank_name_or_note_raises(field):
    with pytest.raises(ValueError):
        DatasetPreset(**{"name": "t", "note": "why", field: "   "})


def test_as_record_flattens_the_note_to_one_line():
    """YAML block scalars keep their newlines; splits.json is read by humans
    grepping it, so the note goes in as one line."""
    p = _ok(note="two\nlines   with  spacing")
    assert p.as_record()["note"] == "two lines with spacing"


def test_the_record_carries_every_knob():
    p = _ok(max_per_generator={"*": 3}, max_real_per_source={"wildfake": 4},
            min_short_side=200, heldout_generators=["VQGAN"],
            exclude_subpaths=["wildfake/real/real_ffhq"])
    rec = p.as_record()
    assert rec == {"name": "t", "note": "why",
                   "max_per_generator": {"*": 3},
                   "max_real_per_source": {"wildfake": 4},
                   "min_short_side": 200,
                   "exclude_subpaths": ["wildfake/real/real_ffhq"],
                   "heldout_generators": ["VQGAN"],
                   "heldout_groups": []}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _write(tmp_path, stem, body: dict) -> str:
    p = tmp_path / f"{stem}.yaml"
    p.write_text(yaml.safe_dump(body))
    return str(p)


def test_load_reads_a_preset(tmp_path):
    path = _write(tmp_path, "demo", {"name": "demo", "note": "n",
                                     "min_short_side": 200})
    assert load_preset(path).min_short_side == 200


def test_an_unknown_field_raises_with_the_path(tmp_path):
    """A misspelled `min_short_size` that was ignored would produce a corpus
    that disagrees with the file describing it."""
    path = _write(tmp_path, "demo", {"name": "demo", "note": "n",
                                     "min_short_size": 200})
    with pytest.raises(TypeError, match="demo.yaml"):
        load_preset(path)


def test_a_name_that_disagrees_with_the_filename_raises(tmp_path):
    path = _write(tmp_path, "demo", {"name": "other", "note": "n"})
    with pytest.raises(ValueError, match="untraceable"):
        load_preset(path)


def test_a_yaml_that_is_not_a_mapping_raises(tmp_path):
    p = tmp_path / "demo.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_preset(str(p))


def test_an_empty_yaml_raises_for_the_missing_fields(tmp_path):
    p = tmp_path / "demo.yaml"
    p.write_text("")
    with pytest.raises(TypeError, match="demo.yaml"):
        load_preset(str(p))


# --------------------------------------------------------------------------
# the presets this repo actually ships
# --------------------------------------------------------------------------

def test_the_repo_ships_presets():
    assert SHIPPED, f"no preset files under {CONFIG_DIR}"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.basename(p))
def test_every_shipped_preset_loads(path):
    """Catches a typo in a committed preset before it costs a GPU night: a bad
    source key, a family name that is a pseudo-generator, an unknown field."""
    preset = load_preset(path)
    assert preset.note.strip()


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.basename(p))
def test_every_shipped_preset_keeps_the_sub_band_floor_at_the_canon_band(path):
    """The floor exists to remove exactly what `augment.canonical` cannot
    equalise, so it is meaningful only at that band. A preset that drifts off
    it is dropping images for a reason nothing in the codebase states."""
    from aigcdet.augment.canonical import CANON_BAND_SIDE
    assert load_preset(path).min_short_side == CANON_BAND_SIDE


# --------------------------------------------------------------------------
# exclude_subpaths -- the level `restricted_buckets` cannot reach
# --------------------------------------------------------------------------

def test_a_subpath_shallower_than_a_bucket_raises():
    """A whole source is excluded by not staging it and a whole bucket by
    `SourceSpec.restricted_buckets`, which also records the licence reason.
    This field is only for what neither can express, and accepting a shallow
    path would quietly duplicate them with no audit trail."""
    for shallow in ("wildfake", "wildfake/real"):
        with pytest.raises(ValueError, match="BELOW a bucket"):
            _ok(exclude_subpaths=[shallow])


def test_a_subpath_under_an_unregistered_source_raises():
    with pytest.raises(ValueError, match="not a registered source"):
        _ok(exclude_subpaths=["wildfke/real/real_ffhq"])


def test_prefixes_are_normalised_and_terminated():
    """The trailing separator is load-bearing: a bare `startswith` on the
    un-terminated string would also match a sibling `real_ffhq_v2`."""
    p = _ok(exclude_subpaths=["wildfake/real/real_ffhq",
                              "wildfake//real/real_afhq/"])
    assert p.excluded_prefixes == ["wildfake/real/real_ffhq/",
                                   "wildfake/real/real_afhq/"]


def test_a_backslash_path_is_normalised_to_posix():
    """`build_dataset` joins `os.path.relpath(...).split(os.sep)` with "/"
    before matching, so a preset written on Windows must still match."""
    assert _ok(exclude_subpaths=["wildfake\\real\\real_ffhq"]).excluded_prefixes \
        == ["wildfake/real/real_ffhq/"]


def test_no_exclusion_means_no_prefixes():
    assert _ok().excluded_prefixes == []


# --- heldout_groups: lineages that must travel together ---------------------
#
# The frozen manifest's random draw held out SDwithAdaptor_controlnet while
# keeping its lora and lycris siblings in training, and VQGAN while keeping
# VQVAE and vqdm. An adapter changes the conditioning, not the decoder that
# leaves the trace, so those held-out scores measure a much easier question
# than their name implies. This field is how a preset says so.

def _p(**kw):
    return DatasetPreset(name="t", note="n", **kw)


def test_groups_flatten_into_heldout_generators():
    """Everything downstream reads one flat list, so the grouping must not
    require assign_splits, splits.json or the bank to learn a new shape."""
    p = _p(heldout_groups=[["VQGAN", "VQVAE", "vqdm"]])
    assert p.heldout_generators == ["VQGAN", "VQVAE", "vqdm"]


def test_groups_merge_with_an_explicit_heldout_list_without_duplicating():
    p = _p(heldout_generators=["VQGAN"], heldout_groups=[["VQGAN", "VQVAE"]])
    assert p.heldout_generators == ["VQGAN", "VQVAE"]


def test_the_record_keeps_the_grouping_the_flat_list_cannot_express():
    """splits.json must be able to say WHY those families travel together;
    a flat list of six names cannot distinguish two lineages from six
    unrelated picks."""
    groups = [["SDwithAdaptor_controlnet", "SDwithAdaptor_lora"],
              ["VQGAN", "VQVAE"]]
    rec = _p(heldout_groups=groups).as_record()
    assert rec["heldout_groups"] == groups
    assert len(rec["heldout_generators"]) == 4


def test_a_bare_string_is_refused_rather_than_iterated_as_characters():
    with pytest.raises(ValueError, match="LISTS"):
        _p(heldout_groups=["VQGAN"])


def test_a_lineage_of_one_is_refused():
    """Otherwise the field means two different things and a reader cannot
    tell a lineage from a single pick."""
    with pytest.raises(ValueError, match="fewer than two"):
        _p(heldout_groups=[["VQGAN"]])


def test_a_family_cannot_belong_to_two_lineages():
    with pytest.raises(ValueError, match="two heldout_groups"):
        _p(heldout_groups=[["VQGAN", "VQVAE"], ["VQVAE", "vqdm"]])


def test_a_pseudo_generator_is_refused_inside_a_group_too():
    """The flat field already refuses this; the group path must not be a way
    round it, or holding out a whole SOURCE becomes spellable again."""
    with pytest.raises(ValueError, match="pseudo-generator"):
        _p(heldout_groups=[["sid_set", "VQGAN"]])


def test_no_groups_leaves_the_flat_field_exactly_as_given():
    p = _p(heldout_generators=["VQGAN", "SDwithAdaptor_controlnet"])
    assert p.heldout_generators == ["VQGAN", "SDwithAdaptor_controlnet"]
    assert p.as_record()["heldout_groups"] == []
