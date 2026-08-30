"""The raw-layout registry: the contract between acquire_data.py (writes the
tree) and build_dataset.py (reads it)."""
from __future__ import annotations

import pytest

from aigcdet.data.sources import (
    PSEUDO_GENERATORS, SOURCES, SourceSpec, classify, is_excluded_from_training,
    is_heldout_eligible, is_restricted_bucket, is_safe_generator, raw_subdir,
    restriction_reason,
)


def test_coco_val2017_is_authentic_in_the_directory_the_zip_extracts_to():
    # C1: the zip extracts to coco_val2017/val2017/, and inferring
    # `label = 0 if bucket == "real" else 1` made every COCO photograph a
    # fake. The registry is what stops that inference existing at all.
    assert classify("coco_val2017", "val2017") == (0, "")


def test_coco_val2017_is_excluded_from_training_regardless_of_label():
    # Spec §4.1(2). The predicate takes only the source, so no labelling
    # mistake anywhere upstream can route a COCO image into training.
    assert is_excluded_from_training("coco_val2017")
    assert not is_excluded_from_training("wildfake")
    assert not is_excluded_from_training("sid_set")


@pytest.mark.parametrize(
    ("source", "bucket", "expected"),
    [
        ("sid_set", "real", (0, "")),
        ("sid_set", "fake", (1, "sid_set")),
        ("sid_set", "sdxl", (1, "sdxl")),      # attributed fakes keep their generator
        ("wildfake", "sdxl", (1, "sdxl")),
        ("wildfake", "midjourney", (1, "midjourney")),
        ("wildfake", "real", (0, "")),
        ("coco_val2017", "val2017", (0, "")),
    ],
)
def test_classify_maps_every_declared_bucket(source, bucket, expected):
    assert classify(source, bucket) == expected


def test_classify_rejects_an_unregistered_source():
    with pytest.raises(ValueError, match="unregistered raw source"):
        classify("some_new_dataset", "real")


def test_classify_rejects_an_undeclared_bucket():
    with pytest.raises(ValueError, match="no rule for bucket"):
        classify("coco_val2017", "fake")


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_raw_subdir_round_trips_through_classify(source):
    """Everything the writer can produce, the reader must understand."""
    spec = SOURCES[source]
    if len(spec.real_buckets) == 1:
        assert classify(source, raw_subdir(source, 0)) == (0, "")
    if spec.pseudo_bucket:
        assert classify(source, raw_subdir(source, 1)) == (1, spec.pseudo_generator)
    if spec.generator_buckets:
        assert classify(source, raw_subdir(source, 1, "sdxl")) == (1, "sdxl")


#: Generator names an untrusted dataset record could plausibly carry,
#: including ones that alias a declared bucket or climb out of the tree.
#: The empty string is deliberately absent: it means "this record names no
#: generator", which routes to the unattributed-fake bucket and is covered by
#: test_raw_subdir_round_trips_through_classify.
_CANDIDATE_GENERATORS = (
    "sdxl", "midjourney", "real", "fake", "val2017", ".", "..", "...",
    "a/b", "..\\b",
)


@pytest.mark.parametrize("source", sorted(SOURCES))
@pytest.mark.parametrize("generator", _CANDIDATE_GENERATORS)
def test_raw_subdir_and_classify_are_inverses_on_every_branch(source, generator):
    """The property that matters, over the whole input domain rather than
    one happy value: whatever `raw_subdir` accepts, `classify` must map back
    to the SAME generator and label 1 -- and whatever it cannot, it must
    refuse rather than return.

    The old test only round-tripped "sdxl", leaving the branch where a
    caller-supplied generator ALIASES a declared bucket completely
    unexercised. `raw_subdir("sid_set", 1, "real")` returned "real", which
    `classify` reads back as `(0, "")`: a fake written into the authentic
    bucket and read as authentic. That is C1's exact defect -- writer and
    reader disagreeing about a directory name -- and the generator string is
    untrusted third-party data lifted from a dataset record field.
    """
    try:
        bucket = raw_subdir(source, 1, generator)
    except ValueError:
        return  # refusing is always a valid answer
    assert classify(source, bucket) == (1, generator)


@pytest.mark.parametrize("generator", ["real", "fake"])
def test_raw_subdir_refuses_a_generator_aliasing_a_declared_bucket(generator):
    # "real" is sid_set's authentic bucket, "fake" its unattributed-fake
    # bucket; either would be read back with the wrong label or the wrong
    # generator.
    with pytest.raises(ValueError, match="not usable as a bucket directory"):
        raw_subdir("sid_set", 1, generator)


@pytest.mark.parametrize("generator", [".", "..", "...", "a/b", "../evil"])
def test_raw_subdir_refuses_a_generator_that_escapes_the_source_tree(generator):
    with pytest.raises(ValueError, match="not usable as a bucket directory"):
        raw_subdir("wildfake", 1, generator)


def test_is_safe_generator_agrees_with_what_raw_subdir_accepts():
    for generator in _CANDIDATE_GENERATORS:
        accepted = True
        try:
            raw_subdir("sid_set", 1, generator)
        except ValueError:
            accepted = False
        assert is_safe_generator("sid_set", generator) is accepted, generator


def test_both_halves_of_the_demo_benchmark_are_excluded_from_training():
    """Spec §4.1(2) forbids training on the organisers' demo benchmark, which
    is COCO val2017 AND DALL-E Advanced. Registering both is what makes
    "the demo set is excluded wholesale by the registry" a true statement --
    a claim that is carried into Plan 4's error-analysis note."""
    assert is_excluded_from_training("coco_val2017")
    assert is_excluded_from_training("dalle_advanced")
    excluded = {n for n, spec in SOURCES.items() if spec.exclude_from_training}
    assert excluded == {"coco_val2017", "dalle_advanced"}


def test_dataset_level_pseudo_generators_are_not_heldout_eligible():
    # Holding out "sid_set" removes an entire SOURCE, which measures dataset
    # shift rather than unseen-generator generalisation (spec §4.6).
    assert PSEUDO_GENERATORS == {"sid_set"}
    assert not is_heldout_eligible("sid_set")
    assert not is_heldout_eligible("")       # authentic rows carry no generator
    assert is_heldout_eligible("sdxl")


# --- Bucket-level licence restriction --------------------------------------
# The mechanism exists for a source whose bucket may not be used. As of the
# organisers' 29 Aug rules slide, which names WildFake and SID_Set as approved
# datasets, NO registered source restricts anything: the 28 Aug bar on
# WildFake's authentic bucket was a stricter reading than the rules require,
# and the frozen manifest the Kaggle banks were extracted against includes
# that bucket. A restriction reappearing here would silently make the next
# rebuild disagree with every bank on disk, so its absence is pinned.

def test_no_registered_source_restricts_any_bucket():
    for name, spec in SOURCES.items():
        assert not spec.restricted_buckets, name
        assert not restriction_reason(name), name
    assert not is_restricted_bucket("wildfake", "real")
    assert not is_restricted_bucket("sid_set", "real")
    assert not is_restricted_bucket("coco_val2017", "val2017")


def test_a_registered_restriction_is_visible_with_its_reason(monkeypatch):
    # The count alone is not an audit trail. Whoever reads docs/splits.json
    # in six months needs to know WHY a bucket's images were dropped.
    spec = SourceSpec("barred", licence="x", real_buckets=frozenset({"real"}),
                      generator_buckets=True,
                      restricted_buckets=frozenset({"real"}),
                      restriction="upstream terms are non-commercial")
    monkeypatch.setitem(SOURCES, "barred", spec)
    assert is_restricted_bucket("barred", "real")
    assert not is_restricted_bucket("barred", "ddim")
    assert "commercial" in restriction_reason("barred").lower()


def test_an_unregistered_source_has_no_restriction_rather_than_raising():
    # is_excluded_from_training is already total over unknown sources; the
    # sibling predicate must be too, or build_dataset's scan loop grows a
    # try/except around a question it asks about every row.
    assert not is_restricted_bucket("nonesuch", "real")
    assert not restriction_reason("nonesuch")


def test_restricting_an_undeclared_bucket_is_a_typo_and_raises():
    # `restricted_buckets={"reals"}` would silently restrict nothing while
    # every authentic image flowed through -- the failure mode is a corpus
    # that looks rebuilt and is not. Only DECLARED buckets may be named.
    with pytest.raises(ValueError, match="reals"):
        SourceSpec(name="x", licence="l", real_buckets=frozenset({"real"}),
                   restricted_buckets=frozenset({"reals"}), restriction="because")


def test_a_restriction_without_a_reason_raises():
    with pytest.raises(ValueError, match="reason"):
        SourceSpec(name="x", licence="l", real_buckets=frozenset({"real"}),
                   restricted_buckets=frozenset({"real"}))


def test_a_reason_without_a_restriction_raises():
    # The mirror case: a reason recorded against nothing reads as a
    # restriction that is in force when it is not.
    with pytest.raises(ValueError, match="restrict"):
        SourceSpec(name="x", licence="l", real_buckets=frozenset({"real"}),
                   restriction="because")


def test_the_two_coco_halves_are_registered_with_opposite_training_status():
    """The pair is the point, so it is asserted as a pair.

    `coco_train2017` is a training source and `coco_val2017` is not. That is a
    deliberate reversal of the rule `data/wildfake.py:_COCO_FORBIDDEN` states
    -- COCO-derived reals barred from training because train/val/test are one
    photographic distribution and the benchmark's real half is COCO val2017 --
    taken for one experiment stream, with `stratified_auc --stratify-by source`
    as the control. If a future change makes these two agree in EITHER
    direction, something has gone wrong: excluding train2017 silently empties
    that stream's real side, and including val2017 trains on the scored
    benchmark, which spec 4.1(2) forbids outright.
    """
    assert not is_excluded_from_training("coco_train2017")
    assert is_excluded_from_training("coco_val2017")
    assert SOURCES["coco_train2017"].real_buckets == frozenset({"train2017"})
    assert SOURCES["coco_val2017"].real_buckets == frozenset({"val2017"})


def test_neither_coco_source_declares_a_generator_bucket():
    """Both are wholly authentic. `generator_buckets=True` would make any
    stray directory under them read as a generator family and label its
    contents 1 -- which is C1 exactly, in the source that produced C1."""
    for name in ("coco_train2017", "coco_val2017"):
        assert not SOURCES[name].generator_buckets
        assert not SOURCES[name].pseudo_bucket
        assert not is_safe_generator(name, "sdxl")
