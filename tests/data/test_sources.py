"""The raw-layout registry: the contract between acquire_data.py (writes the
tree) and build_dataset.py (reads it)."""
from __future__ import annotations

import pytest

from aigcdet.data.sources import (
    PSEUDO_GENERATORS, SOURCES, classify, is_excluded_from_training,
    is_heldout_eligible, is_safe_generator, raw_subdir,
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
