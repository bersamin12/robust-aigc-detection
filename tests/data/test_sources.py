"""The raw-layout registry: the contract between acquire_data.py (writes the
tree) and build_dataset.py (reads it)."""
from __future__ import annotations

import pytest

from aigcdet.data.sources import (
    PSEUDO_GENERATORS, SOURCES, classify, is_excluded_from_training,
    is_heldout_eligible, raw_subdir,
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


def test_dataset_level_pseudo_generators_are_not_heldout_eligible():
    # Holding out "sid_set" removes an entire SOURCE, which measures dataset
    # shift rather than unseen-generator generalisation (spec §4.6).
    assert PSEUDO_GENERATORS == {"sid_set"}
    assert not is_heldout_eligible("sid_set")
    assert not is_heldout_eligible("")       # authentic rows carry no generator
    assert is_heldout_eligible("sdxl")
