"""The probe cutter: what makes a 20k-row A/B trustworthy, and what makes it safe.

A probe manifest is the cheap half of a policy decision -- band-limiting
against cropping -- and it is only worth anything if two properties hold. It
must be a scale model of the corpus, or its verdict does not transfer to the
full run. And it must be impossible to mistake for the frozen manifest, or a
throwaway 20k-row bank ends up somewhere a 175k-row one was meant to be.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cpm = _load_script("cut_probe_manifest")


def _manifest(n_train=600, n_val=200):
    """A frame shaped like the real coco_crop manifest: three sources, one of
    them (wildfake) carrying many small generator families and another
    (sid_set) carrying exactly one. That asymmetry is what separates the two
    samplers, so a fixture without it would let either pass."""
    rows = []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val_internal"
        if i % 4 == 0:
            src, gen, lab = "coco_train2017", "", 0
        elif i % 4 == 1:
            src, gen, lab = "sid_set", "sid_set", 1
        elif i % 4 == 2:
            src, gen, lab = "wildfake", f"wf_gen_{i % 17}", 1
        else:
            src, gen, lab = "wildfake", "", 0
        rows.append(dict(path=f"/root/{src}/{i}.png", label=lab, generator=gen,
                         source=src, licence="x", width=512, height=512,
                         split=split, rel_path=f"{src}/{i}.png",
                         content_sha256=f"c{i:064d}"[:64],
                         pixel_sha256=f"p{i:064d}"[:64]))
    return pd.DataFrame(rows)


def test_budgets_are_honoured_per_split():
    probe, kept = cpm.cut(_manifest(), {"train": 100, "val_internal": 50},
                          "train,val_internal", seed=1)
    assert kept == {"train": 100, "val_internal": 50}
    counts = probe["split"].value_counts().to_dict()
    assert counts == {"train": 100, "val_internal": 50}


def test_the_frame_keeps_the_parents_index_labels():
    """The index label is the per-view RNG key. A `reset_index()` here would
    renumber every row from 0, so two independently-cut copies of one probe
    would agree on every rel_path and disagree on every pixel."""
    parent = _manifest()
    probe, _ = cpm.cut(parent, {"train": 100}, "train", seed=1)
    assert not probe.index.equals(pd.RangeIndex(len(probe)))
    # and each label still points at the row it came from
    for label in probe.index[:20]:
        assert probe.loc[label, "rel_path"] == parent.loc[label, "rel_path"]


def test_probe_is_reproducible_from_the_seed_and_sensitive_to_it():
    parent = _manifest()
    a, _ = cpm.cut(parent, {"train": 100}, "train", seed=7)
    b, _ = cpm.cut(parent, {"train": 100}, "train", seed=7)
    c, _ = cpm.cut(parent, {"train": 100}, "train", seed=8)
    assert list(a.index) == list(b.index)
    assert list(a.index) != list(c.index)


def test_probe_fingerprints_differently_from_its_parent():
    """The guard that keeps a throwaway out of the shipping system. `rel_path`
    in row order IS the bank's identity, so a probe bank must be refused by
    `verify_against_manifest`, `merge_banks` and fusion alike -- and that
    refusal comes from this inequality and nothing else."""
    from aigcdet.features.bank import manifest_fingerprint

    parent = _manifest()
    probe, _ = cpm.cut(parent, {"train": 100, "val_internal": 50},
                       "train,val_internal", seed=1)
    assert manifest_fingerprint(probe) != manifest_fingerprint(parent)


def test_uniform_sampler_tracks_the_corpus_composition():
    """The reason `uniform` is the default. A probe predicts the full run only
    if it is shaped like it; each source's share must survive the cut."""
    parent = _manifest(n_train=2000, n_val=0)
    probe, _ = cpm.cut(parent, {"train": 600}, "train", seed=3, sampler="uniform")
    for src, share in parent["source"].value_counts(normalize=True).items():
        got = (probe["source"] == src).mean()
        assert abs(got - share) < 0.05, (src, got, share)


def test_stratified_sampler_distorts_it_which_is_why_it_is_not_the_default():
    """Not a bug in `stratified_subsample` -- it balances across
    (generator, source) so rare families survive an EVAL cap, which is what it
    is for. On this corpus shape that pulls the generated half toward the
    source with many generator families, and this test pins that difference so
    the default cannot be flipped without someone reading why."""
    parent = _manifest(n_train=2000, n_val=0)
    uni, _ = cpm.cut(parent, {"train": 600}, "train", seed=3, sampler="uniform")
    strat, _ = cpm.cut(parent, {"train": 600}, "train", seed=3, sampler="stratified")

    fakes = lambda d: d[d["label"] == 1]["source"].value_counts(normalize=True)
    truth = fakes(parent).get("sid_set", 0.0)
    assert abs(fakes(uni).get("sid_set", 0.0) - truth) < 0.05
    assert fakes(strat).get("sid_set", 0.0) < truth - 0.05


def test_stratified_path_delegates_rather_than_reimplementing():
    """The eval bank and the training probe must select by the SAME rule when
    asked for the same one. Delegation is what guarantees it; this asserts the
    delegation is real and not a lookalike copy."""
    eb = _load_script("extract_eval_bank")
    parent = _manifest()
    mine, _ = cpm.cut(parent, {"train": 100}, "train", seed=5, sampler="stratified")
    theirs, _ = eb.subsample_manifest(parent[parent["split"] == "train"],
                                      {"train": 100}, seed=5)
    assert list(mine.index) == list(theirs.index)


def test_absent_split_raises_instead_of_writing_a_short_manifest():
    with pytest.raises(ValueError, match="does not contain"):
        cpm.cut(_manifest(), {"train": 10}, "train,benchmark", seed=1)


def test_budget_naming_a_missing_split_raises():
    with pytest.raises(ValueError, match="matches nothing|does not contain"):
        cpm.cut(_manifest(), {"train": 10, "heldout_generator": 5},
                "train", seed=1)


def test_unknown_sampler_is_refused():
    with pytest.raises(ValueError, match="sampler"):
        cpm.cut(_manifest(), {"train": 10}, "train", seed=1, sampler="clever")
