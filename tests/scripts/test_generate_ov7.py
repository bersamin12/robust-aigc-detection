"""The one invariant a second generation run can break silently.

`pool.select` promises one real produces exactly one fake for exactly one
family. It keeps that promise inside a single deal; across two it cannot,
because the family pattern is built from the suite's shares. Running the
lineage supplement over reals the first suite already used re-deals them, and
nothing downstream objects -- `run._done_ids` is per family, so the second
fake is simply generated. The corpus then holds one scene twice on the
generated side against one real.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import os

import pandas as pd
import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate_ov7")


def _rows(tmp_path, family, ids):
    p = tmp_path / f"rows_{family}.jsonl"
    p.write_text("".join(json.dumps({"image_id": i, "family": family}) + "\n"
                         for i in ids))
    return p


def _sel(pairs):
    return pd.DataFrame({"image_id": [i for i, _ in pairs],
                         "family": [f for _, f in pairs]})


def test_a_real_dealt_to_a_second_family_is_caught(tmp_path):
    _rows(tmp_path, "sdxl_t2i", ["aaa", "bbb"])
    clashes = gen.used_elsewhere(tmp_path, _sel([("aaa", "sana1600m_t2i")]))
    assert clashes == [("aaa", "sdxl_t2i", "sana1600m_t2i")]


def test_reselecting_the_same_real_for_the_same_family_is_a_resume(tmp_path):
    """Raising --total re-selects every real already done; that is the resume
    path the whole run depends on and it must not look like a clash."""
    _rows(tmp_path, "sdxl_t2i", ["aaa", "bbb"])
    assert gen.used_elsewhere(tmp_path, _sel([("aaa", "sdxl_t2i"),
                                              ("bbb", "sdxl_t2i")])) == []


def test_a_disjoint_shard_is_clean(tmp_path):
    _rows(tmp_path, "sdxl_t2i", ["aaa"])
    assert gen.used_elsewhere(tmp_path, _sel([("zzz", "kandinsky22_t2i")])) == []


def test_an_empty_rows_dir_is_clean(tmp_path):
    assert gen.used_elsewhere(tmp_path, _sel([("aaa", "sdxl_t2i")])) == []


def test_a_truncated_final_line_does_not_hide_the_rest(tmp_path):
    """A run killed mid-write leaves half a line. `_done_ids` tolerates that,
    so this must too -- and must still report the ids it did read."""
    p = _rows(tmp_path, "sdxl_t2i", ["aaa", "bbb"])
    p.write_text(p.read_text() + '{"image_id": "ccc", "fam')
    clashes = gen.used_elsewhere(tmp_path, _sel([("bbb", "sana1600m_t2i")]))
    assert clashes == [("bbb", "sdxl_t2i", "sana1600m_t2i")]


def test_every_clashing_real_is_reported_not_just_the_first(tmp_path):
    """The error message truncates the list at five, but the COUNT it quotes
    has to be the real one or the operator under-reacts."""
    _rows(tmp_path, "sdxl_t2i", [f"id{n}" for n in range(20)])
    clashes = gen.used_elsewhere(
        tmp_path, _sel([(f"id{n}", "sana1600m_t2i") for n in range(20)]))
    assert len(clashes) == 20


def test_families_subset_must_renormalise_its_shares():
    """`--families` subsets the suite BEFORE resolve_suite, and a strict subset
    sums to less than 1, which validate_suite refuses outright. Without the
    renormalisation the flag is unusable: it raises before generating anything.

    `--run-families` is the other flag and is unaffected -- it filters the
    selection AFTER select(), so its counts come from the full suite. That is
    the flag every production run uses, which is why this never surfaced.
    """
    import dataclasses

    import pytest

    from aigcdet.generate import registry

    # The base suite, so this does not depend on which supplement suites the
    # checkout happens to carry.
    full = dict(registry.SUITE)
    corpus = registry.corpus_of("ov7")
    # Drop the smallest family: any strict subset shows the bug, and the
    # smallest one leaves every lineage still represented, so validate_suite
    # fails on the shares rather than on held-out sanity.
    drop = min(full, key=lambda k: full[k].share)
    sub = {k: v for k, v in full.items() if k != drop}
    assert 0 < sum(f.share for f in sub.values()) < 1.0

    with pytest.raises(ValueError, match="shares sum to"):
        registry.resolve_suite(1000, sub, corpus=corpus)

    tot = sum(f.share for f in sub.values())
    ren = {k: dataclasses.replace(f, share=f.share / tot) for k, f in sub.items()}
    counts = registry.resolve_suite(1000, ren, corpus=corpus)
    assert sum(counts.values()) == 1000
    assert set(counts) == set(sub)


def test_run_families_does_not_touch_the_suite():
    """The production flag must keep full-suite counts, or two boxes running
    different family subsets of the same shard would deal different reals and
    the disjointness the shard blocks guarantee would be gone."""
    src = (Path(__file__).resolve().parents[2]
           / "scripts" / "generate_ov7.py").read_text()
    body = src.split("if args.run_families:", 1)[1].split("caps = caption_pool", 1)[0]
    assert "suite = " not in body, (
        "--run-families must filter the selection, never the suite")
    assert "sel = sel.loc[" in body
