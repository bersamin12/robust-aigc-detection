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
