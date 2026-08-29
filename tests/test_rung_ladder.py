"""The ablation ladder's one-variable contract, read from the YAMLs themselves.

Every rung is `train_rung` with different flags, and a comparison is only a
comparison if the two rungs differ in the thing under test and nothing else.
That is a property of the config FILES, so it is pinned here rather than left
to whoever edits one at 2am: a rung that quietly picks up a second change makes
its pair measure two things and report one.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

RUNGS = pathlib.Path(__file__).resolve().parents[1] / "configs" / "rungs"

#: (base, rung, the one flag the rung turns on). A3 -> A4 and A4 -> A7 read
#: the reconstruction system; A3 -> A7-norecon reads FiLM against the shipping
#: system, so FiLM's verdict does not depend on A4 surviving its kill criterion.
PAIRS = [
    ("a0", "a1", "use_augmented"),
    ("a1", "a2", "use_degradation"),
    ("a2", "a3", "use_consistency"),
    ("a3", "a4", "use_recon"),
    ("a4", "a7", "use_film"),
    ("a3", "a7_norecon", "use_film"),
]


def _load(name: str) -> dict:
    return yaml.safe_load((RUNGS / f"{name}.yaml").read_text(encoding="utf-8"))


def _differing_keys(a: dict, b: dict) -> set[str]:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"name"}


@pytest.mark.parametrize("base, rung, flag", PAIRS, ids=[f"{b}->{r}" for b, r, _ in PAIRS])
def test_each_rung_differs_from_its_base_by_exactly_one_flag(base, rung, flag):
    b, r = _load(base), _load(rung)
    assert _differing_keys(b, r) == {flag}, (
        f"{rung} vs {base} differ in {sorted(_differing_keys(b, r))}; a pair "
        f"that differs in more than {flag!r} measures two things and reports one")
    assert b[flag] is False and r[flag] is True


def test_film_is_asked_both_with_and_without_recon():
    """A7 inherits A4's kill criterion; A7-norecon must not. The two FiLM rungs
    are the same experiment on two bases, so they differ ONLY in use_recon."""
    a7, a7n = _load("a7"), _load("a7_norecon")
    assert a7["use_recon"] is True and a7n["use_recon"] is False
    assert a7["use_film"] is True and a7n["use_film"] is True
    assert _differing_keys(a7, a7n) == {"use_recon"}


def test_every_config_names_itself_after_its_file():
    """`run_ablation` keys checkpoints, scores and selection.json on the YAML's
    `name`, so a copied file that kept the old name would silently overwrite
    another rung's checkpoint."""
    for path in sorted(RUNGS.glob("*.yaml")):
        assert _load(path.stem)["name"] == path.stem, path
