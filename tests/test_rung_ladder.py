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
    # The two autoencoders. a4 and a4vq are SIBLINGS off a3, not a ladder:
    # each asks whether ONE autoencoder's round-trip helps. a4both is one flag
    # off a4 and must be read against a4, never against a3.
    ("a3", "a4vq", "use_recon_vq"),
    ("a4", "a4both", "use_recon_vq"),
    # The frequency branch is a third independent block off the same base, not
    # a step past the autoencoders.
    ("a3", "aF", "use_freq"),
    # A6 is the one pair where the flag changes NOTHING about training: it is
    # applied at inference. The ladder still owns it, because "differs from its
    # base by exactly one thing" is the property being protected and an A6 that
    # also moved `use_recon` would confound TTA with the recon branch just as
    # surely as a training flag would.
    ("a3", "a6", "tta"),
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


def test_the_two_autoencoders_are_asked_as_siblings_off_the_same_base():
    """a4 and a4vq must differ ONLY in which autoencoder they use.

    They are the same experiment run against two decoders -- a continuous KL
    VAE and a vector-quantised one -- so anything else differing would make
    "the VQ autoencoder is better here" mean "and also something else changed".
    """
    a4, a4vq = _load("a4"), _load("a4vq")
    assert a4["use_recon"] is True and a4["use_recon_vq"] is False
    assert a4vq["use_recon"] is False and a4vq["use_recon_vq"] is True
    assert _differing_keys(a4, a4vq) == {"use_recon", "use_recon_vq"}


def test_a4both_turns_on_both_blocks():
    both = _load("a4both")
    assert both["use_recon"] is True and both["use_recon_vq"] is True


def test_a6_is_inference_only_and_therefore_resume_compatible():
    """A6's flag must not be able to orphan a checkpoint.

    `tta` sits in the training dataclass so the ladder can read it out of the
    config file like every other rung's flag, and that placement has a cost:
    `config_differences` compares a stored checkpoint's config against the
    requested one field by field, so a new field is normally enough to make
    every head on disk unresumable. That is the right behaviour for a field
    that changed the weights and the wrong one for a field that cannot have.

    Checked here rather than left to the comment on the field, because the two
    halves live in different files and nothing else fails if they drift.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_ablation import RESUME_IGNORED_KEYS

    from aigcdet.train.train_head import RungConfig

    assert "tta" in {f for f in RungConfig.__dataclass_fields__}
    assert "tta" in RESUME_IGNORED_KEYS, (
        "`tta` is applied at inference and train_rung never reads it, so a "
        "checkpoint trained before the field existed is the same model as one "
        "trained after. Leaving it out of RESUME_IGNORED_KEYS invalidates "
        "every head on disk over a flag that provably did not touch them.")


def test_training_never_reads_the_tta_flag():
    """The claim `a6`'s weights equal `a3`'s rests on this, so state it.

    A source-level check, deliberately: the alternative is training two heads
    in a unit test to compare them, which is slow enough that it would be
    skipped and therefore would not hold the line. `run_ablation` still prints
    the measured max|w_a6 - w_a3| on the real run -- this is the cheap guard
    that fails in CI the moment somebody wires `cfg.tta` into the trainer.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src" / "aigcdet" / "train" / "train_head.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in src.splitlines()
                      if not line.lstrip().startswith("#"))
    assert "cfg.tta" not in body and ".tta" not in body.split("tta: bool")[-1], (
        "train_head.py now reads the `tta` flag. A6 is defined as its base "
        "rung's head scored differently; if training consumes the flag, A6 is "
        "a different model and the one-flag reading of its score is wrong.")
