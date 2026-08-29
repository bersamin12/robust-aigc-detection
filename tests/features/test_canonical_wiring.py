"""Every decode site canonicalises, or none of them do.

`docs/resolution_shortcut.md` records why canonicalisation exists. This file
exists for a narrower reason: the property "all three sites agree" is NOT
covered by the tests that look like they cover it.

`tests/features/test_recon.py` asserts that `attach_recon_to_bank` replays
`extract_bank`'s pixels bit-exactly, but it does so against ground truth it
recomputes by hand -- and that recomputation canonicalises too. So if
`extract.py` alone stopped canonicalising, recon and the hand-rolled
expectation would still agree with each other and the replay test would stay
green while the cached features diverged from the replayed ones. The failure
mode canonicalisation was wired to prevent is invisible to the test that
appears to guard it.

These tests watch the call itself, at each site independently, so removing it
from any one of the three fails here regardless of what the others do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.eval import grid
from aigcdet.features import extract, recon
from aigcdet.features.backbones import BackboneSpec
from aigcdet.features.bank import RECON_DIM, FeatureBank

N_IMAGES = 3


def _tree(tmp_path, n=N_IMAGES):
    """Images deliberately NOT at the canonical size, and not square.

    A fixture already at the nominal side cannot detect a skipped resize, and
    a square one cannot detect an aspect-ratio bug -- both are failure modes
    this project has actually shipped in fixtures before.
    """
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    return pd.DataFrame({
        "path": paths, "label": [i % 2 for i in range(n)],
        "generator": [""] * n, "source": ["test"] * n,
        "split": ["train"] * n, "width": [128] * n, "height": [96] * n,
    })


class _Spy:
    """Counts calls and forwards to the real implementation."""

    def __init__(self, real):
        self.real, self.calls = real, 0

    def __call__(self, img, **kw):
        self.calls += 1
        return self.real(img, **kw)


def _stub_backbone(monkeypatch, module):
    spec = BackboneSpec("fake", "fake/fake", 64, 8, 0, 1)
    monkeypatch.setattr(module, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        module, "embed",
        lambda model, spec, imgs, device="cpu", batch_size=16:
            np.zeros((len(imgs), spec.dim), dtype=np.float32))
    return spec


def test_extract_bank_canonicalises_every_image_it_decodes(tmp_path, monkeypatch):
    df = _tree(tmp_path)
    _stub_backbone(monkeypatch, extract)
    spy = _Spy(extract.canonicalise)
    monkeypatch.setattr(extract, "canonicalise", spy)

    extract.extract_bank(df, "fake", str(tmp_path / "bank"), seed=1, device="cpu")

    assert spy.calls == len(df), (
        f"extract_bank decoded {len(df)} images but canonicalised {spy.calls}")


def test_extract_eval_bank_canonicalises_every_image_it_decodes(tmp_path, monkeypatch):
    df = _tree(tmp_path)
    _stub_backbone(monkeypatch, grid)
    spy = _Spy(grid.canonicalise)
    monkeypatch.setattr(grid, "canonicalise", spy)

    grid.extract_eval_bank(df, "fake", str(tmp_path / "eval"), device="cpu")

    assert spy.calls == len(df), (
        f"extract_eval_bank decoded {len(df)} images but canonicalised {spy.calls}")


def test_attach_recon_to_bank_canonicalises_every_image_it_decodes(tmp_path, monkeypatch):
    df = _tree(tmp_path)
    _stub_backbone(monkeypatch, extract)
    out = str(tmp_path / "bank")
    extract.extract_bank(df, "fake", out, seed=1, device="cpu")
    bank = FeatureBank.open(out)

    monkeypatch.setattr(recon, "load_recon_models", lambda device: (None, None))
    monkeypatch.setattr(
        recon, "recon_features",
        lambda img, vae, lp, device: np.zeros(RECON_DIM, dtype=np.float32))
    spy = _Spy(recon.canonicalise)
    monkeypatch.setattr(recon, "canonicalise", spy)

    recon.attach_recon_to_bank(bank, df, device="cpu", seed=1)

    assert spy.calls == len(bank.meta), (
        f"attach_recon_to_bank decoded {len(bank.meta)} images but "
        f"canonicalised {spy.calls}")


def test_the_three_sites_use_the_same_canonicalise(tmp_path):
    """A site that imported a different implementation would satisfy its own
    call-count test and still produce pixels the others cannot replay."""
    assert extract.canonicalise is grid.canonicalise is recon.canonicalise
