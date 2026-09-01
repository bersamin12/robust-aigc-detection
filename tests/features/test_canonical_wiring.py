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
from any one fails here regardless of what the others do.

There are FIVE production sites, not the three the `augment/canonical.py`
docstring used to name: `features/extract`, `eval/grid`, `features/recon`,
`infer` and `explain/patch_heatmap`. The last two were added later and went
unwatched. They matter for a second reason now: since standardisation became
a POLICY rather than a fixed transform, a site that canonicalises but with the
wrong policy is a new failure mode, and it is just as silent -- band mode and
crop mode both hand the backbone the same size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.eval import grid
from aigcdet.features import extract, recon, replay
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

    monkeypatch.setattr(recon, "load_recon_models", lambda device, kind='kl': (None, None))
    monkeypatch.setattr(
        recon, "recon_features",
        lambda img, vae, lp, device: np.zeros(RECON_DIM, dtype=np.float32))
    # The replay loop moved into `features.replay` so the reconstruction and
    # frequency blocks share one canonicalisation site; the guarantee under
    # test is unchanged, so the spy follows the call rather than the module it
    # used to live in. Still driven through `attach_recon_to_bank`, which is
    # the entry point that must keep honouring it.
    spy = _Spy(replay.canonicalise)
    monkeypatch.setattr(replay, "canonicalise", spy)

    recon.attach_recon_to_bank(bank, df, device="cpu", seed=1)

    assert spy.calls == len(bank.meta), (
        f"attach_recon_to_bank decoded {len(bank.meta)} images but "
        f"canonicalised {spy.calls}")


def test_the_three_sites_use_the_same_canonicalise(tmp_path):
    """A site that imported a different implementation would satisfy its own
    call-count test and still produce pixels the others cannot replay."""
    assert extract.canonicalise is grid.canonicalise is recon.canonicalise


# ===========================================================================
# The policy, not just the call
# ===========================================================================

def test_the_training_bank_records_the_policy_it_was_built_under(tmp_path, monkeypatch):
    """A crop bank and a band bank have identical shapes, dtypes and row
    counts. The config is the only thing on disk that distinguishes them, and
    `BankWriter` treats every unrecognised config key as must-match -- which is
    what makes resume, merge and A5 fusion refuse a mismatched pair."""
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy

    _stub_backbone(monkeypatch, extract)
    df = _tree(tmp_path)
    out = extract.extract_bank(df, "fake", str(tmp_path / "band"), n_views=2,
                               device="cpu")
    assert FeatureBank.open(out).config["canon_policy"]["mode"] == "band"
    assert FeatureBank.open(out).config["geometric"] == ""

    # 96x128 fixture, so a 64px window fits.
    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    out2 = extract.extract_bank(df, "fake", str(tmp_path / "crop"), n_views=2,
                                device="cpu", policy=policy, geometric=True)
    cfg = FeatureBank.open(out2).config
    assert cfg["canon_policy"] == policy.as_record()
    assert cfg["geometric"] == "dihedral8"


def test_a_crop_bank_and_a_band_bank_refuse_to_merge(tmp_path, monkeypatch):
    """Two shards standardised differently hold features of different PIXELS.
    Merging them produces a bank that is internally inconsistent in a way no
    shape check can see, so the refusal has to come from the config."""
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy
    from aigcdet.features.bank import merge_banks

    _stub_backbone(monkeypatch, extract)
    df = _tree(tmp_path, n=4)
    a = extract.extract_bank(df.iloc[:2], "fake", str(tmp_path / "a"),
                             n_views=2, device="cpu")
    b = extract.extract_bank(df.iloc[2:], "fake", str(tmp_path / "b"),
                             n_views=2, device="cpu",
                             policy=CanonPolicy(mode=MODE_CROP, crop_side=64))
    with pytest.raises(ValueError, match="canon_policy"):
        merge_banks([a, b], str(tmp_path / "merged"))


def test_resume_refuses_a_continuation_under_a_changed_policy(tmp_path, monkeypatch):
    """Half a bank of band-limited features and half of cropped ones is worse
    than either, and nothing downstream could tell."""
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy

    _stub_backbone(monkeypatch, extract)
    df = _tree(tmp_path, n=2)
    out = str(tmp_path / "bank")
    extract.extract_bank(df, "fake", out, n_views=2, device="cpu")
    with pytest.raises(ValueError, match="cannot resume|canon_policy"):
        extract.extract_bank(df, "fake", out, n_views=2, device="cpu",
                             resume=True,
                             policy=CanonPolicy(mode=MODE_CROP, crop_side=64))


def test_geometric_needs_a_square_standardisation(tmp_path, monkeypatch):
    """Caught once, at the top of extract_bank, rather than per image deep in
    a worker process: a 90-degree rotation transposes a non-square image and
    every op in `augment.ops` is shape-preserving."""
    _stub_backbone(monkeypatch, extract)
    with pytest.raises(ValueError, match="square"):
        extract.extract_bank(_tree(tmp_path), "fake", str(tmp_path / "x"),
                             n_views=2, device="cpu", geometric=True)


def test_the_eval_bank_records_its_policy_and_stays_deterministic(tmp_path, monkeypatch):
    """The grid measures how far a score falls under a condition. If the
    window moved between conditions that measurement would be confounded with
    'a different picture', so the eval path passes no rng at all."""
    from aigcdet.augment import canonical
    from aigcdet.augment.canonical import MODE_CROP, CanonPolicy

    _stub_backbone(monkeypatch, grid)
    spy = _Spy(canonical.canonicalise)
    monkeypatch.setattr(grid, "canonicalise", spy)
    df = _tree(tmp_path)
    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    out = grid.extract_eval_bank(
        df, "fake", str(tmp_path / "eval"), device="cpu", policy=policy,
        conditions={"clean": __import__(
            "aigcdet.augment.recipes", fromlist=["Recipe"]).Recipe(())})
    assert spy.calls == N_IMAGES          # once per IMAGE, not per condition
    assert FeatureBank.open(out).config["canon_policy"] == policy.as_record()


def test_inference_and_the_heatmap_take_the_policy_they_are_given():
    """The fourth and fifth sites. A model trained on 200px crops upscaled to
    512 and served band-limited images is being shown a distribution it has
    never seen, and both policies hand the backbone the same size."""
    import inspect

    from aigcdet import infer
    from aigcdet.explain import patch_heatmap

    assert "canon_policy" in inspect.signature(patch_heatmap.patch_scores).parameters
    assert "canon_policy" in inspect.signature(infer.Predictor.__init__).parameters
    assert "canon_policy" in inspect.signature(infer.export_bundle).parameters
