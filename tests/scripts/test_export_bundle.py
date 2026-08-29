"""`export_bundle.py` -- fitting the calibrator, EQI and policy, then freezing.

The dangerous failure here is not a crash. It is fitting on the wrong rows:
the benchmark images live in the same eval bank as the internal validation
ones, and a calibration fitted across both is invisible afterwards while
making every reliability number in the report a fiction.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch

from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.features.bank import N_FAMILIES, BankWriter
from aigcdet.models.heads import Detector

_ROOT = pathlib.Path(__file__).resolve().parents[2]
DIM = 6


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eb_script = _load_script("export_bundle")


def _eval_bank(tmp_path, name="eval_bank", n_per_block=60, backbone="fake",
               seed=1, blocks=None) -> str:
    """An eval bank over the full condition grid, in four split blocks.

    Includes `benchmark` rows on purpose: they are what the script must
    exclude, and a bank without them cannot show that it does.
    """
    out = str(tmp_path / name)
    conditions = list(EVAL_GRID)
    blocks = blocks or [("val_internal", 0), ("val_internal", 1),
                        ("heldout_generator", 1), ("benchmark", 0)]
    n = n_per_block * len(blocks)
    w = BankWriter(out, n, len(conditions), DIM, backbone, 0,
                   manifest_sha256="eb",
                   extra_config={"conditions": conditions})
    rng = np.random.default_rng(seed)
    i = 0
    for split, label in blocks:
        for _ in range(n_per_block):
            base = rng.normal(0.5 if label else -0.5, 0.6, DIM)
            feats = np.stack([base] + [base + rng.normal(0, 1.0, DIM)
                                       for _ in range(len(conditions) - 1)]
                             ).astype(np.float32)
            presence = np.zeros((len(conditions), N_FAMILIES), np.float32)
            presence[1:, 0] = 1.0
            severity = np.zeros((len(conditions), N_FAMILIES), np.float32)
            severity[1:, 0] = 0.5
            w.write_image(i, {"path": f"/e{i}.png", "label": label,
                              "generator": "g" if label else "",
                              "source": "s", "split": split},
                          feats=feats, presence=presence, severity=severity,
                          proxies=rng.normal(0, 1, (len(conditions), 3)
                                             ).astype(np.float32),
                          recipes=["[]"] * len(conditions))
            i += 1
    w.close()
    return out


def _checkpoint(tmp_path, backbone="fake", name="checkpoint.pt",
                bank_dir=None) -> str:
    """A checkpoint whose logits actually carry signal.

    An untrained Detector emits logits uncorrelated with the labels, and
    `ConditionalTemperature.fit` rightly refuses those -- the temperature runs
    away to a few hundred trying to squash noise to p=0.5. So the fixture
    trains briefly on the bank it will be calibrated against, which is also
    what the real pipeline does.
    """
    model = Detector(dim_feat=DIM)
    if bank_dir is not None:
        from aigcdet.features.bank import FeatureBank

        bank = FeatureBank.open(bank_dir)
        f = torch.from_numpy(
            np.asarray(bank.feats).reshape(-1, DIM).astype(np.float32))
        y = torch.from_numpy(
            np.repeat(np.asarray(bank.meta["label"], dtype=np.float32),
                      bank.feats.shape[1]))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(120):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(f)["logit"], y)
            loss.backward()
            opt.step()
        model.eval()
    p = tmp_path / name
    torch.save({"state_dict": model.state_dict(),
                "config": {"use_recon": False, "use_film": False},
                "dim_feat": DIM, "backbone": backbone}, p)
    return str(p)


def _run(tmp_path, **over):
    bank = over.pop("--eval-bank", None) or _eval_bank(tmp_path)
    args = {"--checkpoint": over.pop("--checkpoint", None)
                            or _checkpoint(tmp_path, bank_dir=bank),
            "--eval-bank": bank,
            "--out": str(tmp_path / "release")}
    args.update(over)
    return eb_script.main([str(x) for kv in args.items() for x in kv])


def test_it_writes_a_bundle_the_predictor_can_load(tmp_path, monkeypatch):
    from aigcdet import infer
    from aigcdet.features.backbones import BackboneSpec

    assert _run(tmp_path) == 0
    out = tmp_path / "release"
    assert {p.name for p in out.iterdir()} == {
        "checkpoint.pt", "calibrator.joblib", "eqi.joblib", "policy.json",
        "config.json"}

    monkeypatch.setattr(infer, "load_backbone",
                        lambda n, device: (None, BackboneSpec("fake", "none",
                                                              64, DIM, 1, 0)))
    monkeypatch.setattr(infer, "embed",
                        lambda m, s, imgs, device="cpu", batch_size=16:
                        np.zeros((len(imgs), s.dim), np.float32))
    p = infer.Predictor.load(str(out), device="cpu")
    img = np.random.default_rng(0).integers(0, 256, (80, 90, 3), dtype=np.uint8)
    assert 0.0 <= p.predict_array(img)["pred"] <= 1.0


def test_it_fits_on_val_internal_only(tmp_path):
    """The split labels handed to each `fit` must be the ones read off the
    bank for the rows actually kept. If the filter selected anything else,
    `check_fit_split` refuses -- which is the guarantee this asserts."""
    seen = {}
    bank_dir = _eval_bank(tmp_path)

    real_fit = eb_script.ConditionalTemperature.fit

    def spy(self, logits, y, cond, *a, split=None, **kw):
        seen["split"] = np.asarray(split)
        return real_fit(self, logits, y, cond, *a, split=split, **kw)

    eb_script.ConditionalTemperature.fit = spy
    try:
        assert _run(tmp_path, **{"--eval-bank": bank_dir}) == 0
    finally:
        eb_script.ConditionalTemperature.fit = real_fit

    assert set(np.unique(seen["split"])) == {"val_internal"}
    # 2 of the 4 blocks are val_internal, over the whole condition grid.
    assert seen["split"].size == 60 * 2 * len(EVAL_GRID)


def test_it_refuses_an_eval_bank_with_no_val_internal_rows(tmp_path):
    bank = _eval_bank(tmp_path, name="benchmark_only",
                      blocks=[("benchmark", 0), ("benchmark", 1)])
    with pytest.raises(SystemExit, match="val_internal"):
        _run(tmp_path, **{"--eval-bank": bank})


def test_it_refuses_a_bank_from_a_different_backbone(tmp_path):
    """A head reads one feature space. Scored against another backbone's bank
    it emits well-formed nonsense that nothing downstream can detect."""
    bank = _eval_bank(tmp_path, name="other", backbone="siglip2l")
    with pytest.raises(SystemExit, match="siglip2l"):
        _run(tmp_path, **{"--checkpoint": _checkpoint(tmp_path),
                          "--eval-bank": bank})  # checkpoint says "fake"


def test_condition_vectors_follow_the_condition_of_each_row(tmp_path):
    """Indexed by [image, condition], not by image alone: the clean view's
    severity on a JPEG-30 row would tell the conditional temperature the
    opposite of the truth."""
    from aigcdet.features.bank import FeatureBank

    bank = FeatureBank.open(_eval_bank(tmp_path, n_per_block=5))
    names = list(bank.config["conditions"])
    scores = {"image_idx": np.array([0, 0, 3]),
              "condition": [names[0], names[2], names[1]]}

    cv = eb_script.condition_vectors(bank, scores)

    assert cv.shape == (3, N_FAMILIES + 3)
    for row, (i, c) in enumerate(zip(scores["image_idx"], scores["condition"])):
        j = names.index(c)
        np.testing.assert_allclose(
            cv[row], np.concatenate([np.asarray(bank.severity)[i, j],
                                     np.asarray(bank.proxies)[i, j]]), rtol=1e-6)


def test_split_labels_are_read_from_the_bank_not_asserted(tmp_path):
    """`["val_internal"] * n` would satisfy `check_fit_split` and prove
    nothing. The guard is only worth having if the labels are evidence, so
    pin that they come back per row, from the bank, including the splits the
    script is about to exclude."""
    from aigcdet.features.bank import FeatureBank

    bank = FeatureBank.open(_eval_bank(tmp_path, n_per_block=5))
    # One row from each of the four blocks, out of order.
    idx = np.array([12, 0, 18, 7])
    got = eb_script.split_labels(bank, {"image_idx": idx})

    np.testing.assert_array_equal(got, np.asarray(bank.meta["split"])[idx])
    assert set(got) == {"val_internal", "heldout_generator", "benchmark"}


def test_the_policy_it_writes_is_the_policy_it_fitted(tmp_path):
    assert _run(tmp_path) == 0
    written = json.loads((tmp_path / "release" / "policy.json").read_text())
    assert set(written) == {"flag_threshold", "clear_threshold", "eqi_threshold"}
    assert all(isinstance(v, float) for v in written.values())
