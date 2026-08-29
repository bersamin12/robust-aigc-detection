"""The single inference path.

`Predictor` exists so `scripts/predict.py` and the dashboard cannot drift
apart, which makes its own contract the thing worth pinning: what reaches the
backbone (a CANONICALISED image, like every other decode site), what comes out
(a CALIBRATED probability, not a raw sigmoid), and what happens to a directory
containing a file that is not an image.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from aigcdet.augment.canonical import canonicalise
from aigcdet.infer import RESULT_KEYS_MINIMAL

_DIM = 8


class _ConstantConditional:
    """A calibrator with ConditionalTemperature's two-argument `transform`.

    Module level, not a closure, because `export_bundle` pickles it.
    """

    def __init__(self, value: float = 0.123):
        self.value = value
        self.seen_cond = None

    def transform(self, logits, cond):
        self.seen_cond = np.asarray(cond)
        return np.full(np.shape(logits), self.value, dtype=np.float64)


class _ConstantGlobal:
    """A calibrator with GlobalTemperature's one-argument `transform`."""

    def __init__(self, value: float = 0.321):
        self.value = value

    def transform(self, logits):
        return np.full(np.shape(logits), self.value, dtype=np.float64)


class _NotACalibrator:
    """No `transform` at all."""


def _fake_bundle(tmp_path, calibrator=None, eqi=None, use_recon=False):
    """A bundle with a tiny detector and no real backbone weights."""
    import joblib
    import torch

    from aigcdet.models.heads import Detector

    d = Detector(dim_feat=_DIM, use_recon=use_recon)
    out = tmp_path / "bundle"
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": d.state_dict(),
                "config": {"use_recon": use_recon, "use_film": False},
                "dim_feat": _DIM, "backbone": "fake"}, out / "checkpoint.pt")
    joblib.dump(_ConstantConditional() if calibrator is None else calibrator,
                out / "calibrator.joblib")
    joblib.dump(eqi, out / "eqi.joblib")
    with open(out / "policy.json", "w") as f:
        json.dump({"flag_threshold": 0.8, "clear_threshold": 0.2,
                   "eqi_threshold": 0.3}, f)
    with open(out / "config.json", "w") as f:
        json.dump({"backbone": "fake", "use_recon": use_recon,
                   "dim_feat": _DIM}, f)
    return str(out)


def _patch_backbone(monkeypatch, fn=None):
    """Replace the backbone with a stub, so no weights are downloaded."""
    from aigcdet import infer
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, _DIM, 1, 0)
    monkeypatch.setattr(infer, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        infer, "embed",
        fn or (lambda m, s, imgs, device="cpu", batch_size=16:
               np.zeros((len(imgs), s.dim), np.float32)))
    return infer


def _img(seed=0, shape=(120, 200, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_minimal_keys_are_exactly_the_two_required():
    """The brief asks for `image_path` and `pred`. An extra key is how a
    submission fails on a technicality."""
    assert RESULT_KEYS_MINIMAL == ("image_path", "pred")


def test_predictor_loads_and_scores(tmp_path, monkeypatch):
    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")

    out = p.predict_array(_img())

    assert 0.0 <= out["pred"] <= 1.0
    assert out["decision"] in ("clear", "review", "flag")
    assert len(out["severity"]) == 6
    assert len(out["presence"]) == 6
    assert len(out["proxies"]) == 3


def test_predict_array_canonicalises_before_it_embeds(tmp_path, monkeypatch):
    """Every other decode site canonicalises resolution first, because
    resolution leaks the label (docs/resolution_shortcut.md). If inference is
    the one site that does not, the deliverable scores images at a scale the
    head never saw -- and it does so silently, with a confident number.
    """
    seen = {}

    def spy(model, spec, imgs, device="cpu", batch_size=16):
        seen["shape"] = imgs[0].shape
        return np.zeros((len(imgs), spec.dim), np.float32)

    infer = _patch_backbone(monkeypatch, fn=spy)
    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    raw = _img(shape=(120, 200, 3))

    p.predict_array(raw)

    assert seen["shape"] == canonicalise(raw).shape
    assert seen["shape"] != raw.shape        # or the assertion above is vacuous


def test_proxies_describe_the_pixels_the_head_actually_saw(tmp_path, monkeypatch):
    """The proxies are the degradation evidence the calibrator conditions on,
    so they have to describe the CANONICALISED image -- the one the backbone
    embedded -- not the original file. Two of the three (sharpness, noise
    floor) move with resolution, which is the very thing canonicalisation
    neutralises, so measuring them on the original feeds the calibrator a
    description of an image the head never scored."""
    from aigcdet.features.proxies import proxy_vector

    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    raw = _img(seed=3, shape=(140, 260, 3))

    got = p.predict_array(raw)["proxies"]

    np.testing.assert_allclose(got, proxy_vector(canonicalise(raw)), rtol=1e-6)
    assert not np.allclose(got, proxy_vector(raw))


def test_pred_is_the_calibrated_probability_not_a_raw_sigmoid(tmp_path, monkeypatch):
    """`pred` is what a reader interprets as "90% likely AI-generated", so it
    has to be the calibrator's output. A raw sigmoid of the head's logit is a
    different number wearing the same name."""
    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(
        _fake_bundle(tmp_path, calibrator=_ConstantConditional(0.123)),
        device="cpu")

    out = p.predict_array(_img())

    assert out["pred"] == pytest.approx(0.123)
    assert out["pred"] != pytest.approx(1.0 / (1.0 + np.exp(-out["logit"])))


def test_a_conditional_calibrator_is_given_severity_and_proxies(tmp_path, monkeypatch):
    """ConditionalTemperature's whole purpose is that the temperature varies
    with the degradation the image shows, so the condition vector has to
    actually arrive."""
    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(
        _fake_bundle(tmp_path, calibrator=_ConstantConditional()), device="cpu")

    out = p.predict_array(_img())

    # `p.calibrator`, not the instance passed in: the bundle round-trips
    # through joblib, so the object that gets called is a deserialised copy.
    seen = p.calibrator.seen_cond
    assert seen.shape == (1, 6 + 3)
    np.testing.assert_allclose(
        seen[0], np.concatenate([out["severity"], out["proxies"]]), rtol=1e-6)


def test_a_global_calibrator_is_called_with_logits_alone(tmp_path, monkeypatch):
    """GlobalTemperature.transform takes one argument. Choosing the arity by
    inspecting the signature -- rather than calling with two and catching
    TypeError -- keeps a genuine TypeError raised INSIDE transform from being
    mistaken for the one-argument form."""
    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(
        _fake_bundle(tmp_path, calibrator=_ConstantGlobal(0.321)), device="cpu")

    assert p.predict_array(_img())["pred"] == pytest.approx(0.321)


def test_an_unusable_calibrator_raises_rather_than_falling_back(tmp_path, monkeypatch):
    """Falling back to a raw sigmoid here would produce the exact bug this
    class exists to prevent, and produce it silently: a well-formed number in
    the right range that no longer means what the README says it means."""
    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(
        _fake_bundle(tmp_path, calibrator=_NotACalibrator()), device="cpu")

    with pytest.raises(TypeError, match="calibrator"):
        p.predict_array(_img())


def test_predict_paths_survives_a_corrupt_file(tmp_path, monkeypatch):
    infer = _patch_backbone(monkeypatch)
    good = tmp_path / "good.png"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")

    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    res = p.predict_paths([str(good), str(bad)])

    assert len(res) == 2
    assert res[1]["pred"] == 0.5 and res[1]["error"]
    assert res[0]["error"] is None


def test_every_result_carries_the_path_it_was_scored_from(tmp_path, monkeypatch):
    """A failure must not shift the scores of the files after it. Each result
    carries its own `image_path` rather than being re-paired positionally by
    the caller, which is what makes that structurally impossible."""
    infer = _patch_backbone(
        monkeypatch,
        fn=lambda m, s, imgs, device="cpu", batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))
    paths = []
    for i in range(4):
        q = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((64, 64, 3), i * 40, np.uint8)).save(q)
        paths.append(str(q))
    broken = tmp_path / "img_broken.png"
    broken.write_bytes(b"nope")
    paths.insert(2, str(broken))

    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    res = p.predict_paths(paths)

    assert [r["image_path"] for r in res] == paths
    assert res[2]["error"] and all(res[i]["error"] is None
                                   for i in (0, 1, 3, 4))


def test_a_batch_goes_through_the_backbone_in_one_pass(tmp_path, monkeypatch):
    """`batch_size` has to mean something. Embedding one image per forward
    pass is roughly an order of magnitude slower on a 5k-image directory, and
    a signature that accepts the argument and ignores it hides that."""
    calls = []

    def spy(model, spec, imgs, device="cpu", batch_size=16):
        calls.append(len(imgs))
        return np.zeros((len(imgs), spec.dim), np.float32)

    infer = _patch_backbone(monkeypatch, fn=spy)
    paths = []
    for i in range(7):
        q = tmp_path / f"b{i}.png"
        Image.fromarray(np.full((48, 48, 3), i * 30, np.uint8)).save(q)
        paths.append(str(q))

    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    res = p.predict_paths(paths, batch_size=4)

    assert len(res) == 7
    assert calls == [4, 3]


def test_a_failure_does_not_reorder_the_batch_around_it(tmp_path, monkeypatch):
    """The failed row is filled in separately from the scored ones, so the
    two have to be merged back into the INPUT order rather than concatenated.
    Appending scored-then-failed puts every bad file at the end of its batch,
    silently rewriting the order a caller diffing two runs relies on."""
    infer = _patch_backbone(monkeypatch)
    paths = []
    for i in range(4):
        q = tmp_path / f"c{i}.png"
        Image.fromarray(np.full((48, 48, 3), i * 30, np.uint8)).save(q)
        paths.append(str(q))
    broken = tmp_path / "c_broken.png"
    broken.write_bytes(b"nope")
    paths.insert(1, str(broken))

    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    res = p.predict_paths(paths, batch_size=3)

    assert [r["image_path"] for r in res] == paths
    assert res[1]["error"] and res[1]["pred"] == 0.5


def test_predictions_are_deterministic(tmp_path, monkeypatch):
    infer = _patch_backbone(
        monkeypatch,
        fn=lambda m, s, imgs, device="cpu", batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))
    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    img = _img(seed=1, shape=(100, 100, 3))

    assert p.predict_array(img)["pred"] == p.predict_array(img)["pred"]


def test_export_bundle_round_trips(tmp_path, monkeypatch):
    """What `export_bundle` writes is what `Predictor.load` reads. These two
    live in one module precisely so the file names cannot disagree."""
    import torch

    from aigcdet.calibrate.policy import Policy
    from aigcdet.models.heads import Detector

    src = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": Detector(dim_feat=_DIM).state_dict(),
                "config": {"use_recon": False, "use_film": False},
                "dim_feat": _DIM, "backbone": "fake"}, src)

    infer = _patch_backbone(monkeypatch)
    out = infer.export_bundle(
        str(src), _ConstantConditional(0.42), None,
        Policy(flag_threshold=0.9, clear_threshold=0.1, eqi_threshold=0.5),
        str(tmp_path / "release"), backbone_name="fake", use_recon=False,
        dim_feat=_DIM)

    p = infer.Predictor.load(out, device="cpu")
    assert p.policy.flag_threshold == 0.9
    assert p.predict_array(_img())["pred"] == pytest.approx(0.42)


def test_a_recon_bundle_without_recon_features_is_refused(tmp_path, monkeypatch):
    """A recon-trained head fed only backbone features would either crash deep
    inside the model or, worse, be handed zeros and score confidently on
    half its input."""
    infer = _patch_backbone(monkeypatch)
    bundle = _fake_bundle(tmp_path, use_recon=True)
    p = infer.Predictor.load(bundle, device="cpu")
    monkeypatch.setattr(p, "_recon_vector",
                        lambda img: (_ for _ in ()).throw(RuntimeError("no VAE")))

    with pytest.raises(RuntimeError, match="no VAE"):
        p.predict_array(_img())
