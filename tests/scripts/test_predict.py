"""The submission's inference entry point (brief section 5.5.2).

    "A script that takes an image directory as input and outputs a confidence
     score for each image, indicating the likelihood that it is AIGC-generated.
     The output should be a JSON file containing image_path and pred for each
     image."

Everything else in this repository consumes a frozen manifest and a cached
feature bank. This is the one path a judge actually runs, on a directory of
loose files with no manifest, no labels and no bank -- so the properties that
the rest of the pipeline gets from the manifest have to be established here
instead.

The scoring itself is `aigcdet.infer.Predictor`, tested in `tests/test_infer.py`.
What is tested here is what the SCRIPT owns: which files it finds, in what
order, what lands in the JSON, and what it does when a file will not decode.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from aigcdet.calibrate.policy import Policy
from aigcdet.calibrate.temperature import GlobalTemperature
from aigcdet.models.heads import Detector

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
SCRIPT = str(REPO / "scripts" / "predict.py")
DIM = 8


def _images(d, n=5, size=(40, 57)):
    """Deliberately non-square and not at the canonical size, so a missing
    canonicalise() changes the pixels rather than happening to be a no-op."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = d / f"img{i}.png"
        Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def _calibrator():
    """A real, fitted GlobalTemperature.

    A stub class defined in this file would need to be importable by name in
    the subprocess test's fresh interpreter, since the bundle is unpickled
    there. A shipped class avoids that entirely.
    """
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    logits = rng.normal(0, 1, n) + np.where(y == 1, 1.2, -1.2)
    return GlobalTemperature().fit(logits, y,
                                   split=np.array(["val_internal"] * n))


def _bundle(tmp_path, name="bundle", use_recon=False, use_film=False,
            backbone="fake"):
    """A release bundle in the shape `export_bundle` writes."""
    from aigcdet.infer import export_bundle

    torch.manual_seed(0)
    model = Detector(dim_feat=DIM, use_recon=use_recon, use_film=use_film)
    ck = tmp_path / f"{name}_checkpoint.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {"use_recon": use_recon, "use_film": use_film,
                           "name": "a3", "seed": 1},
                "dim_feat": DIM, "backbone": backbone}, ck)
    return export_bundle(
        str(ck), _calibrator(), None,
        Policy(flag_threshold=0.8, clear_threshold=0.2, eqi_threshold=0.3),
        str(tmp_path / name), backbone_name=backbone, use_recon=use_recon,
        dim_feat=DIM)


@pytest.fixture()
def pred(monkeypatch):
    """The script, with a stand-in backbone: no weights, no GPU.

    The stubs go on `aigcdet.infer`, not on the script: the script holds no
    backbone of its own any more, which is the point of routing it through
    `Predictor`.
    """
    import importlib.util

    from aigcdet import infer
    from aigcdet.features.backbones import BackboneSpec

    spec = importlib.util.spec_from_file_location("predict_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bspec = BackboneSpec("fake", "none", 64, DIM, 1, 0)
    monkeypatch.setattr(infer, "load_backbone", lambda n, device: (None, bspec))
    monkeypatch.setattr(
        infer, "embed",
        lambda m, s, imgs, device="cpu", batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))
    mod.infer = infer
    return mod


def _argv(images, bundle, out, *extra):
    return ["--images", str(images), "--bundle", str(bundle), "--out", str(out),
            "--device", "cpu", *extra]


# --- the deliverable's literal contract -------------------------------------

def test_writes_image_path_and_pred_for_every_image(tmp_path, pred):
    paths = _images(tmp_path / "in")
    out = tmp_path / "preds.json"
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path), out))

    rows = json.loads(out.read_text())
    assert isinstance(rows, list) and len(rows) == len(paths)
    assert all(set(r) == {"image_path", "pred"} for r in rows)
    assert {r["image_path"] for r in rows} == {str(p) for p in paths}


def test_pred_is_a_probability(tmp_path, pred):
    _images(tmp_path / "in")
    out = tmp_path / "preds.json"
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path), out))
    for r in json.loads(out.read_text()):
        assert isinstance(r["pred"], float)
        assert 0.0 <= r["pred"] <= 1.0


def test_pred_is_the_calibrated_probability(tmp_path, pred):
    """Not `sigmoid(logit)`. Both are floats in [0, 1] that move with the
    image, so nothing about the output file distinguishes them -- which is
    exactly why it is asserted rather than eyeballed."""
    _images(tmp_path / "in", n=3)
    out = tmp_path / "preds.json"
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path), out, "--full"))

    rows = json.loads(out.read_text())
    assert rows, "nothing scored"
    for r in rows:
        raw = 1.0 / (1.0 + np.exp(-r["logit"]))
        assert r["pred"] != pytest.approx(raw, abs=1e-9)


def test_the_submitted_file_carries_no_extra_keys(tmp_path, pred):
    """`--full` is opt-in. The default output is the two keys the brief names
    and nothing else, because an extra key is how a submission fails on a
    technicality."""
    _images(tmp_path / "in", n=2)
    plain, full = tmp_path / "a.json", tmp_path / "b.json"
    bundle = _bundle(tmp_path)
    pred.main(_argv(tmp_path / "in", bundle, plain))
    pred.main(_argv(tmp_path / "in", bundle, full, "--full"))

    assert all(set(r) == {"image_path", "pred"}
               for r in json.loads(plain.read_text()))
    assert {"logit", "eqi", "decision", "severity"} <= set(
        json.loads(full.read_text())[0])


def test_finds_images_in_nested_directories_and_ignores_non_images(tmp_path, pred):
    _images(tmp_path / "in", n=2)
    _images(tmp_path / "in" / "deeper", n=3)
    (tmp_path / "in" / "notes.txt").write_text("not an image")
    (tmp_path / "in" / "README.md").write_text("nor this")
    out = tmp_path / "preds.json"
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path), out))
    rows = json.loads(out.read_text())
    assert len(rows) == 5
    assert not any(r["image_path"].endswith((".txt", ".md")) for r in rows)


def test_output_order_is_stable_across_runs(tmp_path, pred):
    """A judge may diff two runs. os.walk order is filesystem-dependent, so
    the rows are sorted rather than left to whatever readdir returned."""
    _images(tmp_path / "in", n=6)
    bundle = _bundle(tmp_path)
    got = []
    for i in range(2):
        out = tmp_path / f"p{i}.json"
        pred.main(_argv(tmp_path / "in", bundle, out))
        got.append([r["image_path"] for r in json.loads(out.read_text())])
    assert got[0] == got[1] == sorted(got[0])


# --- the properties that fail silently --------------------------------------

def test_canonicalises_every_image_before_embedding(tmp_path, pred, monkeypatch):
    """The fourth decode site. If inference skips canonicalisation it scores
    images on different pixels than the head was trained on -- no error, just
    wrong numbers. Counted per image, not merely 'called at least once'.

    Also pins that the bundle's OWN standardisation policy is what reaches it,
    and deterministically. A head trained on 200px crops upscaled to 512 and
    then served band-limited images is being shown a distribution it has never
    seen, and nothing about the shapes says so -- both policies hand the
    backbone the same size."""
    seen = []
    real = pred.infer.canonicalise

    def spy(a, **kw):
        seen.append((a.shape, kw.get("policy"), kw.get("rng")))
        return real(a, **kw)

    monkeypatch.setattr(pred.infer, "canonicalise", spy)
    _images(tmp_path / "in", n=4)
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path), tmp_path / "p.json"))
    assert len(seen) == 4
    # Every call carries a policy, and none carries an rng: serving must
    # return the same score for the same file twice.
    assert all(p is not None for _, p, _ in seen)
    assert all(r is None for _, _, r in seen)


def test_inference_uses_the_same_canonicalise_as_the_other_decode_sites(pred):
    from aigcdet.augment import canonical
    from aigcdet.features import extract

    assert pred.infer.canonicalise is canonical.canonicalise \
        is extract.canonicalise


def test_scores_the_clean_view_only(tmp_path, pred, monkeypatch):
    """Inference must embed the image as given, not a sampled augmented view.
    The bank's view 0 is the clean view; this is that view and nothing else,
    so the pixels handed to the backbone are the canonicalised original."""
    from aigcdet.augment.canonical import canonicalise

    handed = []
    monkeypatch.setattr(pred.infer, "embed",
                        lambda m, s, imgs, device="cpu", batch_size=16:
                            (handed.extend(imgs),
                             np.zeros((len(imgs), s.dim), np.float32))[1])
    paths = _images(tmp_path / "in", n=3)
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path), tmp_path / "p.json"))

    assert len(handed) == 3
    for p, got in zip(sorted(paths), handed):
        with Image.open(p) as im:
            want = canonicalise(np.asarray(im.convert("RGB"), dtype=np.uint8))
        assert np.array_equal(got, want)


def test_a_recon_bundle_computes_recon_features(tmp_path, pred, monkeypatch):
    """An A4 head expects `[f | r]`. The bundle records that, so the recon
    branch runs rather than the head being handed half its input -- which is a
    shape error at best and a confident meaningless score at worst."""
    calls = []
    monkeypatch.setattr(
        "aigcdet.infer.Predictor._recon_vector",
        lambda self, img: (calls.append(img.shape),
                           np.zeros(12, np.float32))[1])
    _images(tmp_path / "in", n=2)
    out = tmp_path / "p.json"
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path, use_recon=True), out))

    assert len(calls) == 2
    assert len(json.loads(out.read_text())) == 2


def test_a_film_bundle_scores_normally(tmp_path, pred):
    """FiLM conditions on the degradation head's own embedding, computed
    inside Detector.forward -- it needs nothing extra from the caller."""
    _images(tmp_path / "in", n=2)
    out = tmp_path / "p.json"
    pred.main(_argv(tmp_path / "in", _bundle(tmp_path, use_film=True), out))
    assert len(json.loads(out.read_text())) == 2


def test_an_empty_directory_fails_loudly(tmp_path, pred):
    """Writing `[]` and exiting 0 looks like a successful run of a model that
    scored nothing. The likely cause is a wrong --images path."""
    (tmp_path / "in").mkdir()
    with pytest.raises(SystemExit, match="no images"):
        pred.main(_argv(tmp_path / "in", _bundle(tmp_path), tmp_path / "p.json"))


def test_a_missing_bundle_says_what_a_bundle_is(tmp_path, pred):
    _images(tmp_path / "in", n=1)
    with pytest.raises(SystemExit, match="export_bundle"):
        pred.main(_argv(tmp_path / "in", tmp_path / "nope",
                        tmp_path / "p.json"))


def test_an_unreadable_file_gets_a_row_and_a_non_zero_exit(tmp_path, pred):
    """A judge's directory will contain something PIL cannot open. Every image
    still gets a row -- a result file shorter than the directory is a puzzle
    for whoever reads it -- scored at 0.5, which asserts nothing. The non-zero
    exit is how the run reports that it was not complete."""
    _images(tmp_path / "in", n=3)
    (tmp_path / "in" / "broken.png").write_bytes(b"not a png")
    out = tmp_path / "p.json"
    rc = pred.main(_argv(tmp_path / "in", _bundle(tmp_path), out))

    rows = json.loads(out.read_text())
    assert len(rows) == 4
    bad = [r for r in rows if r["image_path"].endswith("broken.png")]
    assert len(bad) == 1 and bad[0]["pred"] == 0.5
    assert rc != 0


def test_scores_stay_paired_with_their_paths_when_a_colon_named_file_fails(
        tmp_path, pred, monkeypatch):
    """A colon is legal in a POSIX filename, and an error line reads
    `<path>: <error>` -- so any scheme that recovers the pairing by splitting
    such a string truncates the path, and `zip` then pairs each score with the
    WRONG image. Every row still looks well-formed."""
    d = tmp_path / "in"
    _images(d, n=2)
    (d / "a:b.png").write_bytes(b"not a png")

    monkeypatch.setattr(pred.infer, "embed",
                        lambda m, s, imgs, device="cpu", batch_size=16:
                            np.arange(len(imgs) * s.dim, dtype=np.float32
                                      ).reshape(len(imgs), s.dim))
    out = tmp_path / "p.json"
    pred.main(_argv(d, _bundle(tmp_path), out, "--batch-size", "8"))

    rows = json.loads(out.read_text())
    assert [r["image_path"] for r in rows] == [
        str(d / "a:b.png"), str(d / "img0.png"), str(d / "img1.png")]
    assert rows[0]["pred"] == 0.5


def test_runs_as_a_command_line_program(tmp_path):
    """Imported-and-called is not the same as `python scripts/predict.py`.
    The deliverable is the command line, so it is exercised as one."""
    _images(tmp_path / "in", n=2)
    bundle = _bundle(tmp_path)
    out = tmp_path / "p.json"
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import numpy as np, runpy, sys\n"
        "import aigcdet.infer as I\n"
        "import aigcdet.features.backbones as B\n"
        "spec = B.BackboneSpec('fake', 'none', 64, 8, 1, 0)\n"
        # Patched on `infer`, where they are bound: the script imports the
        # Predictor, not the backbone helpers.
        "I.load_backbone = lambda n, device: (None, spec)\n"
        "I.embed = lambda m, s, imgs, device='cpu', batch_size=16: "
        "np.zeros((len(imgs), s.dim), np.float32)\n"
        f"sys.argv = ['predict.py', '--images', {str(tmp_path / 'in')!r}, "
        f"'--bundle', {bundle!r}, '--out', {str(out)!r}, '--device', 'cpu']\n"
        f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n")
    r = subprocess.run([sys.executable, str(stub)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(json.loads(out.read_text())) == 2


# --- the finetuned-checkpoint mode ------------------------------------------

def _dual_ckpt(tmp_path):
    """A checkpoint in `_write_ckpt`'s dual shape, with weightless towers.

    Identity towers with empty state dicts satisfy `_load_models`' load path
    (fp32 cast, load_state_dict, bf16 cast) without a real backbone; the
    forward is stubbed at `_forward_tower`, which is where the real path
    diverges from the bundle mode anyway.
    """
    torch.manual_seed(0)
    head = Detector(dim_feat=2 * DIM)
    ck = {"state_dict": head.state_dict(),
          "tower_state_dicts": [{}, {}],
          "backbones": ["fakeA", "fakeB"],
          "dim_feat": 2 * DIM,
          "epoch": 1,
          "config": {"policy_mode": "crop", "crop_side": 32,
                     "nominal_side": 48, "use_recon": False,
                     "use_film": False, "head_hidden": 512}}
    p = tmp_path / "finetuned.pt"
    torch.save(ck, p)
    return str(p)


@pytest.fixture()
def pred_ckpt(monkeypatch):
    """`predict.py` with the finetuned path's two seams stubbed."""
    import importlib.util

    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.train import finetune

    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import score_plan_splits

    spec = importlib.util.spec_from_file_location("predict_mod_ckpt", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(
        score_plan_splits, "load_backbone",
        lambda n, device="cpu": (torch.nn.Identity(),
                                 BackboneSpec(n, "none", 48, DIM, 1, 0)))
    monkeypatch.setattr(
        finetune, "_forward_tower",
        lambda model, sp, imgs, device, dtype, chunk: torch.stack(
            [torch.full((sp.dim,), float(np.mean(i)) / 255.0) for i in imgs]))
    return mod


def test_ckpt_mode_writes_the_same_contract(tmp_path, pred_ckpt):
    paths = _images(tmp_path / "in")
    ck = _dual_ckpt(tmp_path)
    out = tmp_path / "p.json"
    rc = pred_ckpt.main(["--images", str(tmp_path / "in"), "--ckpt", ck,
                         "--out", str(out), "--device", "cpu"])
    assert rc == 0
    rows = json.loads(out.read_text())
    assert [r["image_path"] for r in rows] == [str(p) for p in sorted(paths)]
    assert all(sorted(r) == ["image_path", "pred"] for r in rows)
    assert all(0.0 <= r["pred"] <= 1.0 for r in rows)


def test_ckpt_mode_scores_a_bad_file_half_and_exits_nonzero(tmp_path, pred_ckpt):
    _images(tmp_path / "in", n=2)
    (tmp_path / "in" / "broken.png").write_bytes(b"not an image")
    ck = _dual_ckpt(tmp_path)
    out = tmp_path / "p.json"
    rc = pred_ckpt.main(["--images", str(tmp_path / "in"), "--ckpt", ck,
                         "--out", str(out), "--device", "cpu"])
    assert rc == 1
    rows = {r["image_path"]: r["pred"] for r in json.loads(out.read_text())}
    assert rows[str(tmp_path / "in" / "broken.png")] == 0.5
    assert len(rows) == 3


def test_ckpt_and_bundle_are_mutually_exclusive(tmp_path, pred_ckpt):
    _images(tmp_path / "in", n=1)
    with pytest.raises(SystemExit):
        pred_ckpt.main(["--images", str(tmp_path / "in"),
                        "--out", str(tmp_path / "p.json"), "--device", "cpu"])


def test_ckpt_mode_refuses_a_recon_head(tmp_path, pred_ckpt):
    torch.manual_seed(0)
    head = Detector(dim_feat=2 * DIM, use_recon=True)
    ck = {"state_dict": head.state_dict(), "tower_state_dicts": [{}, {}],
          "backbones": ["fakeA", "fakeB"], "dim_feat": 2 * DIM, "epoch": 1,
          "config": {"policy_mode": "crop", "crop_side": 32,
                     "nominal_side": 48, "use_recon": True,
                     "use_film": False, "head_hidden": 512}}
    p = tmp_path / "recon.pt"
    torch.save(ck, p)
    _images(tmp_path / "in", n=1)
    with pytest.raises(SystemExit, match="recon"):
        pred_ckpt.main(["--images", str(tmp_path / "in"), "--ckpt", str(p),
                        "--out", str(tmp_path / "p.json"), "--device", "cpu"])
