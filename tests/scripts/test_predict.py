"""The submission's inference entry point (brief §5.5.2).

    "A script that takes an image directory as input and outputs a confidence
     score for each image, indicating the likelihood that it is AIGC-generated.
     The output should be a JSON file containing image_path and pred for each
     image."

Everything else in this repository consumes a frozen manifest and a cached
feature bank. This is the one path a judge actually runs, on a directory of
loose files with no manifest, no labels and no bank -- so the properties that
the rest of the pipeline gets from the manifest have to be established here
instead.

The one that matters most is canonicalisation. `predict` is a FOURTH decode
site, and the resolution shortcut (docs/resolution_shortcut.md) means a decode
site that skips it computes features on different pixels than the head was
trained on -- with no shape error, no warning, and scores that are merely
wrong rather than absent.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from aigcdet.models.heads import Detector

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
SCRIPT = str(REPO / "scripts" / "predict.py")


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


def _checkpoint(path, dim=8, use_recon=False, use_film=False):
    torch.manual_seed(0)
    model = Detector(dim_feat=dim, use_recon=use_recon, use_film=use_film)
    torch.save({"state_dict": model.state_dict(),
                "config": {"use_recon": use_recon, "use_film": use_film,
                           "name": "a3", "seed": 1},
                "dim_feat": dim, "backbone": "fake"}, path)
    return str(path)


@pytest.fixture()
def pred(monkeypatch):
    """Import the script with a stand-in backbone: no weights, no GPU."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("predict_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from aigcdet.features.backbones import BackboneSpec
    bspec = BackboneSpec("fake", "none", 64, 8, 1, 0)
    monkeypatch.setattr(mod, "load_backbone", lambda n, device: (None, bspec))
    monkeypatch.setattr(
        mod, "embed",
        lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))
    return mod


# --- the deliverable's literal contract -------------------------------------

def test_writes_image_path_and_pred_for_every_image(tmp_path, pred):
    paths = _images(tmp_path / "in")
    out = tmp_path / "preds.json"
    pred.main(["--images", str(tmp_path / "in"),
               "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
               "--out", str(out), "--device", "cpu"])

    rows = json.loads(out.read_text())
    assert isinstance(rows, list) and len(rows) == len(paths)
    assert all(set(r) == {"image_path", "pred"} for r in rows)
    assert {r["image_path"] for r in rows} == {str(p) for p in paths}


def test_pred_is_a_probability(tmp_path, pred):
    _images(tmp_path / "in")
    out = tmp_path / "preds.json"
    pred.main(["--images", str(tmp_path / "in"),
               "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
               "--out", str(out), "--device", "cpu"])
    for r in json.loads(out.read_text()):
        assert isinstance(r["pred"], float)
        assert 0.0 <= r["pred"] <= 1.0


def test_finds_images_in_nested_directories_and_ignores_non_images(tmp_path, pred):
    _images(tmp_path / "in", n=2)
    _images(tmp_path / "in" / "deeper", n=3)
    (tmp_path / "in" / "notes.txt").write_text("not an image")
    (tmp_path / "in" / "README.md").write_text("nor this")
    out = tmp_path / "preds.json"
    pred.main(["--images", str(tmp_path / "in"),
               "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
               "--out", str(out), "--device", "cpu"])
    rows = json.loads(out.read_text())
    assert len(rows) == 5
    assert not any(r["image_path"].endswith((".txt", ".md")) for r in rows)


def test_output_order_is_stable_across_runs(tmp_path, pred):
    """A judge may diff two runs. os.walk order is filesystem-dependent, so
    the rows are sorted rather than left to whatever readdir returned."""
    _images(tmp_path / "in", n=6)
    ck = _checkpoint(tmp_path / "ck.pt")
    got = []
    for i in range(2):
        out = tmp_path / f"p{i}.json"
        pred.main(["--images", str(tmp_path / "in"), "--checkpoint", ck,
                   "--out", str(out), "--device", "cpu"])
        got.append([r["image_path"] for r in json.loads(out.read_text())])
    assert got[0] == got[1] == sorted(got[0])


# --- the properties that fail silently --------------------------------------

def test_canonicalises_every_image_before_embedding(tmp_path, pred, monkeypatch):
    """The fourth decode site. If this one skips canonicalisation it scores
    images on different pixels than the head was trained on -- no error, just
    wrong numbers. Counted per image, not merely 'called at least once'."""
    seen = []
    real = pred.canonicalise
    monkeypatch.setattr(pred, "canonicalise",
                        lambda a: (seen.append(a.shape), real(a))[1])
    _images(tmp_path / "in", n=4)
    pred.main(["--images", str(tmp_path / "in"),
               "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
               "--out", str(tmp_path / "p.json"), "--device", "cpu"])
    assert len(seen) == 4


def test_the_script_uses_the_same_canonicalise_as_the_other_decode_sites(pred):
    from aigcdet.augment import canonical
    from aigcdet.features import extract

    assert pred.canonicalise is canonical.canonicalise is extract.canonicalise


def test_scores_the_clean_view_only(tmp_path, pred, monkeypatch):
    """Inference must embed the image as given, not a sampled augmented view.
    The bank's view 0 is the clean view; this is that view and nothing else,
    so the pixels handed to the backbone are the canonicalised original."""
    from aigcdet.augment.canonical import canonicalise

    handed = []
    monkeypatch.setattr(pred, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            (handed.extend(imgs),
                             np.zeros((len(imgs), s.dim), np.float32))[1])
    paths = _images(tmp_path / "in", n=3)
    pred.main(["--images", str(tmp_path / "in"),
               "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
               "--out", str(tmp_path / "p.json"), "--device", "cpu"])

    assert len(handed) == 3
    for p, got in zip(sorted(paths), handed):
        with Image.open(p) as im:
            want = canonicalise(np.asarray(im.convert("RGB"), dtype=np.uint8))
        assert np.array_equal(got, want)


def test_refuses_a_recon_checkpoint_rather_than_scoring_without_recon(tmp_path, pred):
    """An A4 head expects `[f | r]`. Handing it `f` alone is a shape error at
    best and a silently wrong score at worst, so it is refused by name."""
    _images(tmp_path / "in", n=2)
    with pytest.raises(SystemExit, match="recon"):
        pred.main(["--images", str(tmp_path / "in"),
                   "--checkpoint", _checkpoint(tmp_path / "ck.pt", use_recon=True),
                   "--out", str(tmp_path / "p.json"), "--device", "cpu"])


def test_a_film_checkpoint_scores_normally(tmp_path, pred):
    """FiLM conditions on the degradation head's own embedding, computed
    inside Detector.forward -- it needs nothing extra from the caller, so
    unlike recon it must NOT be refused."""
    _images(tmp_path / "in", n=2)
    out = tmp_path / "p.json"
    pred.main(["--images", str(tmp_path / "in"),
               "--checkpoint", _checkpoint(tmp_path / "ck.pt", use_film=True),
               "--out", str(out), "--device", "cpu"])
    assert len(json.loads(out.read_text())) == 2


def test_an_empty_directory_fails_loudly(tmp_path, pred):
    """Writing `[]` and exiting 0 looks like a successful run of a model that
    scored nothing. The likely cause is a wrong --images path."""
    (tmp_path / "in").mkdir()
    with pytest.raises(SystemExit, match="no images"):
        pred.main(["--images", str(tmp_path / "in"),
                   "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
                   "--out", str(tmp_path / "p.json"), "--device", "cpu"])


def test_an_unreadable_file_names_itself_and_does_not_abort_the_run(tmp_path, pred):
    """A judge's directory will contain something PIL cannot open. Losing a
    12-hour run's worth of scores to one bad file is worse than skipping it,
    but skipping it silently is worse than both."""
    _images(tmp_path / "in", n=3)
    (tmp_path / "in" / "broken.png").write_bytes(b"not a png")
    out = tmp_path / "p.json"
    rc = pred.main(["--images", str(tmp_path / "in"),
                    "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
                    "--out", str(out), "--device", "cpu"])
    rows = json.loads(out.read_text())
    assert len(rows) == 3
    assert not any("broken" in r["image_path"] for r in rows)
    assert rc != 0


def test_runs_as_a_command_line_program(tmp_path):
    """Imported-and-called is not the same as `python scripts/predict.py`.
    The deliverable is the command line, so it is exercised as one -- with a
    real (tiny) backbone stubbed out via the module's own hook."""
    _images(tmp_path / "in", n=2)
    ck = _checkpoint(tmp_path / "ck.pt")
    out = tmp_path / "p.json"
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import numpy as np, runpy, sys\n"
        "import aigcdet.features.backbones as B\n"
        "spec = B.BackboneSpec('fake', 'none', 64, 8, 1, 0)\n"
        "B.load_backbone = lambda n, device: (None, spec)\n"
        "B.embed = lambda m, s, imgs, device, batch_size=16: "
        "np.zeros((len(imgs), s.dim), np.float32)\n"
        f"sys.argv = ['predict.py', '--images', {str(tmp_path / 'in')!r}, "
        f"'--checkpoint', {ck!r}, '--out', {str(out)!r}, '--device', 'cpu']\n"
        f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n")
    r = subprocess.run([sys.executable, str(stub)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(json.loads(out.read_text())) == 2


def test_scores_stay_paired_with_their_paths_when_a_colon_named_file_fails(
        tmp_path, pred, monkeypatch):
    """The pairing must not be reconstructed by parsing the failure strings.

    A colon is legal in a POSIX filename, and a failure line reads
    `<path>: <error>` -- so splitting one on ':' truncates such a path, the
    undecodable file survives the filter, and `zip` then silently pairs each
    score with the WRONG image_path. Every row still looks well-formed."""
    d = tmp_path / "in"
    _images(d, n=2)
    (d / "a:b.png").write_bytes(b"not a png")

    # One batch, so the failed file and the good ones are filtered together.
    monkeypatch.setattr(pred, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.arange(len(imgs) * s.dim, dtype=np.float32
                                      ).reshape(len(imgs), s.dim))
    out = tmp_path / "p.json"
    pred.main(["--images", str(d), "--checkpoint", _checkpoint(tmp_path / "ck.pt"),
               "--out", str(out), "--batch-size", "8", "--device", "cpu"])

    rows = json.loads(out.read_text())
    assert [r["image_path"] for r in rows] == [str(d / "img0.png"), str(d / "img1.png")]
