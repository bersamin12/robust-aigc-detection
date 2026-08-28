"""`scripts/acquire_data.py`'s SID_Set routine: mode handling and resume.

A real run died 206 images in with `OSError: cannot write mode CMYK as PNG` —
SID_Set carries at least one CMYK JPEG, and PNG cannot represent CMYK. The
stream is faked here (a list of records is exactly what `load_dataset(...,
streaming=True)` yields), so nothing is downloaded.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from aigcdet.data.sources import classify

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "scripts")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ad = _load_script("acquire_data")


def _image(mode="RGB", size=(8, 8), seed=0):
    if mode == "RGB":
        arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3),
                                                   dtype=np.uint8)
        return Image.fromarray(arr)
    return Image.new(mode, size, color=None)


@pytest.fixture
def stream(monkeypatch):
    """Stand in for `datasets.load_dataset(..., streaming=True)`."""
    records: list[dict] = []

    def install():
        mod = types.ModuleType("datasets")
        mod.load_dataset = lambda *a, **k: records
        monkeypatch.setitem(sys.modules, "datasets", mod)
        return records

    return install


def _written(out):
    found = []
    for d, _, files in os.walk(os.path.join(out, "sid_set")):
        found += [os.path.join(d, f) for f in files if f.endswith(".png")]
    return sorted(found)


def _report(out):
    with open(os.path.join(out, "sid_set", "ingest_report.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# the CMYK crash
# --------------------------------------------------------------------------

def test_a_cmyk_record_is_written_rather_than_ending_the_run(tmp_path, stream):
    """The live failure: one CMYK JPEG raised OSError out of `acquire_sid_set`
    and lost the whole streaming download."""
    recs = stream()
    recs += [{"label": 0, "image": _image("RGB", seed=1)},
             {"label": 1, "image": _image("CMYK")},
             {"label": 0, "image": _image("RGB", seed=2)}]
    out = str(tmp_path / "raw")
    assert ad.acquire_sid_set(out, 10) == 3
    assert len(_written(out)) == 3
    modes = _report(out)["modes"]
    assert modes["CMYK->RGB"] == 1 and modes["RGB->RGB"] == 2
    assert _report(out)["failures"] == []


def test_the_cmyk_image_is_readable_rgb_on_disk(tmp_path, stream):
    recs = stream()
    recs += [{"label": 1, "image": _image("CMYK")}]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 10)
    with Image.open(_written(out)[0]) as im:
        assert im.format == "PNG" and im.mode == "RGB"


def test_an_rgba_record_keeps_its_alpha_rather_than_being_flattened(tmp_path, stream):
    """RGBA is PNG-native, so acquisition must not convert it. A blanket
    `.convert("RGB")` composites alpha against black, changing every
    transparent pixel to a colour that was never in the image — a real pixel
    change in a project whose premise is that pixel-level cues matter."""
    im = Image.new("RGBA", (8, 8), (200, 30, 40, 0))
    recs = stream()
    recs += [{"label": 1, "image": im}]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 10)
    with Image.open(_written(out)[0]) as back:
        assert back.mode == "RGBA"
        assert back.getpixel((0, 0)) == (200, 30, 40, 0)
    assert _report(out)["modes"] == {"RGBA->RGBA": 1}


def test_a_premultiplied_alpha_mode_converts_to_rgba_not_rgb(tmp_path, stream):
    """"RGBa" is not PNG-native and its name does not end in "A"; alpha is
    detected from the bands, so it still lands in RGBA and keeps its alpha."""
    recs = stream()
    recs += [{"label": 1, "image": Image.new("RGBa", (8, 8))}]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 10)
    with Image.open(_written(out)[0]) as back:
        assert back.mode == "RGBA"
    assert _report(out)["modes"] == {"RGBa->RGBA": 1}


def test_a_record_that_cannot_be_saved_at_all_is_recorded_not_fatal(tmp_path, stream):
    class Broken:
        mode = "RGB"

        def getbands(self):
            raise OSError("truncated file")

    recs = stream()
    recs += [{"label": 0, "image": _image("RGB", seed=1)},
             {"label": 1, "image": Broken()},
             {"label": 0, "image": _image("RGB", seed=2)}]
    out = str(tmp_path / "raw")
    assert ad.acquire_sid_set(out, 10) == 2
    failures = _report(out)["failures"]
    assert len(failures) == 1 and failures[0]["index"] == 1
    assert "truncated file" in failures[0]["reason"]


# --------------------------------------------------------------------------
# naming and resume
# --------------------------------------------------------------------------

def test_names_come_from_the_stream_position_not_the_success_count(tmp_path, stream):
    """A counter of successful saves renumbered every later image the moment
    one record was skipped, so a resumed run — which skips a different set —
    disagreed with the first about which file is which."""
    recs = stream()
    recs += [{"label": 2, "image": _image("RGB", seed=1)},   # tampered: skipped
             {"label": 0, "image": _image("RGB", seed=2)},
             {"label": 1, "image": _image("RGB", seed=3)}]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 10)
    assert sorted(os.path.basename(p) for p in _written(out)) == [
        "0000001.png", "0000002.png"]


def test_a_resumed_run_skips_what_is_on_disk_and_respects_the_cap(tmp_path, stream):
    recs = stream()
    recs += [{"label": i % 2, "image": _image("RGB", seed=i)} for i in range(10)]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 4)
    first = _written(out)
    stamps = {p: os.stat(p).st_mtime_ns for p in first}
    assert len(first) == 4

    assert ad.acquire_sid_set(out, 4) == 4
    assert _written(out) == first
    # Untouched: a resume must not rewrite bytes it already has.
    assert {p: os.stat(p).st_mtime_ns for p in first} == stamps


def test_a_partial_run_is_completed_rather_than_restarted(tmp_path, stream):
    recs = stream()
    recs += [{"label": i % 2, "image": _image("RGB", seed=i)} for i in range(10)]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 6)
    full = _written(out)
    for p in full[2:]:
        os.remove(p)
    ad.acquire_sid_set(out, 6)
    assert _written(out) == full


def test_the_ingest_report_accumulates_across_runs(tmp_path, stream):
    recs = stream()
    recs += [{"label": 0, "image": _image("RGB", seed=1)},
             {"label": 1, "image": _image("CMYK")}]
    out = str(tmp_path / "raw")
    ad.acquire_sid_set(out, 1)          # first record only
    assert _report(out)["modes"] == {"RGB->RGB": 1}
    ad.acquire_sid_set(out, 2)          # resumes, then writes the CMYK one
    assert _report(out)["modes"] == {"CMYK->RGB": 1, "RGB->RGB": 1}


def test_tampered_records_are_skipped_and_buckets_read_back_correctly(tmp_path, stream):
    recs = stream()
    recs += [{"label": 0, "image": _image("RGB", seed=1)},
             {"label": 2, "image": _image("RGB", seed=2)},
             {"label": 1, "image": _image("RGB", seed=3)},
             {"label": 1, "image": _image("RGB", seed=4), "generator": "sdxl"}]
    out = str(tmp_path / "raw")
    assert ad.acquire_sid_set(out, 10) == 3
    seen = set()
    for p in _written(out):
        rel = os.path.relpath(p, out).split(os.sep)
        seen.add(classify(rel[0], rel[1]))
    assert seen == {(0, ""), (1, "sid_set"), (1, "sdxl")}
