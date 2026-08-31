"""The guards in the generation loop.

None of these need a GPU: they are the checks that decide whether an output
becomes a manifest row, and every one of them exists because the failure it
catches is invisible in aggregate statistics.
"""
import json

import numpy as np
import pytest
from PIL import Image

from aigcdet.generate.run import MIN_DELTA, MIN_STD, _done_ids, check


def _img(seed=0, w=64, h=96):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_a_flat_frame_is_rejected():
    """The failure mode of an OOM-degraded pipeline: it returns something
    plausibly shaped and entirely black."""
    black = Image.new("RGB", (64, 96), (0, 0, 0))
    with pytest.raises(ValueError, match="near-constant"):
        check(black, _img(), "t2i")


def test_an_output_that_is_its_own_real_is_rejected():
    """A corpus of copies trains a detector to call photographs fake."""
    real = _img(1)
    with pytest.raises(ValueError, match="copy"):
        check(real.copy(), real, "img2img")


def test_a_genuine_redraw_passes():
    check(_img(2), _img(3), "img2img")


def test_t2i_is_exempt_from_the_copy_check():
    """A t2i output shares only content with its real; there is no pixel
    correspondence to measure, and a coincidental match is not a copy."""
    real = _img(4)
    check(real.copy(), real, "t2i")


def test_thresholds_are_the_documented_ones():
    assert MIN_STD == 2.0 and MIN_DELTA == 1.5


def test_done_ids_only_counts_rows_whose_files_survive(tmp_path):
    """Resume must be driven by what is on disk, not by what was logged: a
    killed run can leave a row whose image never landed."""
    out = tmp_path / "root"
    rows = tmp_path / "rows_x.jsonl"
    (out / "f").mkdir(parents=True)
    (out / "r").mkdir(parents=True)
    for i in ("a", "b"):
        (out / "f" / f"{i}.jpg").write_bytes(b"x")
        (out / "r" / f"{i}.jpg").write_bytes(b"x")
    (out / "f" / "b.jpg").unlink()               # fake lost to a kill
    with rows.open("w") as fh:
        for i in ("a", "b", "c"):
            fh.write(json.dumps({"image_id": i, "fake_rel": f"f/{i}.jpg",
                                 "real_rel": f"r/{i}.jpg"}) + "\n")
    assert _done_ids(rows, out, "x") == {"a"}


def test_done_ids_survives_a_truncated_last_line(tmp_path):
    """SIGKILL mid-write leaves half a JSON object. Losing one row is correct;
    refusing to resume the other 9,999 is not."""
    out = tmp_path / "root"
    (out / "f").mkdir(parents=True)
    (out / "r").mkdir(parents=True)
    (out / "f" / "a.jpg").write_bytes(b"x")
    (out / "r" / "a.jpg").write_bytes(b"x")
    rows = tmp_path / "rows_x.jsonl"
    rows.write_text(json.dumps({"image_id": "a", "fake_rel": "f/a.jpg",
                                "real_rel": "r/a.jpg"}) + '\n{"image_id": "b"')
    assert _done_ids(rows, out, "x") == {"a"}


def test_done_ids_on_a_fresh_run(tmp_path):
    assert _done_ids(tmp_path / "nope.jsonl", tmp_path, "x") == set()
