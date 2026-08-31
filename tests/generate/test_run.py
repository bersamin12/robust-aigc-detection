"""The guards in the generation loop.

None of these need a GPU: they are the checks that decide whether an output
becomes a manifest row, and every one of them exists because the failure it
catches is invisible in aggregate statistics.
"""
import json

import numpy as np
import pytest
from PIL import Image

from aigcdet.generate.run import (MIN_DELTA, MIN_STD, _done_ids, _round_up,
                                  check)


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


# --- size granularity -------------------------------------------------------
# Kandinsky 2.2 was measured silently rounding a request UP to a multiple of
# 64: 432x640 and 416x640 both came back 448x640 -- the same image twice. A
# pipeline that decides its own size is how a family ends up with a dimension
# its real does not share, which is the leak the whole corpus exists to avoid.

def test_round_up_is_a_no_op_for_the_sizes_the_frozen_suite_uses():
    """`crop_box` emits multiples of 16, and every model in the frozen suite
    has size_multiple 8. Rounding must therefore change nothing for them, or
    this would silently re-cut the corpus already on disk."""
    for n in (400, 416, 432, 448, 560, 640):
        assert _round_up(n, 8) == n


def test_round_up_never_rounds_down():
    """A smaller output cannot be cropped up to the real's box; `generate`
    raises on it. So the rounding has to go the safe way."""
    for n, mult in ((432, 64), (416, 64), (433, 32), (1, 32)):
        assert _round_up(n, mult) >= n
        assert _round_up(n, mult) % mult == 0


def test_round_up_matches_what_kandinsky_actually_did():
    assert _round_up(432, 64) == 448 and _round_up(416, 64) == 448


def test_round_up_refuses_a_nonsense_multiple():
    with pytest.raises(ValueError, match="size_multiple"):
        _round_up(64, 0)


# --- licence ----------------------------------------------------------------

def test_check_licence_verifies_every_repo_the_pipeline_pulls(monkeypatch):
    """A combined pipeline loads its prior from a SECOND repo. Checking only
    `hf_id` licence-clears half the weights that made the image, and the
    corpus's whole licence position is the registry being true."""
    import aigcdet.generate.run as run_mod
    from aigcdet.generate.registry import MODELS, ModelSpec

    spec = ModelSpec(hf_id="org/decoder", licence_tag="apache-2.0",
                     commercial=True, lineage="x", arch="unet",
                     companion_ids=("org/prior",))
    monkeypatch.setitem(MODELS, "_probe", spec)

    seen = []

    class _Info:
        def __init__(self, lic):
            self.cardData = {"license": lic}

    def fake_info(hf_id, *a, **k):
        seen.append(hf_id)
        return _Info("apache-2.0" if hf_id == "org/decoder" else "other")

    monkeypatch.setattr("huggingface_hub.model_info", fake_info)
    with pytest.raises(RuntimeError, match="org/prior"):
        run_mod.check_licence("_probe")
    assert seen == ["org/decoder", "org/prior"]
