"""The guards in the generation loop.

None of these need a GPU: they are the checks that decide whether an output
becomes a manifest row, and every one of them exists because the failure it
catches is invisible in aggregate statistics.
"""
import json
from pathlib import Path

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


class _FakePipe:
    """Enough of a diffusers pipeline for `load` to finish without weights."""

    def set_progress_bar_config(self, **k):
        pass

    def to(self, device):
        return self


def _capture_load(monkeypatch, model_key, **spec_kw):
    """Run `load` against a stubbed `from_pretrained` and return its kwargs."""
    import aigcdet.generate.run as run_mod
    from aigcdet.generate.registry import MODELS, ModelSpec

    monkeypatch.setitem(MODELS, model_key, ModelSpec(
        hf_id="org/probe", licence_tag="apache-2.0", commercial=True,
        lineage="x", arch="unet", **spec_kw))
    monkeypatch.setattr(run_mod, "check_licence", lambda k: None)
    monkeypatch.setattr(run_mod, "_free_vram_gb", lambda d: 999.0)

    seen = {}

    def fake_from_pretrained(hf_id, **kw):
        seen.update(kw)
        return _FakePipe()

    import diffusers
    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    run_mod.load(model_key, {"t2i"}, device="cpu")
    return seen


def test_sd15s_safety_checker_is_never_loaded(monkeypatch):
    """It returns a BLACK FRAME rather than raising, so it reaches `check` as
    "near-constant output" -- a generation failure that is not one. Three of
    them in sd15_t2i's first 55 crossed run_family's 5% abort and killed a
    lane 40 minutes into a 2.3 h run."""
    seen = _capture_load(monkeypatch, "_probe_checker", safety_checker=True)
    assert seen["safety_checker"] is None
    assert seen["requires_safety_checker"] is False


def test_the_kwarg_is_not_passed_to_models_that_have_no_checker(monkeypatch):
    """Diffusers pipelines swallow unknown kwargs, so a blanket
    `safety_checker=None` would not raise on Sana or Klein -- it would just be
    a claim about every checkpoint in the registry that only one of them
    ships."""
    seen = _capture_load(monkeypatch, "_probe_plain")
    assert "safety_checker" not in seen
    assert "requires_safety_checker" not in seen


def test_sd15_is_the_only_checkpoint_flagged():
    """If a second SD1.x checkpoint is ever added, it needs the flag too --
    this is the assertion that says so out loud."""
    from aigcdet.generate.registry import MODELS

    flagged = {k for k, s in MODELS.items() if s.safety_checker}
    assert flagged == {"sd15_base"}


def test_a_guard_rejection_and_an_aborted_family_differ_by_exit_code():
    """They used to share exit code 1, and `run_real.run_lane` abandons a lane
    on any non-zero code -- so one copy-guard hit in a lane's FIRST shard would
    have silently dropped its second, ~1,900 pairs, while the run still printed
    ALL LANES DONE. An abort raises and Python exits 1; a completed run with
    rejections must therefore not."""
    from aigcdet.generate.run import EXIT_SOME_FAILED

    assert EXIT_SOME_FAILED != 0, "must be distinguishable from a clean run"
    assert EXIT_SOME_FAILED != 1, "1 is what an uncaught abort already exits"


def test_generate_ov7_returns_the_soft_code_not_one(tmp_path, monkeypatch):
    """The call site, not just the constant: `main` must return the soft code
    when images were rejected and 0 when none were."""
    import importlib.util

    from aigcdet.generate.run import EXIT_SOME_FAILED

    root = Path(__file__).resolve().parents[2]
    src = (root / "scripts" / "generate_ov7.py").read_text()
    assert "return EXIT_SOME_FAILED if failed else 0" in src, (
        "generate_ov7.main no longer returns the soft code; run_real.run_lane "
        "would abandon a lane on a single guard rejection again")
    assert "return 1 if failed else 0" not in src
