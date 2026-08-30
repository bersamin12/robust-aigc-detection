import numpy as np
import pytest
from PIL import Image

from aigcdet.data.encoder_parity import (
    GEOMETRY_CROP, GEOMETRY_RESAMPLE, ParityError, conform, crop_to_size,
    read_profile, save_matched,
)

from pilot_commercial_apis import PROVIDERS, sanitise


def _photo(w, h, seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (max(1, h // 8), max(1, w // 8), 3), dtype=np.uint8)
    return Image.fromarray(small).resize((w, h), Image.BICUBIC)


def _real(tmp_path, size=(400, 600), quality=71):
    p = tmp_path / "real.jpg"
    _photo(*size).save(p, format="JPEG", quality=quality, subsampling=2)
    return str(p)


# --- prompt sanitisation (§3.5) --------------------------------------------

def test_strips_the_spoken_boilerplate_every_narrative_opens_with():
    """These captions were transcribed from speech, so they all begin by
    describing the act of looking at a picture rather than its contents."""
    prompt, why = sanitise("In this image we can see two people and many bottles on the shelves.")
    assert why == ""
    assert prompt.startswith("Two people")
    assert "in this image" not in prompt.lower()


@pytest.mark.parametrize("caption", [
    "In this image I can see the human hand and the background is in white color.",
    "In this picture we can see a dog running across a green field.",
    "This image shows a bicycle leaning against a red brick wall.",
])
def test_boilerplate_variants_are_all_recognised(caption):
    prompt, why = sanitise(caption)
    assert why == ""
    assert not prompt.lower().startswith(("in this", "this image", "we can see", "i can see"))


def test_refuses_a_narrative_naming_a_person():
    """§3.5 bars prompting for identifiable individuals, and the photographs
    behind these narratives are of real people."""
    prompt, why = sanitise("In this image we can see Barack standing at a podium.")
    assert prompt == ""
    assert "§3.5" in why and "Barack" in why


def test_does_not_refuse_ordinary_sentence_initial_capitals():
    """A capital after a full stop is grammar, not a name. Over-refusing here
    throws away usable prompts for nothing."""
    prompt, why = sanitise(
        "In this image we can see a beach. There is sand and the sky is cloudy.")
    assert why == "", why
    assert "sand" in prompt


def test_common_nouns_for_people_are_not_treated_as_names():
    prompt, why = sanitise(
        "In this image I can see a man holding a phone and a woman wearing glasses.")
    assert why == ""
    assert "man" in prompt


def test_refuses_a_caption_that_is_only_boilerplate():
    prompt, why = sanitise("In this image we can see.")
    assert prompt == "" and "too short" in why


def test_refuses_empty_input_without_raising():
    assert sanitise("")[0] == ""
    assert sanitise(None)[0] == ""


# --- the provider roster ----------------------------------------------------

def test_every_provider_is_its_own_family():
    """`docs/03` §4: providers must not be merged into one `commercial_api`
    bucket, or the report cannot say which provider we fail on."""
    families = [p.family for p in PROVIDERS]
    assert len(families) == len(set(families))
    assert "commercial_api" not in families


def test_ideogram_is_not_routed_through_openrouter():
    """It is not on OpenRouter, so it keeps its own adapter and key."""
    ideogram = [p for p in PROVIDERS if "ideogram" in p.family][0]
    assert ideogram.adapter == "ideogram"
    assert all(p.adapter == "openrouter" for p in PROVIDERS if p is not ideogram)


def test_google_row_carries_the_synthid_warning():
    """§5.7 requires the flag; the roster is where someone will actually see it."""
    google = [p for p in PROVIDERS if "google" in p.family][0]
    assert "SynthID" in google.note


# --- the two geometries (§3.1) ---------------------------------------------

def test_crop_geometry_resamples_nothing(tmp_path):
    """A crop takes real pixels out of the frame unchanged, so the window must
    be byte-identical to the same window of the source."""
    prof = read_profile(_real(tmp_path, size=(400, 600)))
    src = _photo(1024, 1536, seed=5)
    out = conform(src, prof, GEOMETRY_CROP)
    assert out.size == (400, 600)
    left, top = (1024 - 400) // 2, (1536 - 600) // 2
    expected = src.crop((left, top, left + 400, top + 600)).convert("RGB")
    assert np.array_equal(np.asarray(out), np.asarray(expected))


def test_resample_geometry_keeps_the_whole_frame(tmp_path):
    """The two modes trade against each other: resample keeps field of view and
    pays a resampling signature, crop keeps pixels and pays field of view."""
    prof = read_profile(_real(tmp_path, size=(400, 600)))
    out = conform(_photo(1024, 1536, seed=5), prof, GEOMETRY_RESAMPLE)
    assert out.size == (400, 600)
    crop = conform(_photo(1024, 1536, seed=5), prof, GEOMETRY_CROP)
    assert not np.array_equal(np.asarray(out), np.asarray(crop))


def test_both_geometries_produce_identical_encoder_parity(tmp_path):
    """Geometry is the only thing that differs. Both must still carry the
    real's exact tables, or the gate would be comparing two things at once."""
    real_path = _real(tmp_path, size=(400, 600), quality=63)
    prof = read_profile(real_path)
    paths = {}
    for geometry in (GEOMETRY_RESAMPLE, GEOMETRY_CROP):
        dst = str(tmp_path / f"{geometry}.jpg")
        save_matched(_photo(1024, 1536, seed=7), dst, prof, geometry)
        paths[geometry] = dst
    with Image.open(real_path) as r, \
         Image.open(paths[GEOMETRY_RESAMPLE]) as a, \
         Image.open(paths[GEOMETRY_CROP]) as b:
        assert a.quantization == r.quantization == b.quantization
        assert a.size == r.size == b.size


def test_crop_refuses_rather_than_upscaling(tmp_path):
    prof = read_profile(_real(tmp_path, size=(400, 600)))
    with pytest.raises(ParityError, match="cannot crop"):
        conform(_photo(300, 450), prof, GEOMETRY_CROP)


def test_crop_to_size_is_centred():
    im = _photo(1000, 1000)
    out = crop_to_size(im, 400, 600)
    expected = im.crop((300, 200, 700, 800))
    assert np.array_equal(np.asarray(out), np.asarray(expected))


def test_unknown_geometry_is_refused(tmp_path):
    prof = read_profile(_real(tmp_path))
    with pytest.raises(ParityError, match="unknown geometry"):
        conform(_photo(1024, 1536), prof, "squish")


def test_name_check_runs_before_the_boilerplate_strip():
    """Regression. The subject of a narrative is usually its first content
    word, so stripping first puts a name at position 0 where a sentence-initial
    exclusion ignores it. The first version of `sanitise` had exactly this bug
    and passed a named individual straight through."""
    for caption in (
        "In this image we can see Barack standing at a podium.",
        "In this picture I can see Serena holding a tennis racket.",
        "In this image we can see a crowd and Elvis on the stage.",
    ):
        prompt, why = sanitise(caption)
        assert prompt == "", f"leaked a name: {caption!r} -> {prompt!r}"
        assert "§3.5" in why


def test_capitalised_common_noun_opener_still_passes():
    """The counterpart risk: over-refusing "Shopping trolleys." would throw
    away usable prompts for nothing."""
    prompt, why = sanitise("In this image we can see shopping trolleys and vehicles behind.")
    assert why == "" and "hopping trolleys" in prompt
