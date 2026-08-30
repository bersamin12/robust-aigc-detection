"""Give a generated image the same encoding history as the real it is paired
with, at acquisition time, before anything else touches it.

WHY THIS EXISTS
---------------
`docs/02-open-weight-generators-on-open-images.md` §1 makes this mandatory and
`docs/03-commercial-apis-on-open-images.md` §3.1 makes it the gate that clears
before money is spent. The reasoning is short: the Open Images reals are
`Thumbnail300KURL` thumbnails, i.e. *re-encoded JPEGs*, and this project has
already measured that JPEG history leaks the label
(`docs/low_level_confounds.md`: `jpeg_quality` AUC 0.5532 pooled). A generator
hands back a PNG. Store the two side by side and "was ever JPEG-compressed"
separates the classes perfectly, without a detector reading a single forensic
cue it could generalise from.

`data/normalize.py` does NOT fix this. It re-saves everything as PNG, which
equalises the *container* but not the *pixels*: JPEG blockiness already baked
into a real's samples survives a PNG re-save intact, which is exactly what
`proxies.estimate_jpeg_quality`'s pixel fallback measures. Parity has to be
established before normalisation, by putting the generated image through the
same encoder.

WHY QUANTISATION TABLES AND NOT A QUALITY NUMBER
------------------------------------------------
The obvious implementation samples a quality from the reals' distribution and
re-encodes at that quality. It is worse than it looks. Recovering an integer
quality from a quantisation table means inverting the standard scaling
(`proxies.estimate_jpeg_quality`'s exact branch does this), and the inversion
is lossy — a table that no standard quality produces exactly lands on a
neighbour, and the residual is a systematic, class-correlated offset in the
one statistic being gated.

Pillow will encode with a caller-supplied `qtables`, so the real's tables are
copied verbatim onto the fake instead. That makes `estimate_jpeg_quality`'s
exact branch return the *identical* value for both members of a pair by
construction, not approximately. Subsampling, progressive mode and restart
interval are copied for the same reason: each is a container property that
would otherwise be constant within a class and therefore free signal.

WHY THE PROFILE COMES FROM THE PAIRED REAL, NOT A DISTRIBUTION
--------------------------------------------------------------
Both handoffs pair every generated image to a specific real by `ImageID`.
Sampling encoder settings from the reals' marginal distribution would match the
two classes *in aggregate*; inheriting them from the paired real matches them
*row by row*, which is strictly stronger and costs nothing given the pairing
already exists. It also makes the residual confound measurable as a paired
difference rather than a difference of distributions.

Geometry is inherited for the same reason, and it is not a side concern:
`docs/resolution_shortcut.md` measures short side alone classifying at 72.6%
in the training pool and ~100% on the organisers' benchmark. Reals here have a
median short side of 457 and `normalize_image` never upscales, so a real
usually reaches the corpus at its native size while a 1024px API image is
LANCZOS-downscaled to 512. That is a resampling signature, constant within
each class. Matching the real's exact pixel dimensions removes it.

WHAT THIS DOES NOT FIX, STATED PLAINLY
--------------------------------------
A thumbnail is a re-encode of an already-compressed original, so a real
plausibly carries *double* JPEG history while a fake put through this module
carries *single*. Double-quantisation artefacts are a known forensic signature
and this module cannot reproduce one, because the original's quality is not
recoverable from the thumbnail. The residual is left deliberately unmodelled
rather than guessed at: `scripts/prove_encoder_parity.py` measures whether what
remains actually moves the gate, which is the only question that matters.
Guessing a two-stage encode would be fabricating a history we cannot verify.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image, ImageOps, JpegImagePlugin

#: Chroma subsampling is meaningless for a single-channel encode, and Pillow
#: rejects the keyword for one, so it is omitted below this band count.
_COLOUR_BANDS = 3


class ParityError(ValueError):
    """A profile could not be read, or could not be honoured."""


@dataclass(frozen=True)
class EncodingProfile:
    """Everything about how a real reached disk that a fake must reproduce.

    `qtables` is the raw Pillow quantisation mapping, copied rather than
    summarised — see the module docstring. `mode` is carried because a
    greyscale real encodes with one table and an RGB fake with three would not
    be the same encoder run at all.
    """

    width: int
    height: int
    mode: str
    qtables: dict
    subsampling: int
    progressive: bool
    restart_interval: int

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def aspect(self) -> float:
        return self.width / self.height


def read_profile(real_path: str) -> EncodingProfile:
    """Read the encoding profile off a real JPEG.

    Raises `ParityError` rather than falling back to a default when the file
    carries no quantisation tables. A silent fallback here would produce fakes
    encoded to a house default while the reals kept their own — which is the
    precise confound this module exists to remove, reintroduced by the error
    path. A source this cannot read is a source to skip, loudly.
    """
    try:
        with Image.open(real_path) as im:
            # Read the encoder settings off the ORIGINAL, before any transpose.
            # `ImageOps.exif_transpose` returns a plain copy, and a copy is no
            # longer a `JpegImageFile`: it carries neither `.quantization` nor
            # the sampling factors, so measuring after it silently loses the
            # only thing this function exists to read.
            tables = getattr(im, "quantization", None)
            if not tables:
                raise ParityError(f"no quantisation tables: {real_path}")
            profile = {
                "mode": im.mode,
                "qtables": {k: list(v) for k, v in tables.items()},
                "subsampling": (JpegImagePlugin.get_sampling(im)
                                if len(im.getbands()) >= _COLOUR_BANDS else -1),
                "progressive": bool(im.info.get("progression", False)),
                "restart_interval": int(im.info.get("restart_interval", 0) or 0),
            }
            # Geometry, though, is taken AFTER orientation is applied, because
            # `normalize_image` applies it too: a real tagged as rotated would
            # otherwise hand the fake a transposed target size.
            oriented = ImageOps.exif_transpose(im) or im
            return EncodingProfile(width=oriented.width, height=oriented.height,
                                   **profile)
    except ParityError:
        raise
    except (OSError, ValueError, KeyError) as e:
        raise ParityError(f"unreadable real {real_path}: {type(e).__name__}: {e}") from e


def crop_to_aspect(im: Image.Image, aspect: float) -> Image.Image:
    """Centre-crop `im` to `aspect` (width/height), removing as little as possible.

    Cropping rather than stretching is the whole point. An anisotropic resize
    to the real's exact dimensions would rescale the two axes by different
    factors, and every frequency-domain cue this project measures — blockiness
    anchored on JPEG's 8x8 grid, the Laplacian variance, the noise floor — is
    anisotropic afterwards in a way that is constant within the generated
    class. That trades a resolution confound for a subtler one.
    """
    w, h = im.size
    if abs(w / h - aspect) < 1e-9:
        return im
    if w / h > aspect:                      # too wide: trim the sides
        new_w, new_h = max(1, round(h * aspect)), h
    else:                                   # too tall: trim top and bottom
        new_w, new_h = w, max(1, round(w / aspect))
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return im.crop((left, top, left + new_w, top + new_h))


def conform(im: Image.Image, profile: EncodingProfile) -> Image.Image:
    """Crop and resample `im` to the profile's exact geometry and mode.

    Refuses to upscale. Inventing detail would fabricate the forensic evidence
    the corpus is being built to measure, and it is never necessary in
    practice: API and open-weight output is 1024px or larger while the reals
    are ~360-700px on the short side. An upscale means the pairing is wrong,
    so it raises instead of quietly interpolating.
    """
    im = crop_to_aspect(im, profile.aspect)
    if im.width < profile.width or im.height < profile.height:
        raise ParityError(
            f"would upscale {im.size} -> {(profile.width, profile.height)}; "
            "generated image is smaller than the real it is paired with"
        )
    if im.size != (profile.width, profile.height):
        im = im.resize((profile.width, profile.height), Image.LANCZOS)
    # Mode last, so the resample runs on the richer representation.
    target = "L" if profile.mode == "L" else "RGB"
    if im.mode != target:
        im = im.convert(target)
    return im


def save_matched(im: Image.Image, dst: str, profile: EncodingProfile) -> None:
    """Write `im` to `dst` as a JPEG encoded exactly the way the real was.

    Writes through `dst + ".part"` and renames, matching `normalize.save_png`:
    the acquisition scripts resume by testing `os.path.exists(dst)`, so a
    truncated file that resume treats as done is worse than no file at all.
    """
    im = conform(im, profile)
    kwargs: dict = {
        "format": "JPEG",
        "qtables": profile.qtables,
        "progressive": profile.progressive,
        "optimize": False,
    }
    if profile.subsampling >= 0 and len(im.getbands()) >= _COLOUR_BANDS:
        kwargs["subsampling"] = profile.subsampling
    if profile.restart_interval:
        kwargs["restart_marker_blocks"] = profile.restart_interval

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp = dst + ".part"
    try:
        im.save(tmp, **kwargs)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_matched_to_real(im: Image.Image, dst: str, real_path: str) -> EncodingProfile:
    """Convenience wrapper: read the paired real's profile and apply it.

    Returns the profile so an acquisition script can record what it used —
    the parity claim is only auditable if the settings are written down.
    """
    profile = read_profile(real_path)
    save_matched(im, dst, profile)
    return profile
