"""Which generators we may use, and in what proportion.

The licence on the *weights* is a separate question from the organisers'
allowance to use commercial APIs (`docs/02` §2). An API's terms can permit
commercial use of its outputs; a model whose weights are non-commercial does
not become commercial because you reached it a different way. So every entry
carries the licence tag its model card publishes, `run.check_licence()` asserts
at load time that the card still says that, and `validate_suite()` refuses to
run a family whose model is not `commercial`.

`docs/02` §2's own table has one error, recorded here rather than trusted:
**SDXL-Turbo is `sai-nc-community`, not OpenRAIL++** -- non-commercial, and it
is the fast SDXL variant somebody will reach for first. It is present below,
refused, for exactly that reason. `flux2_klein_9b` is likewise present and
refused: only the 4B of that family is Apache.

Nothing is deleted from this dict when it is ruled out. A deleted entry gets
re-added by the next person who has the same idea.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """One set of weights."""

    hf_id: str
    #: Exactly the string the model card's `license:` publishes. Asserted at
    #: load, so a relicensing upstream fails the run rather than the audit.
    licence_tag: str
    #: False means DO NOT USE. The entry stays for the record; the suite
    #: validator refuses it.
    commercial: bool
    #: The held-out grouping key. `docs/02` §3.4: group by DECODER, not by
    #: name, or holding out a family only measures generalising to a cousin
    #: that shares its VAE.
    lineage: str
    arch: str
    dtype: str = "bfloat16"
    #: Weights resident on the GPU at bf16. Above the card's VRAM, `run.load`
    #: turns on sequential CPU offload rather than quantising: `docs/03` §3
    #: refuses quantisation because a 4-bit model's traces are partly the
    #: compute budget's, and this corpus exists to isolate the generator.
    vram_gb: float = 8.0
    #: Set when this model's decoder is the same one `features/recon.py`
    #: probes with. Not disqualifying -- it is the recon feature working as
    #: designed -- but any A4/A7 recon result must then be reported per
    #: lineage, because "recon separates fakes" does not generalise off the
    #: lineage that shares the probe's VAE.
    recon_probe_collision: bool = False
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "sdxl_base": ModelSpec(
        hf_id="stabilityai/stable-diffusion-xl-base-1.0",
        licence_tag="openrail++", commercial=True,
        lineage="sdxl_vae", arch="unet", vram_gb=7.0,
        note="Volume workhorse. OpenRAIL++ grants output ownership; its "
             "use-based restrictions bind model use, not image redistribution."),
    "sdxl_inpaint": ModelSpec(
        hf_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        licence_tag="openrail++", commercial=True,
        lineage="sdxl_vae", arch="unet", vram_gb=7.0,
        note="9-channel UNet. Used ONLY with an all-zero mask (self_cond); see "
             "METHODS. Asserted 9-channel at load -- a 4-channel checkpoint "
             "here would silently become a plain img2img."),
    "sd15_base": ModelSpec(
        hf_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        licence_tag="creativeml-openrail-m", commercial=True,
        lineage="sd_vae", arch="unet", vram_gb=3.0, recon_probe_collision=True,
        note="Third lineage and cheap volume. Its VAE is the one "
             "features/recon.py loads, verbatim -- so these fakes reconstruct "
             "at near-zero error and the A4/A7 recon rungs separate them for "
             "free. Report recon numbers per lineage or they mean nothing."),
    "flux2_klein_4b": ModelSpec(
        hf_id="black-forest-labs/FLUX.2-klein-4B",
        licence_tag="apache-2.0", commercial=True,
        lineage="flux2_vae", arch="flow_dit", vram_gb=16.0,
        note="The Apache headline and the held-out lineage: a different "
             "decoder AND a different architecture, so holding it out measures "
             "a lineage jump rather than docs/02 §3.4's cousin. 7.75 GB "
             "transformer + 8.05 GB text encoder fits 20 GB at bf16, which is "
             "why it is here and FLUX.1-schnell (12B, 23.8 GB) is not."),
    "kandinsky22": ModelSpec(
        hf_id="kandinsky-community/kandinsky-2-2-decoder",
        licence_tag="apache-2.0", commercial=True,
        lineage="movq", arch="unet_prior", vram_gb=6.0,
        note="AVAILABLE, NOT IN THE SUITE. A genuinely distinct decoder "
             "(MoVQ) and the natural fourth lineage, but it needs a separate "
             "prior pipeline and its own resolution handling, which did not "
             "fit the session that built this. Enable it here first."),
    # --- Refused. Kept so the next person does not re-derive the refusal. ---
    "sdxl_turbo": ModelSpec(
        hf_id="stabilityai/sdxl-turbo",
        licence_tag="sai-nc-community", commercial=False,
        lineage="sdxl_vae", arch="unet", vram_gb=7.0,
        note="REFUSED. 1-4 step SDXL, so it is the obvious way to buy volume, "
             "and docs/02 §2 wrongly lists it as OpenRAIL++. The card says "
             "sai-nc-community: non-commercial."),
    "flux1_schnell": ModelSpec(
        hf_id="black-forest-labs/FLUX.1-schnell",
        licence_tag="apache-2.0", commercial=True,
        lineage="flux1_vae", arch="flow_dit", vram_gb=23.8,
        note="REFUSED ON HARDWARE, not licence. 12B at bf16 does not fit this "
             "20 GB card, and fp8 would put the compute budget into the "
             "traces (docs/03 §3). Restore it on a 40 GB card."),
    "flux2_klein_9b": ModelSpec(
        hf_id="black-forest-labs/FLUX.2-klein-9B",
        licence_tag="other", commercial=False,
        lineage="flux2_vae", arch="flow_dit", vram_gb=34.0,
        note="REFUSED. Only the 4B of this family is Apache; every 9B variant "
             "is under the FLUX non-commercial licence."),
    "flux1_dev": ModelSpec(
        hf_id="black-forest-labs/FLUX.1-dev",
        licence_tag="other", commercial=False,
        lineage="flux1_vae", arch="flow_dit", vram_gb=23.8,
        note="REFUSED. FLUX.1 Non-Commercial. This is the one people reach "
             "for by default."),
}

#: How a fake is conditioned on its real.
#:
#: `inpaint_box` is deliberately absent. It leaked 1.4-21.3% of pixels OUTSIDE
#: the mask at strength 0.99 and ran on an empty prompt (`docs/03` §8) -- a
#: partially-synthetic class whose authentic region is not actually authentic
#: is worse than no partially-synthetic class. `box_mask` survives in
#: `geometry` for whoever fixes it.
METHODS: dict[str, str] = {
    "t2i": "Text-to-image from the real's caption. Shares only content with "
           "its real; every pixel is generated.",
    "img2img": "SDEdit from the real at `strength`. Same composition, redrawn.",
    "ref_image": "The real passed as a REFERENCE image to a Kontext-style "
                 "pipeline, alongside its caption. Not SDEdit: FLUX.2-klein "
                 "takes `image` as conditioning with no `strength`, so the "
                 "output is a fresh generation that has seen the real rather "
                 "than a partially-denoised copy of it.",
    "self_cond": "An ALL-ZERO mask into the 9-channel inpainting UNet: 4 noisy "
                 "latent + 1 mask + 4 masked-image latent, i.e. 'regenerate "
                 "this image'. Every pixel is generated, but conditioned on "
                 "the whole real. Prior art: B-Free (CVPR 2025), DRCT (ICML "
                 "2024), TwinSynths (WACV 2025).",
}


@dataclass(frozen=True)
class FamilySpec:
    """One (model, method) pair. `docs/02` §3.3: `flux_schnell_t2i` and
    `flux_schnell_inpaint` are two families, not one -- the whole held-out
    design depends on family names meaning something. The family name is also
    the directory bucket `build_dataset.py` reads at `rel[1]`."""

    model: str
    method: str
    #: Fraction of the generated side. Validated to sum to 1.
    share: float
    steps: int
    guidance: float
    #: img2img only. High enough that the output is a redraw rather than a
    #: filtered photograph -- `run.check()` rejects near-copies regardless.
    strength: float | None = None
    negative: str = ""


#: The suite this session runs. t2i 0.62 of images; fully synthetic (t2i +
#: self_cond) 0.76, img-conditioned 0.24 -- the ~70/30 `docs/02` §3.1 asks for.
#: Held-out group `flux2_vae` at 0.18, ~2,500 images at the projected volume,
#: well clear of `splits.MIN_HELDOUT_IMAGES = 200`.
SUITE: dict[str, FamilySpec] = {
    "sdxl_t2i":        FamilySpec("sdxl_base",      "t2i",       0.30, 25, 6.0),
    "sd15_t2i":        FamilySpec("sd15_base",      "t2i",       0.20, 30, 7.5),
    "sdxl_self_cond":  FamilySpec("sdxl_inpaint",   "self_cond", 0.14, 25, 6.0),
    "sdxl_img2img":    FamilySpec("sdxl_base",      "img2img",   0.10, 25, 6.0, strength=0.75),
    "sd15_img2img":    FamilySpec("sd15_base",      "img2img",   0.08, 30, 7.5, strength=0.75),
    # klein is step-wise distilled: it warns that `guidance_scale` is ignored,
    # so the value here is recorded for the manifest, not obeyed by the model.
    "klein4b_t2i":       FamilySpec("flux2_klein_4b", "t2i",       0.12, 20, 1.0),
    "klein4b_ref_image": FamilySpec("flux2_klein_4b", "ref_image", 0.06, 20, 1.0),
}

#: Held out as a whole lineage or not at all (`presets.heldout_groups`).
HELDOUT_LINEAGE = "flux2_vae"


def family_of(name: str, suite: dict[str, FamilySpec] | None = None) -> FamilySpec:
    suite = SUITE if suite is None else suite
    try:
        return suite[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; suite has {sorted(suite)}") from None


def lineage_of(name: str, suite: dict[str, FamilySpec] | None = None) -> str:
    return MODELS[family_of(name, suite).model].lineage


def heldout_groups(suite: dict[str, FamilySpec] | None = None) -> list[list[str]]:
    """The `heldout_groups` value for the dataset preset: every family sharing
    `HELDOUT_LINEAGE`, as one inner list."""
    suite = SUITE if suite is None else suite
    return [sorted(n for n in suite if lineage_of(n, suite) == HELDOUT_LINEAGE)]


def validate_suite(suite: dict[str, FamilySpec] | None = None,
                   *, tol: float = 1e-9) -> None:
    """Raise unless the suite is runnable, licence-clean and held-out-sane.

    Called by `run.py` before a single model loads, because every failure here
    is one a six-hour generation run would otherwise surface at the end.
    """
    suite = SUITE if suite is None else suite
    if not suite:
        raise ValueError("empty suite")
    for name, fam in suite.items():
        if fam.model not in MODELS:
            raise KeyError(f"{name}: unknown model {fam.model!r}")
        model = MODELS[fam.model]
        if not model.commercial:
            raise ValueError(
                f"{name}: {model.hf_id} is {model.licence_tag!r}, refused. "
                f"{model.note}")
        if fam.method not in METHODS:
            raise KeyError(f"{name}: unknown method {fam.method!r}; "
                           f"known: {sorted(METHODS)}")
        if (fam.method == "img2img") != (fam.strength is not None):
            raise ValueError(f"{name}: strength is required by img2img and "
                             f"meaningless elsewhere (method={fam.method!r})")
        if fam.share <= 0:
            raise ValueError(f"{name}: share must be positive, got {fam.share}")
    total = sum(f.share for f in suite.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"shares sum to {total!r}, not 1.0")
    groups = heldout_groups(suite)
    if not groups[0]:
        raise ValueError(f"no family has the held-out lineage "
                         f"{HELDOUT_LINEAGE!r}; the split would hold out nothing")
    trained = {lineage_of(n, suite) for n in suite} - {HELDOUT_LINEAGE}
    if len(trained) < 2:
        raise ValueError(
            f"only {len(trained)} lineage(s) left in training ({trained}); "
            f"holding out {HELDOUT_LINEAGE!r} would then measure a jump from a "
            f"single decoder, which is not the unseen-generator question")


def resolve_suite(total: int, suite: dict[str, FamilySpec] | None = None
                  ) -> dict[str, int]:
    """Split `total` images across the suite by share, largest-remainder, so the
    counts sum to exactly `total` and no family silently rounds to zero."""
    suite = SUITE if suite is None else suite
    validate_suite(suite)
    if total < len(suite):
        raise ValueError(f"total {total} cannot cover {len(suite)} families")
    exact = {n: f.share * total for n, f in suite.items()}
    counts = {n: max(1, int(v)) for n, v in exact.items()}
    short = total - sum(counts.values())
    order = sorted(exact, key=lambda n: (exact[n] - int(exact[n])), reverse=True)
    i = 0
    while short > 0:
        counts[order[i % len(order)]] += 1
        short -= 1
        i += 1
    while short < 0:  # only reachable when the max(1, ...) floor bound
        n = max(counts, key=lambda k: counts[k])
        if counts[n] <= 1:
            raise ValueError(f"total {total} too small for suite shares")
        counts[n] -= 1
        short += 1
    return counts
