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
    #: The latent downsampling factor the pipeline requires the requested
    #: height and width to be a multiple of. 8 for every KL-VAE UNet here;
    #: Sana's deep-compression autoencoder is 32, and asking it for a size
    #: this does not divide either raises or silently returns a different
    #: one. `run.generate` rounds the request UP to this and crops the output
    #: back to the real's exact box, because the pair is void the moment the
    #: two sides differ in size (`geometry.crop_box`, and the leak it exists
    #: to prevent). Cropping and not resampling, per
    #: `docs/resolution_shortcut.md`.
    size_multiple: int = 8
    #: Every OTHER repo the pipeline pulls weights from. Kandinsky 2.2's
    #: combined pipeline loads a separate prior repo, and `check_licence`
    #: verified only `hf_id` -- so half the weights that made an image were
    #: licence-checked and half were not. This corpus's entire licence
    #: position rests on the registry being true, so it must name every repo
    #: the pipeline touches, not just the one it is addressed by.
    companion_ids: tuple[str, ...] = ()
    #: Extra keyword arguments for the pipeline CALL, as pairs so the spec
    #: stays hashable.
    #:
    #: Sana defaults `use_resolution_binning=True`, which maps the request to
    #: the nearest 1024-based aspect bin, generates there, and resizes the
    #: result back (`pipeline_sana.py`, `resize_and_crop_tensor`). Measured on
    #: this corpus's own geometry: a 432x640 request is generated at 1216x832
    #: -- LANDSCAPE, for a portrait request -- and squashed down, which leaves
    #: var-Laplacian at 2083.8 against 555.9 for the native generation, near
    #: 4x sharper. That is the resample `docs/resolution_shortcut.md` bans,
    #: applied to the generated class only, in a corpus whose worst remaining
    #: confound IS sharpness (0.5998). It would have handed the head a family
    #: separable on one statistic. Off, which makes 32-divisibility a hard
    #: error instead -- see `size_multiple`.
    call_kwargs: tuple[tuple[str, object], ...] = ()
    #: How to run a model whose weights exceed free VRAM. "sequential" moves
    #: one SUBMODULE at a time and fits almost anything at a large speed cost;
    #: "model" moves one COMPONENT at a time and is far faster, but needs the
    #: single largest component to fit on its own. Neither quantises --
    #: `docs/03` §3 refuses that, because a 4-bit model's traces are partly
    #: the compute budget's and this corpus exists to isolate the generator.
    offload_mode: str = "sequential"
    #: The call kwarg that carries `FamilySpec.guidance`. Wuerstchen's
    #: combined pipeline has no `guidance_scale` at all -- it splits into
    #: `prior_guidance_scale` and `decoder_guidance_scale` -- and diffusers
    #: pipelines swallow unknown kwargs, so passing the usual name would not
    #: raise. It would generate the whole family at the pipeline's DEFAULT
    #: guidance while the manifest recorded ours, which is a silent lie in
    #: every row rather than a failure.
    guidance_kw: str = "guidance_scale"
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
        companion_ids=("kandinsky-community/kandinsky-2-2-prior",),
        licence_tag="apache-2.0", commercial=True,
        lineage="movq", arch="unet_prior", vram_gb=10.0, size_multiple=64,
        note="The fourth lineage, and the only VECTOR-QUANTISED decoder here: "
             "MoVQ is a VQModel, not a KL-VAE, so holding it out is a jump "
             "between decoder classes rather than between two continuous "
             "latents. AutoPipelineForText2Image resolves it to "
             "KandinskyV22CombinedPipeline, which pulls the prior from a "
             "SECOND repo -- hence companion_ids; both are apache-2.0. "
             "Measured 9.2 GB resident, ~1.9 s/image at 20 steps, and it "
             "SILENTLY rounds a request up to a multiple of 64: 432x640 and "
             "416x640 both came back 448x640, the same image twice."),
    "sana_1600m": ModelSpec(
        hf_id="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        licence_tag="apache-2.0", commercial=True,
        lineage="dc_ae", arch="sana_dit", vram_gb=9.0, size_multiple=32,
        call_kwargs=(("use_resolution_binning", False),),
        note="The fifth lineage and the most architecturally distant decoder "
             "in the registry: AutoencoderDC, a 32x deep-compression "
             "autoencoder with 32 latent channels, against everything else's "
             "8x KL-VAE. Apache-2.0 including the bundled text encoder. "
             "Measured 8.5 GB resident, ~4 s/image at 20 steps. Read "
             "call_kwargs before changing anything about how it is called."),
    "wuerstchen": ModelSpec(
        hf_id="warp-ai/wuerstchen",
        companion_ids=("warp-ai/wuerstchen-prior",),
        licence_tag="mit", commercial=True,
        lineage="paella_vq", arch="wuerstchen", vram_gb=6.0, size_multiple=128,
        guidance_kw="prior_guidance_scale",
        call_kwargs=(("decoder_guidance_scale", 0.0),
                     ("prior_num_inference_steps", 30)),
        note="Sixth lineage, and the most compressed decoder here: a "
             "PaellaVQModel at 4 latent channels behind a WuerstchenDiffNeXt, "
             "roughly 42x spatial against everyone else's 8x. MIT, and both "
             "repos are (the prior is a separate one, hence companion_ids). "
             "2023-era, so it buys decoder diversity rather than modern-"
             "generator coverage -- but the held-out rung measures the "
             "decoder. NOT in AutoPipelineForText2Image's mapping, so "
             "`run.load` dispatches it explicitly on arch."),
    "cogview4_6b": ModelSpec(
        hf_id="THUDM/CogView4-6B",
        licence_tag="apache-2.0", commercial=True,
        lineage="cogview_vae", arch="cogview_dit", vram_gb=29.0,
        size_multiple=32, offload_mode="model",
        note="Seventh lineage: its own AutoencoderKL at 16 latent channels "
             "and scaling_factor 1.0, which is neither SDXL's 4-channel nor "
             "FLUX's 16-channel/0.3611. Apache-2.0. 29 GiB across the repo -- "
             "13 GiB of transformer plus an 18 GiB GLM-4 text encoder -- so "
             "it needs offload on a 20 GB card; MODEL offload, not "
             "sequential, because the largest single component still fits and "
             "sequential would cost far more than it needs to."),
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
    "shuttle3": ModelSpec(
        hf_id="shuttleai/shuttle-3-diffusion",
        licence_tag="apache-2.0", commercial=True,
        lineage="flux1_vae", arch="flow_dit", vram_gb=23.8,
        note="REFUSED, and it is the near-miss worth recording: apache-2.0 "
             "and UNGATED, which FLUX.1-schnell no longer is, so it is the "
             "only reachable route to the flux1_vae lineage. But its VAE "
             "config is 16ch/0.3611 -- FLUX's -- and its transformer is "
             "FLUX-dev sized, 53.6 GiB of repo, which on a 20 GB card means "
             "offload for a lineage we would be adding for variety. Take it "
             "on a 40 GB card."),
    "kolors": ModelSpec(
        hf_id="Kwai-Kolors/Kolors-diffusers",
        licence_tag="apache-2.0", commercial=True,
        lineage="sdxl_vae", arch="unet", vram_gb=10.0,
        note="REFUSED as redundant, not as unusable. Apache-2.0 and a "
             "genuinely different model, but its VAE is AutoencoderKL 4ch at "
             "scaling_factor 0.13025 -- SDXL's. It would add volume to a "
             "lineage that already has 5,400 pairs and nothing to the "
             "held-out axis."),
    "lumina2": ModelSpec(
        hf_id="Alpha-VLLM/Lumina-Image-2.0",
        licence_tag="apache-2.0", commercial=True,
        lineage="flux1_vae", arch="flow_dit", vram_gb=5.2,
        note="REFUSED as redundant. Apache-2.0, 5.2 GB, brand-new "
             "architecture -- and its vae/config.json is AutoencoderKL 16ch "
             "at scaling_factor 0.3611, which is FLUX's VAE. Held out it "
             "would measure a cousin of flux2_vae. ARCHITECTURE NOVELTY IS "
             "NOT DECODER NOVELTY, and docs/02 §3.4 made that mistake once "
             "already."),
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

#: Additive second run: two NEW decoder lineages over reals the first run
#: never touched (`--shard`). It exists to make the held-out rung a
#: measurement instead of an anecdote -- with three lineages there is exactly
#: one leave-one-lineage-out rotation and its result is a point estimate;
#: with five there are five, and "generalises to an unseen decoder" becomes a
#: distribution.
#:
#: t2i only, and deliberately: image-conditioned methods share composition
#: with the real, so a fake that is 30% redraw carries less of its decoder's
#: fingerprint than one generated from noise. For an arm whose entire job is
#: to represent a decoder, that dilutes the thing being measured.
#:
#: These are TRAINING lineages, not new held-out ones. `HELDOUT_LINEAGE` stays
#: flux2_vae; which lineage a given rung actually holds out is a rung-level
#: choice (`RungConfig.train_exclude_generators`), and adding these is what
#: makes rotating that choice possible at all.
LINEAGE_SUPPLEMENT: dict[str, FamilySpec] = {
    "kandinsky22_t2i": FamilySpec("kandinsky22", "t2i", 0.5, 25, 4.0),
    "sana1600m_t2i":   FamilySpec("sana_1600m",  "t2i", 0.5, 20, 4.5),
}

#: Third run, same additive shape, two more decoder classes. `steps` is the
#: DECODER step count for Wuerstchen; its prior runs separately and is set in
#: that model's `call_kwargs`.
LINEAGE_SUPPLEMENT_2: dict[str, FamilySpec] = {
    "wuerstchen_t2i": FamilySpec("wuerstchen",  "t2i", 0.5, 12, 4.0),
    "cogview4_t2i":   FamilySpec("cogview4_6b", "t2i", 0.5, 30, 5.0),
}

#: Runnable suites by name (`generate_ov7.py --suite`).
SUITES: dict[str, dict[str, FamilySpec]] = {
    "ov7": SUITE,
    "ov7_lineage": LINEAGE_SUPPLEMENT,
    "ov7_lineage2": LINEAGE_SUPPLEMENT_2,
}

#: Which already-generated suites each one is generated INTO. A supplement is
#: additive, so its run has to be validated against every family the manifest
#: will end up holding, not against its own two -- see `validate_suite`.
SUITE_EXTENDS: dict[str, tuple[str, ...]] = {
    "ov7": (),
    "ov7_lineage": ("ov7",),
    "ov7_lineage2": ("ov7", "ov7_lineage"),
}

#: Held out as a whole lineage or not at all (`presets.heldout_groups`).
HELDOUT_LINEAGE = "flux2_vae"


def corpus_of(suite_name: str) -> dict[str, FamilySpec]:
    """Every family the manifest holds once `suite_name` has run."""
    if suite_name not in SUITES:
        raise KeyError(f"unknown suite {suite_name!r}; known: {sorted(SUITES)}")
    out: dict[str, FamilySpec] = {}
    for base in SUITE_EXTENDS[suite_name]:
        out.update(SUITES[base])
    out.update(SUITES[suite_name])
    return out


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
                   *, tol: float = 1e-9,
                   corpus: dict[str, FamilySpec] | None = None) -> None:
    """Raise unless the suite is runnable, licence-clean and held-out-sane.

    Called by `run.py` before a single model loads, because every failure here
    is one a six-hour generation run would otherwise surface at the end.

    `corpus` is every family the MANIFEST will contain once this run lands,
    and the held-out invariants -- that `HELDOUT_LINEAGE` is represented, and
    that two or more lineages are left to train on -- are properties of that
    rather than of one run's slice. A supplement run adding two new lineages
    to an existing corpus is a perfectly valid run and an invalid corpus on
    its own; checked against itself it would be refused for holding out a
    lineage it was never meant to contain. The per-family and share checks
    stay on `suite`, because those ARE properties of the run: the shares have
    to sum to 1 for `resolve_suite` to divide `--total` by them.

    Defaults to `suite`, which is the single-run case and reproduces the
    original behaviour exactly.
    """
    suite = SUITE if suite is None else suite
    corpus = suite if corpus is None else corpus
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
    groups = heldout_groups(corpus)
    if not groups[0]:
        raise ValueError(f"no family has the held-out lineage "
                         f"{HELDOUT_LINEAGE!r}; the split would hold out nothing")
    trained = {lineage_of(n, corpus) for n in corpus} - {HELDOUT_LINEAGE}
    if len(trained) < 2:
        raise ValueError(
            f"only {len(trained)} lineage(s) left in training ({trained}); "
            f"holding out {HELDOUT_LINEAGE!r} would then measure a jump from a "
            f"single decoder, which is not the unseen-generator question")


def resolve_suite(total: int, suite: dict[str, FamilySpec] | None = None,
                  *, corpus: dict[str, FamilySpec] | None = None
                  ) -> dict[str, int]:
    """Split `total` images across the suite by share, largest-remainder, so the
    counts sum to exactly `total` and no family silently rounds to zero."""
    suite = SUITE if suite is None else suite
    validate_suite(suite, corpus=corpus)
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
