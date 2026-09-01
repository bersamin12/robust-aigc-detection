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

import warnings
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
    #: True when the checkpoint ships Stable Diffusion's NSFW safety checker,
    #: which `run.load` then refuses to load.
    #:
    #: The checker does not raise and does not skip the row -- it replaces the
    #: decoded image with a BLACK FRAME and returns it as a normal result. So
    #: it lands as `check`'s "near-constant output", i.e. as a generation
    #: failure, and 3 unlucky black frames in `sd15_t2i`'s first 55 crossed
    #: `run_family`'s 5% abort and took the lane down 40 minutes into a 2.3 h
    #: run. Measured over the shards that got further, the real rate is ~0.8%
    #: -- benign Open Images captions, on a checker whose false-positive rate
    #: on photographs of people is notorious.
    #:
    #: Off, and not "retry on a new seed": the reals are already a public
    #: curated corpus, the prompts are Florence-2 captions OF those reals, and
    #: a retry would spend wall clock re-rolling a verdict that was never
    #: about the image. Loading it also costs 1.2 GB of VRAM to do nothing.
    #: `check` still rejects a genuinely degenerate frame, so the guard the
    #: black frames tripped is untouched.
    safety_checker: bool = False
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
        safety_checker=True,
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
    "zimage_turbo": ModelSpec(
        hf_id="Tongyi-MAI/Z-Image-Turbo",
        licence_tag="apache-2.0", commercial=True,
        lineage="flux1_vae", arch="zimage_dit", vram_gb=23.0,
        size_multiple=16, offload_mode="model",
        note="Eighth lineage, and the ONE ENTRY HERE THAT CARRIES A KNOWN "
             "RISK. 6B S3-DiT, apache-2.0, 9 steps at guidance 0.0 (turbo "
             "models take no CFG), ~1 s/image. Its decoder is not new: "
             "vae/config.json is AutoencoderKL 16ch at scaling_factor 0.3611 "
             "and carries `_name_or_path: \"flux-dev\"`, so the VAE is "
             "FLUX.1-dev's by the config's own provenance rather than by "
             "inference. It is `flux1_vae` and it is labelled that way, "
             "because mislabelling it would be the registry lying about the "
             "one thing heldout_groups() reads. THE RISK: flux1_vae enters "
             "TRAINING while flux2_vae is the held-out rung, and how far "
             "apart those two decoders actually are has never been measured "
             "(see LINEAGE_COUSINS). If they are close, the held-out rung "
             "stops measuring an unseen decoder. Accepted deliberately; the "
             "cheap way to retire the risk is the recon probe "
             "(`features/recon.py`) on FLUX.1's VAE against FLUX.2's, which "
             "also rules on shuttle3 and lumina2. size_multiple is 16, not "
             "the default 8: the VAE downsamples 8x and "
             "ZImageTransformer2DModel patches at 2 (`all_patch_size: [2]`), "
             "so 8 would be a size the pipeline silently rounds -- the "
             "Kandinsky failure (§11) with a different divisor."),
    # --- Refused. Kept so the next person does not re-derive the refusal. ---
    "hunyuandit_v12": ModelSpec(
        hf_id="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        licence_tag="other", commercial=True,
        lineage="sdxl_vae", arch="unet", vram_gb=10.0,
        note="REFUSED AS REDUNDANT, and recorded because it was previously "
             "refused for the WRONG REASON. `ai_ov7_generation.md` §11 listed "
             "it under 'on licence' with no ModelSpec and no config check -- "
             "the only refusal in this registry that was a bare assertion. "
             "Tencent's Hunyuan Community licence does permit commercial use, "
             "so that refusal was void. It is refused anyway, on the second "
             "gate: vae/config.json fetched 2026-08-31 reads AutoencoderKL, "
             "latent_channels 4, scaling_factor 0.13025 -- SDXL's VAE, the "
             "identical signature to `kolors`. It would add volume to a "
             "lineage that already has 5,400 pairs and nothing to the "
             "held-out axis. ARCHITECTURE NOVELTY IS NOT DECODER NOVELTY. "
             "The Hunyuan models that DO carry their own decoders are "
             "HunyuanImage 2.1 (32x spatial) and 3.0 (f16, 32-dim), both "
             "genuinely new lineages and both out of reach here -- 17B and "
             "an 80B MoE, neither in diffusers layout (vae/config.json 404s "
             "on both). HunyuanVideo has its own 3D causal VAE and is "
             "refused on the Wan2.2 argument: a still from a temporally-"
             "causal video VAE puts motion blur and codec artefacts into the "
             "generated class only."),
    "cogview4_6b": ModelSpec(
        hf_id="THUDM/CogView4-6B",
        licence_tag="apache-2.0", commercial=True,
        lineage="cogview_vae", arch="cogview_dit", vram_gb=29.0,
        size_multiple=32, offload_mode="model",
        note="REFUSED ON THROUGHPUT, not licence and not decoder. The "
             "decoder is genuinely new and this refusal costs the corpus a "
             "lineage: its own AutoencoderKL at 16 latent channels and "
             "scaling_factor 1.0, which is neither SDXL's 4-channel nor "
             "FLUX's 16-channel/0.3611. Apache-2.0. It is refused because it "
             "is 29 GiB across the repo -- 13 GiB of transformer plus an "
             "18 GiB GLM-4 text encoder -- so it does not fit a 24 GB card "
             "resident and runs under MODEL offload, where it measured "
             "~39 s/image at this corpus's ~416x640. That is 13x the next "
             "slowest family (klein4b_ref_image at 6.62) and 33x the "
             "fastest (sana at 1.17): 2,000 images is 21.7 h against 2.3 h "
             "for the whole rest of the outstanding work. THE NUMBER IS AN "
             "OFFLOAD ARTEFACT, NOT THE MODEL -- it is PCIe traffic, not "
             "compute -- so on a single card that holds 29 GiB resident it "
             "should be several times faster and this refusal should be "
             "revisited. It is not rescued by having four 24 GB cards, "
             "because offload is per process and 4x24 is not 1x96; what "
             "four cards buy is four shards in parallel, ~5.4 h of wall "
             "clock for 2,000 images. Restore on a 40 GB+ card, or accept "
             "the wall clock and run it sharded."),
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
        note="REFUSED AS REDUNDANT (2026-08-31). The hardware gate that used "
             "to carry this refusal is GONE -- 23.8 GB at bf16 fits the "
             "24 GB cards now available -- but lifting it does not admit the "
             "model, it only moves the refusal down to the second gate. Its "
             "VAE is AutoencoderKL 16ch/0.3611, i.e. `flux1_vae`, which "
             "`zimage_t2i` already carries at ~1 s/image against this "
             "model's 12B. Adding it would buy volume in a lineage that has "
             "one family already AND would deepen the LINEAGE_COUSINS bet, "
             "since flux1_vae trains against a held-out flux2_vae. Note also "
             "the repo is now gated=auto on the Hub, which shuttle3 is not."),
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
        note="REFUSED AS REDUNDANT (2026-08-31). Was 'take it on a 40 GB "
             "card'; that is no longer the operative reason. apache-2.0 and "
             "UNGATED (re-probed 2026-08-31: gated=False, against "
             "FLUX.1-schnell's gated=auto), so it IS the reachable route to "
             "flux1_vae -- but flux1_vae stopped being a gap the moment "
             "`zimage_t2i` was added, and its VAE config is the same "
             "16ch/0.3611. 53.6 GiB of repo for a lineage the corpus already "
             "has, plus a deeper LINEAGE_COUSINS bet against the held-out "
             "flux2_vae. Revisit only if zimage_t2i is dropped."),
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
    # Qwen-Image-2.0 (7B, announced 2026-02-10) gets no entry because it has
    # no weights to name: `Qwen/Qwen-Image-2.0`, `-2602`, `-2601` and
    # `Qwen-Image-VAE-2.0` all 401 on the Hub from every author, and the
    # third-party GGUF/Nunchaku re-uploads that trail any Qwen image release
    # within days stop at `2512`. It would have been a genuine new lineage --
    # Qwen published a dedicated report (arXiv 2605.13565) for a purpose-built
    # autoencoder benchmarked AGAINST Wan2.2 rather than descended from it --
    # so this comes back the day the weights land. docs/02 U11.
    "qwen_image_2512": ModelSpec(
        hf_id="Qwen/Qwen-Image-2512",
        licence_tag="apache-2.0", commercial=True,
        lineage="wan_vae", arch="flow_dit", vram_gb=40.9,
        offload_mode="sequential",
        note="REFUSED ON HARDWARE, not licence. STILL REFUSED after the "
             "2026-08-31 hardware change to 4x24 GB: the 40.86 GB "
             "transformer does not fit a 24 GB card, sequential offload "
             "cannot rescue it because the largest single component IS that "
             "transformer, and four cards do not help because offload is per "
             "process -- 4x24 is not 1x96 and this harness gives one GPU per "
             "family. It is now the ONLY refused model that would still add "
             "a lineage (`wan_vae`), so it is the one to revisit first on a "
             "48 GB+ card. The 7B Qwen-Image-2.0 that would fit was "
             "re-probed on 2026-08-31 and is STILL 401 (as are -2602/-2601); "
             "Qwen/Qwen-Image-2512 is ungated and real, and it is the 40.9 "
             "GB one. The newest Qwen text-to-image weights that "
             "actually exist (2025-12-30); the 7B Qwen-Image-2.0 was never "
             "published. Its transformer shard index totals 40.86 GB at bf16, "
             "which is `ai_ov7_generation.md` §11's 'Qwen-Image at 40.9 GB on "
             "hardware' -- measured now rather than quoted, and §11's figure "
             "was exact. offload_mode is sequential and not model because the "
             "largest single component IS the 40.9 GB transformer; there is "
             "no component split that rescues it on a 24 GB card. Lineage "
             "`wan_vae` is not a guess: its vae/config.json is "
             "AutoencoderKLQwenImage, base_dim 96, z_dim 16, with "
             "latents_mean identical to four decimals across all sixteen "
             "channels to Wan-AI/Wan2.1-T2V-1.3B's VAE. Latent statistics "
             "fingerprint the trained encoder, so those are the same frozen "
             "weights -- which is why adding ANY Wan2.1-lineage model would "
             "build a cousin of this one rather than a new lineage."),
    "wan22_ti2v_5b": ModelSpec(
        hf_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        licence_tag="apache-2.0", commercial=True,
        lineage="wan22_vae", arch="wan_dit_3d", vram_gb=22.7,
        offload_mode="model",
        note="REFUSED: IT IS A VIDEO MODEL, and this is an image corpus. "
             "Recorded because everything else about it passes and it is the "
             "obvious next reach once Qwen is ruled out. apache-2.0; its "
             "decoder is genuinely new -- despite sharing the "
             "AutoencoderKLWan class name with Wan2.1 it is a different "
             "autoencoder (base_dim 160 against 96, decoder_base_dim 256, "
             "in_channels 12, is_residual), so unlike Wan2.1 it would NOT "
             "collide with qwen_image_2512; and it fits, at ~10 GB "
             "transformer + 11.4 GB UMT5-XXL + 1.3 GB VAE with model-level "
             "offload. It is refused anyway because a still from it is one "
             "frame out of a temporally-causal VAE trained on H.264 video: "
             "motion blur and codec artefacts would enter the GENERATED class "
             "only, in a corpus whose worst remaining confound is sharpness. "
             "That is the shape of the Sana resolution-binning near-miss "
             "(§11), which was caught by a smoke run rather than by the gate. "
             "Reversible for anyone willing to measure it, but two things "
             "have to be smoked before a single share is assigned: that "
             "num_frames=1 survives the temporal compression, and what "
             "`size_multiple` actually is (Wan2.2's VAE is 16x spatial "
             "against Wan2.1's 8x, so it is neither 8 nor obviously 32)."),
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
    "wuerstchen_t2i": FamilySpec("wuerstchen",  "t2i", 1.0, 12, 4.0),
    # cogview4_t2i was the other half of this suite and was REMOVED on
    # throughput (2026-08-31), not on licence or on decoder novelty -- see the
    # `cogview4_6b` note. Removing it costs the corpus the `cogview_vae`
    # lineage outright, taking the rotation from seven points to six, and
    # nothing currently reachable replaces it. That is a deliberate trade of
    # one rotation point for ~21.7 h of wall clock, and it is reversible: put
    # the family back at share 0.5 and rerun this suite on a fresh shard.
}

#: Fourth run, one family. Z-Image-Turbo is the fastest Apache text-to-image
#: model that fits this hardware, and it is the only arm in the corpus whose
#: lineage claim is NOT backed by a measurement -- see `LINEAGE_COUSINS` and
#: the `zimage_turbo` note. It is here because breadth is what moved the gate
#: (§11: `laplacian_var` 0.5998 -> 0.5632 on lineages alone, at no extra
#: volume) and because at ~1 s/image it is the cheapest breadth available.
#:
#: t2i only, on the supplement's argument: an image-conditioned fake shares
#: composition with its real and so carries less of its decoder's
#: fingerprint, and this arm exists to represent a decoder.
LINEAGE_SUPPLEMENT_3: dict[str, FamilySpec] = {
    # 9 steps / guidance 0.0 are the card's own numbers -- turbo models take
    # no CFG, and a nonzero value here would be recorded in the manifest
    # while the model ignored it.
    "zimage_t2i": FamilySpec("zimage_turbo", "t2i", 1.0, 9, 0.0),
}

#: The scale-up: every family at once, over a fresh pool of reals the first
#: four runs never dealt. This is the suite that finishes the corpus.
#:
#: Shares are NOT a style choice -- they are solved backwards from a balanced
#: target, AND THE SOLVE IS TIED TO ONE TOTAL. At the 42,646-real pool this
#: was run on the corpus ends at 54,624 pairs over seven lineages, so a
#: balanced corpus is 7,803 each, and each family's share here is whatever
#: brings its lineage from where it already stands UP to that line. That is
#: why `sdxl_t2i` -- the largest family in the frozen corpus at 3,000 -- has
#: the smallest share here (0.0313): `sdxl_vae` already has 5,400 pairs and
#: needs only 2,403 more, while `paella_vq` and `flux1_vae` start at zero and
#: are owed the whole 7,803.
#:
#: Because the solve is tied to that total, THESE SHARES ARE NOT PORTABLE TO A
#: DIFFERENT ONE. Run them at 32,774 and the spread across lineages is 1,247
#: pairs; at 42,646 it is 8. A later run of a different size must redo the
#: arithmetic, not reuse these numbers;
#: `docs/ai_ov7_generation.md` §17 has the table.
#:
#: Within a lineage, families keep the base suite's ratios (sdxl_t2i 30 :
#: self_cond 14 : img2img 10, and sd15_t2i 20 : img2img 8), so the
#: fully-synthetic/image-conditioned balance of each lineage is preserved
#: rather than re-argued.
#:
#: RUN IT ON EVERY SHARD OF A FREE POOL -- the production run sharded its
#: 42,646 previously-undealt reals four ways and ran shards 0-3, which is
#: legal and is the point: blocks are disjoint by construction, so the boxes
#: need no lock and no shared filesystem, and each shard is a balanced
#: microcosm of the same deal rather than a lineage-shaped slice. What rule 1
#: forbids is a DIFFERENT suite on reals some suite has already been dealt --
#: which is a property of the POOL, not of a shard index. A pool that overlaps
#: an earlier run is backfilled by raising --total on the suite it already
#: owns, never by pointing this suite at it.
LINEAGE_FULL: dict[str, FamilySpec] = {
    "wuerstchen_t2i":    FamilySpec("wuerstchen",     "t2i",       0.1831, 12, 4.0),
    "zimage_t2i":        FamilySpec("zimage_turbo",   "t2i",       0.1830,  9, 0.0),
    "kandinsky22_t2i":   FamilySpec("kandinsky22",    "t2i",       0.1595, 25, 4.0),
    "sana1600m_t2i":     FamilySpec("sana_1600m",     "t2i",       0.1595, 20, 4.5),
    "klein4b_t2i":       FamilySpec("flux2_klein_4b", "t2i",       0.0938, 20, 1.0),
    "sd15_t2i":          FamilySpec("sd15_base",      "t2i",       0.0842, 30, 7.5),
    "klein4b_ref_image": FamilySpec("flux2_klein_4b", "ref_image", 0.0469, 20, 1.0),
    "sd15_img2img":      FamilySpec("sd15_base",      "img2img",   0.0337, 30, 7.5, strength=0.75),
    "sdxl_t2i":          FamilySpec("sdxl_base",      "t2i",       0.0313, 25, 6.0),
    "sdxl_self_cond":    FamilySpec("sdxl_inpaint",   "self_cond", 0.0146, 25, 6.0),
    "sdxl_img2img":      FamilySpec("sdxl_base",      "img2img",   0.0104, 25, 6.0, strength=0.75),
}

#: Runnable suites by name (`generate_ov7.py --suite`).
SUITES: dict[str, dict[str, FamilySpec]] = {
    "ov7": SUITE,
    "ov7_lineage": LINEAGE_SUPPLEMENT,
    "ov7_lineage2": LINEAGE_SUPPLEMENT_2,
    "ov7_lineage3": LINEAGE_SUPPLEMENT_3,
    "ov7_full": LINEAGE_FULL,
}

#: Which already-generated suites each one is generated INTO. A supplement is
#: additive, so its run has to be validated against every family the manifest
#: will end up holding, not against its own two -- see `validate_suite`.
SUITE_EXTENDS: dict[str, tuple[str, ...]] = {
    "ov7": (),
    "ov7_lineage": ("ov7",),
    "ov7_lineage2": ("ov7", "ov7_lineage"),
    "ov7_lineage3": ("ov7", "ov7_lineage", "ov7_lineage2"),
    # ov7_full contains every family itself, so it extends nothing:
    # corpus_of("ov7_full") is already the whole corpus.
    "ov7_full": (),
}

#: Held out as a whole lineage or not at all (`presets.heldout_groups`).
HELDOUT_LINEAGE = "flux2_vae"

#: Lineages that MIGHT be the same decoder, and have never been measured.
#:
#: A lineage key is a claim that two decoders are different. Most of this
#: registry's keys are backed by evidence -- `movq` is a VQModel against
#: everyone else's KL-VAE, `dc_ae` is 32x/32ch, `wan_vae` and `wan22_vae` have
#: different `base_dim` and different latent statistics. This pair is not.
#: Nobody has run FLUX.1's VAE and FLUX.2's against each other.
#:
#: CORRECTED 2026-08-31: this used to read "both AutoencoderKL at 16 latent
#: channels", which is false and was inherited from a wrong cell in
#: `ai_ov7_generation.md` §10. FLUX.1's is `AutoencoderKL` 16ch at
#: scaling_factor 0.3611; FLUX.2-klein-4B's is `AutoencoderKLFlux2`, 32
#: channels, 2x2-patchified to 128 effective, BatchNorm, and no
#: scaling_factor. They are architecturally distinct, verified from both
#: configs. That LOWERS this risk and does not retire it: different
#: architectures are not proof of different artefacts, and what the held-out
#: rung needs is a fingerprint difference, not a config one. The entry stays
#: until `features/recon.py` measures the two.
#:
#: It matters because `zimage_turbo` puts `flux1_vae` into TRAINING while
#: `flux2_vae` is the held-out rung. If the two are close, that rung stops
#: measuring an unseen decoder and starts measuring `docs/02` §3.4's cousin --
#: the exact mistake this design exists to avoid, arrived at from the other
#: direction. `validate_suite` warns about it rather than refusing, because it
#: is a deliberate choice and not an accident; the way to retire it is
#: `features/recon.py` on the two VAEs, which also rules on `shuttle3` and
#: `lumina2`.
LINEAGE_COUSINS: dict[str, tuple[str, ...]] = {
    "flux2_vae": ("flux1_vae",),
    "flux1_vae": ("flux2_vae",),
}


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
    cousins = trained & set(LINEAGE_COUSINS.get(HELDOUT_LINEAGE, ()))
    if cousins:
        # A warning and not an error: this is a deliberate choice (zimage_t2i
        # buys the cheapest lineage breadth available) and the registry's job
        # is to make sure nobody arrives at it by accident. It stops being a
        # warning the moment somebody runs the recon probe on the two VAEs.
        warnings.warn(
            f"{sorted(cousins)} is in TRAINING and may be the same decoder as "
            f"the held-out {HELDOUT_LINEAGE!r} -- the two have never been "
            f"measured against each other (registry.LINEAGE_COUSINS). If they "
            f"are close, the held-out rung measures a cousin rather than an "
            f"unseen decoder. Retire this with features/recon.py on the two "
            f"VAEs; report any held-out result with this caveat until then.",
            stacklevel=2)
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
