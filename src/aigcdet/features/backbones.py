"""Frozen backbones and pooled embedding (spec §3.2).

Pooling is global average over final-layer patch tokens. The SigLIP2 authors
found this beat CLS-token, attention pooling, and multi-layer concatenation
head-to-head, so it is the default rather than an option (see module docstring
of the task brief for the citation this design follows).

Preprocessing is a "squish" resize to a fixed square, ignoring aspect ratio.
Random resized cropping is deliberately avoided: it can remove or distort the
localised forensic cues (compression grids, resampling artefacts, generator
fingerprints) that this project's detector depends on.

The three registered backbones do NOT share one forward() input contract --
see `INPUT_FORMATS` -- so `BackboneSpec` carries an `input_format` strategy and
`embed` builds each tower's kwargs from it. `embed`'s own signature is
unchanged: callers still hand it a list of HWC uint8 images.

Licence provenance for every weight loaded here is recorded in
`docs/model_licences.md` (spec §4.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

#: Pretraining normalisation, carried PER BACKBONE on `BackboneSpec` rather
#: than applied globally.
#:
#: This was one shared ImageNet constant until 2026-08-31, under a comment
#: calling it "an unverified simplification ... Revisit with a real comparison
#: if a backbone's features underperform expectation". The backbone probe is
#: that revisit: its entire purpose is RANKING towers, and EVA-02 was
#: pretrained on CLIP's statistics while SigLIP was pretrained on [0.5]*3, so
#: handing either ImageNet's numbers handicaps it and the ranking would be
#: measuring the preprocessing.
#:
#: ImageNet stays the DEFAULT, so every entry that predates this change is
#: unchanged bit-for-bit and no bank on disk is orphaned. That leaves a
#: deliberate asymmetry worth stating rather than leaving as a puzzle: `clipl`
#: and `siglip2l` were ALSO pretrained on their own statistics and keep
#: ImageNet's anyway, because their banks were built that way and re-extracting
#: them is a separate and much larger job. So a `siglip2l` number and a
#: `siglipso400m` number differ in preprocessing as well as in tower, and that
#: particular pair cannot be read as an architecture comparison.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: OpenAI CLIP's statistics, which EVA-02's `pretrained_cfg` also reports --
#: EVA-02 distils CLIP features, so it inherits CLIP's input distribution.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
#: SigLIP normalises to [-1, 1], not to zero mean and unit variance.
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


#: The two input contracts the registry's vision towers actually accept. They
#: are NOT interchangeable. `model(pixel_values=(B, 3, H, W))` is correct for
#: DINOv3, CLIP and the FIXED-RESOLUTION SigLIP2 checkpoints
#: (`siglip2-*-patch16-{224,256,384,512}`, which report `model_type: siglip`
#: and reuse SigLIP v1's `SiglipVisionModel`). It is impossible for the NaFlex
#: checkpoints (`siglip2-*-patch16-naflex`, `model_type: siglip2`), whose
#: transformer signature is `forward(pixel_values, attention_mask,
#: spatial_shapes)` over an ALREADY patchified
#: `(B, num_patches, C * patch * patch)` tensor -- their `patch_embedding` is
#: an `nn.Linear`, not a `nn.Conv2d`.
#:
#: The registry pins the fixed-resolution variant deliberately. Every image
#: reaches a backbone already canonicalised to one size (see
#: `augment.canonical` -- resolution leaks the label, spec 4.4a), so NaFlex's
#: native-resolution handling would buy nothing and cost a second contract.
#: INPUT_SIGLIP2_PATCHES stays implemented and tested against the transformers
#: reference so that switching to NaFlex is a one-line registry change, but no
#: entry uses it today.
INPUT_IMAGE_TENSOR = "image_tensor"       # (B, 3, H, W), pixel_values only
INPUT_SIGLIP2_PATCHES = "siglip2_patches"  # patchified + attention_mask + spatial_shapes
INPUT_FORMATS = (INPUT_IMAGE_TENSOR, INPUT_SIGLIP2_PATCHES)


#: How a tower's output is reduced to one vector per image. Separate from
#: `input_format` because they vary independently: a CNN shares the ViTs'
#: `pixel_values` input contract but emits a (B, C, H, W) feature MAP, not a
#: (B, T, D) token sequence, so there are no prefix tokens to strip and no
#: token axis to average.
#:
#: POOL_SPATIAL_MS concatenates the per-channel MEAN and STANDARD DEVIATION
#: over spatial positions, for each stage in `BackboneSpec.stages`. The std is
#: the point, not a free extra: the ViT path's `mean` over tokens averages
#: local texture away, and texture -- noise floor, resampling lattice,
#: compression grid -- is the forensic evidence a conv stack is here to read.
#: A mean-only conv bank would be a semantic feature extractor competing with
#: SigLIP2 on SigLIP2's own ground.
POOL_TOKENS = "tokens"           # (B, T, D) -> mean over T, after num_prefix_tokens
POOL_SPATIAL_MS = "spatial_ms"   # (B, C, H, W) per stage -> [mean, std] over H*W
POOLS = (POOL_TOKENS, POOL_SPATIAL_MS)


#: Which package publishes the weights. `AutoModel.from_pretrained` resolves a
#: `timm/*` repo to `TimmWrapperModel`, whose `last_hidden_state` IS
#: `timm_model.forward_features(x)` -- a (B, 1 + N, D) token sequence -- so
#: `embed`, `_pool` and `model_inputs` need no timm-specific branch at all and
#: none exists. The field earns its place for two things `load_backbone` must
#: do that the transformers path does not: turn off timm's classifier head,
#: and fail with an install line rather than a bare ImportError when the
#: optional package is absent.
LOADER_TRANSFORMERS = "transformers"
LOADER_TIMM = "timm"
LOADERS = (LOADER_TRANSFORMERS, LOADER_TIMM)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    image_size: int
    dim: int
    num_prefix_tokens: int   # CLS + register tokens to strip before pooling
    params: int               # real vision-tower parameter count, for the 2B budget check
    #: How `embed` must build this tower's forward() kwargs -- one of
    #: `INPUT_FORMATS`. The registry holds backbones with genuinely different
    #: input contracts, so the strategy is per-spec data rather than an
    #: assumption baked into `embed`.
    input_format: str = INPUT_IMAGE_TENSOR
    #: Patch side length, required by (and only used by) INPUT_SIGLIP2_PATCHES,
    #: which has to patchify the image itself before the model sees it.
    patch_size: int = 0
    #: Whether the HuggingFace repo requires accepting a licence on the
    #: downloading account before the weights can be fetched. A property of the
    #: checkpoint, so it lives here rather than in a notebook's prose: it
    #: decides whether a run needs an HF token at all, and the fleet running an
    #: ungated backbone should not be stopped for one. See
    #: `docs/model_licences.md` for each entry's terms.
    gated: bool = False
    #: The dtype the tower is loaded and run in. float16 halves memory and
    #: roughly triples throughput over float32 on every GPU this project uses,
    #: and SigLIP2 and CLIP are finite in it. DINOv3-L is NOT: its activations
    #: overflow float16 by hidden layer 1, and every pooled vector comes out NaN
    #: -- silently, at full speed, for an entire 5-hour bank (2026-08-29).
    #: bfloat16 keeps float32's exponent range, is finite on DINOv3 at float16's
    #: throughput, and its pooled output (|x| < 2) stores in the bank's float16
    #: without loss. Per backbone, because it is a property of the checkpoint;
    #: see `run_dtype` for the one device-dependent fallback.
    dtype: torch.dtype = torch.float16
    #: How `embed` reduces this tower's output -- one of `POOLS`. Defaults to
    #: the ViT contract so every existing entry is unchanged.
    pool: str = POOL_TOKENS
    #: For POOL_SPATIAL_MS only: which entries of `output_hidden_states` to
    #: pool, in the order they are concatenated. Indices into the tuple HF
    #: returns, where 0 is the stem embedding and the last is the final stage.
    #: `dim` must equal 2 * sum(channels of these stages) -- asserted against
    #: the real model in tests/features/test_backbones.py, since the channel
    #: widths are a property of the checkpoint and cannot be checked here.
    stages: tuple[int, ...] = ()
    #: Pretraining normalisation. Defaults to ImageNet so every entry that
    #: predates 2026-08-31 is unchanged bit-for-bit; see IMAGENET_MEAN above.
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    #: Which package publishes the weights -- one of `LOADERS`.
    loader: str = LOADER_TRANSFORMERS

    def __post_init__(self):
        if self.input_format not in INPUT_FORMATS:
            raise ValueError(
                f"{self.name}: input_format must be one of {INPUT_FORMATS}, "
                f"got {self.input_format!r}")
        if self.pool not in POOLS:
            raise ValueError(
                f"{self.name}: pool must be one of {POOLS}, got {self.pool!r}")
        if self.loader not in LOADERS:
            raise ValueError(
                f"{self.name}: loader must be one of {LOADERS}, "
                f"got {self.loader!r}")
        for field_name, value in (("mean", self.mean), ("std", self.std)):
            if len(value) != 3:
                raise ValueError(
                    f"{self.name}: {field_name} must be 3 per-channel values, "
                    f"got {value!r}")
        if any(v == 0 for v in self.std):
            raise ValueError(f"{self.name}: std has a zero channel: {self.std!r}")
        if (self.pool == POOL_SPATIAL_MS) != bool(self.stages):
            raise ValueError(
                f"{self.name}: `stages` is required by {POOL_SPATIAL_MS} and "
                f"meaningless without it; got pool={self.pool!r} "
                f"stages={self.stages!r}")
        if self.pool == POOL_SPATIAL_MS and self.num_prefix_tokens:
            raise ValueError(
                f"{self.name}: a feature map has no prefix tokens to strip, "
                f"got num_prefix_tokens={self.num_prefix_tokens}")
        if self.input_format == INPUT_SIGLIP2_PATCHES:
            if self.patch_size <= 0:
                raise ValueError(
                    f"{self.name}: {INPUT_SIGLIP2_PATCHES} needs a positive patch_size")
            if self.image_size % self.patch_size != 0:
                raise ValueError(
                    f"{self.name}: image_size {self.image_size} is not divisible by "
                    f"patch_size {self.patch_size}, so the image cannot be patchified")


# Parameter counts below are the vision-tower-only counts actually loaded and
# run by `load_backbone`/`embed` (not the full dual text+image checkpoint size
# reported by some model cards). They were measured, not guessed:
#   - dinov3l: HF Hub safetensors metadata for the (vision-only) checkpoint
#     reports total F32 parameters = 303_129_600.
#   - siglip2l / clipl: instantiated `AutoModel.from_config(...)` (architecture
#     only, no weights downloaded) from each model's published config, then
#     summed `.vision_model.parameters()`. siglip2l -> 316_283_904,
#     clipl -> 303_179_776.
BACKBONES: dict[str, BackboneSpec] = {
    # Gated: Meta's custom DINOv3 licence must be accepted per ACCOUNT.
    "dinov3l": BackboneSpec("dinov3l", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                             384, 1024, 5, 303_129_600, gated=True,
                             dtype=torch.bfloat16),
    # DINOv2-L is here for a LICENCE reason, not an architectural one. DINOv3-L
    # is the strongest backbone measured (a0 0.8611 against ConvNeXt's 0.4244
    # and SigLIP2's 0.1870) but ships under Meta's custom DINOv3 licence, gated
    # per ACCOUNT -- so a fleet of five cannot run it without five acceptances,
    # and a submission cannot rely on it without that term. DINOv2-L is the same
    # self-supervised lineage under Apache-2.0 and ungated. Whether it retains
    # DINOv3's advantage is rung-level evidence nobody has yet; that is the
    # question the entry exists to let us ask.
    #
    # 518, not 384. `canonicalise` delivers a 512-px nominal side, so 518 is the
    # only registered image_size that does not throw pixels away before the
    # tower sees them -- it upsamples 512 -> 518 rather than downsampling. That
    # matters more here than elsewhere: the forensic cues this project detects
    # live in high-frequency detail, which is exactly what a downsample removes.
    # It is not free. Measured on the A4500, GPU forward only, 24 real images:
    #   336px (576 tok) 140.4 img/s | 392 (784)  99.3 | 448 (1024) 74.3
    #   518px (1369 tok) 54.0 img/s
    # so 518 costs 2.6x 336's GPU time. The 131,116 x 11 bank projects to ~7.4 h
    # of GPU against dinov3l's ~2.9 h at matched tokens; the real dinov3l run
    # took 5 h 09, i.e. ~1.8x its GPU-only figure, because the pipeline is half
    # CPU-bound on decode/augment/proxies. Budget ~8-9 h wall, and drop to 448
    # if that does not fit rather than to 336.
    #
    # float16, and this was MEASURED rather than inherited from dinov3l's entry
    # (2026-08-30, 24 canonicalised images, pooled vectors vs a float32 run):
    # every value finite, max|diff| 2.3e-02 -- and CLOSER to float32 than
    # bfloat16's 1.0e-01. Pooled |x| peaks at 12.29, well inside float16 both in
    # the tower and in the bank's float16 storage. DINOv2 does not have DINOv3's
    # layer-1 overflow, so it must not pay for the bfloat16 workaround.
    "dinov2l": BackboneSpec("dinov2l", "facebook/dinov2-large",
                             518, 1024, 1, 304_368_640),
    "siglip2l": BackboneSpec("siglip2l", "google/siglip2-large-patch16-384",
                              384, 1024, 0, 316_283_904),
    "clipl": BackboneSpec("clipl", "openai/clip-vit-large-patch14",
                           224, 1024, 1, 303_179_776),
    # --- Convolutional towers (spec 6.4: A5 wants DECORRELATED paradigms, not
    # three strong ones). Both share the ViTs' `pixel_values` contract and the
    # ImageNet normalisation default -- which, unlike for CLIP and SigLIP2, is
    # these checkpoints' OWN pretraining normalisation, so the asymmetry
    # described at IMAGENET_MEAN does not apply to them.
    #
    # Nothing upstream changes: they consume the same canonicalised, augmented
    # pixels as every other bank, so views stay bit-identical and
    # `assert_fusion_parents` still holds. Only the pooling differs.
    #
    # Parameter counts measured by instantiating from the published config
    # (architecture only, no weights), same discipline as the ViTs above.
    #
    # `dim` is 2 * sum(stage channels): mean and std per channel per stage.
    #   convnextt  stages (3, 4) -> (384 + 768) * 2 = 2304
    #   resnet50   stage  (4,)   -> 2048 * 2        = 4096
    # Both are float16-safe: ConvNeXt is LayerNorm throughout and ResNet's
    # BatchNorm runs on frozen eval statistics. Neither has DINOv3's overflow.
    #
    # THESE TWO are also ~50x less compute per image than SigLIP2-L, so a bank
    # costs one Kaggle session rather than a five-account fleet. That is a
    # property of THESE checkpoints and not of the conv paradigm: `convnextv2h`
    # below is a conv tower at ~338 GFLOPs, more than DINOv2-L at 518 px. It is
    # a probe candidate, not a cheap bank.
    "convnextt": BackboneSpec("convnextt", "facebook/convnext-tiny-224",
                               224, 2304, 0, 27_820_128,
                               pool=POOL_SPATIAL_MS, stages=(3, 4)),
    # Stage 4 only, at 4096 dims, because stages (3, 4) would be 6144 -- a
    # 16.59 GiB bank against a 20 GiB Kaggle working quota, with the repo and
    # pip's cache to fit alongside it. At 4096 it is 11.08 GiB, against
    # convnextt's 6.27 GiB (measured with kaggle_bootstrap.bank_bytes over the
    # real 131,116-row split x 11 views). ResNet's 2048-channel final stage is
    # the old-style width the modern nets dropped, and it is what makes the
    # multiplier bite. Widen to (3, 4) here only if the convnextt result earns
    # the storage -- it would need a second session's worth of shards.
    "resnet50": BackboneSpec("resnet50", "microsoft/resnet-50",
                              224, 4096, 0, 23_508_032,
                              pool=POOL_SPATIAL_MS, stages=(4,)),

    # --- Backbone-probe candidates, added 2026-08-31 ----------------------
    # Four towers the ladder has never been run on, registered to be RANKED on
    # the 20,000-row union probe rather than to ship. Licence and gating for
    # each were read via the HF Hub API on 2026-08-31 and are recorded in
    # docs/model_licences.md; all four are ungated, so no arm needs a token.
    #
    # The 2B parameter cap does NOT bind this list. It is a constraint on the
    # architecture that ships -- the spec's own wording is "Final model uses at
    # most two backbones" -- not on the menu of candidates an ablation may
    # consider. See test_the_heaviest_shippable_configuration_stays_under_2b.
    #
    # Parameter counts measured the same way as every entry above: instantiated
    # from the published config (architecture only, no weights) and summed over
    # the vision tower actually loaded and run.

    # DINOv2 WITH REGISTERS, against plain `dinov2l` above. This is a directed
    # hypothesis about THIS task, not a version bump. Registers exist to absorb
    # the high-norm artefact tokens DINOv2 develops in low-information patches
    # -- flat sky, blur, smooth gradients -- and those are exactly the patches a
    # generator's decoder leaves its trace in. If the artefact tokens were
    # polluting the patch mean, the pooled vector this project reads should get
    # cleaner; if they were carrying the forensic signal, it should get worse.
    # Either result is informative, which is why the entry is worth a session.
    #
    # num_prefix_tokens is 5, NOT dinov2l's 1: `num_register_tokens: 4` in the
    # published config, plus the CLS token. Copying the 1 across would average
    # four register tokens into every pooled vector, silently.
    #
    # float16 MEASURED, not inherited (2026-08-31, 24 canonicalised images vs a
    # float32 run of the same path): every value finite, max|diff| 5.29e-03,
    # pooled |x| peaks at 7.31 -- the tightest of the four candidates, and far
    # inside float16 both in the tower and in the bank's float16 storage.
    # docs/backbone_dtype_probe.json.
    "dinov2regl": BackboneSpec("dinov2regl",
                                "facebook/dinov2-with-registers-large",
                                518, 1024, 5, 304_372_736),

    # The SAME tower at 224 instead of 518, for the two-tower experiment. It is
    # a separate registry entry rather than an `--input-size` flag on purpose:
    # `image_size` is recorded in every bank's config and is what makes a bank
    # and a checkpoint comparable, so a run that could silently change it would
    # let two banks claim the same backbone while holding embeddings of
    # different-resolution images.
    #
    # 224/14 = 16, so the tower sees 256 patches + 1 CLS + 4 registers against
    # 518's 1369 + 5 -- a 5.3x cut in tokens, which is the whole reason two of
    # these fit on one card where one at 518 barely does. DINOv2 interpolates
    # its position embeddings, so 224 needs no checkpoint surgery.
    #
    # What it costs is NOT free and must be said: `canonicalise` delivers a
    # 512-px nominal side and the true information content is capped at the
    # 200-px crop either way, so at 518 the tower UPSAMPLES 512 -> 518 and at
    # 224 it DOWNSAMPLES 512 -> 224. This entry therefore throws away real
    # resolution that `dinov2regl` keeps, and a comparison between the two is a
    # resolution ablation, not a free speedup.
    "dinov2regl224": BackboneSpec("dinov2regl224",
                                  "facebook/dinov2-with-registers-large",
                                  224, 1024, 5, 304_372_736),

    # EVA-02-L/14 at 448. A third pretraining PARADIGM, which is why it is here
    # rather than as a fourth ViT: masked image modelling distilled from a CLIP
    # teacher, then supervised fine-tuning on IN-22k/IN-1k -- neither DINOv2's
    # self-distillation nor SigLIP's contrastive image-text objective.
    #
    # 448 is not a choice. `pretrained_cfg.fixed_input_size` is true, so the
    # position embedding admits exactly 448 and `canonicalise`'s 512 nominal
    # side is DOWNSAMPLED on the way in -- the one arm of the probe whose input
    # is degraded, and a stated handicap rather than noise. It is also the
    # cheapest ViT arm for the same reason (1024 tokens against dinov2l's 1369
    # at 518; measured 74.3 vs 54.0 img/s on the A4500), so the handicap and the
    # discount are one fact.
    #
    # CLIP's normalisation, not ImageNet's -- EVA-02 distils CLIP features and
    # its pretrained_cfg reports CLIP's mean/std. See IMAGENET_MEAN.
    #
    # 304,055,232 is the tower with `reset_classifier(0)` applied, which
    # `load_backbone` does: TimmWrapperModel materialises the 1000-way IN-1k
    # head that this project never reads.
    #
    # float16 MEASURED (2026-08-31, same 24 images): finite, max|diff| 3.36e-01
    # against float32, pooled |x| peaks at 102.00. Note the RELATIVE error is
    # ~0.33%, five times dinov2regl's 0.07% -- EVA-02 pools larger activations,
    # so it has less float16 headroom than the others. Still two orders inside
    # the 65504 the bank stores at, but it is the entry to re-measure first if
    # a future checkpoint or resolution moves. docs/backbone_dtype_probe.json.
    "eva02l": BackboneSpec("eva02l",
                            "timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                            448, 1024, 1, 304_055_232,
                            mean=CLIP_MEAN, std=CLIP_STD, loader=LOADER_TIMM),

    # ConvNeXt V2 Huge at 384. The conv paradigm at a scale that can actually
    # compete, against `convnextt`'s 27.8M -- convnextt lost the A5 comparison
    # (a0 0.4244) and "the conv paradigm is weak here" and "we ran a tiny conv
    # tower" are not the same claim. This entry separates them.
    #
    # stages=(4,) and NOT (3, 4). (3, 4) would give 2 * (1408 + 2816) = 8448,
    # an 8.25x wider bank than the 1024-d ViT arms, and `train_head` takes
    # `dim_feat=bank.config["dim"]` -- so a wider bank silently buys a bigger
    # head and a win here would have two explanations. (4,) is 2 * 2816 = 5632,
    # which REDUCES that confound and does not remove it: 5.5x is still 5.5x,
    # so a convnextv2h win requires the matched-capacity control (re-run the
    # best ViT rung with `hidden` raised to match this head's parameter count)
    # before it is a finding.
    #
    # float16 was the entry this probe existed to doubt, and it PASSED. GRN
    # takes a global L2 norm over all spatial positions, and at 384^2 with 2816
    # channels that is a far larger reduction than convnextt's, so convnextt's
    # "LayerNorm throughout" argument does not transfer and could not be
    # borrowed. Measured instead (2026-08-31, 24 canonicalised images):
    # every value finite, max|diff| 2.26e-01 against float32, pooled |x| peaks
    # at 80.15. docs/backbone_dtype_probe.json.
    "convnextv2h": BackboneSpec("convnextv2h", "facebook/convnextv2-huge-22k-384",
                                 384, 5632, 0, 657_472_640,
                                 pool=POOL_SPATIAL_MS, stages=(4,)),

    # SigLIP SO400M/14 at 384, against `siglip2l` above. A shape-optimised
    # tower -- 27 layers at width 1152, sized by architecture search rather than
    # by the ViT-L/H ladder -- and the strongest open contrastive image encoder
    # available under Apache-2.0.
    #
    # [0.5]*3 normalisation, which is SigLIP's own: it maps to [-1, 1] rather
    # than to zero mean and unit variance. NOTE THE ASYMMETRY described at
    # IMAGENET_MEAN: `siglip2l` keeps ImageNet's statistics because its banks
    # were built that way, so a siglip2l-vs-siglipso400m gap confounds the tower
    # with the preprocessing and that ONE pair is not an architecture
    # comparison.
    #
    # float16 MEASURED (2026-08-31, same 24 images): finite, max|diff| 9.70e-02
    # against float32, pooled |x| peaks at 36.66.
    # docs/backbone_dtype_probe.json.
    "siglipso400m": BackboneSpec("siglipso400m",
                                  "google/siglip-so400m-patch14-384",
                                  384, 1152, 0, 428_225_600,
                                  mean=SIGLIP_MEAN, std=SIGLIP_STD),

    # SigLIP2-L/16 at 512. The SAME TOWER as `siglip2l`, at the resolution
    # `canonicalise` actually emits.
    #
    # This is a resolution entry, not a capacity one, and the parameter counts
    # say so: 316,742,656 against the 384 variant's 316,283,904. The whole
    # difference is 458,752 values of position embedding. Raising the input
    # side costs COMPUTE -- (512/16)^2 = 1024 tokens against 576, so ~1.78x --
    # and buys no parameters at all, which is why "use the bigger SigLIP2" is
    # not the same proposal as "use a bigger backbone".
    #
    # It is here because `canonicalise` emits a 512 nominal side and `siglip2l`
    # therefore throws away 25% of it before the tower ever sees it -- the same
    # handicap `eva02l` carries at 448, except that here it is avoidable. If a
    # SigLIP2 arm is going to be ranked against DINOv2 at 518, it should be
    # ranked at a resolution that is not a self-inflicted wound.
    #
    # NORMALISATION IS ITS OWN, AND THAT COSTS A COMPARISON. This entry takes
    # SigLIP's [0.5]*3 because it is correct and because no bank on disk
    # constrains it. `siglip2l` keeps ImageNet's statistics only because its
    # banks were built that way (see IMAGENET_MEAN). So `siglip2l` ->
    # `siglip2l512` varies BOTH resolution and normalisation and is not a clean
    # resolution A/B; the comparison this entry can carry is against
    # `siglipso400m`, which shares its statistics.
    #
    # float16 MEASURED, not inherited (2026-08-31, 24 canonicalised images vs a
    # float32 run of the same path): every value finite, max|diff| 2.709e-02,
    # pooled |x| peaks at 15.38 -- far inside float16 both in the tower and in
    # the bank's float16 storage. Notably tighter than `siglipso400m`'s 36.66
    # despite the longer sequence. docs/backbone_dtype_probe_new.json.
    "siglip2l512": BackboneSpec("siglip2l512",
                                 "google/siglip2-large-patch16-512",
                                 512, 1024, 0, 316_742_656,
                                 mean=SIGLIP_MEAN, std=SIGLIP_STD),

    # DINOv2-with-registers ViT-g/14 at 518. THE CAPACITY QUESTION, asked once.
    #
    # Every tower in this registry sits between 300M and 660M, so the ablation
    # has never actually tested whether capacity is what binds. Two results say
    # it might be the only thing that does: A1/A2 beat A3 on the frozen corpus
    # -- the training-side variations bought nothing -- and the DINOv3 head
    # ablation found the frozen features so strong that fitting the classifier
    # barely moved the number. If the head is not where the signal is decided,
    # the backbone is, and 300M is where we have been looking for it.
    #
    # `dinov2regl` -> `dinov2regg` is a ONE-VARIABLE capacity comparison: same
    # lineage, same self-distillation objective, same 518 input, same
    # ImageNet normalisation, and the same 5 prefix tokens
    # (`num_register_tokens: 4` plus CLS -- confirmed from the published config,
    # not assumed from the large's entry). Only width (1024 -> 1536) and depth
    # (24 -> 40) move.
    #
    # WHAT IT COSTS. Roughly depth x width^2: (40/24) * (1536/1024)^2 = 3.75x a
    # ViT-L. Against dinov2l's measured 7.4 img/s at 518 on a 4090 that is
    # ~2 img/s, so a 20k probe arm is ~2h45 and the 375,358-row union is ~13h
    # across four cards. This is the one entry whose full extraction is a whole
    # night, and it must earn that from the probe first.
    #
    # WHAT IT FORBIDS. At 1,136,486,912 it can only ship beside a partner of
    # ~780M or less once the SD 1.5 VAE (84M) and LPIPS (2.5M) are counted. Two
    # giants are 2.30B and are barred outright -- see
    # tests/features/test_backbones.py::test_the_heaviest_shippable_configuration_stays_under_2b.
    #
    # float16 MEASURED (2026-08-31, same 24 images): finite, max|diff|
    # 7.612e-03, pooled |x| peaks at 6.85 -- the TIGHTEST of any entry in this
    # registry, and the expectation going in was the opposite. 40 layers is 40
    # chances to accumulate and DINOv3's NaN bank was an overflow in layer 1 of
    # a tower half this deep, so depth was the stated risk. It did not
    # materialise: the overflow is a property of DINOv3's checkpoint, not of
    # ViT depth, and DINOv2's lineage does not inherit it at any scale.
    # docs/backbone_dtype_probe_new.json. This is exactly why the probe exists
    # rather than a rule of thumb -- the rule of thumb would have cost this
    # entry a needless bfloat16 fallback and ~3x its runtime on a T4.
    "dinov2regg": BackboneSpec("dinov2regg",
                                "facebook/dinov2-with-registers-giant",
                                518, 1536, 5, 1_136_486_912),
}


def squish(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size), ignoring aspect ratio."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _bf16_is_native() -> bool:
    """True when the current CUDA device runs bfloat16 in hardware (Ampere,
    sm_80, and later). Kaggle's T4 (sm_75) and P100 (sm_60) do not."""
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:                       # torch < 2.3 has no such keyword
        return bool(torch.cuda.is_bf16_supported())


def run_dtype(spec: BackboneSpec, device: str) -> torch.dtype:
    """The dtype `spec` actually runs in on `device`.

    `spec.dtype`, except that a bfloat16 spec on a CUDA device without native
    bfloat16 falls back to float32: slower, but finite. It never falls back to
    float16 -- that is the dtype a bfloat16 spec exists to avoid, and a
    fallback into it would reproduce the all-NaN bank at the one site meant to
    prevent it.
    """
    if (spec.dtype is torch.bfloat16 and str(device).startswith("cuda")
            and not _bf16_is_native()):
        return torch.float32
    return spec.dtype


def _require_timm(spec: BackboneSpec) -> None:
    """Fail on the missing optional package with an install line.

    Without this the failure is a transformers ImportError raised from inside
    `AutoModel.from_pretrained`, several cells into a Kaggle session, naming
    neither the backbone that asked for it nor what to do about it.
    """
    import importlib.util

    if importlib.util.find_spec("timm") is None:
        raise ImportError(
            f"backbone {spec.name!r} ({spec.hf_id}) is published by timm, which "
            f"is not installed. Install it with `pip install 'timm>=1.0'` -- it "
            f"is a declared dependency in pyproject.toml, and Kaggle's image "
            f"already ships it.")


def _strip_timm_head(model):
    """Drop the classifier `TimmWrapperModel` materialises, and stop it running.

    Two separate things, both needed. `reset_classifier(0)` frees the 1000-way
    IN-1k head, which is what makes `eva02l`'s recorded parameter count the
    count of what is actually loaded. `do_pooling=False` stops
    `TimmWrapperModel.forward` calling `forward_head` at all -- `_pool` reads
    `last_hidden_state`, which is `forward_features(x)` either way, so the
    pooler output is compute nobody reads.
    """
    inner = getattr(model, "timm_model", None)
    if inner is not None and hasattr(inner, "reset_classifier"):
        inner.reset_classifier(0)
    if hasattr(model, "config"):
        model.config.do_pooling = False
    return model


def load_backbone(name: str, device: str = "cuda") -> tuple[torch.nn.Module, BackboneSpec]:
    """Load a frozen backbone's vision tower in eval mode on `device`, in
    `run_dtype(spec, device)`."""
    from transformers import AutoModel

    spec = BACKBONES[name]
    if spec.loader == LOADER_TIMM:
        _require_timm(spec)
    model = AutoModel.from_pretrained(spec.hf_id, dtype=run_dtype(spec, device))
    # CLIP and SigLIP wrap a vision tower alongside a text tower; DINOv3 is
    # already a vision-only model.
    model = getattr(model, "vision_model", model)
    if spec.loader == LOADER_TIMM:
        model = _strip_timm_head(model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec


def _normalised_batch(imgs: list[np.ndarray], size: int,
                      mean: tuple[float, float, float],
                      std: tuple[float, float, float]) -> np.ndarray:
    """Squish, scale to [0, 1] and normalise into (B, size, size, 3) float32,
    channels LAST -- the layout both input formats below start from.

    `mean`/`std` come from the spec, not from a module constant: see
    IMAGENET_MEAN above for why they are per-backbone.
    """
    arr = np.stack([squish(i, size) for i in imgs]).astype(np.float32) / 255.0
    return ((arr - np.asarray(mean, dtype=np.float32))
            / np.asarray(std, dtype=np.float32))


def _patchify(arr: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    """(B, H, W, C) -> ((B, n_h * n_w, patch*patch*C), n_h, n_w).

    Element order matches `transformers.models.siglip2.image_processing_siglip2
    .convert_image_to_patches` exactly: raster-scan over patches, then
    (row, col, channel) within a patch. That order is what SigLIP2's
    `patch_embedding` Linear's weights were trained against, so it is a
    contract, not a formatting choice.
    """
    b, h, w, c = arr.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"({h}, {w}) is not divisible by patch_size {patch_size}")
    n_h, n_w = h // patch_size, w // patch_size
    p = arr.reshape(b, n_h, patch_size, n_w, patch_size, c).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(p.reshape(b, n_h * n_w, patch_size * patch_size * c)), n_h, n_w


def model_inputs(spec: BackboneSpec, imgs: list[np.ndarray], device: str,
                 dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Build the forward() kwargs `spec`'s vision tower actually accepts.

    Only `pixel_values` is cast to `dtype`: `attention_mask` and
    `spatial_shapes` are integer index/mask tensors and must stay integral.
    """
    arr = _normalised_batch(imgs, spec.image_size, spec.mean, spec.std)
    if spec.input_format == INPUT_IMAGE_TENSOR:
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return {"pixel_values": x.to(device, dtype)}
    if spec.input_format == INPUT_SIGLIP2_PATCHES:
        patches, n_h, n_w = _patchify(arr, spec.patch_size)
        b = patches.shape[0]
        return {
            "pixel_values": torch.from_numpy(patches).to(device, dtype),
            # Every image here is squished to the same fixed square, so no
            # patch is padding and the mask is all-ones. It is still passed
            # because Siglip2VisionTransformer.forward requires it positionally.
            "attention_mask": torch.ones((b, n_h * n_w), dtype=torch.long, device=device),
            "spatial_shapes": torch.tensor([[n_h, n_w]] * b, dtype=torch.long, device=device),
        }
    raise ValueError(f"unknown input_format {spec.input_format!r}")


def _pool(model, spec: BackboneSpec, inputs: dict) -> "torch.Tensor":
    """Run one batch through `model` and reduce it per `spec.pool`, to
    (B, spec.dim) float32."""
    if spec.pool == POOL_TOKENS:
        h = model(**inputs).last_hidden_state             # (B, T, D)
        patches = h[:, spec.num_prefix_tokens:, :]        # drop CLS + registers
        return patches.mean(dim=1).float()

    hidden = model(**inputs, output_hidden_states=True).hidden_states
    parts = []
    for stage in spec.stages:
        # (B, C, H, W) -> (B, C, H*W). float() BEFORE the moments: a float16
        # variance over a few hundred positions loses precision in the tail,
        # and the std is the whole reason this pooling exists.
        flat = hidden[stage].flatten(2).float()
        parts.append(flat.mean(dim=-1))
        # Population std (correction=0): the spatial positions are the whole
        # feature map, not a sample from a larger one, and a 1x1 stage would
        # otherwise divide by zero rather than yielding 0.
        parts.append(flat.std(dim=-1, correction=0))
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def embed(model, spec: BackboneSpec, imgs: list[np.ndarray],
          device: str = "cuda", batch_size: int = 16) -> np.ndarray:
    """Global-average-pool the final-layer patch tokens (CLS/register tokens
    stripped per `spec.num_prefix_tokens`) into an (N, spec.dim) float32 array.

    The forward() call is built per `spec.input_format`, because the registry's
    backbones do not share one input contract (see `INPUT_FORMATS`).
    """
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:                      # a parameterless stand-in
        dtype = torch.float32
    out = []
    for i in range(0, len(imgs), batch_size):
        inputs = model_inputs(spec, imgs[i:i + batch_size], device, dtype)
        pooled = _pool(model, spec, inputs)
        # Checked here, on the first batch, rather than discovered by Stage B:
        # a tower running in a dtype it overflows produces NaN for EVERY image
        # at full speed, and nothing downstream of this line -- the bank
        # writer, the row count a chained job checks, `check_invariants` --
        # looked at a single value until 2026-08-29's 5-hour all-NaN bank.
        bad = ~torch.isfinite(pooled).all(dim=1)
        if bool(bad.any()):
            raise ValueError(
                f"{spec.name} produced non-finite features for "
                f"{int(bad.sum())} of {pooled.shape[0]} images in this batch, "
                f"running in {dtype}. This is the backbone overflowing its "
                f"dtype, not a bad image: DINOv3-L overflows float16 at hidden "
                f"layer 1 and needs bfloat16 (or float32). Nothing was "
                f"written; fix BackboneSpec.dtype / run_dtype and re-run.")
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
