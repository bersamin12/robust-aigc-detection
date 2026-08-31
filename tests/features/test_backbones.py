import os

import numpy as np
import pytest

from aigcdet.features.backbones import (BACKBONES, POOL_SPATIAL_MS,
                                        POOL_TOKENS, squish)

_MIN_FREE_BYTES = 4 * 1024**3  # 4 GB; a ViT-L backbone needs headroom beyond its weights


def _skip_unless_gpu_has_headroom():
    """Real @pytest.mark.gpu tests must skip honestly on a near-full GPU rather
    than attempting a load that OOMs. torch.cuda.is_available() is True even at
    a few hundred MiB free, so the guard checks free VRAM via mem_get_info(),
    not just device availability."""
    if os.environ.get("AIGCDET_ALLOW_GPU_TESTS") != "1":
        pytest.skip(
            "GPU tests are opt-in: set AIGCDET_ALLOW_GPU_TESTS=1 to run them. "
            "They load real backbone weights and will download them if absent."
        )

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if free_bytes < _MIN_FREE_BYTES:
        pytest.skip(
            f"only {free_bytes / 1024**2:.0f} MiB free on GPU, "
            f"need at least {_MIN_FREE_BYTES / 1024**2:.0f} MiB to load a ViT-L backbone"
        )


def test_registry_has_the_planned_backbones():
    from aigcdet.features.backbones import POOL_SPATIAL_MS, POOL_TOKENS

    assert set(BACKBONES) == {"dinov3l", "dinov2l", "siglip2l", "clipl",
                              "convnextt", "resnet50",
                              # backbone-probe candidates, added 2026-08-31
                              "dinov2regl", "eva02l", "convnextv2h",
                              "siglipso400m",
                              # resolution and capacity arms, added 2026-08-31.
                              # siglip2l512 is siglip2l's tower at the side
                              # canonicalise actually emits; dinov2regg is the
                              # first entry above 1B, and asks whether capacity
                              # is what binds at all.
                              "siglip2l512", "dinov2regg",
                              # The two-tower arm, added 2026-09-01.
                              # dinov2regl224 is dinov2regl's tower at 224
                              # instead of 518 -- 256 patches against 1369, so
                              # two of them fit where one at 518 barely does.
                              # It is a registry entry rather than a runtime
                              # flag because image_size is recorded in every
                              # bank config and is what makes a bank and a
                              # checkpoint comparable.
                              "dinov2regl224"}
    for spec in BACKBONES.values():
        # 518 is dinov2l's and dinov2regl's: `canonicalise` emits a 512-px
        # nominal side, so 518 is the one registered size that upsamples into
        # the tower instead of discarding high-frequency detail on the way in.
        #
        # 448 is eva02l's, and it is not a choice -- its pretrained_cfg sets
        # `fixed_input_size`, so the position embedding admits exactly 448 and
        # the 512 nominal side is downsampled on the way in. That is a stated
        # handicap on that one arm, recorded on its registry entry.
        #
        # Widening this set is a real decision -- an entry at an unintended
        # resolution silently changes what the bank measures.
        # 512 is siglip2l512's, and it is the ONE entry that sees the nominal
        # side exactly: `canonicalise` emits 512, so 518 upsamples into the
        # tower, 448 and 384 discard, and 512 does neither. That it took until
        # 2026-08-31 to register a backbone at the resolution the pipeline
        # actually produces is itself the reason the entry exists.
        assert spec.dim > 0 and spec.image_size in (224, 384, 448, 512, 518)
        if spec.pool == POOL_TOKENS:
            # At least a CLS token to strip, except the SigLIP towers, whose
            # architecture has no prefix token at all.
            assert (spec.num_prefix_tokens >= 1
                    or spec.name in ("siglip2l", "siglip2l512", "siglipso400m"))
        else:
            # A feature map has no token axis, so there is nothing to strip;
            # __post_init__ rejects a non-zero count outright.
            assert spec.pool == POOL_SPATIAL_MS and spec.num_prefix_tokens == 0


def test_the_convolutional_entries_are_a_different_paradigm_not_a_third_vit():
    """A5 fuses paradigms, not checkpoints (spec 6.4). Two ViTs pooled the same
    way are one paradigm wearing two hats; the conv entries earn their place in
    the registry only by differing in BOTH the tower and the pooling."""
    from aigcdet.features.backbones import POOL_SPATIAL_MS, POOL_TOKENS

    conv = {"convnextt", "resnet50", "convnextv2h"}
    assert {n for n, s in BACKBONES.items() if s.pool == POOL_SPATIAL_MS} == conv
    assert {n for n, s in BACKBONES.items() if s.pool == POOL_TOKENS} == set(BACKBONES) - conv
    for name in conv:
        # The std half is the reason the pooling exists: mean-only would make
        # these semantic extractors competing with SigLIP2 on its own ground.
        # dim is even because every stage contributes a mean AND a std.
        assert BACKBONES[name].dim % 2 == 0
        assert BACKBONES[name].stages


def test_two_shipped_backbones_stay_under_1b_working_ceiling():
    # A tighter working budget for the two backbones this project trains against
    # day to day (dinov3l + siglip2l). clipl is a third registry entry not summed
    # here -- see test_full_registry_stays_under_the_2b_hackathon_hard_limit for
    # the constraint that actually gates the hackathon submission.
    assert BACKBONES["dinov3l"].params + BACKBONES["siglip2l"].params < 1_000_000_000


def test_the_heaviest_shippable_configuration_stays_under_2b():
    # The hackathon's real, hard constraint (project-constraints.md): models
    # under 2B parameters total.
    #
    # This summed the WHOLE REGISTRY until 2026-08-31, which conflated two
    # different things. The constraint binds the architecture that SHIPS -- the
    # spec's own wording is "Final model uses at most two backbones", and its
    # exclusions name "any model at or above 2B parameters" -- not the menu of
    # candidates an ablation is allowed to consider. Under the old reading the
    # registry's 722M of headroom would have vetoed a four-backbone probe that
    # ships none of the four, which is a constraint the rules do not impose.
    #
    # What must actually hold is that no bundle export can exceed the cap. An
    # A5 fusion loads at most two towers, and Task 4 attaches the SD 1.5 VAE
    # (~84M) and LPIPS AlexNet (~2.5M) alongside, so the worst case is the two
    # heaviest entries plus both auxiliaries.
    # TWO autoencoders since 2026-08-31, not one. The reconstruction branch
    # now has a second block (`a4vq`/`a4both`): a bundle that ships the VQ
    # features carries the VQ autoencoder alongside the KL one, and a bundle
    # shipping `a4both` carries both. Counted at the measured values rather
    # than round numbers, and raised HERE on purpose rather than discovered as
    # a red test after an extraction:
    #
    #     SD 1.5 KL VAE                       83,653,863
    #     CompVis LDM VQ autoencoder          55,322,782
    #     LPIPS (AlexNet), shared by both      2,470,848
    KL_VAE, VQ_AE, LPIPS = 83_653_863, 55_322_782, 2_470_848
    heaviest = sorted((spec.params for spec in BACKBONES.values()), reverse=True)
    worst_case = sum(heaviest[:2]) + KL_VAE + VQ_AE + LPIPS
    assert worst_case < 2_000_000_000, f"{worst_case:,} parameters"


def test_backbone_params_are_positive_and_real_looking():
    # Guards against a placeholder value that happens to satisfy the budget
    # inequality above: every recorded count must be a plausible measured one.
    #
    # ONE band, not one per paradigm. The split used to be 200M-500M for
    # token-pooled and 10M-150M for spatial-pooled, and the narrow conv band was
    # justified in this file by "a conv tower is one to two orders smaller ...
    # ~50x less compute per image than SigLIP2-L". `convnextv2h` is a conv tower
    # at 657M and ~338 GFLOPs -- MORE compute per image than DINOv2-L at 518 --
    # so that sentence is a property of convnextt and resnet50 and not of the
    # paradigm. It now lives on their own registry entries, where it is still
    # exactly right, and what is left here is a plausibility check.
    #
    # The upper bound moved from 1B to 1.5B on 2026-08-31 for `dinov2regg`
    # (1,136,486,912). It is a plausibility guard, not a budget: the budget is
    # test_the_heaviest_shippable_configuration_stays_under_2b, and THAT test
    # is now nearly binding -- dinov2regg + convnextv2h + the two auxiliaries
    # is 1.880B against the 2B cap, where before this entry the worst pair left
    # 828M of headroom and now leaves 119M. A third auxiliary model would break
    # it. Widening this band does not widen that one.
    for spec in BACKBONES.values():
        assert 10_000_000 < spec.params < 1_500_000_000, spec.name


def test_squish_ignores_aspect_ratio():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    out = squish(img, 384)
    assert out.shape == (384, 384, 3) and out.dtype == np.uint8


def test_squish_preserves_pixel_content():
    # A regression guard against a stub that returns a black/constant image:
    # squish must actually resample the input, not fabricate output.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
    out = squish(img, 64)
    assert out.std() > 0
    assert not np.array_equal(out, np.zeros_like(out))


def test_squish_distorts_aspect_ratio_by_known_factor():
    # squish must stretch x and y independently, not letterbox/pad-and-resize.
    # A resize that preserved content aspect ratio would pass
    # test_squish_ignores_aspect_ratio (it only checks output shape) while
    # quietly changing what every backbone sees -- random resized cropping and
    # aspect-preserving resizes were both rejected for this project because
    # they can destroy the localised forensic cues detection depends on. Pin
    # the distortion itself: a square marker must come out as a rectangle whose
    # dimensions match squish's independent x/y scale factors.
    h, w, size = 100, 300, 384
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255  # a 10x10 white square, aspect ratio 1:1

    out = squish(img, size)

    mask = out[..., 0] > 128
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    out_height = int(rows.max() - rows.min() + 1)
    out_width = int(cols.max() - cols.min() + 1)

    expected_height = round(10 * size / h)  # independent y scale: 384/100 -> ~38
    expected_width = round(10 * size / w)   # independent x scale: 384/300 -> ~13
    assert abs(out_height - expected_height) <= 3
    assert abs(out_width - expected_width) <= 3
    # The square must become a visibly non-square rectangle: a letterboxed or
    # otherwise aspect-preserving resize would keep it square here.
    assert out_height != out_width


@pytest.mark.gpu
def test_embed_returns_pooled_vectors_of_the_right_width():
    _skip_unless_gpu_has_headroom()
    from aigcdet.features.backbones import embed, load_backbone

    model, spec = load_backbone("clipl", device="cuda")
    imgs = [np.random.default_rng(i).integers(0, 256, (512, 640, 3), dtype=np.uint8)
            for i in range(3)]
    out = embed(model, spec, imgs, device="cuda", batch_size=2)
    assert out.shape == (3, spec.dim) and out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.gpu
def test_embedding_is_deterministic():
    _skip_unless_gpu_has_headroom()
    from aigcdet.features.backbones import embed, load_backbone

    model, spec = load_backbone("clipl", device="cuda")
    img = [np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)]
    a = embed(model, spec, img, device="cuda")
    b = embed(model, spec, img, device="cuda")
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# embed()'s tensor contract, per backbone, on CPU.
#
# These build the SAME architecture each registry entry names, at toy width,
# from a locally-constructed config: no download, no GPU, no weights. That is
# what would have caught C1 on day 1 -- `embed` asserted one input contract
# (`model(pixel_values=(B, 3, H, W))`) across three backbones that do not share
# one, and SigLIP2 could not be embedded at all.
# --------------------------------------------------------------------------

_TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
             num_attention_heads=4)
_TINY_PATCH = 16
_TINY_IMAGE = 64          # 4 x 4 = 16 patches
#: Toy stage widths for the conv towers. Four stages, each halving spatially:
#: 64 -> 16 (stride-4 stem) -> 8 -> 4 -> 2, so every stage keeps a real spatial
#: extent for the std to be computed over. A 1x1 stage would make the std
#: identically zero and hide a broken pooling.
_TINY_CONV_WIDTHS = [8, 16, 32, 64]


def _tiny_tower(name: str):
    """(vision tower, spec) for `name`'s real architecture at toy width.

    The returned spec keeps every contract-bearing field of the registry entry
    -- `input_format`, `patch_size`, `num_prefix_tokens` -- and only shrinks
    `image_size`/`dim`/`params`, so a wrong input_format or prefix-token count
    in `BACKBONES` fails here.
    """
    from dataclasses import replace

    real = BACKBONES[name]
    if name == "dinov3l":
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        # num_register_tokens=4 plus DINOv3's CLS token == the registry's
        # num_prefix_tokens=5, so the real value is exercised, not a stand-in.
        model = DINOv3ViTModel(DINOv3ViTConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_register_tokens=real.num_prefix_tokens - 1, **_TINY))
    elif name == "dinov2l":
        # Dinov2Model, and NOT DINOv3's class: the two lineages share a name and
        # nothing else in `transformers`. DINOv2 has a CLS token and NO register
        # tokens, so `num_prefix_tokens` must be exactly 1 -- asserting it here
        # is what catches a 5 copied across from the dinov3l entry, which would
        # silently drop four real patch tokens out of every pooled vector.
        from transformers import Dinov2Config, Dinov2Model
        assert real.num_prefix_tokens == 1
        model = Dinov2Model(Dinov2Config(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE, **_TINY))
    elif name == "siglip2l":
        # SiglipVisionModel, NOT Siglip2VisionModel. SigLIP2's FIXED-RESOLUTION
        # checkpoints (`siglip2-*-patch16-384`) publish `model_type: siglip`
        # and load as SigLIP v1's tower; only the `-naflex` ones are
        # Siglip2VisionModel. Building the wrong class here is not a cosmetic
        # slip -- it is how a wrong `input_format` in BACKBONES stays green:
        # the stub agrees with the spec, both disagree with the checkpoint, and
        # the first real forward pass 4 hours into an extraction is what tells
        # you. `test_declared_input_format_matches_the_published_config` is the
        # guard that actually consults the repo.
        from transformers import SiglipVisionConfig, SiglipVisionModel
        model = SiglipVisionModel(SiglipVisionConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_channels=3, **_TINY))
    elif name == "clipl":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        model = CLIPVisionModel(CLIPVisionConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE, **_TINY))
    elif name in ("dinov2regl", "dinov2regl224"):
        # `dinov2regl224` is the SAME tower and the same recipe -- only the
        # registry's `image_size` differs, and the tiny stand-in overrides that
        # with `_TINY_IMAGE` anyway, so sharing the branch is honest rather
        # than a shortcut: the contract being checked (register count, prefix
        # stripping, tensor shapes) is genuinely identical.
        # Dinov2WithRegistersModel, and NOT Dinov2Model: the registers are the
        # entire reason this entry exists beside `dinov2l`. num_register_tokens
        # 4 plus the CLS token == the registry's num_prefix_tokens of 5, so the
        # real value is exercised rather than a stand-in. The assert mirrors
        # dinov2l's `== 1` in the opposite direction, and for the same reason:
        # a 1 copied across from dinov2l would average four register tokens
        # into every pooled vector, and nothing downstream would notice.
        from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
        assert real.num_prefix_tokens == 5
        model = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_register_tokens=real.num_prefix_tokens - 1, **_TINY))
    elif name == "dinov2regg":
        # Identical recipe to dinov2regl, and that is the point: dinov2regl ->
        # dinov2regg is a one-variable capacity comparison, so if the giant
        # needed a DIFFERENT tower class here the comparison would be varying
        # the architecture too. `num_register_tokens: 4` is confirmed from the
        # giant's own published config, not inherited from the large's entry.
        from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
        assert real.num_prefix_tokens == 5
        model = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_register_tokens=real.num_prefix_tokens - 1, **_TINY))
    elif name == "siglip2l512":
        # SiglipVisionModel, same class as siglip2l: the 512 checkpoint is the
        # SAME TOWER at a longer position embedding and publishes the same
        # `model_type: siglip`. A different class here would mean the entry was
        # mis-registered as a capacity change rather than a resolution one.
        from transformers import SiglipVisionConfig, SiglipVisionModel
        model = SiglipVisionModel(SiglipVisionConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_channels=3, **_TINY))
    elif name == "eva02l":
        # Through TimmWrapperModel, which is what `load_backbone` actually gets
        # back from AutoModel for a `timm/*` repo -- a bare timm module has
        # `forward(x)`, not `forward(pixel_values=...)`, so stubbing one
        # directly would test a contract the production path never uses.
        #
        # `eva02_tiny_patch14_224` is the real EVA-02 architecture (SwiGLU, RoPE,
        # sub-LN) shrunk by model_args rather than a plain ViT wearing its name,
        # and nothing is downloaded: TimmWrapperConfig builds from the
        # architecture string alone. patch_size 16 over img_size 64 keeps the
        # 4x4 = 16 patches every other recipe here uses.
        pytest.importorskip("timm")
        from transformers import TimmWrapperConfig, TimmWrapperModel
        assert real.num_prefix_tokens == 1
        model = TimmWrapperModel(TimmWrapperConfig(
            architecture="eva02_tiny_patch14_224", num_classes=0,
            do_pooling=False,
            model_args={"img_size": _TINY_IMAGE, "patch_size": _TINY_PATCH,
                        "embed_dim": _TINY["hidden_size"],
                        "depth": _TINY["num_hidden_layers"],
                        "num_heads": _TINY["num_attention_heads"]}))
        # timm's own count of its own architecture, against the registry's.
        assert model.timm_model.num_prefix_tokens == real.num_prefix_tokens
    elif name == "convnextv2h":
        # ConvNextV2Config, NOT ConvNextConfig. V2 replaces V1's LayerScale
        # with GRN (a global L2 norm over spatial positions), which is both the
        # architectural difference and the float16 risk recorded on the registry
        # entry -- so building V1 here would test the wrong tower.
        from transformers import ConvNextV2Config, ConvNextV2Model
        model = ConvNextV2Model(ConvNextV2Config(
            num_channels=3, hidden_sizes=_TINY_CONV_WIDTHS, depths=[1, 1, 1, 1]))
    elif name == "siglipso400m":
        # SiglipVisionModel, same as siglip2l: so400m publishes
        # `model_type: siglip` and loads as SigLIP v1's tower.
        from transformers import SiglipVisionConfig, SiglipVisionModel
        model = SiglipVisionModel(SiglipVisionConfig(
            patch_size=_TINY_PATCH, image_size=_TINY_IMAGE,
            num_channels=3, **_TINY))
    elif name == "convnextt":
        from transformers import ConvNextConfig, ConvNextModel
        model = ConvNextModel(ConvNextConfig(
            num_channels=3, hidden_sizes=_TINY_CONV_WIDTHS, depths=[1, 1, 1, 1]))
    elif name == "resnet50":
        from transformers import ResNetConfig, ResNetModel
        model = ResNetModel(ResNetConfig(
            num_channels=3, embedding_size=_TINY_CONV_WIDTHS[0],
            hidden_sizes=_TINY_CONV_WIDTHS, depths=[1, 1, 1, 1],
            layer_type="bottleneck"))
    else:
        raise AssertionError(f"no tiny tower recipe for {name!r}")

    # The same unwrapping load_backbone does.
    model = getattr(model, "vision_model", model).eval()
    dim = _TINY["hidden_size"]
    if real.pool == POOL_SPATIAL_MS:
        # DISCOVERED from the toy tower, not asserted: the whole contract under
        # test is "spec.dim == 2 * sum(channels at spec.stages)", so hardcoding
        # a width here would let a wrong `stages` agree with a wrong `dim` and
        # pass. `test_spatial_pooling_width_is_two_moments_per_stage_channel`
        # is what pins the relationship.
        import torch
        with torch.inference_mode():
            probe = model(pixel_values=torch.zeros(1, 3, _TINY_IMAGE, _TINY_IMAGE),
                          output_hidden_states=True).hidden_states
        dim = 2 * sum(probe[i].shape[1] for i in real.stages)
    spec = replace(real, image_size=_TINY_IMAGE, dim=dim, params=0)
    return model, spec


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_embed_tensor_contract_holds_for_every_backbone_on_cpu(name):
    from aigcdet.features.backbones import embed

    model, spec = _tiny_tower(name)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (120, 200, 3), dtype=np.uint8) for _ in range(3)]

    out = embed(model, spec, imgs, device="cpu", batch_size=2)

    assert out.shape == (3, spec.dim)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Not a constant/stub: three different images must give three different
    # pooled vectors.
    assert len({tuple(np.round(r, 5)) for r in out}) == 3


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_num_prefix_tokens_matches_the_real_architecture(name):
    """`embed` strips `spec.num_prefix_tokens` before pooling. If that count is
    wrong for an architecture, pooling silently averages a CLS/register token
    into the patch mean (or drops a real patch) and nothing else notices."""
    import torch

    from aigcdet.features.backbones import model_inputs

    if BACKBONES[name].pool != POOL_TOKENS:
        pytest.skip(f"{name} pools a feature map; it has no token axis")

    model, spec = _tiny_tower(name)
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 256, (90, 90, 3), dtype=np.uint8) for _ in range(2)]

    with torch.inference_mode():
        h = model(**model_inputs(spec, imgs, "cpu", torch.float32)).last_hidden_state

    expected_patches = (spec.image_size // _TINY_PATCH) ** 2
    assert h.shape[1] - spec.num_prefix_tokens == expected_patches
    assert h.shape[-1] == spec.dim


def test_every_registry_entry_declares_a_supported_input_format():
    from aigcdet.features.backbones import INPUT_FORMATS, INPUT_SIGLIP2_PATCHES

    for spec in BACKBONES.values():
        assert spec.input_format in INPUT_FORMATS
        if spec.input_format == INPUT_SIGLIP2_PATCHES:
            assert spec.patch_size > 0
            assert spec.image_size % spec.patch_size == 0


def test_spec_rejects_an_unusable_input_contract():
    from aigcdet.features.backbones import INPUT_SIGLIP2_PATCHES, BackboneSpec

    with pytest.raises(ValueError, match="input_format"):
        BackboneSpec("bad", "none", 384, 1024, 0, 1, input_format="conv2d")
    with pytest.raises(ValueError, match="patch_size"):
        BackboneSpec("bad", "none", 384, 1024, 0, 1,
                     input_format=INPUT_SIGLIP2_PATCHES)
    with pytest.raises(ValueError, match="divisible"):
        BackboneSpec("bad", "none", 100, 1024, 0, 1,
                     input_format=INPUT_SIGLIP2_PATCHES, patch_size=16)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_declared_input_format_matches_the_published_config(name):
    """Every registry entry's `input_format` must match the architecture its
    hf_id actually publishes.

    The tiny towers above are hermetic, which is their virtue and their blind
    spot: they assert an architecture rather than discover one, so a registry
    entry pointing at a checkpoint of a different class passes every one of
    them. This test is the only place the claim is checked against the repo,
    which is why it needs the network and carries the gpu marker (config only
    -- no weights are downloaded).
    """
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_MAPPING

    from aigcdet.features.backbones import INPUT_SIGLIP2_PATCHES

    if os.environ.get("AIGCDET_ALLOW_GPU_TESTS") != "1":
        pytest.skip("opt-in: set AIGCDET_ALLOW_GPU_TESTS=1 (downloads configs)")

    spec = BACKBONES[name]
    cfg = AutoConfig.from_pretrained(spec.hf_id)
    cls_name = MODEL_MAPPING[type(cfg)].__name__

    # Only the Siglip2* family takes patchified input + attention_mask +
    # spatial_shapes; everything else takes pixel_values=(B, 3, H, W).
    needs_patches = cls_name.startswith("Siglip2")
    declared_patches = spec.input_format == INPUT_SIGLIP2_PATCHES
    assert needs_patches == declared_patches, (
        f"{name}: {spec.hf_id} loads as {cls_name}, which "
        f"{'requires' if needs_patches else 'does not accept'} patchified "
        f"input, but the registry declares input_format={spec.input_format!r}")


def test_patchify_matches_the_transformers_reference_implementation():
    """SigLIP2's patch_embedding is a Linear over flattened patches, so the
    element ORDER inside each patch is part of the weight contract. Pin it
    against the transformers implementation the real processor uses."""
    from transformers.models.siglip2.image_processing_siglip2 import (
        convert_image_to_patches,
    )

    from aigcdet.features.backbones import _patchify

    rng = np.random.default_rng(3)
    arr = rng.normal(size=(2, 64, 48, 3)).astype(np.float32)
    ours, n_h, n_w = _patchify(arr, 16)
    assert (n_h, n_w) == (4, 3)
    for b in range(2):
        np.testing.assert_array_equal(ours[b], convert_image_to_patches(arr[b], 16))


# --- which weights are gated ------------------------------------------------

def test_only_dinov3_is_gated():
    """Gating is a property of the checkpoint, so it belongs on the spec rather
    than in a notebook's prose.

    It decides whether a run needs a HuggingFace token at all, and the
    allocation now has five teammates on SigLIP2 (Apache-2.0) while DINOv3
    runs locally -- so a blanket token requirement would block four people over
    a licence acceptance none of their runs need. See docs/model_licences.md."""
    from aigcdet.features.backbones import BACKBONES

    assert BACKBONES["dinov3l"].gated is True
    assert BACKBONES["siglip2l"].gated is False
    assert BACKBONES["clipl"].gated is False
    # dinov2l is ungated Apache-2.0, which is its whole reason for being in the
    # registry beside the gated dinov3l -- if this ever flips to True the
    # substitute has stopped being a substitute.
    assert BACKBONES["dinov2l"].gated is False
    # All four backbone-probe candidates are ungated (Hub API, 2026-08-31), so
    # the probe's auth cell reports "not gated" and moves on. Named here rather
    # than left uncovered: an ungated entry missing from the gating test is how
    # a session ends up blocked on a token none of its arms need.
    for probe_name in ("dinov2regl", "eva02l", "convnextv2h", "siglipso400m"):
        assert BACKBONES[probe_name].gated is False, probe_name


def test_each_spec_declares_its_own_pretraining_normalisation():
    """`_normalised_batch` reads mean/std off the spec, not off a module
    constant. Handing a tower the wrong input distribution does not raise -- it
    quietly costs that arm accuracy, and in a comparison whose purpose is
    RANKING towers that is indistinguishable from the tower being worse."""
    from aigcdet.features.backbones import (BACKBONES, CLIP_MEAN, CLIP_STD,
                                            IMAGENET_MEAN, IMAGENET_STD,
                                            SIGLIP_MEAN, SIGLIP_STD)

    # Read from each checkpoint's published preprocessor_config.json,
    # 2026-08-31.
    assert BACKBONES["eva02l"].mean == CLIP_MEAN
    assert BACKBONES["eva02l"].std == CLIP_STD
    assert BACKBONES["siglipso400m"].mean == SIGLIP_MEAN
    assert BACKBONES["siglipso400m"].std == SIGLIP_STD
    assert BACKBONES["dinov2regl"].mean == IMAGENET_MEAN
    assert BACKBONES["convnextv2h"].mean == IMAGENET_MEAN

    # The DELIBERATE asymmetry, pinned so it stays a decision and not a bug:
    # clipl and siglip2l were pretrained on their own statistics too, and keep
    # ImageNet's, because their banks were built that way and re-extracting
    # them is a separate job. The cost is that siglip2l vs siglipso400m
    # confounds tower with preprocessing -- see the registry comment.
    for legacy in ("dinov3l", "dinov2l", "siglip2l", "clipl",
                   "convnextt", "resnet50"):
        assert BACKBONES[legacy].mean == IMAGENET_MEAN, legacy
        assert BACKBONES[legacy].std == IMAGENET_STD, legacy


def test_normalisation_defaults_leave_every_pre_existing_entry_bit_identical():
    """The whole reason mean/std default to ImageNet: an entry that predates
    the change must produce the same pixels it always did, or every bank on
    disk is orphaned."""
    import numpy as np

    from aigcdet.features.backbones import (BACKBONES, IMAGENET_MEAN,
                                            IMAGENET_STD, _normalised_batch)

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (90, 140, 3), dtype=np.uint8) for _ in range(2)]
    spec = BACKBONES["siglip2l"]

    got = _normalised_batch(imgs, 64, spec.mean, spec.std)
    want = _normalised_batch(imgs, 64, IMAGENET_MEAN, IMAGENET_STD)
    assert np.array_equal(got, want)


def test_only_eva02_uses_the_timm_loader():
    """`loader` exists for the two things load_backbone must do differently for
    a timm repo: strip the classifier TimmWrapperModel materialises, and fail
    with an install line rather than a bare ImportError."""
    from aigcdet.features.backbones import (BACKBONES, LOADER_TIMM,
                                            LOADER_TRANSFORMERS)

    timm_backed = {n for n, s in BACKBONES.items() if s.loader == LOADER_TIMM}
    assert timm_backed == {"eva02l"}
    assert all(s.loader in (LOADER_TIMM, LOADER_TRANSFORMERS)
               for s in BACKBONES.values())
    # The hf_id and the loader must agree: a `timm/` repo declared as
    # transformers would reach AutoModel without the head-strip, and a
    # transformers repo declared as timm would raise on a package it does not
    # need.
    for name, spec in BACKBONES.items():
        assert spec.hf_id.startswith("timm/") == (spec.loader == LOADER_TIMM), name


def test_a_spec_rejects_a_degenerate_normalisation():
    from aigcdet.features.backbones import BackboneSpec

    with pytest.raises(ValueError, match="3 per-channel values"):
        BackboneSpec("x", "y", 224, 8, 0, 1, mean=(0.5, 0.5))
    with pytest.raises(ValueError, match="zero channel"):
        BackboneSpec("x", "y", 224, 8, 0, 1, std=(0.5, 0.0, 0.5))
    with pytest.raises(ValueError, match="loader must be one of"):
        BackboneSpec("x", "y", 224, 8, 0, 1, loader="hub")


# --------------------------------------------------------------------------
# dtype. DINOv3-L overflows float16 at hidden layer 1 and emits NaN for every
# image; the 2026-08-29 bank was 131,116 x 11 vectors of NaN, produced at full
# speed with nothing raising. These pin the fix and the guard that would have
# caught it on the first batch.
# --------------------------------------------------------------------------

def test_dinov3_runs_in_bfloat16_never_float16():
    import torch
    assert BACKBONES["dinov3l"].dtype is torch.bfloat16
    for name, spec in BACKBONES.items():
        assert spec.dtype in (torch.bfloat16, torch.float32) or name != "dinov3l"


def test_dinov2_runs_in_float16_because_that_was_measured():
    """dinov2l must NOT inherit dinov3l's bfloat16 workaround.

    The two share a name and a lineage, so copying the dtype across is the
    obvious mistake. It would be wrong twice over: DINOv2-L has no float16
    overflow (2026-08-30, 24 canonicalised images, every pooled value finite,
    max |diff| against a float32 run 2.3e-02), and bfloat16 is measurably
    FURTHER from float32 here at 1.0e-01. Paying bfloat16's cost would buy
    accuracy loss, not safety.
    """
    import torch
    assert BACKBONES["dinov2l"].dtype is torch.float16


def test_load_backbone_loads_the_spec_dtype(monkeypatch):
    """`from_pretrained` must be handed each spec's own dtype -- not one
    literal for every backbone, which is how DINOv3 ended up in float16."""
    import torch
    import transformers

    seen = {}

    def fake_from_pretrained(hf_id, dtype=None, **_):
        seen[hf_id] = dtype
        return torch.nn.Linear(2, 2).to(dtype)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    from aigcdet.features.backbones import load_backbone

    for name, spec in BACKBONES.items():
        model, got = load_backbone(name, device="cpu")
        assert got is spec
        assert seen[spec.hf_id] is spec.dtype, name
        assert next(model.parameters()).dtype is spec.dtype
        assert not any(p.requires_grad for p in model.parameters())


def test_bfloat16_falls_back_to_float32_not_float16_without_native_support(monkeypatch):
    """Kaggle's T4/P100 have no hardware bfloat16. There a bfloat16 spec must
    run in float32 (finite, slower) -- never float16, the dtype it exists to
    avoid -- and a float16 spec is untouched, and CPU needs no fallback."""
    import torch
    from aigcdet.features import backbones

    monkeypatch.setattr(backbones, "_bf16_is_native", lambda: False)
    assert backbones.run_dtype(BACKBONES["dinov3l"], "cuda") is torch.float32
    assert backbones.run_dtype(BACKBONES["dinov3l"], "cuda:1") is torch.float32
    assert backbones.run_dtype(BACKBONES["siglip2l"], "cuda") is torch.float16
    assert backbones.run_dtype(BACKBONES["dinov3l"], "cpu") is torch.bfloat16

    monkeypatch.setattr(backbones, "_bf16_is_native", lambda: True)
    assert backbones.run_dtype(BACKBONES["dinov3l"], "cuda") is torch.bfloat16


def test_embed_refuses_non_finite_features_on_the_batch_that_produced_them():
    """A tower that overflows its dtype must fail at the first batch, naming
    the backbone and the dtype, instead of writing NaN for five hours."""
    from dataclasses import replace
    from types import SimpleNamespace

    import torch
    from aigcdet.features.backbones import embed

    spec = replace(BACKBONES["dinov3l"], image_size=_TINY_IMAGE, dim=8, params=0)

    class Overflowing(torch.nn.Module):
        def forward(self, pixel_values):
            b = pixel_values.shape[0]
            h = torch.full((b, spec.num_prefix_tokens + 16, 8), float("nan"))
            h[0] = 1.0                           # one healthy image in the batch
            return SimpleNamespace(last_hidden_state=h)

    imgs = [np.zeros((_TINY_IMAGE, _TINY_IMAGE, 3), np.uint8)] * 3
    with pytest.raises(ValueError, match=r"dinov3l.*non-finite.*2 of 3.*float32"):
        embed(Overflowing(), spec, imgs, device="cpu", batch_size=3)


# ---------------------------------------------------------------------------
# Spatial mean+std pooling (the conv entries). The contract is
# `spec.dim == 2 * sum(channels at spec.stages)`, and the ORDER within it is
# [mean(stage_a), std(stage_a), mean(stage_b), std(stage_b), ...] -- a bank is
# written once and read for the rest of the project, so a silent reordering
# between extraction and a later re-extraction would misalign every column
# against a head trained on the earlier one, with no shape error anywhere.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(
    n for n, s in BACKBONES.items() if s.pool == POOL_SPATIAL_MS))
def test_spatial_pooling_width_is_two_moments_per_stage_channel(name):
    import torch

    from aigcdet.features.backbones import _pool, model_inputs

    model, spec = _tiny_tower(name)
    rng = np.random.default_rng(2)
    imgs = [rng.integers(0, 256, (90, 140, 3), dtype=np.uint8) for _ in range(2)]
    inputs = model_inputs(spec, imgs, "cpu", torch.float32)

    with torch.inference_mode():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
        pooled = _pool(model, spec, inputs)

    channels = [hidden[i].shape[1] for i in spec.stages]
    assert pooled.shape == (2, 2 * sum(channels))
    assert pooled.shape[1] == spec.dim


@pytest.mark.parametrize("name", sorted(
    n for n, s in BACKBONES.items() if s.pool == POOL_SPATIAL_MS))
def test_spatial_pooling_emits_mean_then_std_per_stage_in_stage_order(name):
    """Recomputes both moments by hand from the same hidden states and matches
    them slot for slot, so a swapped pair, a dropped stage or a stage read in
    the wrong order all fail here rather than in a bank six hours later."""
    import torch

    from aigcdet.features.backbones import _pool, model_inputs

    model, spec = _tiny_tower(name)
    rng = np.random.default_rng(3)
    imgs = [rng.integers(0, 256, (110, 90, 3), dtype=np.uint8) for _ in range(2)]
    inputs = model_inputs(spec, imgs, "cpu", torch.float32)

    with torch.inference_mode():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
        pooled = _pool(model, spec, inputs).numpy()

    at = 0
    for stage in spec.stages:
        flat = hidden[stage].flatten(2).float().numpy()      # (B, C, H*W)
        c = flat.shape[1]
        np.testing.assert_allclose(pooled[:, at:at + c], flat.mean(-1),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(pooled[:, at + c:at + 2 * c], flat.std(-1),
                                   rtol=1e-5, atol=1e-6)
        at += 2 * c
    assert at == pooled.shape[1]


def test_the_std_half_carries_signal_the_mean_half_discards():
    """The justification for this pooling, made falsifiable.

    The ViT path averages over tokens, so two images whose feature maps share a
    per-channel mean but differ in spatial VARIANCE -- flat versus textured --
    pool to the same vector and are indistinguishable downstream. If the std
    half were dropped (or were a constant, or a duplicate of the mean), this
    project's stated reason for adding a conv tower would be false. Uses a
    stand-in tower so the property is tested, not a checkpoint's luck.
    """
    import torch
    from dataclasses import replace

    from aigcdet.features.backbones import _pool

    class TwoStageMap(torch.nn.Module):
        """Emits a fixed pair of feature maps, ignoring its input."""
        def __init__(self, maps):
            super().__init__()
            self.maps = maps

        def forward(self, pixel_values=None, output_hidden_states=False):
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=self.maps[-1],
                                   hidden_states=tuple(self.maps))

    # Two 1-channel 2x2 maps with IDENTICAL means (0.5) and different spreads.
    flat = torch.full((1, 1, 2, 2), 0.5)
    textured = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    spec = replace(BACKBONES["resnet50"], dim=2, stages=(0,))

    a = _pool(TwoStageMap([flat]), spec, {})
    b = _pool(TwoStageMap([textured]), spec, {})

    assert a[0, 0].item() == pytest.approx(b[0, 0].item())   # means agree
    assert a[0, 1].item() == pytest.approx(0.0)              # flat map, no spread
    assert b[0, 1].item() == pytest.approx(0.5)              # textured map
    assert not torch.allclose(a, b), (
        "mean+std pooling failed to separate a flat map from a textured one "
        "with the same mean -- the std half is not doing its job")


def test_spec_rejects_a_pooling_and_stages_mismatch():
    from aigcdet.features.backbones import BackboneSpec

    with pytest.raises(ValueError, match="stages"):
        BackboneSpec("bad", "x", 224, 64, 0, 1, pool=POOL_SPATIAL_MS)
    with pytest.raises(ValueError, match="stages"):
        BackboneSpec("bad", "x", 224, 64, 0, 1, pool=POOL_TOKENS, stages=(3,))
    with pytest.raises(ValueError, match="pool"):
        BackboneSpec("bad", "x", 224, 64, 0, 1, pool="mean_of_everything")
    with pytest.raises(ValueError, match="prefix tokens"):
        BackboneSpec("bad", "x", 224, 64, 2, 1, pool=POOL_SPATIAL_MS, stages=(3,))


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(
    n for n, s in BACKBONES.items() if s.pool == POOL_SPATIAL_MS))
def test_registry_dim_matches_the_published_architecture(name):
    """The tiny towers above discover `dim` from a toy width, which is what
    makes them hermetic and what makes them blind: they cannot catch a `dim`
    that is wrong for the REAL checkpoint. A wrong `dim` is not a crash -- it
    is a BankWriter allocating the wrong stride and an extraction that dies on
    its first batch, hours after the session was paid for. Config only; no
    weights are downloaded.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    if os.environ.get("AIGCDET_ALLOW_GPU_TESTS") != "1":
        pytest.skip("opt-in: set AIGCDET_ALLOW_GPU_TESTS=1 (downloads configs)")

    spec = BACKBONES[name]
    model = AutoModel.from_config(AutoConfig.from_pretrained(spec.hf_id)).eval()
    with torch.inference_mode():
        hidden = model(pixel_values=torch.zeros(1, 3, spec.image_size, spec.image_size),
                       output_hidden_states=True).hidden_states

    expected = 2 * sum(hidden[i].shape[1] for i in spec.stages)
    assert spec.dim == expected, (
        f"{name}: registry says dim={spec.dim}, but {spec.hf_id} at stages "
        f"{spec.stages} pools to {expected}")
