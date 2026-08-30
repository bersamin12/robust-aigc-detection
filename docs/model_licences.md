# Model weight licences

Recorded per spec §4.5 (model-weight provenance, same discipline as dataset provenance).
Licence text was read in full at the source below, not inferred from a model-card
summary line. Checked 2026-08-27/28; the two convolutional backbones and DINOv2 added
2026-08-30; the four backbone-probe candidates added 2026-08-31.

| Model | HF id | Licence | Source | Permits public repo + hackathon use? |
| --- | --- | --- | --- | --- |
| DINOv3 ViT-L/16 | facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3 License (custom, Meta) | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m ; full text https://ai.meta.com/resources/models-and-libraries/dinov3-license/ (mirrored at https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md, "Last Updated: August 19, 2025") | Yes |
| DINOv2 ViT-L/14 | facebook/dinov2-large | Apache License 2.0 | https://huggingface.co/facebook/dinov2-large (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-30; `gated: False`) | Yes |
| SigLIP2-L/16-384 | google/siglip2-large-patch16-384 | Apache License 2.0 | https://huggingface.co/google/siglip2-large-patch16-384 (`license: apache-2.0` in card metadata) | Yes |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | MIT License | https://github.com/openai/CLIP (`LICENSE` file; the HF mirror at https://huggingface.co/openai/clip-vit-large-patch14 carries no `license:` tag of its own but ships the same weights as the MIT-licensed OpenAI repo) | Yes |
| ConvNeXt-Tiny-224 | facebook/convnext-tiny-224 | Apache License 2.0 | https://huggingface.co/facebook/convnext-tiny-224 (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-30; ungated) | Yes |
| ResNet-50 | microsoft/resnet-50 | Apache License 2.0 | https://huggingface.co/microsoft/resnet-50 (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-30; ungated) | Yes |
| SD 1.5 VAE | stable-diffusion-v1-5/stable-diffusion-v1-5 (vae) | CreativeML Open RAIL-M | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (`license: creativeml-openrail-m` in card metadata); full text https://huggingface.co/spaces/CompVis/stable-diffusion-license | Yes |
| LPIPS (AlexNet) | richzhang/PerceptualSimilarity | BSD-2-Clause | https://github.com/richzhang/PerceptualSimilarity (`LICENSE` file) | Yes |

### Backbone-probe candidates (added 2026-08-31)

Four towers registered to be RANKED on the 20,000-row union probe, not to ship.
Licence and gating for each were read via the HuggingFace Hub API on 2026-08-31.
All four are **ungated**, so no arm of the probe needs a token.

| Model | HF id | Licence | Source | Permits public repo + hackathon use? |
| --- | --- | --- | --- | --- |
| DINOv2-with-registers ViT-L/14 | facebook/dinov2-with-registers-large | Apache License 2.0 | https://huggingface.co/facebook/dinov2-with-registers-large (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-31; `gated: False`) | Yes |
| EVA-02 ViT-L/14 @448 | timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | MIT License | https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k (`license: mit` in card metadata AND in the checkpoint's own `pretrained_cfg`, read 2026-08-31; `gated: False`) | Yes |
| ConvNeXt V2 Huge @384 | facebook/convnextv2-huge-22k-384 | Apache License 2.0 (card) / MIT (upstream repo) | https://huggingface.co/facebook/convnextv2-huge-22k-384 (`license: apache-2.0`, read via the Hub API 2026-08-31; ungated) and https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE (`MIT License`, read in full 2026-08-31) | Yes |
| SigLIP SO400M/14 @384 | google/siglip-so400m-patch14-384 | Apache License 2.0 | https://huggingface.co/google/siglip-so400m-patch14-384 (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-31; `gated: False`) | Yes |

## Notes on each verdict

- **DINOv3 License.** Read the full `LICENSE.md` text (not the model-card summary).
  It is a non-exclusive, worldwide, royalty-free grant to use, reproduce, distribute,
  and create derivative works — no non-commercial or research-only restriction.
  Conditions that bind this project: (1) if we publish results of research performed
  using DINOv3, we must acknowledge the use of "DINO Materials" in the publication —
  satisfied by this file plus attribution in the project README/report; (2) distributing
  the weights or derivative *materials* to a third party requires including a copy of
  this agreement — does not apply to us, since we only load the weights via the HF Hub
  at runtime and never redistribute them; (3) no reverse engineering of the underlying
  components; (4) compliance with Trade Controls/ITAR and a prohibition on military,
  nuclear, espionage, or weapons end-uses — none of which apply to forensic AIGC
  detection. Access is gated (requires accepting the licence on an HF account before
  downloading), which is an access-control step, not a use restriction. Verdict: usable
  as the primary backbone; no need to fall back to SigLIP2/DINOv2 per the brief's
  contingency.
- **DINOv2.** Apache-2.0 and ungated, which is the entire reason it was added on
  2026-08-30. DINOv3 is the strongest backbone we have measured, but its custom
  Meta licence is gated per ACCOUNT: a five-person fleet needs five acceptances,
  and any submission that depends on those weights inherits terms Apache-2.0 does
  not impose. DINOv2 is the same self-supervised lineage without that condition.
  Whether it retains DINOv3's accuracy is an open question, not an assumption --
  see the backbone ladder.

- **SigLIP2.** Apache-2.0 is unconditionally permissive for this use.
- **CLIP ViT-L/14.** The HF mirror's model card omits a `license:` metadata tag, but
  the weights are the same ones distributed in OpenAI's `openai/CLIP` GitHub repository,
  which carries an MIT `LICENSE` file. Treating the HF mirror as MIT-licensed is the
  standard reading used throughout the ecosystem (e.g. `openai/clip-vit-base-patch32`
  and siblings all ship the same code/weights under that repo's licence). Note the
  ambiguity this reading carries: the MIT text speaks of "the Software" and never
  explicitly names trained model weights, so applying a source-code licence to a
  checkpoint is an ecosystem-standard practice, not a textually settled one.
- **ConvNeXt-Tiny and ResNet-50.** Apache-2.0 in both cases, unconditionally
  permissive for this use, and neither is gated — no HuggingFace token is needed
  for a run that uses only these, which is why `notebooks/kaggle_stage_a_cnn.ipynb`
  has no auth step to fail on. Both are **generic ImageNet classifiers**, not
  AIGC detectors: the competition rules disqualify "using pre-trained AIGC
  detection models directly", and a general-purpose vision backbone frozen as a
  feature extractor is the same relationship this project already has with
  DINOv3, SigLIP2 and CLIP. Nothing forensic is inherited from either checkpoint.
- **SD 1.5 VAE.** CreativeML Open RAIL-M permits commercial and non-commercial use but
  attaches use-based restrictions (Attachment A) prohibiting specific harmful end-uses
  (e.g. generating illegal, discriminatory, or non-consensual content). This project
  uses only the VAE encoder as a frozen feature extractor for forensic reconstruction-
  error features (Task 4); it does not generate images with the diffusion model, so none
  of the prohibited end-uses apply.
- **LPIPS (AlexNet weights).** BSD-2-Clause is unconditionally permissive.

- **DINOv2-with-registers.** Apache-2.0 and ungated, same terms as plain DINOv2.
  It is in the registry for an architectural reason rather than a licence one:
  registers absorb the high-norm artefact tokens DINOv2 develops in
  low-information patches, which is where a generator's decoder leaves its trace.
- **EVA-02.** MIT, and ungated. `handoffs/08-ablation-rungs.md` recorded
  `timm/eva02_*` earlier as a licence note about a candidate that was NOT
  adopted; it is adopted now because the backbone probe is asking a different
  question. The earlier note was about shipping it; this is about measuring it.
  It is the only entry loaded through timm rather than transformers directly
  (`AutoModel` resolves a `timm/*` repo to `TimmWrapperModel`), which is why
  `timm>=1.0` became a declared dependency in `pyproject.toml`.
- **ConvNeXt V2 Huge. TWO PERMISSIVE READINGS, AND THEY DISAGREE ON WHICH.** The
  HF card metadata says `apache-2.0`; the upstream `facebookresearch/ConvNeXt-V2`
  repository ships an `MIT License` file. Both were read in full on 2026-08-31.
  Neither carries a non-commercial or research-only restriction, so the
  divergence does not change the verdict and nothing is blocked — but it is
  recorded rather than resolved by picking the more convenient one, because the
  discipline in this file is that the licence is read, not assumed. Note the
  contrast with ConvNeXt V1 (`facebook/convnext-tiny-224`), whose card and
  upstream agree on Apache-2.0.
- **SigLIP SO400M.** Apache-2.0, unconditionally permissive, ungated.
- **All four are general-purpose vision backbones**, frozen as feature
  extractors — the same relationship this project already has with DINOv3,
  SigLIP2 and CLIP. The competition rules disqualify "using pre-trained AIGC
  detection models directly"; nothing forensic is inherited from any of these
  checkpoints.

## The 2B parameter cap and this file

The registry now holds ten entries summing to ~2.97B parameters, which is NOT a
breach. The hackathon's constraint binds the architecture that SHIPS — the spec's
wording is "Final model uses at most two backbones", and its exclusions name "any
model at or above 2B parameters" — not the menu of candidates an ablation may
consider. The test that enforces it
(`tests/features/test_backbones.py::test_the_heaviest_shippable_configuration_stays_under_2b`)
sums the two heaviest entries plus the SD 1.5 VAE and LPIPS: 1.17B, with 828M of
margin. It summed the whole registry until 2026-08-31, which would have vetoed a
four-backbone probe that ships none of the four.

None of the models above blocks public-repo or hackathon use. The registry in
`src/aigcdet/features/backbones.py` ships DINOv3 ViT-L/16 as planned, with
SigLIP2-L/16-384 and CLIP ViT-L/14 as the other two entries; the rest are
ablation candidates.
