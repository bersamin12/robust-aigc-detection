# Model weight licences

Recorded per spec §4.5 (model-weight provenance, same discipline as dataset provenance).
Licence text was read in full at the source below, not inferred from a model-card
summary line. Checked 2026-08-27/28; the two convolutional backbones added 2026-08-30.

| Model | HF id | Licence | Source | Permits public repo + hackathon use? |
| --- | --- | --- | --- | --- |
| DINOv3 ViT-L/16 | facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3 License (custom, Meta) | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m ; full text https://ai.meta.com/resources/models-and-libraries/dinov3-license/ (mirrored at https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md, "Last Updated: August 19, 2025") | Yes |
| SigLIP2-L/16-384 | google/siglip2-large-patch16-384 | Apache License 2.0 | https://huggingface.co/google/siglip2-large-patch16-384 (`license: apache-2.0` in card metadata) | Yes |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | MIT License | https://github.com/openai/CLIP (`LICENSE` file; the HF mirror at https://huggingface.co/openai/clip-vit-large-patch14 carries no `license:` tag of its own but ships the same weights as the MIT-licensed OpenAI repo) | Yes |
| ConvNeXt-Tiny-224 | facebook/convnext-tiny-224 | Apache License 2.0 | https://huggingface.co/facebook/convnext-tiny-224 (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-30; ungated) | Yes |
| ResNet-50 | microsoft/resnet-50 | Apache License 2.0 | https://huggingface.co/microsoft/resnet-50 (`license: apache-2.0` in card metadata, read via the Hub API 2026-08-30; ungated) | Yes |
| SD 1.5 VAE | stable-diffusion-v1-5/stable-diffusion-v1-5 (vae) | CreativeML Open RAIL-M | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (`license: creativeml-openrail-m` in card metadata); full text https://huggingface.co/spaces/CompVis/stable-diffusion-license | Yes |
| LPIPS (AlexNet) | richzhang/PerceptualSimilarity | BSD-2-Clause | https://github.com/richzhang/PerceptualSimilarity (`LICENSE` file) | Yes |

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

None of the four backbone/auxiliary models block public-repo or hackathon use. The
registry in `src/aigcdet/features/backbones.py` therefore ships DINOv3 ViT-L/16 as
planned, with SigLIP2-L/16-384 and CLIP ViT-L/14 as the other two entries.
