# Related work and novelty check: FOV / crop-vs-squish in AIGC detection

Search date: 2026-10-12. Sources: arXiv full texts (PDFs downloaded and grepped with pdftotext), CVF open access, GitHub repos/code, Hugging Face discussions, web search.

**Evidence labels used below**
- **[FT]** = verified by reading the relevant passage in the paper's full text (quotes are verbatim from the extracted PDF text; minor hyphenation from PDF extraction removed).
- **[CODE]** = verified from the official code or README.
- **[SNIP]** = from an abstract, venue page, or search snippet only; not checked against the full text.

---

## Verdict

**Partially anticipated. The remedy is not new. The diagnosis and its controlled quantification appear to be new.** No paper I found names the field of view (crop side / image short side) as a train-to-test covariate shift in AIGC detection, or shows that it, rather than the photometric degradations, limits robust AUC. Three bodies of work come close, and the paper has to credit all three:

1. **The general principle is old.** FixRes (Touvron et al., NeurIPS 2019) says preprocessing must standardise "apparent object size" because "networks do not have built-in scale invariance". The FOV argument is this principle applied to the forensic habit of taking fixed-pixel crops at native resolution.
2. **The closest forensic statement is Grommelt et al. (ECCVW 2024, "Fake or JPEG?").** It says in one sentence that "Detectors employing cropping to achieve a uniform input size are still subject to bias in terms of object size." Its inference pipeline (resize to 512, crop 450, resize to 224) also normalises FOV. However, it treats size as a label-correlated *dataset shortcut*. It does not treat it as a train/test covariate shift, and it never measures robustness or the crop-vs-resize choice.
3. **The fix is already used, without analysis.** In the same NTIRE 2026 challenge, the 1st-place team (MICV) resizes the whole image at inference "to preserve the global spatial context". The 5th-place team (Vincentlc) calls its preprocessing "Squish" ("ignoring aspect ratio"). Neither runs an ablation or names a mechanism. Separately, AI-GenBench (IJCNN 2025) reports that whole-image resize "is often preferable" to multi-crop for CLIP and DINOv2 ViT-L/14.

Against this, the dominant forensic position (Corvi 2023, SAFE, B-Free, SPAI, TextureCrop, HiDA-Net, Konstantinidou 2026) is that resizing destroys traces. One 2026 paper (GlobalForge) reports the *opposite* cross-preprocessing result: crop-trained UnivFD loses about 9 BAcc points when tested with resize. The paper must engage that result directly.

The two dataset-leak observations were not reported anywhere I could find:
- the narrow short-side band of the resized RAISE reals in Synthbuster;
- the JPEG-vs-PNG and size confound in AIGIBench CommunityAI.

The RAISE resizing itself *is* documented by B-Free, where it was a deliberate choice to make the image area about 1024x1024. A related Synthbuster format leak is also public: the Firefly ".png" files are actually JPEG (GitHub issue #1, cited by BIAS-ID).

---

## Q1. Has FOV / scale / crop-vs-resize preprocessing been identified as a covariate shift limiting AIGC detection robustness?

**Short answer: not as a covariate shift.** Several papers touch the ingredients:
- object size under cropping (Grommelt);
- size-dependent preprocessing loss (TextureCrop);
- resolution and resampling shortcuts (Grommelt, Rajan, BIAS-ID);
- resize-vs-crop tables (AIGIBench, AI-GenBench, GlobalForge, Chu et al.).

None of them treats crop-side/short-side as the variable that changes between training and test.

### Closest conceptual precursor (general vision)
**FixRes — Touvron, Vedaldi, Douze, Jégou, "Fixing the train-test resolution discrepancy", NeurIPS 2019, arXiv:1906.06423. [FT]**
- Abstract: "existing augmentations induce a significant discrepancy between the size of the objects seen by the classifier at train and test time".
- Sec. 3: "preprocessing standardizes the apparent size, which otherwise would depend on the input image resolution. This is important as networks do not have built-in scale invariance."
- *Overlap:* the same mechanism (apparent size / FOV mismatch between train and test preprocessing). *Not covered:* forensics, native-resolution fixed-pixel crops, degradations. Cite this as the conceptual root of the FOV argument.

### Closest forensic statement
**Grommelt, Weiss, Pfreundt, Keuper, "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets", arXiv:2403.17608 (ECCV 2024 Workshops per mlanthology [SNIP]). [FT]**
- "Furthermore, for non-square images, resizing results in significant information loss along one axis, leading to frequency artifacts. However, this problem extends beyond detection methods that rely solely on resizing as a preprocessing step. **Detectors employing cropping to achieve a uniform input size are still subject to bias in terms of object size.**"
- Fix (constrained setting): "we center-cropped images to 450 - the lower bound of the size interval - before resizing to 224 ... During inference, for images of any size, we first resize the input to 512 to allow cropping to 450 and then resizing." This pipeline holds FOV roughly constant.
- *Overlap:* the object-size consequence of cropping, and a FOV-normalising inference pipeline. *Not covered:* framing as train/test covariate shift, robustness or degradations, any crop-vs-resize ablation, ViT backbones. Their concern is a size shortcut correlated with the label, which is the opposite of the "size not predictive of label" control in our work.
- *Caution for us:* the sentence about non-square resizing ("information loss along one axis ... frequency artifacts") is a direct objection to *squish* (aspect discarded), as opposed to whole-image resizing in general.

### NTIRE 2026 Robust AIGC Detection report — what top teams did
**Gushchin et al., "NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild", CVPRW 2026, arXiv:2604.11487. [FT]**

| Rank | Team | Robust AUC (avg) | Preprocessing (verbatim from report) |
|---|---|---|---|
| 1 | MICV (DINOv3 committees) | 0.9723 | Train: "images are randomly cropped and resized to 512 × 512 pixels". **Inference: "images are directly resized to 512 × 512 to preserve the global spatial context."** |
| 2 | Ant International (2× DINOv3-7B) | 0.9721 | "Input resolution: 512 × 512" (Expert 1), "288 × 288" (Expert 2); inference "Test-time augmentation (TTA): each expert predicts on multiple augmented variants". The TTA views are not specified. |
| 3 | TeleAI-TeleGuard (EVA-CLIP + LoRA) | 0.9251 | Not specified in the report. |
| 4 | INTSIG (DINOv3-H, MetaCLIP2-G) | 0.9130 | "random resized crop (256 × 256, bicubic, scale (0.08, 1.0), ratio (0.75, 1.3333))"; Model 4 "448 × 448"; Model 5 "Inputs are resized to 378 × 378". |
| 5 | Vincentlc (SigLIP2-giant) | 0.8730 | **"Spatial Transforms ('Squish' Strategy). Instead of random resized cropping (which may remove localized forensic cues), we apply a 'squish' strategy: all images are directly resized to 384 × 384 (ignoring aspect ratio) ... This preserves full image content".** Fig. 5 also says "Resize (Squish) to 384x384". |

Other relevant facts in the report [FT]:
- **Size alignment.** "we also align its distributions of resolutions, aspect ratios, JPEG compression quality factors, and other statistics to those of the real subset." This is consistent with our finding that native size is not predictive of label in training (AUC 0.53).
- **Geometric transforms appear only after the train split** (Table 6). The train split's 12 transforms are photometric, blur, noise, and JPEG only. Validation adds "Random Crop", "Random Aspect Crop", "Downscale". Validation-hard and test add "Perspective Transform", among others. **The geometric degradations that create the FOV/size shift are themselves part of the challenge's degradation set.** See "Framing" below.
- No team runs an ablation of crop vs resize, and the report does not discuss it.

### Other papers on crop-vs-resize, size, and preprocessing (forensics)

**AI-GenBench — Pellegrini, Cozzolino, Pandolfini, Maltoni, Ferrara, Verdoliva, Prati, Ramilli, IJCNN 2025, arXiv:2504.20865. [FT]**
- Compares "i) resize, where images are resized to a fixed size during both training and evaluation; and ii) multi-cropping, where a single random crop is used during training, while five crops ... are employed during evaluation".
- "The resize strategy better preserves the semantic content of the images, at the cost of altered resolution and aspect ratio" (so aspect ratio is discarded).
- Conclusion: "**iii) resize is often preferable with respect to multi-crop.**"
- Table III (Next Period, AUROC resize / multicrop): ViT-L/14 DINOv2 tune 94.24 / 93.44; probe 88.47 / 88.74. ViT-L/14 CLIP tune 92.04 / 87.66; probe 88.47 / 68.74.
- *Overlap:* whole-image squish ≥ crops for exactly our backbone family. *Not covered:* a test-time-only swap, FOV mechanism, degradations, size stratification.

**AIGIBench — Li, Yan, He, Zeng, Jiang, Xiong, Fu, "Is Artificial Intelligence Generated Image Detection a Solved Problem?", NeurIPS 2025 Datasets & Benchmarks, arXiv:2505.12335. [FT]**
- Task 4 "Test Data Pre-processing Assessment": Resize vs Crop on six detectors. Finding: "the performance gains from Crop are predominantly attributed to improvements in R.Acc., while F.Acc. remains largely unchanged or even declines". Crop "retains high-frequency local features ... thereby enhancing the model's ability to recognize real content".
- Table 6 (Setting-II): **the one CLIP-based detector, "CLIPDetection" (= Ojha et al. UnivFD, their ref. [19]), is better with Resize**: Acc/AP 72.5/75.6 (Resize) vs 66.5/68.4 (Crop). All five low-level detectors are better with Crop. The authors do not comment on this exception.
- Resize/crop target sizes are not stated in the text.

**TextureCrop — Konstantinidou, Koutlis, Papadopoulos, WACV 2025 Workshops, arXiv:2407.15500. [FT]** (venue from CVF URL [SNIP])
- "resizing entails image resampling and interpolation, which may erase the subtle high-frequency traces ... and center cropping an image to a fraction of its original size can lead to significant loss of information ... When detectors are trained and evaluated on small images, resizing or cropping has little effect on their performance ... but when dealing with images of higher resolution, this effect is much more pronounced."
- This is the closest *forensic* acknowledgment that preprocessing loss grows with native size. It is framed as information loss, not as a FOV distribution shift.
- Test-only plug-in on ≥1024 px images: TextureCrop +14.9 AUC vs Resizing and +6.1 vs CenterCrop on average. **For the CLIP detectors, CenterCrop > Resizing** (AUC: UnivFD 70.31 vs 66.48; RINE(4) 73.44 vs 69.21; RINE(LDM) 81.90 vs 68.05). Exception: "the DALL-E 3 images from Synthbuster, where Resizing seems to lead to slightly better results."

**WaRPAD — Choi, Lee, Lee, "Training-free Detection of AI-generated images via Cropping Robustness", NeurIPS 2025, arXiv:2511.14030. [FT]**
- "RescaleNPatchify(x) = Patchify(Rescale(x, d_rescale), d_patch) ... Rescale(x, d) is the image rescaling operation to dimension d × d". This is a whole-image *square* resize (aspect discarded) to 896 or 1344, then 224 tiles.
- Because the image is resized to a fixed canvas before tiling, **each patch has a constant FOV regardless of native size**. The paper motivates this as simulating RandomResizedCrop: "deterministically simulate multiple instances of RRC". It never mentions FOV or covariate shift.
- Ablation: RIGID + RescaleNPatchify only 0.587 → 0.656 AUROC on Synthbuster.

**SPAI — Karageorgiou, Papadopoulos, Kompatsiaris, Gavves, "Any-Resolution AI-Generated Image Detection by Spectral Learning", CVPR 2025, arXiv:2411.19417. [FT]**
- "a resizing operation, that effectively discards the high-frequency information of the image, is problematic"
- "pre-processing operations are required, effectively dropping a significant amount of discriminative information"
- Notably: "**any resizing or cropping approach would introduce unwanted biases**" (Sec. 3, augmented-views training).
- Ablation "w/o SCA + TenCrop (mean)" 85.3 vs full 91.0 AUC.
- Reports results by image-size bins but has no FOV analysis.

**B-Free — Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva, CVPR 2025, arXiv:2412.17671. [FT]**
- Uses DINOv2+reg (same backbone family as ours). "During training, we avoid resizing the image and rely on large crops of 504 × 504 pixels. At inference time, we also extract crops of the same dimension ... (if the image is larger we average the results of multiple crops)."
- "the model avoids resizing operations that can be particularly harmful by washing out the subtle low-level forensic clues".
- Training reals: "we first extracted the largest central crop and resize it to 512 x 512" [CODE/README; also FT appendix]. So training FOV is about 1, while test-time crops on large images have small FOV. This latent mismatch is not discussed.

**Rajan, Ojha, Schloesser, Lee, "Aligned Datasets Improve Detection of Latent Diffusion-Generated Images", ICLR 2025, arXiv:2410.11835. [FT]**
- With RandomResizedCrop and real/fake size discrepancy, "the random resized crop introduces different signals to each distribution". The Corvi detector "changes its prediction if the same image is saved at a different resolutions".
- Relevant to the "size not predictive" control. It concerns resampling artifacts, not FOV.

**BIAS-ID — Ricker, Fischer, Quiring, arXiv:2605.31153 (May 2026). [FT]**
- Score-shift analysis: "UnivFD, SPAI, and B-Free show a tendency for classifying downscaled images as real."
- Also see Q3.

**Chu et al., "When Semantics Regulate: Rethinking Patch Shuffle and Internal Bias for Generated Image Detection with CLIP", arXiv:2511.19126. [FT]**
- On GenImage with Swin-T: "cropping without resizing consistently yields higher or comparable accuracy relative to direct resizing", attributed to interpolation disrupting low-level statistics.
- Confounded by GenImage's fixed generator sizes (Grommelt).

**GlobalForge — Cui, Liu, Zou, Qin, Xu, Wang, Wei, et al., arXiv:2607.14684 (Jul 2026). [FT]** See Q2. This is the most direct *contrary* evidence.

**Zhou & Wang, "How Fragile Are Training-Free AI-Generated Image Detectors? ...", arXiv:2606.20488 (Jun 2026). [FT]**
- "switching between resize-to-512 and native-resolution preprocessing flips per-generator conclusions by up to 0.38 AUROC". Recommends that papers "report backbone, resolution policy, and crop policy as first-class experimental variables".
- Covers training-free scores only.

**TAP into the Patch Tokens — Abdullah, Ebert, Wasenmüller, arXiv:2604.26772. [FT]**
- "For variable-resolution models, increasing the input image resolution rarely correlates with increased performance. On the contrary, we observe a decline in accuracy in the cases of DINOv2 and DINOv3 when scaling up input image sizes." No mechanism given.
- Tangential, but consistent with semantic ViTs not benefiting from more native pixels.

**Searched with nothing closer found:** "field of view" + detection, "covariate shift" + image size, "scale invariance deepfake detector", "squish" + detection, "multi-scale crops generated image detection 2025/2026".
- The NTIRE 2026 *deepfake* report (arXiv:2604.24163 [SNIP]) lists top-team preprocessing (e.g., ShallowReal random resolution in [384, 1152]) with no FOV analysis.
- "Look Closer" (arXiv:2609.37937 [SNIP]), GLASS (arXiv:2601.00141 [FT]) and HiDA-Net (Q2) combine a resized global view with local crops but do not frame FOV as a shift.

---

## Q2. Is "resize the whole image instead of cropping" reported as better, worse, or harmful?

### Strongest "resizing destroys artifacts" statements (must be engaged)
- **Corvi et al., ICASSP 2023, arXiv:2211.00680 [FT]:** "Another central discovery concerns the need to avoid any loss of information in the pre-processing of training and test images ... **Most of all, it is important to avoid resizing**, a common practice in deep learning to adapt images to fixed input layers, as this entails image resampling and interpolation, which may erase the subtle high-frequency traces left by the generation process." Proposed remedies: "training the networks on local patches, cropped from the image with no resizing ... making the final decision on the whole image through some fusion strategy".
- **SAFE — Li, Cai, Hao, Jiang, Hu, Feng, KDD 2025, arXiv:2408.06741 [FT]:** the down-sampling operator "can inadvertently smooth the noticeable local correlations in synthetic images and thereby weaken those subtle discriminative artifacts ... we propose to replace the down-sampling operator with the crop operator, with RandomCrop at training and CenterCrop at inference" (256²). Ablation conclusion: "The resize operator indeed diminishes the subtle artifact features and degrades the detection performance".
  - *Useful for us:* "The similar performance between CC, RC, and SI [source image, full native size] during testing indicates that our pipeline can achieve accurate detection through center-cropped regions only". So **for a low-level CNN detector, FOV does not matter**. This supports scoping our claim to semantic ViT backbones.
- **B-Free [FT]:** "avoids resizing operations that can be particularly harmful by washing out the subtle low-level forensic clues".
- **SPAI [FT]:** "a resizing operation, that effectively discards the high-frequency information of the image, is problematic".
- **TextureCrop [FT]:** resizing "may erase the subtle high-frequency traces left by the generation process". It is 14.9 AUC worse than TextureCrop on ≥1024 px images, including for CLIP detectors.
- **HiDA-Net — Mu, Zou, Bai, Hu, Zheng, Ye, Zhuang, Ali, Wang, Hu, "No Pixel Left Behind: A Detail-Preserving Architecture for Robust High-Resolution AI-Generated Image Detection", arXiv:2508.17346; ICLR 2026 per iclr.cc listing [SNIP] (the arXiv PDF uses an AAAI template). [FT]**
  - "resizing introduces a strong low-pass effect, irreversibly erasing the [high-frequency cues]".
  - Global-only branch: "the accuracy on the GenImage benchmark drops significantly to 84.24%" vs 96.10% with full-coverage tiles (frozen CLIP ViT-L/14, clean data).
- **Konstantinidou, Karageorgiou, Koutlis, Papadopoulou, Schinas, Papadopoulos, "Navigating the Challenges of AI-Generated Image Detection in the Wild: What Truly Matters?", ACM MAD '26 workshop, arXiv:2507.10236 [FT]:** "With resizing being considered detrimental for AID—due to its tendency to erase subtle high-frequency traces left by the generation process—cropping techniques become essential". Uses DINOv2-L/14. Compares center vs texture crops only and never tests resize.
- **GLASS — Han, arXiv:2601.00141 [FT]:** resizing "can inadvertently discard fine-grained details produced by the image generator". However, it also concedes "a high classification accuracy was achieved even with standard transfer learning using only the resized image".

### Direct counter-evidence to our result
**GlobalForge, arXiv:2607.14684 [FT]:**
- "Existing AIGI detectors almost universally adopt center crop rather than resize at inference time [8, 9, 15], because resize causes a substantial drop in detection accuracy."
- "A model trained with center crop loses about 8.8% of BAcc when tested with resize, while a model trained with resize barely degrades when tested with resize at all." Setup: Fig. 2, "UnivFD on GenImage".
- Fig. 2 cells as extracted: Test-Resize 67.50 / 58.70; Test-Crop 67.10 / 67.90. My reading of the column order (Train-Resize | Train-Crop) is **inferred** from the text, so check the figure before quoting.
- **This is the opposite of our "test-time squish on crop-trained checkpoints helps" result.** Plausible reasons it differs, which should be argued explicitly:
  1. their test set is clean GenImage, so high-frequency cues are intact and the most useful;
  2. UnivFD is a linear probe on frozen CLIP, while ours is DINOv2-L+reg fine-tuned;
  3. GenImage reals (ImageNet, ~450 px) vs fixed-size fakes make the FOV range narrow, and resizing interacts with the known size bias (Grommelt).
  
  Under the NTIRE degradations (JPEG, blur, neural codecs, downscale) the high-frequency cues are already damaged, and FOV becomes the dominant shift.

### Reports where whole-image resize is better or competitive
- **AI-GenBench [FT]:** "resize is often preferable with respect to multi-crop" (CLIP and DINOv2 ViT-L/14; numbers above).
- **AIGIBench Table 6 [FT]:** UnivFD is better with Resize than with Crop (the only semantic detector tested).
- **NTIRE 2026 [FT]:** 1st place resizes at inference "to preserve the global spatial context"; 5th place uses "Squish".
- **Cozzolino et al. CLIP detector, CVPRW 2024, arXiv:2312.00195 [CODE]:** the released `Clip224` config does `Resize(224, interpolation=BICUBIC)` then `CenterCrop((224, 224))`. This is a short-side resize, i.e. near-whole-image, aspect preserved. Paper [FT]: "the proposed CLIP-based detector keeps working well under these conditions [resizing/compression], suggesting that it relies on higher-level semantic features."
- **WaRPAD [FT]:** whole-image square rescale before patching improves Synthbuster AUROC.
- **TextureCrop [FT]:** the DALL-E 3 subset of Synthbuster is an exception where resizing is slightly better.

### UnivFD preprocessing, for the record [CODE]
- Training (`data/datasets.py`): `custom_resize` to `loadSize=256` (short side), then `RandomCrop(224)`.
- Test (`validate.py`): `transforms.CenterCrop(224)` with **no resize**.
- So UnivFD's train FOV is about 224/256 of the short side, while its test FOV is 224/native-short-side: a built-in FOV mismatch on large test images. The paper text does not describe this preprocessing; it is from code only.

**Summary for Q2:** the field's consensus is "crop, don't resize", based on low-level/high-frequency reasoning. Evidence that resize helps exists but is scattered: AI-GenBench, AIGIBench's UnivFD row, and NTIRE practice. One 2026 paper (GlobalForge) explicitly reports test-time resize *hurting* a crop-trained CLIP probe on clean data. Our result needs to be scoped to (a) semantic ViT backbones and (b) degraded test data with a wide FOV range, and it should engage GlobalForge and TextureCrop's CLIP rows head-on.

---

## Q3. Synthbuster/RAISE size uniformity; AIGIBench CommunityAI format/size leakage

### Synthbuster / RAISE
- **Original Synthbuster (Bammey, IEEE OJSP vol. 5, 2024 [SNIP]) does not ship the reals.** README [CODE]: "download raise-1k from RAISE". Zenodo record [SNIP]: "Raise-1k images are not included in the dataset". Full-size RAISE is roughly 4928×3264 (range quoted in WaRPAD [FT]: "Varying (256×256 ∼ 4928×3264)").
- **The narrow size band comes from B-Free's redistribution.** B-Free README [CODE]: extended Synthbuster includes "1K images from **RAISE**, which we resized to a dimension so that the area is about 1024x1024". For a 3:2 RAISE frame, that gives about 1254×836, consistent with your 833–835 px short side. **Verify which copy you used.** If it is B-Free's extended Synthbuster, say so and attribute the resizing to them.
- **BIAS-ID (Ricker et al. 2026) [FT]** uses this copy on purpose: "To align the resolution of real and fake images, we use the resized version of RAISE-1k provided by [26] [B-Free]. Our comparison of original and resized images in Appendix D shows that this does not significantly change the results."
- **Related, already-public Synthbuster leak.** BIAS-ID footnote [FT]: "We exclude the Firefly subset since, despite being saved as PNGs, they were found to be JPEGs, see https://github.com/qbammey/synthbuster/issues/1". GitHub issue #1 [CODE] (open, 2025-12-03): "even though all fake images have the `.png` extension, they aren't all PNG internally. This only affects Firefly generated images."
- **Not found anywhere:** the observation that resized-RAISE reals occupy a ~3 px short-side band, so a *non-monotone* size rule nearly separates labels while the monotone size-AUC stays ≈0.52. As far as I can tell this point is new. Frame it as a side effect of a deliberate, documented alignment choice, not as a hidden error.

### AIGIBench CommunityAI
- Paper [FT]: "CommunityAI, is compiled from AI art communities including ArtStation, Civitai, and Liblib."
- Real images: "sampled from three sources: FFHQ, CelebA-HQ, and Open Images V7". The appendix separately says SocialRF reals were crawled via "#nature, #photography, and #realphoto".
- The paper says nothing about file format or resolution for these subsets.
- The GitHub README [CODE] corrects counts: "The test set count in the paper contained some errors"; it lists CommunityAI at 6000 and SocialRF at 3000, whereas the paper appendix says 4,500/4,500.
- No GitHub issue (4 total) or Hugging Face discussion (3 total) reports a format or size confound. HF discussion #2 confirms reals are non-overlapping across subsets.
- **Not found anywhere:** the CommunityAI JPEG-real / PNG-fake confound with size AUC 0.92. It appears new.
- **General precedent to cite:**
  - Grommelt et al. (JPEG and size bias in GenImage) [FT].
  - B-Free intro: "synthetic images are often created in a lossless format (PNG), while the latter are typically compressed in lossy formats like JPEG" [FT, paraphrase checked].
  - Chen et al. (DDA), arXiv:2505.14359: "real images are JPEG-encoded and vary in size, whereas synthetic images are uniformly PNG-encoded and fixed in size" [SNIP, from WebFetch of HTML; not re-verified in PDF].

---

## Q4. Test-time preprocessing alone (no retraining) giving large robustness gains

- **TextureCrop [FT]:** an explicitly plug-in, test-only component for pretrained detectors: "+6.1% [AUC] compared to center cropping and by 15% compared to resizing" on ≥1024 px images. This is clean high-res data, not degradation robustness.
- **AIGIBench Table 6 [FT]:** a test-time Resize→Crop swap on fixed checkpoints changes Acc by up to +15 points (SAFE 64.9→79.9) and by −6 for UnivFD.
- **GlobalForge Fig. 2 [FT]:** a test-time Crop→Resize swap costs a crop-trained UnivFD about 9 BAcc points. Large test-time sensitivity in the other direction.
- **Zhou & Wang 2026 [FT]:** a preprocessing switch moves per-generator AUROC by up to 0.38 (training-free scores).
- **WaRPAD [FT]:** RescaleNPatchify is a test-time procedure; the full method is training-free.
- **SAFE Fig. 7 [FT]:** "Plug & Play application with existing pipelines". Note that this changes *training* preprocessing, so it does not count as test-time only.
- **NTIRE 2026 [FT]:** Ant International uses test-time augmentation, but the views are unspecified and there is no ablation.
- **Not found:** a test-time preprocessing change on frozen checkpoints that yields a robust-AUC gain of +0.077 (0.886→0.963) under a degradation benchmark, *together with* the observation that retraining under the same preprocessing gives no extra benefit. The "no retraining needed" control is our strongest distinctive element.

---

## Framing recommendations (what should change)

1. **Do not present squish as new.** The NTIRE 2026 winner resizes the whole image at inference, and the 5th-place team literally calls it "Squish". AI-GenBench already reports resize ≥ multi-crop for DINOv2/CLIP ViT-L/14. Present the contribution as *diagnosis plus controls*: FOV is the uncontrolled covariate, it explains why whole-image inputs work, and it needs no retraining.
2. **Anchor the mechanism in FixRes** ("apparent object size"; "networks do not have built-in scale invariance"). Credit Grommelt's one-line "object size" remark as the forensic precursor. The phrase "pixel count is equalised but FOV is not" is a crisp statement of what both missed in the robustness setting.
3. **Re-word "covariate shift, not the degradations themselves."** In NTIRE 2026, "Random Crop", "Random Aspect Crop", "Downscale" (and "Perspective Transform") are absent from train and present in val/val-hard/test. So part of the FOV shift is *induced by* challenge degradations. Two improvements:
   - stratify the squish gain by transform family (geometric vs photometric), or report gain within images that received no geometric transform;
   - phrase the claim as "the geometric degradations act mainly through FOV; under FOV-normalised input the photometric ones cost little".
4. **Scope the claim to semantic ViT backbones and degraded inputs,** and engage the counter-evidence directly:
   - GlobalForge (test-time resize hurts crop-trained UnivFD on clean GenImage);
   - TextureCrop (CenterCrop > Resizing for UnivFD/RINE on ≥1024 px);
   - HiDA-Net (global-only branch 84% vs 96% with tiles);
   - SAFE's SI≈CC result, which shows low-level CNNs are FOV-insensitive and is useful support for the scoping.
5. **Defend aspect distortion separately.** Grommelt explicitly warns that non-square resizing causes "information loss along one axis, leading to frequency artifacts". Add an aspect-preserving FOV=1 control so the gain is attributed to FOV rather than to squish specifically: letterbox/pad to square, or short-side resize + center crop as in Cozzolino's `Clip224`.
6. **Add the leak findings as caveats for others, with attribution.** The RAISE resize to "area about 1024x1024" is B-Free's documented choice, which BIAS-ID adopted "to align the resolution". The new point is the non-monotone leak it creates. Also cite the known Firefly-JPEG Synthbuster issue as a precedent for format leaks in the same benchmark.
7. **Optional, cheap, and persuasive:** B-Free (DINOv2+reg) and UnivFD both have a built-in train-vs-test FOV mismatch:
   - B-Free trains on 512-px crops at FOV≈1 and tests by native-resolution multi-crop;
   - UnivFD trains with resize-256 + crop-224 and tests with a native center-crop-224.
   
   Showing a squish gain on a public checkpoint of either would generalise the claim beyond your own models.

---

## BibTeX

Fields marked `% UNVERIFIED` were not checked against the publisher page.

```bibtex
@inproceedings{touvron2019fixres,
  title     = {Fixing the train-test resolution discrepancy},
  author    = {Touvron, Hugo and Vedaldi, Andrea and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
  eprint    = {1906.06423},
  archivePrefix = {arXiv}
  % pages UNVERIFIED (venue confirmed by arXiv v4 footnote "publication of this paper at Neurips")
}

@inproceedings{grommelt2024fakeorjpeg,
  title     = {Fake or {JPEG}? Revealing Common Biases in Generated Image Detection Datasets},
  author    = {Grommelt, Patrick and Weiss, Louis and Pfreundt, Franz-Josef and Keuper, Janis},
  booktitle = {European Conference on Computer Vision (ECCV) Workshops},  % UNVERIFIED (from mlanthology listing only)
  year      = {2024},
  eprint    = {2403.17608},
  archivePrefix = {arXiv}
}

@article{gushchin2026ntire,
  title   = {{NTIRE} 2026 Challenge on Robust {AI}-Generated Image Detection in the Wild},
  author  = {Gushchin, Aleksandr and Abud, Khaled and Shumitskaya, Ekaterina and Filippov, Artem and Bychkov, Georgii and Lavrushkin, Sergey and Erofeev, Mikhail and others},
  journal = {arXiv preprint arXiv:2604.11487},
  year    = {2026},
  note    = {CVPR 2026 Workshops (NTIRE); CVF open access version exists}  % pages UNVERIFIED
}

@inproceedings{pellegrini2025aigenbench,
  title     = {{AI-GenBench}: A New Ongoing Benchmark for {AI}-Generated Image Detection},
  author    = {Pellegrini, Lorenzo and Cozzolino, Davide and Pandolfini, Serafino and Maltoni, Davide and Ferrara, Matteo and Verdoliva, Luisa and Prati, Marco and Ramilli, Marco},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025},
  doi       = {10.1109/IJCNN64981.2025.11228377},
  eprint    = {2504.20865},
  archivePrefix = {arXiv}
}

@inproceedings{li2025aigibench,
  title     = {Is Artificial Intelligence Generated Image Detection a Solved Problem?},
  author    = {Li, Ziqiang and Yan, Jiazhen and He, Ziwen and Zeng, Kai and Jiang, Weiwei and Xiong, Lizhi and Fu, Zhangjie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2025},
  eprint    = {2505.12335},
  archivePrefix = {arXiv}
}

@inproceedings{konstantinidou2025texturecrop,
  title     = {{TextureCrop}: Enhancing Synthetic Image Detection through Texture-based Cropping},
  author    = {Konstantinidou, Despina and Koutlis, Christos and Papadopoulos, Symeon},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops},  % from CVF URL; pages UNVERIFIED
  year      = {2025},
  eprint    = {2407.15500},
  archivePrefix = {arXiv}
}

@inproceedings{choi2025warpad,
  title     = {Training-free Detection of {AI}-generated images via Cropping Robustness},
  author    = {Choi, Sungik and Lee, Hankook and Lee, Moontae},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2511.14030},
  archivePrefix = {arXiv}
}

@inproceedings{karageorgiou2025spai,
  title     = {Any-Resolution {AI}-Generated Image Detection by Spectral Learning},
  author    = {Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  eprint    = {2411.19417},
  archivePrefix = {arXiv}
  % pages UNVERIFIED
}

@inproceedings{guillaro2025bfree,
  title     = {A Bias-Free Training Paradigm for More General {AI}-generated Image Detection},
  author    = {Guillaro, Fabrizio and Zingarini, Giada and Usman, Ben and Sud, Avneesh and Cozzolino, Davide and Verdoliva, Luisa},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  eprint    = {2412.17671},
  archivePrefix = {arXiv}
  % pages UNVERIFIED
}

@inproceedings{corvi2023detection,
  title     = {On the Detection of Synthetic Images Generated by Diffusion Models},
  author    = {Corvi, Riccardo and Cozzolino, Davide and Zingarini, Giada and Poggi, Giovanni and Nagano, Koki and Verdoliva, Luisa},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10095167},  % from search snippet
  eprint    = {2211.00680},
  archivePrefix = {arXiv}
  % pages UNVERIFIED
}

@inproceedings{corvi2023intriguing,
  title     = {Intriguing Properties of Synthetic Images: From Generative Adversarial Networks to Diffusion Models},
  author    = {Corvi, Riccardo and Cozzolino, Davide and Poggi, Giovanni and Nagano, Koki and Verdoliva, Luisa},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2023},
  pages     = {973--982},  % from a citation inside Cozzolino et al. 2024
  eprint    = {2304.06408},
  archivePrefix = {arXiv}
}

@inproceedings{cozzolino2024raising,
  title     = {Raising the Bar of {AI}-generated Image Detection with {CLIP}},
  author    = {Cozzolino, Davide and Poggi, Giovanni and Corvi, Riccardo and Nie{\ss}ner, Matthias and Verdoliva, Luisa},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2024},
  pages     = {4356--4366},  % from search snippet, UNVERIFIED
  eprint    = {2312.00195},
  archivePrefix = {arXiv}
}

@inproceedings{ojha2023univfd,
  title     = {Towards Universal Fake Image Detectors that Generalize Across Generative Models},
  author    = {Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  pages     = {24480--24489},  % from AIGIBench reference list
  eprint    = {2302.10174},
  archivePrefix = {arXiv}
}

@inproceedings{li2025safe,
  title     = {Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective},
  author    = {Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Feng, Fuli},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2025},
  doi       = {10.1145/3690624.3709392},
  eprint    = {2408.06741},
  archivePrefix = {arXiv}
}

@inproceedings{rajan2025aligned,
  title     = {Aligned Datasets Improve Detection of Latent Diffusion-Generated Images},
  author    = {Rajan, Anirudh Sundara and Ojha, Utkarsh and Schloesser, Jedidiah and Lee, Yong Jae},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.11835},
  archivePrefix = {arXiv}
}

@article{ricker2026biasid,
  title   = {{BIAS-ID}: A Framework for Analyzing Transformation Biases in {AI}-Generated Image Detectors},
  author  = {Ricker, Jonas and Fischer, Asja and Quiring, Erwin},
  journal = {arXiv preprint arXiv:2605.31153},
  year    = {2026}
}

@article{bammey2024synthbuster,
  title   = {Synthbuster: Towards Detection of Diffusion Model Generated Images},
  author  = {Bammey, Quentin},
  journal = {IEEE Open Journal of Signal Processing},
  volume  = {5},       % from search snippet, UNVERIFIED
  pages   = {1--9},    % from search snippet, UNVERIFIED
  year    = {2024}     % Zenodo record cites 2023; IEEE SPS page lists Jan 2024 issue
}

@article{cui2026globalforge,
  title   = {{GlobalForge}: Towards Robust {AI}-Generated Image Detection},
  author  = {Cui, Manni and Liu, Ruiqi and Zou, Dianyuan and Qin, Ziheng and Xu, Jingrui and Wang, ZiAn and Wei, Jianglan and Zhou, Han and Liu, Yu and Wang, Yan and Wu, Shu},
  journal = {arXiv preprint arXiv:2607.14684},
  year    = {2026}
}

@inproceedings{mu2026hidanet,
  title     = {No Pixel Left Behind: A Detail-Preserving Architecture for Robust High-Resolution {AI}-Generated Image Detection},
  author    = {Mu, Lianrui and Zou, Xingze and Bai, Jianhong and Hu, Jiaqi and Zheng, Wenjie and Ye, Jiangnan and Zhuang, Jiedong and Ali, Mudassar and Wang, Jing and Hu, Haoji},
  booktitle = {International Conference on Learning Representations (ICLR)},  % UNVERIFIED (iclr.cc poster listing via search; arXiv PDF is AAAI-formatted)
  year      = {2026},
  eprint    = {2508.17346},
  archivePrefix = {arXiv}
}

@inproceedings{konstantinidou2026whatmatters,
  title     = {Navigating the Challenges of {AI}-Generated Image Detection in the Wild: What Truly Matters?},
  author    = {Konstantinidou, Despina and Karageorgiou, Dimitrios and Koutlis, Christos and Papadopoulou, Olga and Schinas, Emmanouil and Papadopoulos, Symeon},
  booktitle = {Proceedings of the 5th ACM International Workshop on Multimedia AI against Disinformation (MAD '26)},
  year      = {2026},
  doi       = {10.1145/3810988.3812665},
  eprint    = {2507.10236},
  archivePrefix = {arXiv}
}

@article{chu2025semantics,
  title   = {When Semantics Regulate: Rethinking Patch Shuffle and Internal Bias for Generated Image Detection with {CLIP}},
  author  = {Chu, Beilin and You, Weike and Li, Mengtao and Zheng, Tingting and Zhao, Kehan and Xu, Xuan and Lu, Zhigao and Song, Jia and Xu, Moxuan and Zhou, Linna},
  journal = {arXiv preprint arXiv:2511.19126},
  year    = {2025}
}

@article{zhou2026fragile,
  title   = {How Fragile Are Training-Free {AI}-Generated Image Detectors? A Controlled Audit of Score Direction, Preprocessing, and Compression},
  author  = {Zhou, Jingwen and Wang, Mingzhe},
  journal = {arXiv preprint arXiv:2606.20488},
  year    = {2026}
}

@article{abdullah2026tap,
  title   = {{TAP} into the Patch Tokens: Leveraging Vision Foundation Model Features for {AI}-Generated Image Detection},
  author  = {Abdullah, Ahmed and Ebert, Nikolas and Wasenm{\"u}ller, Oliver},
  journal = {arXiv preprint arXiv:2604.26772},
  year    = {2026}
}

@article{han2026glass,
  title   = {Attention to Detail: Global-Local Attention for High-Resolution {AI}-Generated Image Detection},
  author  = {Han, Lawrence},
  journal = {arXiv preprint arXiv:2601.00141},
  year    = {2026}
}
```
