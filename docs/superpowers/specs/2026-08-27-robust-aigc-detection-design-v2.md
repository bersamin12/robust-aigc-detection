# Robust Detection of AI-Generated Images Under Real-World Transformations. Design v2

**Date:** 2026-08-27
**Status:** Revised design, supersedes v1
**Context:** Hackathon Track 5. One week, team of 2 to 4, single RTX A4500 (20 GB, partially occupied) plus Kaggle free tier. Hard constraint: models under 2B parameters.

---

## 0. What changed from v1

Fixes:

- **Consistency loss target corrected (§3.5).** v1 wrote `MSE(f_clean, f_degraded)` on the frozen cached embedding, which has no trainable path and is a constant. The feature-consistency term now applies to the classifier head's 512-d hidden state. Expected gain is revised down accordingly: DCPT's +9 pp was measured on a trainable model, and here the loss only shapes a ~2M-parameter head.
- **"No hidden leaderboard" is now an assumption to confirm, not a fact (§1.1).** The brief says the demo set does not count. It does not say the judges will not run predict.py on their own images. Confirm at the 28 Aug webinar.

Additions:

- **Low-level reconstruction branch (§3.3).** SD 1.5 VAE round-trip error statistics as a ~12-d feature, concatenated to the ViT embedding. This adds evidence the ViT embeddings do not carry, at ~84M parameters and around 2 GPU-hours. It enters the ablation ladder with a stated hypothesis and a kill criterion.
- **Handcrafted degradation proxies (§3.4).** Estimated JPEG quality, Laplacian variance, noise floor. Three numbers, no training, computed in predict.py. They validate the learned degradation head, feed calibration, and act as fallback.
- **Held-out severity bands (§4.6, §5).** Training never sees JPEG q in [65, 75] or blur σ in [0.85, 1.15], so the brief's q=70 and σ=1.0 are unseen severities at evaluation. Cheap, and it turns "robust to the transforms we trained on" into "robust across severity."
- **Leakage guard against the demo set (§4.1).** Perceptual-hash dedupe of all training data against COCO val2017 and DALL·E Advanced, and exclusion of any COCO-derived real source from training.
- **Reviewer-load number (§1.3, §6.1).** Risk-coverage results are converted into "fraction of the queue auto-decided at target FPR." This is the concrete impact figure the 20% Impact criterion asks for and v1 did not produce.
- **Model selection rule stated before results exist (§6.4).**

Patched after review (v2.1):

- **§3.3 — `r` now covers all 11 views.** Caching reconstruction features for only 4 of 11 views would have made A3→A4 a comparison across different augmentation budgets, invalidating the kill criterion. General rule added: any two rungs being compared must share identical view coverage. Cost rises from ~3 h to ~8 h, scheduled as an overnight run.
- **§4.4a — evaluation extraction is now budgeted.** Neither v1 nor the first v2 draft allocated GPU time for eval-set features (~5 to 6 h across backbones). A two-tier cap is stated explicitly: 5k+5k on the full 20-condition grid for all ablation and model selection, and the full 13.8k benchmark on the 15 core conditions once, on day 6. Every reported number names its tier.
- **§4.5 — model weight licences, not just dataset licences.** DINOv3's Meta licence is checked on day 1, before the backbone is locked; SigLIP2/DINOv2 is the fallback.

Cuts, to fit the week:

- Backbone bake-off reduced from four to two trained backbones plus CLIP as the free baseline. ConvNeXt-L and the dual-resolution 384/224 pair move to stretch.
- Ablation ladder core is A0 to A6. FiLM conditioning and multi-layer mixing are stretch.
- CNNDetection baseline is run-only from pretrained weights, not retrained.

---

## 1. Problem and Framing

Distinguish AI-generated images from authentic ones, and keep that accuracy after the post-processing that real redistribution applies: JPEG re-encoding, blur, rescaling, noise, colour adjustment, and cropping.

### 1.1 What actually wins here

Judging is qualitative (35% technical execution, 20% innovation, 20% impact, 15% feasibility, 10% presentation). The provided WildFake/COCO validation subset explicitly does not contribute to the final score.

**RESOLVED at the 28 Aug webinar (`docs/techjam webinar - Google Docs.pdf`).** The assumption recorded here — that judging is qualitative and "marginal AUC is worth little" — is **false**. The organisers announced a formula:

> **Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust**, with ROC AUC as the primary metric (threshold-free, robust to imbalance).

Consequences for this design:

- The score is implemented as `eval.report.challenge_score` and recorded per rung in `selection.json` by `run_ablation.py`. Its robust half averages the **brief's required transforms only** — `CHALLENGE_ROBUST_CONDITIONS`, i.e. `CORE_CONDITIONS` minus `clean` — and deliberately *not* the table's `robust_auc` column, which at the ablation tier also averages the five composed scenarios this project invented.
- `AUC_robust` is not defined by the organisers. This project defines it as the unweighted mean over those fourteen conditions and records that definition next to every number.
- **The §6.4 selection rule is unchanged.** `heldout_robust_tpr_at_1pct` follows from the §1.3 deployment assumption and is the metric the calibration and abstention branch exists to serve. The announced score is reported *beside* it, and `run_ablation.py` prints a NOTE when the two prefer different rungs. That disagreement is a finding for the error-analysis note, not something to resolve by quietly adopting whichever rule flatters the run.

Still unconfirmed: whether there is a hidden test set. The design is unchanged either way — the same `predict.py` is the deliverable, and its behaviour under transforms is measured rather than assumed.

The webinar also confirmed the transform table verbatim (it matches `augment.recipes.FAMILIES` and the eval grid exactly), and added three pointers this design should answer rather than ignore: a recommended FFT/DCT frequency branch, SAFE (KDD 2025) preferring **crop over down-sample** to preserve high-frequency artefacts — which is in direct tension with §3.2 canonicalisation — and RandomRotation as a shortcut-killer, which the recipe pool does not contain.

A detector that visibly collapses at JPEG-30 loses on technical execution regardless, so the robustness numbers must be real.

### 1.2 Positioning

> We reproduced the mechanisms that won a CVPR 2026 challenge under a 1B-parameter, single-GPU budget, then asked the question that challenge did not: when the evidence has been destroyed, does the detector know?

The detection contribution is a competent reproduction of known-good practice (frozen self-supervised ViT, aggressive chained augmentation, clean/degraded consistency), plus one piece of low-level evidence the ViT cannot see (autoencoder reconstruction error). The original contribution is degradation-aware calibration and abstention, plus two evaluation protocols the surveyed literature does not report: leave-one-transform-out and a content-blind control on the official benchmark.

### 1.3 Deployment assumption (stated, per the brief's allowance)

A content-moderation triage queue. Images arrive already redistributed. False positives on authentic images are the expensive error, so the operating point of interest is low FPR, and deferring uncertain cases to human review is acceptable and desirable.

This assumption produces a number. At a target FPR (1% on authentic images), the abstention policy auto-decides some fraction of the queue and defers the rest. That fraction, broken down by transform condition, is the impact figure: "under messaging-app compression, the system still auto-clears X% of authentic images and auto-flags Y% of synthetic ones, deferring Z% to a reviewer." It comes straight out of the risk-coverage evaluation in §6.1.

---

## 2. Prior Art That Shapes This Design

Verify every citation before use in the writeup.

| Source | What we take from it |
| --- | --- |
| NTIRE 2026 Challenge on Robust AIGC Detection (arXiv 2604.11487) | Same problem, same transforms, 20 documented team methods. Top result 0.997 clean / 0.972 robust ROC-AUC. DINOv3 backbones dominated; aggressive chained augmentation was called a key driver; TTA and ensembling were near-universal. |
| PRISM (PSU team, same report) | Paradigm-diverse ensemble of frozen or LayerNorm-only encoders with small heads. Validates the cached-feature design. |
| SigLIP2 team (same report) | Global average pooling over final-layer patch tokens beat CLS-token, attention pooling, and multi-layer concatenation. "Squish" resize beat random resized crop. |
| DCPT (arXiv 2604.10102) | Clean/degraded paired consistency training: +9.1 pp degraded accuracy for −0.9 pp clean, with a trainable model. Ablation warns that adding architectural components overfits on limited training data. |
| TeleAI LPT (NTIRE report) | Loss form: `CE + α·KL(clean ‖ distorted) + β·MSE(feat_clean, feat_distorted)`. |
| Fake or JPEG? (arXiv 2403.17608) | Detection datasets carry JPEG-quality and image-size bias; removing it improves cross-generator performance by >11 pp. |
| AEROBLADE (arXiv 2401.17879) | Training-free detection from latent-autoencoder reconstruction error. Signal is autoencoder-specific: strong on generators sharing the AE, weak elsewhere. Also the source of the DIRE JPEG/PNG bias finding. |
| LaRE² (CVPR 2024), FIRE (arXiv 2412.07140) | One-step latent error as a cheaper reconstruction signal; mid-frequency bands of the error survive compression better than the highest frequencies. |
| Open-detector benchmark (arXiv 2602.07814) | Training-data alignment explains 20 to 60% of variance within identical architectures. No universal winner. Ensembles beat single models. |
| "…If Calibrated" (arXiv 2602.01973) | Post-hoc calibration substantially improves AIGC detectors on clean data. Degradation-conditional calibration remains open. |
| UniversalFakeDetect (Ojha 2023), CNNDetection (Wang 2020), NPR (2024) | Baselines. |

---

## 3. Architecture

### 3.1 Two-stage split

**Stage A (once, on the A4500).** For each training image, sample K=10 augmentation recipes, apply them, and push the K augmented views **plus the undegraded view** through a **frozen** backbone. Store all K+1 embeddings together with the augmentation parameters that produced them. The undegraded view is required: the consistency loss in §3.5 needs clean/degraded pairs from the same source image in the same batch.

**Stage B (many times, anywhere).** Train small heads on the cached embeddings.

Rationale: a bank of 100k images × 11 views × 1024-d fp16 is roughly 2.3 GB per backbone. One backbone's bank fits a Kaggle Dataset, loads into RAM, and trains a head in minutes. Teammates run different ablation rungs concurrently without contending for the single shared GPU. The ablation ladder in §6.4 is the evidence for the project's central claim, and it is only affordable because of this split.

Backbones stay frozen. This is a design choice, not only a budget concession: full fine-tuning overfits to the generators present in training, which is exactly what the leave-one-generator-out evaluation exposes. LayerNorm-only tuning is permitted as a stretch ablation, following PRISM.

**Trade-off accepted:** a fixed augmentation bank is less diverse than fresh on-the-fly augmentation. Mitigated by K independently sampled recipes per image with continuous parameters and chained compositions. Fallback if the bank proves limiting: a LoRA fine-tune of the last blocks as a single "best model" run, with the cached-feature system retained as the ablation platform.

**Inference collapses to one path:** image → backbone(s) → reconstruction features → heads → score. The two-stage split is a training-time device only.

### 3.2 Backbones

Two trained backbones, one free baseline. Paradigm diversity over scale.

| Backbone | Params | Paradigm | Role |
| --- | --- | --- | --- |
| DINOv3 ViT-L/16 @384 | ~300M | Self-supervised | Primary. NTIRE winners' choice; texture discontinuity |
| SigLIP2-L/16 @384 | ~400M | Contrastive VL | Second. Semantic implausibility; ensemble partner in A5 |
| CLIP ViT-L/14 @224 | 304M | Contrastive VL | Baseline only. Cached because UnivFD needs it; costs nothing extra to also run the ladder on it |

Cut to stretch, with reasons: ConvNeXt-L (hypothesis was "wins clean and JPEG-90, loses under blur/resize"; interesting but a third 1.5 h extraction and a fourth bank the week does not have room for) and the 384/224 dual-resolution expert pair (doubles extraction for one backbone; revisit only if A5 shows the ensemble helps and time remains).

Final model uses at most two ViT backbones, to hold total parameters and inference latency at defensible levels.

**Pooling:** global average pooling over final-layer patch tokens. **Preprocessing:** "squish" resize from the stored 512-short-side image to a fixed 384×384, ignoring aspect ratio; no random resized crop.

### 3.3 Low-level reconstruction branch `R` (new)

The ViT embeddings are global, semantic, and pooled. None of them carry pixel-level evidence about whether the image has already passed through a generator's decoder. The reconstruction branch does.

**Method.** Take a 256×256 **native-pixel** center crop of the stored 512-short-side image (no resize: resizing would attenuate exactly the signal being measured). Encode and decode through the Stable Diffusion 1.5 VAE (~84M params). Compute:

- L1 and LPIPS distance between input and reconstruction
- error-map statistics: mean, variance, top-decile energy
- azimuthally averaged power spectrum of the error map in 4 radial bands, following FIRE's observation that mid-frequency error survives compression better than high-frequency error

Result: `r`, a ~12-d vector. **Cached for all 11 views**, the same coverage as the ViT embedding. Cost at 256px is around 25 ms per view on the A4500, so 100k × 11 views is roughly 8 h: one overnight run, scheduled day 3.

**Why all 11 and not a subset.** An earlier draft cached `r` for the clean view plus 3 augmented views. That breaks the ladder: A4 would then train on 4 views where A3 trains on 11, so A3→A4 would measure a difference in augmentation budget rather than the contribution of `r`, and the kill criterion below would be testing the wrong thing. The rule is general and applies to every rung: **any two rungs being compared must be trained on identical view coverage.** If the 8 h extraction cannot be afforded, the fallback is to train *both* A3 and A4 on the recon-covered subset and report that reduced-coverage pair alongside the full-coverage A3, never to compare across coverages. LPIPS adds ~15M frozen parameters.

**Hypothesis.** Latent-diffusion outputs from generators sharing this VAE round-trip with anomalously low error. Real photographs, and outputs from generators with a different decoder, do not.

**Known failure modes, stated up front.**

1. Autoencoder-specific. Strong on SD-family generators, expected weak on DALL·E and proprietary models. The official demo set is DALL·E, so `r` may contribute little exactly on the number judges look at. This is reported, not hidden.
2. Reconstruction error shrinks when a real image is blurred or heavily compressed, because degraded images are easier to reconstruct. So `r` drifts toward false positives under strong transforms. The degradation head and calibration exist to learn when to distrust it.

**Kill criterion.** `R` ships only if rung A4 beats A3 on the **held-out-generator** split with non-overlapping bootstrap CIs on robust AUC. If it only helps on seen generators, that is memorisation of the VAE signature, and the branch is dropped with the negative result reported. Either way the VAE error map is kept for the dashboard as a second heatmap.

### 3.4 Heads

All heads operate on the cached ViT embedding `f` (1024-d for one ViT-L; concatenated width when ensembled in A5), optionally concatenated with `r` from A4 onward. Total trainable parameters ≈ 2M; full system ≈ 400M to 800M depending on backbone count.

**Degradation head `D`:** MLP `dim(f) → 256 → outputs`. Per transform family (JPEG, blur, resize, noise, jitter, crop): one presence logit and one severity regression. Loss: BCE on presence + masked smooth-L1 on severity. All labels are free; the augmentation pipeline generated them. The 256-d hidden layer is the **degradation embedding `d`**.

**Handcrafted degradation proxies `h` (new):** three numbers computed from pixels in predict.py with no training: estimated JPEG quality (blockiness plus quantisation-table read when the file is a JPEG), Laplacian variance as a blur proxy, and noise floor as the median absolute deviation of a high-pass residual. Roles:

- validate `D`: report Spearman correlation between `D`'s severity outputs and `h` on validation; if `D` does not agree with the proxies, the degradation readout on the dashboard is not trustworthy and that gets found out on day 4, not in the video
- extra input to calibration (§3.7), so the temperature can depend on cheap, model-independent evidence
- fallback: if `D` underperforms, `h` replaces `d` in the calibration and EQI paths with no other change

**Classifier head `C`:** MLP on `f` (⊕ `r`) to a 512-d hidden state `h_c`, then a second MLP to the AIGC logit. In stretch rung A7 only, `h_c` is FiLM-modulated by `γ(d)`, `β(d)`. **The headline model does not use FiLM.** Its degradation head feeds calibration, EQI, and the dashboard, not the classifier. Conditioning is a hypothesis under test, not a committed design element.

**Stop-gradient** on the path from `C` into `D` wherever conditioning is enabled. Without it the classifier reshapes `d` into a general-purpose feature that no longer means "degradation," costing the interpretability claim and the dashboard readout. In the core rungs `D` and `C` share only the frozen features, so the question does not arise.

### 3.5 Losses (corrected)

```
L = L_cls                                  # BCE, class-balanced
  + λ_deg · L_deg                          # degradation multi-task
  + α · KL(p_clean ‖ p_degraded)           # prediction consistency
  + β · MSE(h_c_clean, h_c_degraded)       # feature consistency on the HEAD's hidden state
```

The feature-consistency term acts on `h_c`, the classifier's trainable 512-d hidden state. In v1 it was written on the frozen embedding `f`, where it is a constant with no gradient.

**Expectation, revised.** DCPT and LPT measured their gains with a trainable backbone, where the consistency loss can reshape the representation. Here the backbone is frozen and the loss only regularises a ~2M-parameter head. The mechanism is the same, the leverage is smaller. Plan for a fraction of DCPT's +9 pp and let A2→A3 report the honest number. If A3 does not separate from A2 with non-overlapping CIs, the writeup says so, and the project's claim rests on calibration and abstention, which stand independently.

**Batch sampler.** Each batch is built from `n` source images × (clean view + `m` of its degraded views), so every clean embedding has degraded partners in the same batch. Sampling is class-balanced and generator-balanced so no generator family dominates a gradient step.

### 3.6 Evidence Quality Index

EQI is fitted, not hand-defined: the model's probability of being correct given `d` (and `h`), estimated on validation data. Interpretable ("this image retains ~40% usable evidence"), directly useful for abstention, and not hand-waved.

### 3.7 Calibration and abstention

Degradation-conditional temperature scaling: `T(d, h) = softplus(Linear([d; h]))`, `p = σ(z / T(d, h))`, fitted on validation with the classifier frozen. Baselines: uncalibrated, and global temperature scaling.

Output `(p, EQI)` maps to a three-way decision: **Clear / Review / Flag**. Thresholds chosen on internal validation against a target FPR of 1% on authentic images. predict.py's `pred` is the calibrated `p`, so 0.9 means roughly 90%.

### 3.8 Explainability

Because the head consumes global-average-pooled patch tokens, applying the same head to each token individually yields a per-patch AIGC heatmap with no extra machinery. This is a heuristic, since the head was trained on the pooled vector, so its spatial coherence is checked on a handful of images before it goes on the dashboard. From A4 onward the VAE error map provides a second, exact heatmap for free. Combined with the degradation readout and EQI, these are the dashboard's core display.

---

## 4. Data

### 4.1 Sources

- **WildFake** (ModelScope). Many generator families with hierarchical labels; enables leave-one-generator-out. Download from Singapore may be slow; pull selected generator folders, not the whole repository.
- **SID_Set** (HuggingFace). Real and fully-synthetic splits. The tampered third is out of scope for the binary task; retained as a bonus evaluation, never trained on.
- **CIFAKE, excluded.** At 32×32, "center crop 80%" and "JPEG-30" are not the same problem the brief describes, and models trained there will not transfer to megapixel images. The exclusion and its reason are stated in the README.
- **External benchmark, never trained on:** COCO val2017 (4998 real) + DALL·E Advanced (8843 AIGC).

**Leakage guard (new).** Before any split is frozen:

1. Perceptual-hash (pHash, Hamming distance ≤ 4) every training candidate against both halves of the demo set, and drop matches. WildFake's real and DALL·E subsets are plausible overlap sources.
2. Exclude any real-image source that is COCO-derived from training entirely, so demo-set numbers measure generalisation rather than distribution memorisation. Record which real sources remain in the manifest.

### 4.2 The confound, and the two defences

If reals arrive as 640×480 JPEGs and fakes as 1024×1024 near-lossless files, a model can score ~99% on clean data using resolution and encoding history alone, then collapse the moment anything is re-encoded, which is precisely the tested condition. The augmentations do not merely degrade the image; they erase the shortcut the model was using. This is the documented mechanism behind the >11 pp cross-generator gap in *Fake or JPEG?*, and NTIRE 2026 organisers now align resolution, aspect ratio, and JPEG-quality distributions between classes as standard practice.

1. **Audit, then normalise at ingest.** Profile file format, resolution, and estimated JPEG quality per class per source first, and keep the audit table for the README (it is a good figure). Then decode everything, resize short side to a fixed value with a fixed resampler, store as PNG. Identical resolution and encoding history across classes before augmentation is applied.
2. **Content-blind control, published.** Train a deliberately crippled classifier that cannot see content: 16×16 thumbnails, or JPEG quantisation-table features alone. High score ⇒ the dataset is broken and every headline number is suspect. Near-chance ⇒ positive evidence that the signal is content. Reported either way.

### 4.3 Honest caveat

The "clean" real images are already JPEGs of unknown quality. The degradation head therefore estimates degradation **applied on top of an unknown baseline**, not absolute image quality. The writeup frames EQI as relative evidence loss.

### 4.4 Scale

Target ~100k images (50k real / 50k fake), floor 20k. **Stored at short side 512**, which must exceed the model input (384px) so every expert sees a downscale rather than an upscale. Storage is PNG (~50 to 75 GB, within the 270 GB available); if disk becomes tight, fall back to a uniform JPEG q97 re-encode applied identically to both classes, accepting that this attenuates fine forensic evidence for everyone equally rather than differentially. Extraction for the **training** set is ~1.5 h per ViT backbone and ~8 h for the reconstruction branch. The binding risk on acquisition is download volume, not compute. The full pipeline is designed to work on a 20k-image subset if acquisition underdelivers.

### 4.4a Evaluation extraction budget, and an explicit cap

Evaluating a rung on the grid needs cached features for every eval image under every condition, which v1 and the first draft of v2 both omitted. Naively: (10k internal validation + 13.8k external benchmark) × 20 conditions ≈ 480k forwards per backbone, ~2 h each, plus ~3.3 h for the reconstruction branch. Across two backbones that is 5 to 6 unallocated GPU-hours landing on an already-full day 3.

**Explicit two-tier cap, stated rather than silently applied:**

- **Ablation and model-selection tier.** The full 20-condition grid runs on 5k internal validation images plus a **5k stratified subsample** of the external benchmark (stratified by class, generator, and source). ≈200k forwards per backbone, ~50 min. Every rung comparison and the §6.4 selection rule use this tier.
- **Final-report tier.** The complete 13.8k external benchmark runs on the **15 core conditions** (clean + the brief's 14), for the headline model and the baselines only. ≈207k forwards, ~50 min, run once on day 6.

The robustness table states which tier produced each number, and the subsample seed is fixed and committed. Bootstrap CIs (§6.1) are computed on whichever tier the row came from, so a 5k-tier row carries visibly wider intervals than a 13.8k-tier row. No number is reported without its tier.

### 4.5 Licensing and provenance

The brief requires public or properly licensed datasets, and this repository must be public. That applies to **model weights as well as data**. DINOv3 ships under a Meta licence more restrictive than Apache-2.0; check its terms on day 1, before it becomes the primary backbone, and check SD 1.5's VAE and LPIPS terms at the same time. If DINOv3's licence does not permit this use, SigLIP2 becomes primary and DINOv2 the second backbone. Nothing structural changes, which is why the check is cheap on day 1 and expensive on day 6. Record every model's licence and source in the README alongside the datasets.

Record the licence and source URL for every dataset in the manifest and reproduce them in the README. WildFake, SID_Set, and COCO each carry their own terms; check them at acquisition time rather than at submission time, and drop any source whose licence does not permit this use. No scraped or self-collected imagery.

The brief also forbids third-party trademarks or copyrighted content in the demo video. The dashboard's sample gallery uses only images whose licence permits display, and the UI shows generator names as plain text only, no logos.

### 4.6 Splits, frozen before any training

- **Train:** most generators, most sources.
- **Held-out generators:** 2 families excluded from training entirely.
- **Held-out severity bands (new):** training recipes never draw JPEG q ∈ [65, 75] or blur σ ∈ [0.85, 1.15]. The brief's q=70 and σ=1.0 conditions are therefore unseen severities at evaluation, and the robustness table marks them as such.
- **Held-out transform family:** a separate training run (A3-LOTO) excludes one whole family (Gaussian noise); it is tested at evaluation. Answers whether robustness generalises to degradations that were not anticipated at all. No prior work found reporting this.
- **Internal validation:** for all hyperparameter, threshold, and calibration fitting.
- **External benchmark:** touched once, at the end.

---

## 5. Augmentation

**Training recipes:** `p = 1.0` (every training view is distorted), 1 to 3 chained operations sampled from different families, continuous parameter ranges that cover but are not limited to the brief's discrete settings, minus the held-out severity bands in §4.6. Chained composition matters because real redistribution stacks transforms, and the NTIRE report identifies aggressive chained augmentation as a key driver of results.

**Evaluation grid:** the brief's exact 14 single conditions plus clean, then five named composite scenarios:

| Scenario | Chain |
| --- | --- |
| Social repost | resize 0.5× → JPEG 70 |
| Messaging app | resize 0.25× → JPEG 30 |
| Screenshot | crop 80% → resize → JPEG 50 |
| Filtered upload | colour jitter → JPEG 70 |
| Low-light share | noise σ=0.05 → JPEG 50 |

Named scenarios communicate better than parameter grids. Two of them (social repost, filtered upload) land in the held-out JPEG band, which is deliberate.

Training and evaluation augmentation live in the same module but are separately configured, so eval conditions are never silently drawn from the training distribution. Unit tests assert the eval grid reproduces the brief's parameter values exactly.

---

## 6. Evaluation

### 6.1 Metrics

- **ROC-AUC**, reported as *clean AUC* and *robust AUC* (mean over transformed conditions). Matches the NTIRE protocol, so results are comparable to a published leaderboard.
- **TPR @ FPR = 1%.** A moderation system operates at low FPR; AUC hides that corner.
- **Two accuracy columns: oracle-threshold and fixed-clean-threshold.** Most papers re-tune the threshold per condition, implicitly assuming test-time knowledge of the degradation. The gap between the two columns isolates score drift under degradation, invisible to AUC, and exactly what degradation-conditional calibration targets.
- **ECE and Brier**, per condition, for uncalibrated / global TS / conditional TS.
- **Risk-coverage:** AURC, accuracy at 100 / 90 / 80% coverage.
- **Auto-decided fraction at target FPR (new):** per condition, the share of images the Clear/Review/Flag policy decides without a reviewer while holding FPR at 1% on authentic images. This is the impact number for §1.3.
- **Bootstrap 95% CIs** on every AUC (1000 resamples). With ~14k eval images, differences under ~0.5 pp are noise, and several ablation rungs will land inside that band.

### 6.2 Axes

Transform condition (20) × generator (seen / held-out) × dataset (in-domain / external) × severity (seen / held-out). Marginals plus one method × condition heatmap in the report; the full cube ships as CSV.

### 6.3 Baselines, in priority order

| Method | Cost | Why included |
| --- | --- | --- |
| UniversalFakeDetect (frozen CLIP + linear probe) | Free (= rung A0 on the CLIP bank) | The ladder is measured against published work |
| NPR (neighbouring-pixel up-sampling artifacts) | Near-free | Expected to collapse under resize and blur: the most informative failure in the set |
| AEROBLADE (reconstruction error, thresholded) | Free (= branch `R` alone, no training) | Training-free, orthogonal hypothesis |
| CNNDetection (ResNet-50, blur+JPEG aug) | Run-only from pretrained weights, day 6 if time | The canonical robustness baseline |
| DIRE | **Excluded**, with reason stated | Seconds per image and reportedly brittle under compression, hence infeasible for a redistribution setting |

Target pattern: forensic methods win clean, semantic methods hold under degradation, and our model tracks the winner at every operating point.

### 6.4 Ablation ladder

Every rung on the identical grid with identical seeds, reported with CIs. Each rung is minutes on cached features.

| Rung | Configuration | Question it answers |
| --- | --- | --- |
| A0 | Linear probe, clean training only (= UnivFD) | Reference |
| A1 | + augmented training views | How much is plain augmentation worth |
| A2 | + auxiliary degradation loss (no conditioning) | Does multi-task hurt or help the classifier |
| A3 | + clean/degraded consistency loss | Does DCPT's mechanism survive a frozen backbone |
| A4 | + reconstruction features `r` | Does low-level evidence add to semantic evidence, on **held-out** generators |
| A5 | + second backbone (SigLIP2, paradigm-diverse ensemble) | Does paradigm diversity beat a single strong backbone |
| A6 | + 8-view degradation-aware TTA | Cheap inference-time gain or noise |
| A7 (stretch) | + FiLM conditioning on `d` | Tests DCPT's warning that architecture overfits on limited data |
| A8 (stretch) | + multi-layer feature mixing | Tests the SigLIP2 team's finding that GAP beats multi-layer concat |

A0→A1 quantifies plain augmentation. A2→A3 must be positive for the consistency claim. A3→A4 must be positive on held-out generators for `R` to ship. A7 and A8 are hypothesis tests, not expected wins; negative results are reported.

**Model selection rule, fixed now:** the headline model is the rung in A3 to A6 with the highest **robust TPR @ 1% FPR on internal validation, held-out generators**. Not clean AUC, not the external benchmark. Stated in the README so the choice cannot be accused of being fitted to the demo set.

### 6.5 The official benchmark is confounded. Get ahead of it

COCO val2017 images are already JPEG at modest resolution; DALL·E outputs arrive large and near-lossless. The provided demonstration set is therefore confounded along exactly the axis *Fake or JPEG?* identifies, and a suspiciously high score should be expected.

Run the content-blind control on that benchmark specifically and report the number. Report both raw and resolution-normalised results. Tone is factual: "we observed X, here is the control experiment," never a complaint about the organisers.

### 6.6 Error analysis

Automated: rank by score, extract top-k false positives and false negatives, cluster by embedding, emit a contact sheet annotated with score, EQI, estimated degradation, and (from A4) the per-branch contribution, so each failure can be attributed to semantic or reconstruction evidence.

Hypothesised buckets, **to be verified rather than assumed**:

- **False positives:** heavily denoised or beauty-filtered photos, low-light phone shots, HDR landscapes, shallow-DoF macro, upscaled thumbnails, smooth low-texture scenes (sky, skin, minimalist interiors), screenshots and digital graphics. The pattern to discuss: authentic images that have been through aggressive processing look synthetic. This is the central trade-off of the whole problem.
- **False negatives:** heavily compressed AIGC, photorealistic portraits, low-strength img2img edits of real photos, older GAN outputs, and outputs from generators whose decoder differs from SD's VAE (where `r` is blind by construction).

Also report FP rate on the real subset **broken down by source**. False positives concentrated in one dataset indicate a confound, not a detector weakness.

### 6.7 Hygiene

All hyperparameters, thresholds, calibration, and model selection fitted on internal validation only. The external benchmark is evaluated once, at the end. Stated in the README.

### 6.8 Realistic targets

Top NTIRE teams reached 0.997 clean / 0.972 robust ROC-AUC; mid-tier 0.98 to 0.99 clean but 0.91 to 0.93 robust; below 0.88 robust ranked last. For a one-week prototype, ~0.95 clean / ~0.90 robust on our own held-out benchmark is a respectable and honest result. The clean-to-robust gap is the number that matters, and the held-out-severity and held-out-generator columns are where the gap is most likely to open.

---

## 7. Repository

```
aigc-robust-detect/
├── src/aigcdet/
│   ├── data/         acquire · audit · normalize · dedupe · manifest · splits
│   ├── augment/      ops · recipes · scenarios · heldout_bands
│   ├── features/     backbones · extract (Stage A) · recon (VAE branch) · proxies (h) · bank
│   ├── models/       heads · losses · sampler
│   ├── train/        train_head (Stage B) · finetune_lora (stretch)
│   ├── calibrate/    temperature · eqi · policy
│   ├── eval/         grid · metrics · controls · report
│   ├── explain/      patch_heatmap · recon_heatmap
│   └── baselines/    univfd · npr · aeroblade · cnndetection
├── scripts/          predict.py ★ · run_ablation.py · make_error_sheet.py
├── app/              dashboard.py
├── configs/          one YAML per ablation rung
├── docs/             specs · data_audit.md · robustness_table.md · error_analysis.md
└── tests/            test_augment_matches_brief · test_predict_schema · test_sampler_pairs
```

### 7.1 The three contracts, frozen on day 1

1. **Manifest** (parquet): `path, label, generator, source, licence, width, height, split`. A 500-image dummy manifest is published in hour one so training and evaluation code can be built before real data finishes downloading.
2. **Feature bank**: fp16 array + parallel parquet of per-view augmentation parameters, plus the `r` and `h` arrays keyed by the same view index. Separates Stage A from Stage B.
3. **Predictions JSON**: `[{"image_path": ..., "pred": 0.87}]`.

### 7.2 predict.py

The required deliverable and the one file a judge will execute.

- Default output contains **exactly** `image_path` and `pred`, with `pred` a calibrated probability in [0, 1]. EQI, degradation estimates, per-branch scores, and the decision go behind `--rich`. Extra keys are how a submission fails on a technicality. `tests/test_predict_schema` asserts the default schema.
- Recurse subdirectories; skip corrupt and non-image files with a warning and score them 0.5 in the output rather than dropping them, never a traceback.
- Batched inference, progress bar, CPU fallback when no GPU is present, fixed seed, deterministic given the weights.
- Weights are downloaded on first run from a pinned URL with a checksum; a `--weights` override exists for offline use.
- Target: a 5k-image directory in a couple of minutes on the A4500.

---

## 8. Demo

Gradio, single screen.

- **Left:** drag-and-drop upload, plus a sample gallery drawn from licensed data only.
- **Centre:** live sliders for JPEG quality, blur σ, resize scale, noise σ, colour jitter, crop; preview updates as they move.
- **Right:** AIGC probability with calibrated confidence band, EQI gauge, degradation readout ("estimated: JPEG q≈52, downscale 0.5×", with the proxy values `h` alongside for comparison), Clear / Review / Flag chip, per-patch heatmap overlay, and the VAE error map as a second heatmap.
- **Bottom:** score plotted against the active slider, abstention band shaded.
- **Batch tab:** point at a directory, show the JSON, display the table. Doubles as proof the required deliverable works.

A single 384px forward plus the 256px VAE round trip is well under 150 ms on the A4500; debounce at ~150 ms for a live feel.

**The key interaction:** dragging JPEG quality from 90 to 30 shows the degradation readout tracking it, the confidence band widening, and the decision flipping Flag → Review while the underlying score stays roughly right. That demonstrates the degradation head, the calibration, and the abstention policy in one gesture.

**Video (~2.5 min):** problem (15 s) → dataset-confound insight with the audit figure (20 s) → architecture (30 s) → live slider demo (60 s) → robustness table, held-out generator and severity, calibration, auto-decided fraction (30 s) → limitations (15 s). No third-party logos or copyrighted imagery anywhere in it.

---

## 9. Plan

### 9.1 Workstreams

| Stream | Owner | Days | Depends on |
| --- | --- | --- | --- |
| W1 Data, audit, augmentation | A | 1 to 2 (critical path) | |
| W2 Features (ViT + recon), training | B | 2 to 5 | manifest |
| W3 Eval, calibration, baselines | C | 1 to 6 | dummy manifest (d1), real features (d3) |
| W4 Demo, packaging, writeup | D | 3 to 7 | predictions JSON |

Two-person variant: A+B on one side, C+D on the other. W3 starting day 1 against the dummy manifest keeps the evaluation harness off the critical path.

### 9.2 Days

- **1 (27 Aug):** Data acquisition starts first, before anything else. **Licence check on DINOv3, SD 1.5 VAE, and LPIPS weights (§4.5), before the backbone choice is locked.** Manifest schema frozen, dummy manifest published. Augmentation ops implemented and unit-tested against the brief's exact parameter values, held-out bands wired in. Proxies `h` implemented (pure numpy/PIL, no model). Metrics module.
- **2 (28 Aug):** Data audit table, normalisation, pHash dedupe against the demo set. Stage A extraction for DINOv3 runs overnight on the A4500. Head trainer working end-to-end on dummy features, including the pairing sampler. UnivFD and NPR baselines. **Webinar 17:00:** confirm hidden-test-set assumption, per-model vs total parameter limit, and which transforms judges apply. Adjust §1.1 that evening.
- **3:** Rungs A0 to A3 trained in parallel on Kaggle. **Eval-tier extraction for DINOv3 (~50 min, §4.4a) runs before the grid.** Eval grid runs. **First complete robustness table.** SigLIP2 and CLIP extraction on the A4500; reconstruction features (~8 h) start in the evening and run overnight into day 4.
- **4:** Degradation head validated against `h`. Calibration and EQI. Content-blind control. Leave-one-generator-out. Eval-tier extraction for SigLIP2 and for the reconstruction branch. A4, and the kill decision on `R` — which is only valid if A3 and A4 share view coverage (§3.3).
- **5:** A5, A6, A3-LOTO. Error-analysis contact sheets. Dashboard v1. Selection rule applied; headline model fixed.
- **6:** Single touch of the external benchmark: final-tier run, full 13.8k × 15 core conditions (§4.4a). Dashboard sliders. predict.py hardening, schema test, weight hosting. README. CNNDetection baseline if time.
- **7:** Video, Devpost writeup, buffer.

### 9.3 The scheduling rule that dominates the rest

**By end of day 3 the team holds a complete, mediocre, submittable entry:** trained model, robustness table, working predict.py, README stub. Everything from day 4 onward is improvement on a submission that already exists. Hackathons are lost by having six days of excellent components and no working whole.

### 9.4 Risks

| Risk | Mitigation |
| --- | --- |
| WildFake/SID_Set download time from Singapore | Start hour one; pull selected generator folders only; pipeline works on 20k images |
| A4500 already 16.7 GB occupied by another process | Stage A is chunked and checkpointable; can run on Kaggle |
| Kaggle 12 h session / ~30 h week caps | Cached features make Stage B minutes; only Stage A needs sustained GPU |
| Consistency loss underdelivers on a frozen backbone | Expected and budgeted for; one loss term; fall back to aggressive augmentation, already strong; report the number |
| Reconstruction branch adds nothing on held-out generators | Kill criterion in §3.3; drop it, keep the heatmap, report the negative |
| VAE / LPIPS weight download blocked or slow | Fetch on day 2 while ViT extraction runs; `R` is A4, so nothing before it depends on the weights |
| Reconstruction extraction (8 h) overruns into day 4 | It is scheduled as an overnight run; A0–A3 and the first robustness table do not depend on it, so an overrun delays only the A4 kill decision |
| Eval-tier extraction squeezes an already-full day 3 | Two-tier cap in §4.4a holds ablation eval to ~50 min per backbone; the expensive full-benchmark run is deferred to day 6 |
| DINOv3 licence incompatible with a public repo | Checked day 1, before the choice is locked; SigLIP2 becomes primary and DINOv2 second, with no structural change |
| Hidden test set exists after all | predict.py is already the deliverable; robust numbers are measured on held-out severity and generators, not tuned to the demo set |
| 2B limit is total, not per model | Full system stays under 1B in every configuration |
| Teammate unavailable | The three contracts make each workstream independently completable |
| Ablation differences fall inside noise | Bootstrap CIs are reported; claims are made only where CIs separate |

---

## 10. Deliverables Map

| Required | Produced by |
| --- | --- |
| Devpost written description | §1 to 3 condensed; tools, models, libraries, datasets enumerated |
| Public repo, commented code | §7 |
| Directory → JSON confidence script | `scripts/predict.py`, §7.2 |
| README: overview, setup, reproduction, limitations, contributions | Written day 6; contributions section maps to the workstream table |
| Demo video | §8 |
| Robustness evaluation summary | `docs/robustness_table.md`, §6.1 to 6.4 |
| Error analysis note | `docs/error_analysis.md`, §6.6 |

---

## 11. Explicitly Out of Scope

Production deployment, platform-wide moderation systems, video and audio modalities, tamper localisation (SID_Set's tampered split is evaluation-only), adversarial robustness, and any model at or above 2B parameters.

## 12. Known Limitations to State in the README

1. Frozen backbones cap achievable accuracy relative to full fine-tuning of a large model; the trade-off buys generalisation and fits the compute budget. It also limits how much the consistency loss can do.
2. The fixed augmentation bank is less diverse than on-the-fly augmentation.
3. The degradation head estimates degradation relative to an unknown baseline, since source images carry prior compression.
4. The reconstruction branch is specific to the SD 1.5 autoencoder. It is expected to be weak on DALL·E and proprietary generators, including the official demo set.
5. Training data covers the generators present in WildFake and SID_Set; the 2026 benchmark literature reports sharp declines on the newest commercial generators (Flux, Firefly v4, Midjourney v7), which are not represented.
6. No adversarial robustness: an attacker who knows the detector can evade it. Only incidental redistribution transforms are modelled.
