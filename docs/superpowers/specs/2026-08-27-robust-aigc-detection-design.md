# Robust Detection of AI-Generated Images Under Real-World Transformations — Design

**Date:** 2026-08-27
**Status:** Approved design, pending implementation plan
**Context:** Hackathon Track 5. One week, team of 2–4, single RTX A4500 (20 GB, partially occupied) plus Kaggle free tier. Hard constraint: models under 2B parameters.

---

## 1. Problem and Framing

Distinguish AI-generated images from authentic ones, and keep that accuracy after the post-processing that real redistribution applies: JPEG re-encoding, blur, rescaling, noise, colour adjustment, and cropping.

### 1.1 What actually wins here

Judging is qualitative (35% technical execution, 20% innovation, 20% impact, 15% feasibility, 10% presentation). The provided WildFake/COCO validation subset explicitly does not contribute to the final score. There is no hidden leaderboard, so marginal AUC is worth little and a defensible, well-instrumented, well-explained system is worth a lot. A detector that visibly collapses at JPEG-30 still loses on technical execution, so the robustness numbers must be real.

### 1.2 Positioning

> We reproduced the mechanisms that won a CVPR 2026 challenge under a 2B-parameter, single-GPU budget, then asked the question that challenge did not: when the evidence has been destroyed, does the detector know?

The detection contribution is a competent reproduction of known-good practice. The original contribution is degradation-aware calibration and abstention, plus two evaluation protocols (leave-one-transform-out, and a content-blind control on the official benchmark) that the surveyed literature does not report.

### 1.3 Deployment assumption (stated, per the brief's allowance)

A content-moderation triage queue. Images arrive already redistributed. False positives on authentic images are the expensive error, so the operating point of interest is low-FPR, and deferring uncertain cases to human review is acceptable and desirable.

---

## 2. Prior Art That Shapes This Design

Verify all citations before use in the writeup.

| Source | What we take from it |
| --- | --- |
| NTIRE 2026 Challenge on Robust AIGC Detection (arXiv 2604.11487) | Same problem, same transforms, 20 documented team methods. Top result 0.997 clean / 0.972 robust ROC-AUC. DINOv3 backbones dominated; aggressive chained augmentation was called a key driver; TTA and ensembling were near-universal. |
| PRISM (PSU team, same report) | Paradigm-diverse ensemble of frozen/LayerNorm-only encoders with small heads — validates our cached-feature design. Hypothesis: contrastive objectives capture semantic inconsistency; self-supervised patch objectives preserve texture discontinuity; supervised CNNs encode low-frequency spectral anomalies. |
| SigLIP2 team (same report) | Global average pooling over final-layer patch tokens beat CLS-token, attention pooling, and multi-layer concatenation. "Squish" resize to fixed size beat random resized crop, which can remove localised forensic cues. |
| DCPT (arXiv 2604.10102) | Clean/degraded paired consistency training: +9.1 pp degraded accuracy for −0.9 pp clean, +15.7–17.9% under JPEG, zero added parameters. Ablation warns that adding architectural components overfits on limited training data. |
| TeleAI LPT (NTIRE report) | Loss form: `L = CE + α·KL(clean ‖ distorted) + β·MSE(f_clean, f_distorted)`. |
| Fake or JPEG? (arXiv 2403.17608) | Detection datasets carry JPEG-quality and image-size bias; removing it improves cross-generator performance by >11 pp. |
| Open-detector benchmark (arXiv 2602.07814) | Training-data alignment explains 20–60% of variance within identical architectures. No universal winner. Ensembles beat single models. |
| "…If Calibrated" (arXiv 2602.01973) | Post-hoc calibration substantially improves AIGC detectors — on clean data. Degradation-conditional calibration remains open. |
| UniversalFakeDetect (Ojha et al. 2023), CNNDetection (Wang et al. 2020), NPR (2024), AEROBLADE (2024) | Baselines. |

---

## 3. Architecture

### 3.1 Two-stage split

**Stage A (once, on the A4500).** For each training image, sample K=10 augmentation recipes, apply them, and push the K augmented views **plus the undegraded view** through a **frozen** backbone. Store all K+1 embeddings together with the augmentation parameters that produced them. The undegraded view is not optional: the consistency loss in §3.4 requires clean/degraded pairs drawn from the same source image within a batch.

**Stage B (many times, anywhere).** Train small heads on the cached embeddings.

Rationale: a bank of 120k images × 11 views × 1024-d fp16 is roughly 2.7 GB per backbone (~11 GB across the four-backbone bake-off, but each is independently usable). One backbone's bank fits a Kaggle Dataset, loads into RAM, and trains a head in minutes. Four teammates can run different ablation rungs concurrently without contending for the single shared GPU. The ablation ladder in §6 is the evidence for the project's central claim, and it is only affordable because of this split.

Backbones stay frozen. This is a design choice, not only a budget concession: full fine-tuning overfits to the generators present in training, which is exactly what the leave-one-generator-out evaluation is built to expose. LayerNorm-only tuning is permitted as an ablation, following PRISM.

**Trade-off accepted:** a fixed augmentation bank is less diverse than fresh on-the-fly augmentation. Mitigated by K independently sampled recipes per image with continuous parameters and chained compositions. Fallback if the bank proves limiting: a LoRA fine-tune of the last blocks as a single "best model" run, with the cached-feature system retained as the ablation platform.

**Inference collapses to one path:** image → backbone → heads → score. The two-stage split is a training-time device only.

### 3.2 Backbones

Paradigm diversity over scale. Bake-off, all under the 2B cap, all cached once:

| Backbone | Params | Paradigm | Hypothesis |
| --- | --- | --- | --- |
| DINOv3 ViT-L | ~300M | Self-supervised | NTIRE winners' choice at larger scale; texture discontinuity |
| SigLIP2-L | ~400M | Contrastive VL | Semantic implausibility; robust under degradation |
| CLIP ViT-L/14 | 304M | Contrastive VL | Reference point; equals the UniversalFakeDetect baseline |
| ConvNeXt-L | ~200M | Supervised CNN | Stride-4 stem retains high-frequency detail; expected to win clean and JPEG-90, lose under blur/resize |

Final model uses at most two backbones, to hold total parameters and inference latency at defensible levels.

**Pooling:** global average pooling over final-layer patch tokens. **Preprocessing:** "squish" resize to a fixed size ignoring aspect ratio; no random resized crop. **Dual resolution:** a 384px expert and a 224px expert, following the NTIRE runner-up's high-resolution/robustness specialist pair, scaled to budget.

### 3.3 Heads

All heads operate on the cached embedding `f`. Total trainable parameters ≈ 2M; full system ≈ 310M–700M depending on backbone count.

**Degradation head `D`:** MLP `dim(f) → 256 → outputs`, where `dim(f)` is 1024 for a single ViT-L backbone and the concatenated width when backbones are ensembled (A4). Per transform family (JPEG, blur, resize, noise, jitter, crop): one presence logit and one severity regression. Loss: BCE on presence + masked smooth-L1 on severity. All labels are free — the augmentation pipeline generated them. The 256-d hidden layer is the **degradation embedding `d`**.

**Classifier head `C`:** MLP on `f` to 512-d, then a second MLP to the AIGC logit. In rung A6 only, the 512-d hidden state is additionally FiLM-modulated by `γ(d)`, `β(d)` derived from the degradation embedding. **The headline model (A3) does not use FiLM** — its degradation head feeds calibration, EQI, and the dashboard, not the classifier. Conditioning is a hypothesis under test (§6.4), not a committed design element.

**Stop-gradient** on the path from `C` into `D` wherever conditioning is enabled (A6). Without it the classifier reshapes `d` into a general-purpose feature that no longer means "degradation," costing the interpretability claim and the dashboard readout. The alternative is ablated so the cost is reported rather than assumed. In A3, `D` and `C` share only the frozen backbone features, so the question does not arise.

### 3.4 Losses

```
L = L_cls                                  # BCE, class-balanced
  + λ_deg · L_deg                          # degradation multi-task
  + α · KL(p_clean ‖ p_degraded)           # prediction consistency
  + β · MSE(f_clean, f_degraded)           # feature consistency
```

The consistency terms are the **primary robustness mechanism**, per DCPT and LPT: largest documented gain, zero added parameters, zero inference overhead. Requires clean/degraded pairs in each batch, so the feature bank must store the clean view alongside each augmented view.

### 3.5 Evidence Quality Index

EQI is fitted, not hand-defined: the model's probability of being correct given `d`, estimated on validation data. Interpretable ("this image retains ~40% usable evidence"), directly useful for abstention, and not hand-waved.

### 3.6 Calibration and abstention

Degradation-conditional temperature scaling: `T(d) = softplus(Linear(d))`, `p = σ(z / T(d))`, fitted on validation with the classifier frozen. Baselines: uncalibrated, and global temperature scaling.

Output `(p, EQI)` maps to a three-way decision: **Clear / Review / Flag**. Thresholds chosen on internal validation against a target FPR.

### 3.7 Explainability

Because the head consumes global-average-pooled patch tokens, applying the same head to each token individually yields a per-patch AIGC heatmap with no extra machinery and no additional training. Combined with the degradation readout and EQI, this is the dashboard's core display.

---

## 4. Data

### 4.1 Sources

- **WildFake** (ModelScope) — many generator families with hierarchical labels; enables leave-one-generator-out.
- **SID_Set** (HuggingFace) — real and fully-synthetic splits. The tampered third is out of scope for the binary task; retained as a bonus evaluation, never trained on.
- **CIFAKE — excluded.** At 32×32, "center crop 80%" and "JPEG-30" are not the same problem the brief describes, and models trained there will not transfer to megapixel images. The exclusion and its reason are stated in the README.
- **External benchmark, never trained on:** COCO val2017 (4998 real) + DALL·E Advanced (8843 AIGC).

### 4.2 The confound, and the two defences

If reals arrive as 640×480 JPEGs and fakes as 1024×1024 near-lossless files, a model can score ~99% on clean data using resolution and encoding history alone, then collapse the moment anything is re-encoded — which is precisely the tested condition. The augmentations do not merely degrade the image; they erase the shortcut the model was using. This is the documented mechanism behind the >11 pp cross-generator gap in *Fake or JPEG?*, and NTIRE 2026 organisers now align resolution, aspect ratio, and JPEG-quality distributions between classes as standard practice.

1. **Normalise at ingest.** Decode everything, resize short side to a fixed value with a fixed resampler, store as raw uint8 or PNG. Identical resolution and encoding history across classes before augmentation is applied.
2. **Content-blind control, published.** Train a deliberately crippled classifier that cannot see content — 16×16 thumbnails, or JPEG quantisation-table features alone. High score ⇒ the dataset is broken and every headline number is suspect. Near-chance ⇒ positive evidence that the signal is content. Reported either way.

### 4.3 Honest caveat

The "clean" real images are already JPEGs of unknown quality. The degradation head therefore estimates degradation **applied on top of an unknown baseline**, not absolute image quality. The writeup frames EQI as relative evidence loss.

### 4.4 Scale

~120k images (60k real / 60k fake). **Stored at short side 512**, which must exceed the largest model input (384px) so that both the 384px and 224px experts see a downscale rather than an upscale — storing at 256 would silently upsample every input to the high-resolution expert and destroy the point of having one. Storage format is PNG (~60-90 GB on disk, within the 270 GB available); if disk becomes tight, fall back to a uniform JPEG q97 re-encode applied identically to both classes, accepting that this attenuates fine forensic evidence for everyone equally rather than differentially. Extraction is ~1.5 h per backbone. The binding risk is download volume, not compute — pull selected generator folders and parquet shards, not whole repositories. The full pipeline is designed to work on a 20k-image subset if acquisition underdelivers.

### 4.5 Licensing and provenance

The brief requires public or properly licensed datasets. Record the licence and source URL for every dataset in the manifest and reproduce them in the README. WildFake, SID_Set, and COCO each carry their own terms; check them at acquisition time rather than at submission time, and drop any source whose licence does not permit this use. No scraped or self-collected imagery.

### 4.6 Splits, frozen before any training

- **Train** — most generators, most sources.
- **Held-out generators** — 2 families excluded from training entirely.
- **Held-out transform** — training views exclude one transform family (e.g. Gaussian noise); it is tested at evaluation. Answers whether robustness generalises to degradations that were not anticipated, rather than only to the ones augmented for. No prior work found reporting this.
- **Internal validation** — for all hyperparameter, threshold, and calibration fitting.
- **External benchmark** — touched once, at the end.

---

## 5. Augmentation

**Training recipes:** `p = 1.0` (every training image is distorted), 1–3 chained operations sampled from different families, 5 severity levels, continuous parameter ranges that cover but are not limited to the brief's discrete settings. Chained composition matters because real redistribution stacks transforms, and the NTIRE report identifies aggressive chained augmentation as a key driver of results.

**Evaluation grid:** the brief's exact 14 single conditions plus clean, then five named composite scenarios:

| Scenario | Chain |
| --- | --- |
| Social repost | resize 0.5× → JPEG 70 |
| Messaging app | resize 0.25× → JPEG 30 |
| Screenshot | crop 80% → resize → JPEG 50 |
| Filtered upload | colour jitter → JPEG 70 |
| Low-light share | noise σ=0.05 → JPEG 50 |

Named scenarios communicate better than parameter grids.

Training and evaluation augmentation live in the same module but are separately configured, so eval conditions are never silently drawn from the training distribution.

---

## 6. Evaluation

### 6.1 Metrics

- **ROC-AUC**, reported as *clean AUC* and *robust AUC* (mean over transformed conditions). Matches the NTIRE protocol, so results are comparable to a published leaderboard.
- **TPR @ FPR = 1%.** A moderation system operates at low FPR; AUC hides that corner. This reframes results as a product decision.
- **Two accuracy columns: oracle-threshold and fixed-clean-threshold.** Most papers re-tune the threshold per condition, implicitly assuming test-time knowledge of the degradation. The gap between the two columns isolates score drift under degradation — invisible to AUC, and exactly what degradation-conditional calibration targets.
- **ECE and Brier**, per condition.
- **Risk–coverage**: AURC, accuracy at 100 / 90 / 80% coverage.
- **Bootstrap 95% CIs** on every AUC (1000 resamples). With ~14k eval images, differences under ~0.5 pp are noise, and several ablation rungs will land inside that band.

### 6.2 Axes

Transform condition (20) × generator (seen / held-out) × dataset (in-domain / external). Marginals plus one method × condition heatmap in the report; the full cube ships as CSV.

### 6.3 Baselines

| Method | Why included |
| --- | --- |
| UniversalFakeDetect (frozen CLIP + linear probe) | Identical to rung A0, so the ladder is measured against published work |
| CNNDetection (ResNet-50, blur+JPEG augmentation) | The canonical robustness baseline; minimum bar |
| NPR (neighbouring-pixel up-sampling artifacts) | Near-free to implement, expected to collapse under resize and blur — the most informative failure in the set |
| AEROBLADE (LDM autoencoder reconstruction error) | Training-free, orthogonal hypothesis |
| DIRE | **Excluded**, with reason stated: ~seconds per image and reportedly brittle under compression, hence infeasible for a redistribution setting |

Target pattern: forensic methods win clean, semantic methods hold under degradation, and our model tracks the winner at every operating point.

### 6.4 Ablation ladder

Every rung on the identical grid with identical seeds, reported with CIs:

| Rung | Configuration |
| --- | --- |
| A0 | Linear probe, clean training only (= UniversalFakeDetect) |
| A1 | + augmented training views |
| A2 | + auxiliary degradation loss (no conditioning) |
| A3 | + clean/degraded consistency loss (**headline**) |
| A4 | + second backbone (paradigm-diverse ensemble) |
| A5 | + 8-view degradation-aware TTA |
| A6 | + FiLM conditioning / degradation-gated routing |
| A7 | + multi-layer feature mixing |

A0→A1 quantifies plain augmentation. A2→A3 must be positive for the central claim. **A6 and A7 are hypothesis tests, not expected wins:** A6 tests DCPT's finding that architectural components overfit on limited data; A7 tests the SigLIP2 team's finding that GAP-over-patch-tokens beats multi-layer concatenation. Negative results are reported.

If A3 fails to reproduce, the project's claim falls back to calibration and abstention, which stand independently, and the negative result is reported.

### 6.5 The official benchmark is confounded — get ahead of it

COCO val2017 images are already JPEG at modest resolution; DALL·E outputs arrive large and near-lossless. The provided demonstration set is therefore confounded along exactly the axis *Fake or JPEG?* identifies, and a suspiciously high score should be expected.

Run the content-blind control on that benchmark specifically and report the number. Report both raw and resolution-normalised results. Tone is factual — "we observed X, here is the control experiment" — never a complaint about the organisers.

### 6.6 Error analysis

Automated: rank by score, extract top-k false positives and false negatives, cluster by embedding, emit a contact sheet annotated with score, EQI, and estimated degradation.

Hypothesised buckets, **to be verified rather than assumed**:

- **False positives** — heavily denoised or beauty-filtered photos, low-light phone shots, HDR landscapes, shallow-DoF macro, upscaled thumbnails. The pattern to discuss: authentic images that have been through aggressive processing look synthetic. This is the central trade-off of the whole problem.
- **False negatives** — heavily compressed AIGC, photorealistic portraits, low-strength img2img edits of real photos, older GAN outputs.

Also report FP rate on the real subset **broken down by source**. False positives concentrated in one dataset indicate a confound, not a detector weakness.

### 6.7 Hygiene

All hyperparameters, thresholds, and calibration fitted on internal validation only. The external benchmark is evaluated once, at the end. Stated in the README.

### 6.8 Realistic targets

Top NTIRE teams reached 0.997 clean / 0.972 robust ROC-AUC; mid-tier 0.98–0.99 clean but 0.91–0.93 robust; below 0.88 robust ranked last. For a one-week prototype, ~0.95 clean / ~0.90 robust on our own held-out benchmark is a respectable and honest result. The clean-to-robust gap is the number that matters.

---

## 7. Repository

```
aigc-robust-detect/
├── src/aigcdet/
│   ├── data/         acquire · normalize · manifest · splits
│   ├── augment/      ops · recipes · scenarios
│   ├── features/     backbones · extract (Stage A) · bank
│   ├── models/       heads · losses
│   ├── train/        train_head (Stage B) · finetune_lora (stretch)
│   ├── calibrate/    temperature · eqi
│   ├── eval/         grid · metrics · controls · report
│   ├── explain/      patch_heatmap
│   └── baselines/    univfd · npr · cnndetection · aeroblade
├── scripts/          predict.py ★ · run_ablation.py · make_error_sheet.py
├── app/              dashboard.py
├── configs/          one YAML per ablation rung
├── docs/             specs · robustness_table.md · error_analysis.md
└── tests/
```

### 7.1 The three contracts, frozen on day 1

1. **Manifest** (parquet): `path, label, generator, source, width, height, split`. A 500-image dummy manifest is published in hour one so training and evaluation code can be built before real data finishes downloading.
2. **Feature bank**: fp16 array + parallel parquet of per-view augmentation parameters. Separates Stage A from Stage B.
3. **Predictions JSON**: `[{"image_path": ..., "pred": 0.87}]`.

### 7.2 predict.py

The required deliverable and the one file a judge will execute.

- Default output contains **exactly** `image_path` and `pred`. EQI, degradation estimates, and the decision go behind `--rich`. Extra keys are how a submission fails on a technicality.
- Recurse subdirectories; skip corrupt and non-image files with a warning, never a traceback.
- Batched inference, progress bar, CPU fallback when no GPU is present.
- Target: a 5k-image directory in a couple of minutes.

---

## 8. Demo

Gradio, single screen.

- **Left** — drag-and-drop upload, plus a sample gallery.
- **Centre** — live sliders for JPEG quality, blur σ, resize scale, noise σ, colour jitter, crop; preview updates as they move.
- **Right** — AIGC probability with calibrated confidence band, EQI gauge, degradation readout ("estimated: JPEG q≈52, downscale 0.5×"), Clear / Review / Flag chip, per-patch heatmap overlay.
- **Bottom** — score plotted against the active slider, abstention band shaded.
- **Batch tab** — point at a directory, show the JSON, display the table. Doubles as proof the required deliverable works.

A single 384px forward is well under 100 ms on the A4500; debounce at ~150 ms for a live feel.

**The key interaction:** dragging JPEG quality from 90 to 30 shows the degradation readout tracking it, the confidence band widening, and the decision flipping Flag → Review while the underlying score stays roughly right. That demonstrates the degradation head, the calibration, and the abstention policy in one gesture.

**Video (~2.5 min):** problem (15 s) → dataset-confound insight (20 s) → architecture (30 s) → live slider demo (60 s) → robustness table, held-out generator, calibration (30 s) → limitations (15 s).

---

## 9. Plan

### 9.1 Workstreams

| Stream | Owner | Days | Depends on |
| --- | --- | --- | --- |
| W1 Data & augmentation | A | 1–2 (critical path) | — |
| W2 Features & training | B | 2–5 | manifest |
| W3 Eval, calibration, baselines | C | 1–6 | dummy manifest (d1), real features (d3) |
| W4 Demo, packaging, writeup | D | 3–7 | predictions JSON |

Two-person variant: A+B on one side, C+D on the other. W3 starting day 1 against the dummy manifest keeps the evaluation harness off the critical path.

### 9.2 Days

- **1** — Data acquisition starts first, before anything else. Manifest schema frozen, dummy manifest published. Augmentation ops implemented and unit-tested against the brief's exact parameter values. Metrics module.
- **2** — Normalisation done. Stage A extraction for backbone 1 runs overnight on the A4500. Head trainer working end-to-end on dummy features. UnivFD and NPR baselines.
- **3** — Rungs A0–A2 trained in parallel on Kaggle. Eval grid runs. **First complete robustness table.** Remaining backbones extracted.
- **4** — Consistency loss and degradation head. Calibration and EQI. Content-blind control. Leave-one-generator-out.
- **5** — A4–A7, TTA, leave-one-transform-out. Error-analysis contact sheets. Dashboard v1.
- **6** — Model selection. Single touch of the external benchmark. Dashboard sliders. predict.py hardening. README.
- **7** — Video, Devpost writeup, buffer.

### 9.3 The scheduling rule that dominates the rest

**By end of day 3 the team holds a complete, mediocre, submittable entry:** trained model, robustness table, working predict.py, README. Everything from day 4 onward is improvement on a submission that already exists. Hackathons are lost by having six days of excellent components and no working whole.

### 9.4 Risks

| Risk | Mitigation |
| --- | --- |
| WildFake/SID_Set download time | Start hour one; subset aggressively; pipeline works on 20k images |
| A4500 already 16.7 GB occupied by another process | Stage A is chunked and checkpointable; can run on Kaggle |
| Kaggle 12 h session / ~30 h week caps | Cached features make Stage B minutes; only Stage A needs sustained GPU |
| Consistency loss does not reproduce +9 pp | One loss term; fall back to aggressive augmentation, already strong |
| Teammate unavailable | The three contracts make each workstream independently completable |
| Ablation differences fall inside noise | Bootstrap CIs are reported; claims are made only where CIs separate |

---

## 10. Deliverables Map

| Required | Produced by |
| --- | --- |
| Devpost written description | §1–3 condensed; tools, models, libraries, datasets enumerated |
| Public repo, commented code | §7 |
| Directory → JSON confidence script | `scripts/predict.py`, §7.2 |
| README: overview, setup, reproduction, limitations, contributions | Written day 6 |
| Demo video | §8 |
| Robustness evaluation summary | `docs/robustness_table.md`, §6.1–6.4 |
| Error analysis note | `docs/error_analysis.md`, §6.6 |

---

## 11. Explicitly Out of Scope

Production deployment, platform-wide moderation systems, video and audio modalities, tamper localisation (SID_Set's tampered split is evaluation-only), and any model at or above 2B parameters.

## 12. Known Limitations to State in the README

1. Frozen backbones cap achievable accuracy relative to full fine-tuning of a large model; the trade-off buys generalisation and fits the compute budget.
2. The fixed augmentation bank is less diverse than on-the-fly augmentation.
3. The degradation head estimates degradation relative to an unknown baseline, since source images carry prior compression.
4. Training data covers the generators present in WildFake and SID_Set; the 2026 benchmark literature reports sharp declines on the newest commercial generators (Flux, Firefly v4, Midjourney v7), which are not represented.
5. No adversarial robustness: an attacker who knows the detector can evade it. Only incidental redistribution transforms are modelled.
