# Resolution standardisation: crop vs band-limit

**A controlled ablation on the `coco_crop` corpus, 2026-08-30.**

Measured by: two Kaggle T4 sessions, one per arm, over an identical 10,000-row
subsample. Raw outputs in `ablation/docs/<arm>/` of each kernel's output;
`combined_table.csv` holds every number below in one frame.

---

## 1. Summary

Two policies for removing native-resolution information from images before a
frozen backbone sees them were compared under single-variable control.

**Band-limit wins both decision metrics; crop wins every confound metric.** The
gap is not caused by cropping as a technique but by how much of the frame this
particular crop configuration discards: at full frame coverage the two policies
are statistically indistinguishable (ΔTPR = −0.011), and the entire deficit
appears where the crop window covers less than half the image (ΔTPR ≈ −0.23).

| | band | crop | winner |
| --- | --- | --- | --- |
| worst low-level confound (lower better) | 0.7508 | **0.6345** | crop |
| resolution-stratified AUC (higher better) | **0.9366** | 0.8567 | band |
| §6.4 selection metric (higher better) | **0.1903** | 0.1542 | band |
| clean AUC (higher better) | 0.7962 | **0.8094** | crop |

A hypothesis that band's advantage was itself the resolution shortcut was
tested and **refuted** (§7.1). A hypothesis that crop's deficit is field-of-view
loss was tested and **supported** (§8).

---

## 2. Background: the problem being solved

Native image resolution leaks the label in this domain, and severely.

In the frozen 138,116-row manifest, 40,982 images (29.7%) sit at a short side
whose bucket is 100% one class (128 → 0 real / 1,262 fake; 224 → 0 / 7,002;
256 → 0 / 25,728; 450 → 0 / 6,158). Short side alone classifies at 72.6%
against a 52.9% majority baseline. On the organisers' scored benchmark it is
worse: COCO (real) is short side 200 for every image while DALL·E 3 (fake) runs
618–1024. The two classes do not overlap in resolution at all.

The cause is mundane: generative models emit at their native resolution (128,
224, 256, 450 are standard generator output sizes) while the authentic corpora
were stored at 200 or 512.

**Why this survives the backbone.** The backbone squashes every input to a fixed
square (384 for SigLIP2-L). Absolute pixel dimensions therefore never reach the
model. The leak survives as a *resampling signature*: a 200px image reaching a
384px backbone is upscaled 1.92× and arrives soft; a 512px image is downscaled
0.75× and arrives sharp. Softness is near-perfectly correlated with the label.

A model trained and scored on this can post an excellent number having learned
"soft equals authentic" and nothing about generation artefacts. The failure is
silent: it looks like success. It is also *directionally consistent* between
training pool and benchmark, which is the worst case — an inconsistent shortcut
would show up as a bad benchmark score and be caught.

---

## 3. The two policies

Both map every image to a common presentation size (`nominal_side = 512`) and
both hand the backbone identical pixel dimensions for every input. They differ
in what they destroy to get there.

### band (`CanonPolicy(mode="band")`)

1. If short side > `band_side` (200): downscale to 200 with `INTER_AREA`.
2. Resample to `nominal_side` (512) with `INTER_CUBIC`. Always an upscale.

Equalises the *bandwidth* of the corpus. Cost: irreversible destruction of all
detail above 200px Nyquist — which is exactly where GAN upsampling checkerboard
and diffusion spectral peaks live.

Optional `jitter` (0.10) draws the ceiling from [0.9·200, 200] per view, always
downward, so the head cannot memorise one exact resampling signature.

### crop (`CanonPolicy(mode="crop")`)

1. Take a `crop_side` × `crop_side` (200×200) window at **native** resolution.
   Random per view at extraction; centre at inference.
2. Resample to `nominal_side` (512) with `INTER_CUBIC`. Always an upscale.

Equalises the *number of genuine pixels* each image contributes without
box-filtering any of them. A generator's high-frequency signature survives
inside the window. Cost: field of view. A 200px image contributes its whole
frame; a 640px photograph contributes a detail.

### The shared invariant, and why it matters

In both policies **step 2 is always an upscale, with one kernel, for every
image alike.** This is load-bearing. If the final resize were sometimes a
downscale and sometimes an upscale, the kernel used would depend on the input's
original size, and native resolution would be re-recorded as an interpolation
signature — undoing the entire exercise. `CanonPolicy.__post_init__` enforces
`band_side < nominal_side` and `crop_side < nominal_side` for this reason.

Any proposed variant (§10) must preserve this invariant or explicitly account
for breaking it.

---

## 4. Experimental design

The two policies are normally welded to different corpora (`STREAM` in
`kaggle_all_experiments.ipynb`), which makes a corpus-vs-policy comparison
uninterpretable — the streams differ in composition, standardisation *and*
augmentation simultaneously. This ablation breaks that welding under control.

**One corpus, one row set, two policies.** Held fixed:

- corpus (`coco_crop`), backbone (`siglip2l`), seed (20260827)
- the exact 10,000 training rows, written to their own manifest and
  fingerprinted (`b6b2603e6befb944`); each arm was launched with
  `--expect-manifest-sha256` set to that value and refuses to start otherwise
- the 6,000-image eval subsample, drawn stratified from a fixed seed
- rungs (a0, a3), all training hyperparameters

Varied: `--canon-mode` only.

**Dihedral augmentation was disabled in both arms.** The `coco_crop` stream
normally pairs crop with per-view dihedral augmentation; including it would
have made the comparison two-variable. This means the crop arm here is *not*
identical to the production `coco_crop` stream.

**Row selection.** `stratified_subsample` per split, not `--shard i/N`. The
manifest is ordered by source, so a contiguous slice is not a random sample —
shard 3 of the frozen manifest contains zero fakes. Stratification balances
classes first, then (generator × source) strata within each class.

**Eligibility filter.** Crop mode raises rather than upscaling an image too
small to fill the window, so rows below `crop_side` must be dropped. On
`coco_crop` this is a no-op (`min_short_side: 200` in the preset; zero rows
below 200 confirmed). On the frozen corpus it is not: 1,308 rows (0.95%) are
below, all generated, 1,260 of them BigGAN at exactly 128px. Since BigGAN is
bimodal at 128/200, filtering deletes one entire mode. **This is why the frozen
corpus was not used.**

---

## 5. Data

`coco_crop`, built by `configs/datasets/coco_crop.yaml`:

| | rows |
| --- | --- |
| train | 157,429 |
| val_internal | 17,721 |
| heldout_generator | 7,000 |
| **total** | **182,150** |
| train+val real / fake | 91,032 / 84,118 |

Authentic: COCO train2017 (46,800) + LAION (15,000) + SID_Set (29,318).
Generated: WildFake (61,680) + SID_Set (29,438).
Held-out generators: `SDwithAdaptor_controlnet`, `VQGAN` (pinned to the frozen
manifest's own draw).

Short-side distribution is strongly bimodal and **class-correlated**:

| short side | rows | dominant class |
| --- | --- | --- |
| 200 | 11,519 | generated |
| 224 | 7,005 | generated |
| 256 | 22,232 | generated |
| 424–480 | ~28,000 | authentic (COCO) |
| 512 | 83,852 | mixed |

This distribution is the reason the field-of-view effect in §8 has such a large
effect size: **small images are overwhelmingly fake and large images are
overwhelmingly real**, so any policy whose behaviour depends on image size is
partly reading the label.

---

## 6. Metrics

Three families, two directions. Confusing them is the main way to misread the
results.

### 6.1 Confound proxies — Gate 1 (LOWER is better)

`scripts/gate_confounds.py`. Decodes a proportional sample (n=6,000), runs the
pipeline's real front half — `canonicalise` under the arm's policy, then
`dihedral`, then `proxy_vector` — and reports **orientation-corrected AUC**,
`max(a, 1−a)`, of each single pixel statistic against the label. 0.5 means the
statistic carries no label information.

Orientation-corrected because a confound that predicts the label *backwards* is
exactly as usable to a head as one that predicts it forwards; separability is
the quantity of interest, not direction.

Proxies: `jpeg_quality`, `laplacian_var` (sharpness), `noise_floor`. Plus
`short_side` as a **control** — it is read from the manifest, not from
standardised pixels, so it is identical across policies by construction and
measures the corpus, not the transform.

The sample is *proportional*, not stratified: the threshold is read against a
figure measured on the corpus as it actually is.

CPU-only, runs before any GPU time, and `--max-auc` makes it a gate that can
cancel a run.

### 6.2 Detection — per condition and aggregate (HIGHER is better)

ROC AUC of the trained head over 6,000 eval images × 20 conditions (the brief's
JPEG/blur/resize/noise/jitter/crop grid plus five composite scenarios).
Computed over every scored row: internal-validation, held-out-generator and
benchmark alike. Bootstrap 95% CIs, 1,000 resamples, seed 20260827.

Four conditions use severities the training sampler never drew and are marked
*(unseen)*: `jpeg_q70`, `blur_s1.0`, `social_repost`, `filtered_upload`.

### 6.3 Selection metric — §6.4 (HIGHER is better)

`heldout_robust_tpr_at_1pct`: mean TPR at 1% FPR over the degraded conditions,
computed on **val_internal authentic images vs heldout_generator generated
images**. This is the rule the headline model is chosen by. It was fixed before
any result existed. It is not clean AUC, not val AUC, and not the benchmark.

### 6.4 Controls

- **Resolution-stratified AUC** (`stratified_auc.py --stratify-by resolution`).
  Within a fixed short-side stratum resolution explains nothing by
  construction, so surviving AUC is detection rather than resolution reading.
  Reported against a **dimensions-only baseline** — the AUC of short side alone
  — which the model must beat to have been shown to detect anything.
- **Per-source FPR spread** (`--stratify-by source`). One threshold at 1% FPR
  over all authentic rows, applied unchanged to each authentic source. A model
  reading generation artefacts has roughly the same rate on COCO, LAION and
  SID_Set. A model reading "is this a COCO photograph" has a far lower rate on
  COCO while the benchmark — whose real half *is* COCO val2017 — looks
  excellent. Mandatory for this stream.

---

## 7. Results

### 7.1 Confound leak — Gate 1 (n=6,000, lower better)

| proxy | band | crop | Δ | winner |
| --- | --- | --- | --- | --- |
| `jpeg_quality` | 0.5548 | 0.5170 | −0.0378 | crop |
| `laplacian_var` | 0.7508 | **0.6345** | **−0.1163** | **crop** |
| `noise_floor` | 0.6005 | 0.5747 | −0.0258 | crop |
| `short_side` *(control)* | 0.8618 | 0.8618 | 0.0000 | tie |
| **worst proxy** | 0.7508 | **0.6345** | **−0.1163** | **crop** |

Crop is cleaner on every measurable channel. The effect is concentrated in
`laplacian_var` (sharpness) and acts by the predicted mechanism: crop removes
no detail from the pixels it keeps, so the gap between a 200px-native image and
a 512px-native one shrinks.

Under a `--max-auc 0.70` gate, **band fails on this corpus and crop passes.**

### 7.2 Controls

| control | dir | band a0 | band a3 | crop a0 | crop a3 | winner |
| --- | --- | --- | --- | --- | --- | --- |
| resolution-stratified AUC | ↑ | 0.9189 | **0.9366** | 0.8466 | 0.8567 | **band** |
| *dimensions-only baseline* | ↑ | 0.8651 | 0.8651 | 0.8651 | 0.8651 | — |
| per-source FPR spread | ↓ | 0.0119 | 0.0179 | 0.0118 | 0.0179 | tie |

Two findings, and the first refuted the working hypothesis.

**Band is not winning by reading resolution.** The prior hypothesis was that
band's advantage was the shortcut itself, predicting a wider per-source FPR
spread for band. The spreads are identical (0.0119/0.0118 at a0,
0.0179/0.0179 at a3). More decisively, with resolution *held constant* — the
one condition where a resolution shortcut cannot operate — band is 0.080 ahead
at a3. The hypothesis is not supported.

**Crop fails the dimensions-only baseline at both rungs** (0.8466 and 0.8567
against 0.8651). Under stratification, crop cannot be shown to beat a
classifier that never sees a pixel.

*Limitation:* only 433 of 1,012 val_internal rows (42.8%) fall in a
two-class stratum, and all of them are the single 512px stratum. The remaining
57.2% are in single-class strata where no within-stratum AUC exists. The
baseline (0.8651) is computed over all 1,012 rows while the stratified figure
is over 433. This is the script's intended comparison but it is one stratum,
not a sweep.

### 7.3 Detection per condition — AUC (higher better)

| condition | band a0 | band a3 | crop a0 | crop a3 | Δ a3 | winner |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.7597 | 0.7962 | 0.7679 | **0.8094** | +0.0132 | crop |
| jpeg_q90 | 0.7700 | 0.8037 | 0.7617 | 0.8071 | +0.0034 | crop |
| jpeg_q70 *(unseen)* | 0.7804 | 0.8075 | 0.7516 | 0.8021 | −0.0054 | band |
| jpeg_q50 | 0.7847 | 0.8072 | 0.7407 | 0.7936 | −0.0136 | band |
| jpeg_q30 | 0.7864 | 0.8048 | 0.7195 | 0.7818 | −0.0230 | band |
| blur_s0.5 | 0.7570 | 0.7979 | 0.7638 | 0.8069 | +0.0090 | crop |
| blur_s1.0 *(unseen)* | 0.7390 | 0.7982 | 0.7472 | 0.8003 | +0.0021 | crop |
| blur_s2.0 | 0.6663 | 0.7872 | 0.6986 | 0.7633 | −0.0239 | band |
| resize_0.5 | 0.7475 | 0.7953 | 0.7598 | 0.8065 | +0.0112 | crop |
| resize_0.25 | 0.6406 | 0.7599 | 0.6934 | 0.7467 | −0.0132 | band |
| noise_s0.02 | 0.7366 | 0.7898 | 0.6995 | 0.7795 | −0.0103 | band |
| noise_s0.05 | 0.7086 | 0.7813 | 0.6513 | 0.7493 | −0.0320 | band |
| noise_s0.1 | 0.6812 | 0.7760 | 0.6436 | 0.7369 | −0.0391 | band |
| jitter_20 | 0.6323 | 0.6541 | 0.6561 | **0.6788** | +0.0247 | crop |
| crop_80 | 0.6812 | 0.7528 | 0.7196 | 0.7638 | +0.0110 | crop |
| social_repost *(unseen)* | 0.7713 | 0.8050 | 0.7451 | 0.7975 | −0.0075 | band |
| messaging_app | 0.6876 | 0.7589 | 0.6497 | 0.7241 | −0.0348 | band |
| screenshot | 0.6992 | 0.7625 | 0.6944 | 0.7537 | −0.0088 | band |
| filtered_upload *(unseen)* | 0.6451 | 0.6697 | 0.6425 | 0.6736 | +0.0039 | crop |
| low_light_share | 0.7570 | 0.8074 | 0.6798 | 0.7615 | −0.0459 | band |

Split 9 crop / 11 band, but not randomly: **crop wins clean and every purely
geometric condition** (blur, resize, jitter, crop_80); **band wins every
frequency-destroying condition** (jpeg q50/q30, all noise levels, the
composites that chain them).

An anomaly worth recording. Band a0's AUC *rises monotonically* under JPEG
compression: 0.7597 (clean) → 0.7700 → 0.7804 → 0.7847 → 0.7864 (q30), i.e.
+0.027 as compression increases. Crop a0 falls, 0.7679 → 0.7195. JPEG destroys
high-frequency content, so a signal that improves under it is not a
high-frequency signal. This was initially read as evidence of a shortcut, but
§7.2 refutes that reading; the anomaly remains unexplained and is a candidate
for follow-up.

### 7.4 Aggregates and selection metric

| metric | dir | band a0 | band a3 | crop a0 | crop a3 | Δ a3 | winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| robust_auc | ↑ | 0.7196 | 0.7747 | 0.7062 | 0.7646 | −0.0101 | band |
| heldout_auc | ↑ | 0.7339 | 0.7701 | 0.7216 | 0.7684 | −0.0017 | band |
| seen_auc | ↑ | 0.7157 | 0.7759 | 0.7021 | 0.7636 | −0.0123 | band |
| val AUC (clean view) | ↑ | 0.9604 | 0.9529 | 0.9121 | 0.9285 | −0.0244 | band |
| val AUC (mean views) | ↑ | 0.8871 | 0.9333 | 0.8562 | 0.9011 | −0.0322 | band |
| **`heldout_robust_tpr_at_1pct`** | ↑ | 0.1852 | **0.1903** | 0.0870 | 0.1542 | **−0.0361** | **band** |

The selection-metric gap narrows sharply from a0 (−0.0982) to a3 (−0.0361):
crop's disadvantage is largest where the head is a bare linear probe and
smaller once a3's augmentation and consistency objectives can compensate.

---

## 8. Diagnostic: why crop lost

The hypothesis: crop's 200×200 window is uninformative on large images. A
200px window is nearly the whole frame for a 256px image and a small patch of a
512px one, so the head is shown a detail — potentially featureless — rather
than a picture.

**Coverage** is defined as `crop_side / short_side`, the linear fraction of the
short side retained. Band always sees the entire frame, so it is the fixed
reference.

### TPR at 1% FPR on fakes, by coverage (a3)

| native short side | coverage | n | band | crop | Δ |
| --- | --- | --- | --- | --- | --- |
| ≤256px | 78–100% | 360 | 0.6583 | 0.6472 | **−0.0111** |
| 261–470px | 43–77% | 57 | 0.7193 | 0.4912 | −0.2281 |
| ≥471px | ≤42% | 89 | 0.3371 | 0.1124 | −0.2247 |

**When the window is essentially the whole picture, the policies are
equivalent** (−0.011, well inside noise at n=360). When the window is a patch,
crop catches a third as many fakes as band on the same images.

Confirmed on a matched-source test with every compositional variable removed —
SID_Set only, both classes present, every image exactly 512px, coverage 0.39:

| | band | crop | Δ |
| --- | --- | --- | --- |
| AUC, sid_set only, n=199 | 0.9329 | 0.8947 | −0.0382 |

### What was *not* found

The same effect does not appear on the authentic class. Mean logit for reals
shows no monotone coverage relationship in either arm, and COCO reals run the
*opposite* way for crop — more confidently classified at 471–600px (−1.709)
than at 300–420px (−0.949). The mechanism acts on the detection of generated
images, not on false positives.

### Conclusion of the diagnostic

The result is a property of `crop_side = 200`, not of cropping. The measured
comparison is *"crop at 200px loses on this corpus"*, and it does not license
the broader claim that crop standardisation is inferior.

---

## 9. Limitations

1. **Scale.** 10,000 training rows against a 175,150-row corpus; 6,000 eval
   images. Directional, not final. Absolute `heldout_robust_tpr_at_1pct` is
   ~0.19 at best, which is low, and both arms would move under full extraction.
2. **Bootstrap intervals were computed but not differenced.** Each arm's
   `robustness_table.md` carries 1,000-resample 95% CIs; the arm-to-arm
   differences here are point estimates. At 1% FPR over ~500 authentic rows the
   threshold rests on roughly 5 false positives, so the selection metric is the
   noisiest number in this report.
3. **Single stratum.** The resolution-stratified control uses 433 of 1,012
   rows, all at 512px.
4. **Two rungs only** (a0, a3). a4/a7 need a reconstruction pass; a5 needs a
   second bank; a6 is inference-only and was not run.
5. **One corpus.** `docs/dataset_presets.md` shows corpus and policy interact —
   band fails a 0.70 gate on `coco_crop` but the frozen corpus scores 0.6721
   under band. These conclusions are `coco_crop`-specific.
6. **`metadata_control` was not run.** It requires the original image files;
   only the feature banks were retrieved. Crop makes every image the same size
   by construction, so it should sit at chance — an unverified prediction.
7. **No dihedral augmentation**, so the crop arm is not the production
   `coco_crop` stream.

---

## 10. Proposed variants

All of these are untested. Each is stated with the invariant it must preserve
(§3) and the confound it risks introducing. **Every one can be screened on CPU
by `gate_confounds.py` before any GPU time is spent**, and should be.

### 10.1 Multi-crop tiling with feature averaging — *recommended first*

Instead of one window, tile the image with N non-overlapping `crop_side`
windows and average the resulting feature vectors. A 512px image yields ~6
windows; a 200px image yields 1.

- **Preserves the invariant exactly.** Every window is standardised
  identically — same size, same kernel, same direction. Nothing about
  per-window processing changes.
- **Recovers field of view**, which §8 identifies as the entire deficit.
- Output dimensionality is fixed regardless of N, so the varying window count
  does not reach the head as a feature.
- **Cost:** extraction time scales with image area. A 512px image costs ~6×
  a 200px one.
- **Risk:** N correlates with native resolution. Averaging hides it from the
  feature vector, but the *variance* across windows would not be hidden if it
  were also passed. Pass the mean only.
- **Cheap partial test available now:** rung **a6** is inference-only TTA over
  multiple crops and needs no re-extraction. Running a6 against the existing
  crop bank tests the mechanism at near-zero cost.

### 10.2 Larger uniform `crop_side` — *measured, and expensive*

Raising `crop_side` directly increases coverage but requires
`min_short_side ≥ crop_side` in the preset, dropping every image below it.
Measured on the `coco_crop` train+val pool (175,150 rows):

| `crop_side` | rows kept | % kept | % of reals kept | % of fakes kept |
| --- | --- | --- | --- | --- |
| 200 | 175,150 | 100.0 | 100.0 | 100.0 |
| 224 | 163,560 | 93.4 | 99.9 | 86.3 |
| 256 | 156,203 | 89.2 | 99.8 | 77.7 |
| 288 | 133,675 | 76.3 | 99.5 | 51.3 |
| 320 | 132,939 | 75.9 | 99.0 | 50.9 |
| 384 | 125,725 | 71.8 | 91.3 | 50.6 |
| 448 | 107,563 | 61.4 | 71.4 | 50.6 |
| 512 | 83,852 | 47.9 | 52.1 | 43.3 |

**The row loss is severely class-asymmetric.** At `crop_side = 320` the corpus
keeps 99.0% of authentic images but only 50.9% of generated ones, because
generated images are systematically smaller. This is not a size reduction, it
is a **change of corpus composition** — it deletes half the fakes and skews
which generator families survive.

`crop_side = 224` is the only setting with a tolerable cost (86.3% of fakes
retained) and it barely raises coverage. **This axis is largely closed.**

### 10.3 Resolution-tiered crops

The proposal: several tiers (e.g. 256 / 512 / 1024), crop each image at the
largest tier it can fill, then bring all tiers to a common presentation size.
Motivation: crop *a little* from large images instead of cropping *a lot*.

The motivation is sound — it directly targets §8 — but the mechanism must be
chosen carefully, because the obvious implementation breaks the §3 invariant.

**Variant A — tiers resized to a fixed nominal (512): breaks the invariant.**
A 1024 crop downscales to 512, a 512 crop is unchanged, a 256 crop upscales 2×.
Different directions and different kernels, selected by the image's original
size. This re-records native resolution as an interpolation signature and
recreates precisely the confound the module exists to remove. Expect Gate 1
`laplacian_var` to rise toward or past band's 0.7508. **Do not use.**

**Variant B — tiers all upscaled to a nominal above the largest tier.** With
tiers 256/512/1024 and `nominal_side = 1280`, every tier upscales with one
kernel, so the invariant's letter is kept. But the scale *factor* still varies
(5.0× / 2.5× / 1.25×) and scale factor is itself a resampling signature — a 5×
upscale is visibly softer than a 1.25× one. The confound is attenuated, not
removed, and the cost is a 1280px working size. **Measurable, but expect
partial reintroduction. Gate 1 will quantify it.**

**Variant C — tiers padded, not resized.** Crop at the tier, then reflect- or
zero-pad to a common canvas. No resampling occurs at all, so the resampling
signature is eliminated completely. Two new risks:
  - **Pad fraction correlates with native resolution.** A 200px crop on a 512px
    canvas is 85% padding; a 512px crop is 0%. "How much of this is padding" is
    a trivially learnable cue and is a direct proxy for the label.
  - Reflection padding creates artificial mirror symmetries at the seams, which
    are themselves synthetic-looking artefacts and may be read as evidence of
    generation.
  Zero-padding makes the pad fraction *more* visible, not less. **The pad
  fraction confound is likely worse than what it replaces**, but it is directly
  measurable — `content_blind_auc` on the padding mask alone would quantify it
  before any extraction.

**Assessment.** Tiering trades a bandwidth confound for either a resampling
confound (A, B) or a padding confound (C). §10.1 achieves the same field-of-view
recovery without introducing a new confound, and is the better first
experiment. If tiering is pursued anyway, **B is the only variant worth
measuring**, and it must be gated.

### 10.4 Hybrid: crop then band-limit

Take a large crop (preserving field of view) then band-limit the window to a
common ceiling (equalising bandwidth). Preserves the invariant. But it pays
*both* costs — lost field of view *and* destroyed detail — for the union of the
benefits. Worth stating only to record that it was considered.

### 10.5 Not a standardisation fix

`noise_floor` is essentially untouched by policy choice (0.6005 band vs 0.5747
crop) and is driven by SID_Set's within-source leak (0.7341 in
`docs/dataset_presets.md`). No standardisation policy addresses it; that is a
source-composition question, and the sweep recorded in
`docs/low_level_confounds.md` shows no two-source mix lowers both channels at
once.

---

## 11. Recommendation

**On the evidence as it stands, ship band.** It wins the §6.4 selection metric
(0.1903 vs 0.1542) and the resolution-stratified control (0.9366 vs 0.8567),
and it is the only arm that beats the dimensions-only baseline at all.

Two qualifications:

1. **Band carries the worse confound profile** (`laplacian_var` 0.7508 vs
   0.6345) and fails a 0.70 Gate 1 threshold on this corpus. The obligation in
   `docs/resolution_shortcut.md` to quote a dimensions-only baseline beside
   every headline is therefore binding, not optional.
2. **Crop has not been fairly tested.** §8 shows the deficit vanishes at full
   coverage. The measured claim is about `crop_side = 200`, not about cropping.
   §10.1 is a cheap, invariant-preserving route to crop's confound profile with
   band's detection, and the a6 TTA test costs no extraction at all.

The honest summary for a reader: *band-limiting currently detects better; crop
currently confounds less; the gap is field of view and is plausibly closable.*

---

## 12. Reproduction

```bash
# 1. Screen any policy on CPU first. This can and should cancel a run.
python scripts/gate_confounds.py \
    --manifest data/manifest_coco_crop.parquet \
    --canon-mode crop --crop-side 200 --n 6000 --max-auc 0.70

# 2. Extract one bank per arm, from ONE manifest. --canon-mode is the variable.
python scripts/extract_features.py --manifest <manifest> --backbone siglip2l \
    --out banks/band --split train,val_internal --canon-mode band
python scripts/extract_features.py --manifest <manifest> --backbone siglip2l \
    --out banks/crop --split train,val_internal --canon-mode crop --crop-side 200

# 3. One eval bank per canon mode. MUST match the training bank's mode.
python scripts/extract_eval_bank.py --manifest <eval manifest> \
    --backbone siglip2l --out banks/eval_band --tier ablation --canon-mode band

# 4. Ladder per arm.
python scripts/run_ablation.py --bank banks/band --eval-bank banks/eval_band \
    --rungs configs/rungs/a0.yaml configs/rungs/a3.yaml --tier ablation \
    --out docs/band/robustness_table.md --selection docs/band/selection.json

# 5. Controls. CPU-only, and both are required before quoting a headline.
python scripts/stratified_auc.py --stratify-by source \
    --checkpoint <rung>/checkpoint.pt --bank banks/<arm> --manifest <manifest>
python scripts/stratified_auc.py --stratify-by resolution \
    --checkpoint <rung>/checkpoint.pt --bank banks/<arm> --manifest <manifest>
```

`notebooks/kaggle_canon_ablation.ipynb` runs steps 1–4 for a list of arms in one
Kaggle session, with the row set written to its own fingerprinted manifest so
every arm provably covers the same images.

**Three things that will silently corrupt a comparison:**

- Extracting the two arms against different row sets. The manifest fingerprint
  must be identical; `--expect-manifest-sha256` enforces it.
- Scoring a rung against an eval bank built under a different `--canon-mode`.
  A robustness curve measured on pixels the head never saw is not that head's
  curve.
- Using `--shard i/N` to subsample. The manifest is source-ordered; contiguous
  slices are not random samples.

### Related documents

- `src/aigcdet/augment/canonical.py` — the policies, and the argument for each
- `docs/resolution_shortcut.md` — the leak this exists to remove
- `docs/dataset_presets.md` — corpus composition, and the band/crop measurement
  on the full corpus
- `docs/low_level_confounds.md` — why source balancing does not help
