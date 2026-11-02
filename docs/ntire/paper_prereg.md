# Pre-registration: training under whole-image canonicalisation ("squish")

Written and committed 2026-10-08, before any squish-trained model exists.
`scripts/paper_pipeline.sh` refuses to start unless this file is committed,
and `scripts/paper_results.py` evaluates the gates below exactly as written.
If the two ever disagree, this file is right and the script has a bug.

## What is already known (and was found on the fit splits)

`docs/ntire/crop_probe.md`, 2026-10-07. The round-two models were trained under
`CanonPolicy(mode="crop", crop_side=200)`: a 200px window at native
resolution. That policy equalises pixel count but leaves field of view
(200 / short side) uncontrolled. In training the field of view is 0.39-0.52,
because every training image is 384-512px. On `ntire_val_hard` 86% of rows
fall outside that band, down to 0.05.

When the frozen crop-trained checkpoints were shown the whole image at test
time instead, fused robust AUC on `ntire_val_hard` rose from 0.8856 to 0.9631.
The z-scores and fusion weight were fitted on `ntire_val` alone for both
numbers. The paired-bootstrap interval on branch B's gain is
[+0.0567, +0.0901]. Three controls argue against a resolution shortcut: the
cue is absent from training, the gain holds within deciles of native size,
and it survives band-limiting to 160px.

That probe chose the policy using `ntire_val_hard`. **So `ntire_val_hard` is
not an unbiased readout of squish.** This study's held-out evidence is the
external benchmarks, plus one declared test read at the end.

## The change under test

Train with `--canon-mode squish`: the whole frame at `nominal_side` square,
aspect discarded (`augment/canonical.py`, `MODE_SQUISH`). Nothing else
changes: same banks, same recipes, same seeds, same schedules, same fusion
procedure.

## Gates

**G1, proxy, decides whether full training happens.** The proxy protocol is
the one every earlier ablation used:

- one dinov2regl224 tower at depth 24;
- 1 epoch over 25% of train, bank `ntire_meta`;
- mean pool, cosine schedule with SWA, final weights;
- scored on `ntire_val_hard` and `ntire_val`.

There are six arms: crop and squish at seeds 20260827, 20260828 and 20260829.
Within a seed, the training subset, head initialisation and batch order are
identical, and only `--canon-mode` differs. A paired bootstrap on
`ntire_val_hard` (2000 resamples, stratified by label × is_distorted, 95%
percentile CI) gives the squish minus crop difference for each seed.

G1 passes if and only if all three hold:

1. The mean over seeds of the robust AUC difference is **> +0.0100**.
2. The CI on the robust difference lies **entirely above zero** for at least
   **2 of 3** seeds.
3. The mean over seeds of the clean AUC difference is **≥ -0.0050**.

If G1 fails, the pipeline stops before full training. The null is written up
as a null: the probe's gain was then a test-time effect that training does not
reproduce, and that finding gets reported.

**G2, full models, NTIRE.** Both branches are retrained under squish exactly
as round two was trained:

- B: dual dinov2regl224 for 3 epochs.
- A: dinov2regl at nominal 512 for 2 epochs.
- Both use bank `ntire2_meta` and seed 20260827.

For each branch, final or SWA weights are picked on `ntire_val` by
mean(robust, clean), with ties going to final. Both the squish fusion and the
crop fusion (round two's A2 + B2swa, unchanged) are re-fitted on `ntire_val`
alone.

G2 passes if the fused robust AUC difference on `ntire_val_hard`
(squish minus crop) is > 0 with a paired 95% CI entirely above zero. Given the
selection caveat above, G2 is necessary but not sufficient evidence.

**G3, external, the held-out test of the policy.** The external sets are
AIGIBench SocialRF, AIGIBench CommunityAI, and Synthbuster with RAISE-1k
reals. They are deduplicated by content digest against every row of
`manifest_ntire`, and nothing was ever trained or selected on them.

G3 is judged on pooled ROC AUC of the fused models on each set's files as
distributed (`ext_<set>`). For each set, G3 passes if squish minus crop is > 0
with a paired 95% CI entirely above zero. **Every set is reported whether it
passes or fails.**

These readings are reported alongside, but are not gates:

- `_jpeg95`: every file re-encoded at JPEG q95. AIGIBench CommunityAI ships
  every real as JPEG and every fake as PNG, so container format alone
  separates its classes.
- `_deg`: one seeded ntire-pool chain per file, applied to the native image.

Short-side-vs-label and format-vs-label AUC are reported for every set.

## Secondary, reported regardless of outcome

- Per branch on `ntire_val_hard`: crop as trained, crop presented whole at
  test time, and trained under squish. Did training add anything beyond the
  test-time change?
- Robust AUC by native short-side bucket for both fusions. Is the size
  gradient gone in a trained model?
- Proxy results on `ntire_val` for each seed.

## The declared second test read

`ntire_test` was read once, on 2026-10-04, with the crop fusion: robust
0.7931, clean 0.9856. One further read is declared here:

- the squish fusion, frozen as above;
- run by `scripts/ntire_test_read2.sh`;
- only after G2, G3 and every table above are written;
- exactly once.

Both reads are reported with this history. There is no third read. Nothing
the read shows can change a choice, because every choice is fixed before it
runs.

## Not done, deliberately

- No sweep of `nominal_side` and no band-limited squish variant in training.
  Those would be post-hoc rescues if squish disappoints.
- No re-picking of the crop baseline. A2 and B2swa were picked in round two on
  pooled `ntire_val` + `ntire_val_hard`, which favours crop, so this
  comparison is conservative.
- No change to the recipe pool, loss or schedule.

## Amendment 1 (2026-10-08, before any model scored an external set)

The external manifest was built from the AIGIBench files on arrival, without
any model scoring them. Its leak diagnostics are:

| set | n | short side p50 | >512px | short-side-vs-label AUC | format-vs-label AUC |
|---|---|---|---|---|---|
| `ext_aigi_socialrf` | 5,994 | 1366 | 96% | 0.694 | 0.500 |
| `ext_aigi_community` | 12,000 | 500 | 42% | **0.916** | **1.000** |

In CommunityAI, both native resolution and container format nearly separate
the classes. Squish does not band-limit, so a squish win on that set's pooled
AUC could come from reading resolution. Two changes follow.

1. **Every external split except `_deg` also gets a resolution-matched
   reading.** This is the n-weighted AUC within equal-count deciles of native
   short side, with a paired bootstrap that resamples within
   (decile × label) cells. It is the same control the crop probe used on
   `ntire_val_hard`.
2. **The G3 verdict for `ext_aigi_community` is reported with an explicit
   caveat.** A pooled-AUC pass there counts as evidence for squish only if the
   resolution-matched difference on its `_jpeg95` split also has a CI above
   zero. If it doesn't, the CommunityAI result is reported as confounded and
   not as support.

The G3 rule for the other sets, and every other gate, is unchanged.

## Amendment 2 (2026-10-12, after G and read 2, before any number for these arms)

A literature check after the pipeline finished found that whole-image resize
is already in use: the NTIRE 2026 winner resizes to 512, the 5th-placed team
squishes, and AI-GenBench (IJCNN 2025) prefers resize to multi-crop. The
paper therefore claims the field-of-view diagnosis, not the resize. Grommelt
et al. (2024) warn that non-square resizing distorts one axis. Squish
discards aspect, so its gain could come partly from that distortion or from
the extra edge context, not from field of view as such. Two
aspect-preserving test-time arms are added to `scripts/ntire_crop_probe.py`,
on the same frozen round-two checkpoints (A2 final, B2swa) and the same
`ntire_val_hard` and `ntire_val` rows:

- `short_center`: short side to `nominal_side`, then the centre square
  (resize then centre-crop). The whole short axis, the middle of the long one.
- `letterbox`: long side to `nominal_side`, padded black to a square. The
  whole frame, no stretch.

**Prediction, written before the numbers.** If field of view is the lever,
each arm recovers at least 75% of squish's robust-AUC gain over `center` on
`ntire_val_hard`, on both branches. A paired bootstrap against `center` and
against `squish` is reported for every arm.

**Reading.** If both arms clear 75% on both branches, the paper says "the
whole field of view", and squish is one way to deliver it. If either falls
short, the paper says which share of the gain needs the stretch or the full
long axis, and narrows the claim to match. This amendment gates nothing
already reported: G1, G2, G3 and read 2 stand as written, and `ntire_test`
is not read again.

## Amendment 3 (2026-10-12, before any public detector has scored any image here)

The squish result covers only our own two DINOv2 branches. Two published
detectors have a train/test field-of-view mismatch in their own released
code. Both are scored here with their released weights, unchanged, by
`scripts/public_detector_probe.py`:

- UnivFD (Ojha et al., CVPR 2023): trained on short side 256 then a 224 crop
  (FOV 0.875); tested with a native 224 centre crop.
- B-Free (Guillaro et al., CVPR 2025, `BFREE_dino2reg4`): trained on reals
  resized to 512 with 504 crops; tested by averaging five native 504 crops.
  Our re-implementation of its five-crop inference reproduces the authors'
  demo logits to 3e-5.

Neither was trained on NTIRE data. For these detectors `ntire_val_hard` has
never been used to choose anything.

There are three presentations per detector, and nothing else changes:

- `native`: the authors' test pipeline.
- `train_fov`: short side to the authors' training size, then the authors'
  crop.
- `squish`: the whole image at the crop size.

**Primary, per detector.** On `ntire_val_hard`, the robust AUC of `squish`
minus `native` is > 0 with a paired 95% CI entirely above zero (2000
resamples stratified by label × is_distorted).

**Secondary, per detector.** The same comparison for `train_fov` minus
`native`. This is the most direct test of the field-of-view account, because
it gives each detector back the field of view it was trained at.

**Mechanism, per detector and resized arm.** The robust gain over `native` in
the largest native-size bucket (752-4016px) exceeds the gain in the smallest
(80-336px).

**Floor.** If a detector's best arm has robust AUC below 0.60 on
`ntire_val_hard`, that detector is reported but not interpreted: near
chance, a difference between presentations says nothing about field of view.

**Reported regardless:** clean AUC for every arm, `ntire_val`, and the three
external sets as distributed, re-encoded at JPEG q95, and degraded (seed
20261008, as in the main pipeline), with size-matched AUC on the non-degraded
external splits.

**Stated expectation.** B-Free was designed to read low-level traces at
native resolution and to avoid resizing. The field-of-view account still
predicts a gain on degraded inputs. If B-Free loses under both resized arms,
the paper's claim is narrowed to detectors that rely on content cues, and the
paper says so.

`ntire_test` is not read.

## Amendment 4 (2026-10-14, before any CLIP model has been trained or scored here)

Amendment 3 left the field-of-view account resting on one detector family.
UnivFD was at chance on this data, and B-Free, built to read native-resolution
traces, failed its primary test. Both were also trained on other data. This
amendment asks the narrower question the paper can answer: does a second
pretraining family, trained by our own recipe on the same data, show the same
effect?

**Model.** `clipl` (`openai/clip-vit-large-patch14`, vision tower, 224 px,
24 blocks), as registered in `src/aigcdet/features/backbones.py`, including
its ImageNet normalisation (the same in every arm). CLIP's contrastive
image-text pretraining differs from DINOv2's self-distillation, and its input
size and token count match `dinov2regl224`.

**Protocol.** `scripts/clip_proxy.sh` reruns stage P of `paper_pipeline.sh`
with every flag unchanged except `--backbone clipl`: bank `ntire_meta`, depth
24, one epoch over 25% of train, mean pool, cosine schedule with SWA, final
weights, seeds 20260827-29. Arms `clip_crop_s{1,2,3}` (`--canon-mode crop
--crop-side 200 --nominal-side 224`) and `clip_squish_s{1,2,3}`
(`--canon-mode squish --nominal-side 224`). Each is scored on
`ntire_val_hard` and `ntire_val`.

**Primary (G1-C).** G1's criteria, unchanged, on the CLIP arms
(`paper_results.py g1 --prefix clip`): mean robust gain of squish over crop
on `ntire_val_hard` above +0.010; a paired 95% CI above zero in at least two
of three seeds; mean clean change at least -0.005.

**Secondary (T-C).** Each crop-trained CLIP checkpoint is presented whole at
test time by `scripts/ntire_crop_probe.py --arms center,squish`. Mean robust
gain of `squish` over `center` above +0.010, with a CI above zero in at least
two of three seeds (`paper_results.py testtime --prefix clip`).

**Mechanism.** For both the trained and the test-time comparison, the mean
robust gain in the 752-4016 px bucket exceeds the gain in the 80-336 px
bucket.

**Floor.** If the crop-trained CLIP checkpoints' mean robust AUC on
`ntire_val_hard` is below 0.60, the results are reported but not
interpreted.

**Reading.** If G1-C passes, the paper says the effect holds for two
pretraining families trained by the same recipe on the same data. If it
fails, the paper keeps its claim to the DINOv2 detector and reports the CLIP
result. This amendment gates nothing already reported, and the DINOv2 proxy
seeds are not rerun.

`ntire_test` is not read.

## Amendment 5 (2026-10-19, before any number for the tests below exists)

A literature check after amendment 4 changed what the paper can claim. No
published study crosses input policy with damage level and detector type on
the same data. AIGIBench found a CLIP probe preferred resize on clean images,
which contradicts "native pixels win on clean". The benchmarks where crops win
have large images, so "crop vs resize" may be "how much the image is shrunk".
The paper is reframed around the missing grid: input policy x damage level x
image size. This amendment registers twelve tests (1-12) that decide the
reframed claims. Every pass condition below is written before the test's
first score or analysis exists. Where a test reuses per-image scores that
already exist, the analysis on them has not been run.

### Rules that apply to every test

- **`ntire_test` is read at most once more, by Test 6 only.** This reverses
  "There is no third read" above, and the paper reports the reversal. Read 3
  is declared here, before any routed-fusion number exists, under the
  conditions in Test 6. Every other script written for this amendment refuses
  the `ntire_test` split by name.
- **Allowed splits:** `ntire_val`, `ntire_val_hard`, `ext_aigi_socialrf*`,
  `ext_aigi_community*`, `ext_synthbuster*`. `ntire_val_hard` was used to
  choose squish, so every number on it is a fit-split number and is labelled
  as such.
- **Statistics:** 95% percentile intervals from 2,000 paired bootstrap
  resamples, stratified by label x is_distorted, with the same resample
  indices for every model or presentation in a comparison
  (`scripts/paired_bootstrap.py`, seed 20260827). "CI above zero" means the
  whole interval is above zero.
- **Frozen checkpoints:** "Branch A" is the round-two crop-trained A2 (final,
  512 px). "Branch B" is B2swa (SWA, dual 224 px). "Squish-trained" means
  A3sqswa and B3sqswa as picked in `docs/ntire/paper/picks.json`. The proxy
  protocol is G1's, unchanged: one dinov2regl224 tower, depth 24, 1 epoch over
  25% of train, bank `ntire_meta`, mean pool, cosine with SWA, final weights,
  seeds 20260827, 20260828 and 20260829.
- **Size buckets:** native short side 80-336, 352-512, 528-736, 752+ px
  (`ntire_crop_probe.BUCKETS`).
- **Every result is reported whether it passes or fails.**
- **Hardware:** one RTX 4090 (24 GB). The handover's timings assumed an RTX
  4060 Ti; nothing below depends on the GPU type.

### Test 1: coverage vs cleaning (`freqmatch`)

Squish shows the whole image and also shrinks it, which averages away noise
and JPEG blocking. Both explanations predict larger gains on larger images.

**New test-time presentation `freqmatch`.** For an image of size h x w and a
branch input size n: (1) apply `squish` exactly as `views_squish` does, giving
n x n; (2) resize back to h x w with the crop policy's upscale kernel; (3)
apply the branch's own crop policy (`canonicalise`, 200 px centre window
resized to n). The image then has squish's frequency content, including the
extra shrink on the long axis, and the centre window's field of view.

**Run:** frozen Branch A and Branch B on `ntire_val_hard` (primary) and
`ntire_val` (secondary). Report robust and clean AUC with paired intervals
against `center` and against `squish`, in the format of Table 2.

**Statistic:** recovered share R = (robust AUC freqmatch - centre) /
(robust AUC squish - centre), per branch, on `ntire_val_hard`.

- **Coverage holds:** R < 0.40 on both branches.
- **Cleaning explains it:** R > 0.70 on both branches.
- **Otherwise:** both effects contribute; report R for each branch.

### Test 2: regime grid with a damage sweep

**Models and presentations.** Branch A and Branch B: `center`, `grid9`,
`squish`. B-Free (authors' checkpoint, amendment 3): `native` (authors'
five-crop), `train_fov`, `squish`. For our branches, "native" in the
statistic below means `center`, the presentation they were trained on.

**Damage.** Applied to the native image before any presentation, with the
project's own ops (`augment/ops.py`), one op type per arm:

| level | JPEG (`ops.jpeg`) | blur (`ops.blur`) | noise (`ops.noise`) |
|---|---|---|---|
| none | - | - | - |
| light | q90 | sigma 0.5 | sigma 0.02 |
| medium | q70 | sigma 1.0 | sigma 0.05 |
| heavy | q30 | sigma 2.0 | sigma 0.10 |

That gives 10 conditions (none + 3 types x 3 levels). Noise is seeded per
image by `default_rng([20261019, int(content_sha256[:15], 16)])`, so every
model sees identical pixels. JPEG and blur are deterministic.

**Source images.** Undistorted originals only:

- primary: `ntire_val` + `ntire_val_hard` rows with is_distorted = 0
  (6,250 images, pooled);
- secondary: `ext_aigi_socialrf` and `ext_synthbuster` as distributed. Each
  is subsampled to 1,000 per label (seed 20261019), or all rows if a label
  has fewer. Without this cap the grid9 arm of Branch A costs more than a day
  of GPU time.

**Statistic.** D = robust AUC(squish) - AUC(native) within each (model x
condition x size bucket) cell, with a paired CI, on the primary source. The
"none" level has no distorted rows, so AUC over all its rows is used there.
"Damage level" in the pass rule averages D over the three op types at each
level. D per op type is reported alongside.

- **Pass (regime rule holds):** for both of our branches, D does not
  decrease from none -> light -> medium -> heavy in at least 3 of 4 size
  buckets, and D changes sign along that axis in at least one bucket.
- **Secondary (generality):** B-Free's D rises with level, even if it never
  crosses zero.

**Secondary sources are not size-bucketed.** Synthbuster's RAISE reals all
have a short side of 833-835 px, so most size buckets would contain no reals
and have no AUC. SocialRF's fakes were scraped from social platforms and were
likely re-encoded already, and its reals come from FFHQ, CelebA-HQ and Open
Images. For both sets, "none" means "no added degradation", not "clean". Both
are reported pooled over size, with that caveat.

### Test 3: shrink ratio vs field of view

**Problem in the design.** For one branch, the crop window's field of view
(FOV = 200 / short) and squish's shrink ratio (S = short / n) are both
functions of the native short side alone: log FOV = -log S + const. Within a
branch they cannot be separated. The test therefore uses the two places where
they come apart:

1. **Across branches.** On the same image, A and B have the same FOV but S
   differs by 512/224 = 2.29.
2. **Across aspect.** Squish shrinks the long axis by long / n. The window's
   area coverage (200^2 / (h * w)) differs between images with the same
   short side.

**Data.** The existing per-image `center` and `squish` logits of A2 and B2swa
on `ntire_val_hard` (primary) and `ntire_val` (secondary). No new scoring.

**Model.** Per-image signed gain g = (2y - 1)(z_squish - z_center), with each
arm's logits z-scored on `ntire_val`. Ordinary least squares on rows pooled
over both branches:
g ~ branch + log S_short + log S_long + log area_coverage + is_distorted,
with a bootstrap over images (2,000 resamples, stratified as above) for the
coefficient intervals. The Figure 1 analysis is repeated with S on the x-axis,
one curve per branch.

- **Pass (FOV is the variable):** the area-coverage coefficient's CI excludes
  zero, and the S_short and S_long coefficients' CIs include zero. In the
  Figure 1 re-plot, the two branches' curves line up on FOV and not on S.
- **Fail (shrink dominates):** the S coefficients' CIs exclude zero and area
  coverage's does not. Then the paper's variable is wrong and the effect is
  renamed.
- **Otherwise:** report both, and say the two variables cannot be separated
  on this data.

### Test 4: gain by damage family

**Data.** The existing per-image `center` and `squish` logits of A2 and
B2swa, distorted rows of `ntire_val_hard` (primary) and `ntire_val`
(secondary). No new scoring.

**Family map of the NTIRE labels**, written before any regrouping:

| family | NTIRE tokens |
|---|---|
| JPEG | jpeg, jpeg_ai, jpeg_recompression_1, jpeg_recompression_2, jpeg_recompression_comb_jpegai, quantization |
| blur | gausblur, lensblur, motionblur, glassblur |
| noise | whitenoise, impulsenoise, multnoise, isonoise |
| colour | lincontrchange, darken, brighten, rgbshift, colorshift, colorsat, jitter, randomtonecurve, clahe |
| downscale | downscale, pixelate |
| geometric | perspective, randomcrop, randomaspectcrop |
| adversarial | adv_embed_resnet, adv_embed_clip |

A token not in this table goes to "other" and is listed by name in the
results. `quantization` is put in JPEG as agreed; it is also reported as its
own row, because it may be colour-depth quantisation.

**Counting.** Chains hold 1-4 operations, and `downscale` is in 74% of
distorted `ntire_val_hard` rows. Two tables are reported: (a) every row
counted in every family it contains; (b) single-family rows only, with n per
cell. The primary reading uses (a).

**Statistic.** Squish robust gain over `center` per family, per branch, with
paired intervals (resampled by label within the family).

- **Supports coverage:** every family with n >= 100 has a gain whose CI is
  above zero, and the largest family gain is less than 2x the smallest.
- **Supports cleaning:** the JPEG and noise gains each exceed both the blur
  and the colour gains, with paired CIs of those differences above zero.
- **Otherwise:** mixed; report per family.

Read jointly with Test 1.

### Test 5: training field of view after augmentation

**Run.** On a seeded 20,000-image sample of `train` (seed 20261019), run the
training view pipeline exactly as the trainer does (recipe pool `ntire2`,
crop policy, crop_side 200, n = 512 and n = 224), all 11 views per image. For
each view, log: native short side; short side at the moment the window is
taken (after the pre-window downscale); realised FOV = window / that short
side; area coverage = window area / image area at that moment. CPU only.

**Report.** p5, p50 and p95 of native FOV and of post-augmentation FOV, in
the format of Table 1, plus area coverage.

- **Reading:** if the post-augmentation p5-p95 range is wider than Table 1's
  native range by more than 0.10 FOV at either end, the "narrow training
  band" sentence is rewritten with the true range. This test gates no claim.

### Test 6: two-view routed fusion

**Models.** Frozen Branch A and Branch B (crop-trained). Their `grid9` view
is native windows of the kind they were trained on; their `squish` view is
the whole image.

**Rule.** Per image, per branch: s = w * z_squish + (1 - w) * z_grid9, with
w = sigmoid(a + b * d), where d is the branch's predicted overall damage
severity on the squish view (the degradation head's output), and z is the
z-score fitted on `ntire_val`. The branches are then fused as in the paper.
(a, b) are chosen by grid search on `ntire_val` only (a in [-4, 4],
b in [0, 8], step 0.25), by mean(robust, clean). Evaluated on
`ntire_val_hard` and the three external sets as distributed, `_jpeg95` and
`_deg`.

**Pass**, both required:

1. Robust AUC on `ntire_val_hard` is within the paired CI of squish alone
   (the CI of routed - squish includes zero or is above it).
2. **Clean cost recovers, on `ntire_test` (read 3).** On clean `ntire_test`
   rows with native short side > 1100 px, the routed fusion's clean AUC is
   within 0.010 of read 1's crop fusion on the same rows. Sec 8 measured the
   cost on these rows: 0.974 crop against 0.880 squish, branch A.

**Read 3 conditions.** (i) The routing rule, (a, b), z-scores and the branch
fusion weight are fitted on `ntire_val` only and frozen. They are written to
`docs/ntire/amend5/test6_frozen.json`, staged, and committed before the read.
(ii) Read 3 runs only after every other test in this amendment has its
results written. (iii) It runs exactly once, by one script that scores the
frozen routed fusion and nothing else. (iv) The paper reports all three reads
with their dates. Nothing read 3 shows can change a choice. Robust and pooled
AUC on `ntire_test` are reported as well, but only condition 2 is judged
there.

**Also reported, not judged:** the same clean > 1100 px comparison on
`ntire_val`, `ntire_val_hard`, and Synthbuster fakes against all Synthbuster
reals.

This differs from the failed severity-weighted fusion of Sec 11, which mixed
two crop branches. Here a whole-image view is mixed with a patch view.

### Test 7: multi-scale training baseline

**Arms.** Three proxy seeds trained with random resized crop: area fraction
U(0.20, 1.00) of the native image, aspect ratio log-uniform in [3/4, 4/3],
resized to 224. Applied identically to real and fake images. Everything else
is the G1 proxy protocol. Before the full run, 100 steps are timed and GPU
utilisation is logged. Under 70% means the CPU augmentation is the
bottleneck; workers are then raised, and nothing else changes.

**Test-time presentation** for these arms: one of `center`, `short_center` or
`squish`, chosen per seed on `ntire_val` by mean(robust, clean) before
`ntire_val_hard` or any external set is scored.

**Comparators.** The existing `abl_crop_s{1,2,3}` and `abl_squish_s{1,2,3}`
proxy scores. Evaluated on `ntire_val_hard` and the three external sets as
distributed (the existing proxies are scored on the external sets too, if
they have not been).

- **Pass:** mean over seeds of robust AUC (squish proxy - multi-scale proxy)
  on `ntire_val_hard` > 0, with a CI above zero in at least 2 of 3 seeds.
- **Fail:** multi-scale matches or beats squish (mean difference <= 0).
  The contribution is then restated as "training on one size band breaks
  detectors", with squish as one fix among several.

### Test 8: fair native crop baseline

**Arms.** Three proxy seeds trained with a native 224 px window at the 224
input size (`crop_side` 224, `nominal_side` 224, crop_clamp on). There is no
enlargement for any image whose short side is at least 224. Smaller images
are taken whole and resized, as the current policy does below 200 px. Tested
with the same 224 px native centre window. Everything else is the G1 proxy
protocol.

- **Pass:** mean over seeds of robust AUC (squish proxy - native-224 proxy)
  on `ntire_val_hard` > 0, with a CI above zero in at least 2 of 3 seeds.
  The 200 px crop proxies are reported alongside.

### Test 9: learned combiner over nine windows

**Run.** For frozen Branch A and Branch B, extract the pooled features (the
head's input) of each `grid9` window. Train a gated-attention MIL head
(Ilse et al. 2018; hidden 128, dropout 0.1) over the windows, per branch,
on `ntire_val` (90% fit / 10% early stopping, seed 20261019). Evaluate on
`ntire_val_hard`.

- **Pass:** robust AUC (squish - combiner) has a CI above zero on both
  branches.
- **Tie:** the CI includes zero. Report it, and soften "nine native windows
  barely help" to match.
- **Fail:** the combiner's CI is above squish's.

The combiner is fitted on `ntire_val` while squish is zero-shot. That favours
the combiner, and is stated.

### Test 10: masking test

**Run.** Under `squish`, at n x n, fill one region with the ImageNet mean
colour before normalisation:

- `mask_centre`: the centred square of side n/2 (25% of the area);
- `mask_periphery`: everything outside the centred square of side
  round(n * sqrt(0.75)) (also 25%).

Frozen Branch A and Branch B on `ntire_val_hard`, split by clean and
distorted.

- **Pass:** on distorted rows, robust AUC (mask_centre - mask_periphery) has
  a CI above zero on both branches. Masking the periphery costs more.
- The same difference on clean rows is reported, not used as a condition.

### Test 11: clean cost on large images

**Data.** Existing per-image scores where they exist (`A2_*`, `A3sqswa_*`,
`B2swa_*`, `B3sqswa_*`, crop vs squish-trained). Otherwise the squish-trained
and crop-trained branches are scored on `ext_synthbuster`. Synthbuster has
generator labels.

**Analysis.**

- Per generator: clean AUC (each generator's fakes against all RAISE reals),
  crop vs squish, per branch, with paired CIs.
- Fakes only: mean logit and the true positive rate at the threshold that
  gives 1% FPR on the reals, by fake short-side bucket (<= 1100, > 1100).
- Clean `ntire_val` and `ntire_val_hard` rows above 1100 px, crop vs squish.

The RAISE reals all have a short side of 833-835 px, so reals are never
pooled into a size analysis.

- **Pass:** at least one generator or size bucket where squish's clean AUC is
  below crop's by more than 0.020 with a CI below zero, and no such drop
  outside it. That pattern is reported as the cost's location.
- **Otherwise:** the cost is reported as unexplained, and Test 6 is checked
  for whether it removes it.

### Test 12: writing fixes

No scoring. The output is a list of proposed edits to `docs/paper/main.tex`
with the numbers that support each one, for review. The edits are not
applied. Numbers that need computing come only from per-image scores that
already exist:

- TPR@1%FPR for every table;
- the 416-496 px bin per frozen branch;
- crop's final-vs-SWA choice redone on `ntire_val` alone (A2 vs A2swa,
  B2 vs B2swa).

The selection asymmetry for test read 1 is disclosed, not rescored.

### Order

Analysis-only tests (3, 4, 5, 11, 12) run first. Scoring tests (1, 2, 6, 9,
10) follow, then training tests (7, 8), as GPU time allows. Test 6's read 3
of `ntire_test` runs last of all.

### Decisions settled before this amendment was committed

- Test 2 keeps SocialRF and Synthbuster. They are the only held-out sources
  in the grid, and are reported pooled over size.
- Test 6 reads `ntire_test` once (read 3), under the conditions above. Each test lives
on its own branch off this amendment's commit, `dion/test<N>-<slug>`. Its
result goes to `docs/ntire/amend5/test<N>_<slug>.md`, with JSON and
per-image parquets beside it.

### Open questions (unresolved when this amendment was written)

1. **Access.** The frozen checkpoints, the proxy checkpoints, the external
   images and the B-Free weights are under `/scratch/justin_r13`, which this
   account cannot read. Tests 1, 2, 6-10, and the scoring parts of 11, wait
   for read access or a copy. Tests 3, 4, 5, the existing-score parts of 11,
   and 12 do not.
2. **Test 3 collinearity.** Within a branch, FOV and shrink ratio are the
   same variable. The test separates them only through the branch contrast
   and aspect ratio. If those leave too little variation (aspect is narrow in
   NTIRE), the result will be "cannot separate", not a pass or a fail.
3. **Test 6 model choice.** The crop-trained branches are used, because
   `grid9` is in their training regime. Routing the squish-trained branches
   instead is a different test and is not registered.
4. **Test 11 data.** The handover asks for checks on "large clean images in
   `ntire_val`". These are already reported in Sec 8 (871 and 295 images).
   They are recomputed here only as a check.

## Amendment 6 (2026-10-19, after Test 3's rank check, before any Test 3 coefficient exists)

**Test 3's registered regression is not identifiable.** With the registered
covariates, log area_coverage = 2 log 200 - log S_short - log S_long
- 2 log n, and n is a function of `branch` alone. The design matrix
[1, branch, log S_short, log S_long, log area_coverage, is_distorted] is
therefore singular by construction: rank 5 of 6 on both `ntire_val_hard` and
`ntire_val`, with log area recovered from the other columns at R^2 = 1.000000.
The coefficients the pass condition names have no unique values. Amendment 5
expected aspect ratio to separate area coverage from shrink ratio, but S_long
absorbs that variation. It expected the branch contrast to separate them, but
`branch` absorbs the constant cross-branch offset in log S.

**Decision.** Test 3 is reported as "not identifiable on this design". The
OLS and its bootstrap are not run, and no reparametrised or substitute model
is fitted for it. The Figure 1 re-plot (gain against FOV and against S, one
curve per branch) is reported as registered, as a description. It judges
nothing. Whether squish's gain comes from coverage or from cleaning is left to
Test 1 (`freqmatch`), whose pass conditions are unchanged.

**What was seen before this amendment.** These were seen: the rank check, the
z-score parameters, and the Figure 1 re-plot (the curves line up on FOV more
closely than on S: mean absolute gap 0.024 against 0.057 on `ntire_val_hard`).
No regression coefficient was computed under any parametrisation. Details are
in `docs/ntire/amend5/test3_shrink-ratio.md` on branch
`dion/test3-shrink-ratio`.

Nothing else in Amendment 5 changes.

## Amendment 7 (2026-10-20, before any number for Tests 13-15 exists)

A code review of Tests 1-12 raised design questions about the
coverage-vs-cleaning evidence. Tests 1, 2 and 4 each apply damage at native
scale and present it through a resize, so a presentation that shrinks the
image also attenuates the damage. A test of "coverage, not cleaning" has to
hold one of the two fixed while the other varies. Tests 1-12 do not do this
cleanly. This amendment registers three new tests that do, and five
re-analyses of existing results. **No verdict of Tests 1-12 is changed by this
amendment.** The re-analyses are reported as supplements beside the original
write-ups, which stay as they are.

The rules of amendment 5 apply unchanged (allowed splits, frozen checkpoints,
bootstrap, every result reported). Every new script refuses `ntire_test` by
name. Every bootstrap below uses the index scheme of
`paper_analyses.paired_diff` (seed 20260827, `rng.integers` within each
stratum, cells in `np.unique` order), with several statistics computed on
each draw. Each script asserts that one of its differences reproduces
`paired_diff`'s interval exactly.

Each verdict below is decided on the 95% interval, not the point estimate.
An interval that straddles a threshold gives "unresolved", with the point
estimate reported.

### Test 13: damage after squish (`postsquish`)

**Claim decided:** whether squish's rising advantage under damage (Test 2's
D curve) comes from the shrink attenuating the damage (cleaning) or from the
whole field of view (coverage).

**Run.** Test 2's primary source (6,250 clean NTIRE rows), Test 2's nine
damage conditions and seeds, frozen Branch A and Branch B. New arm
`squish_post`: `views_squish` on the clean native image, then the damage op
on the n x n view. The noise generator is seeded per image exactly as in
Test 2. The `center` arm and the native-damage `squish` arm are Test 2's
existing per-image logits, unchanged. Only `squish_post` is scored.

**Statistics,** per branch, pooled over size (primary) and per size bucket
(reported):

- D_nat(c) = AUC(squish) - AUC(center), from Test 2;
  D_post(c) = AUC(squish_post) - AUC(center).
- Rise_nat = mean over op types of D_nat(heavy) - D(none); Rise_post the same
  with D_post. D(none) is the same number in both, because no damage is
  applied.
- Retained share Q = Rise_post / Rise_nat. Q is also reported per op type.

**Conditions** (both branches):

- **Not decidable:** the interval of Rise_nat includes zero.
- **Cleaning explains the rise:** the interval of Q lies below 0.30.
- **Coverage explains the rise:** the interval of Q lies above 0.60.
- **Otherwise:** both contribute, or unresolved; report Q with its interval.

The thresholds mirror Test 1's (R < 0.40 coverage, R > 0.70 cleaning, with
R the share recovered by cleaning, so Q = 1 - R).

**Known limit.** Damage on the n x n view is not "the same damage" as native
damage seen through the centre window, which enlarges it 2.56x (A) or 1.12x
(B). The test removes squish's attenuation; it does not equalise damage
strength across arms.

**Not done.** Re-applying each `ntire_val_hard` row's recorded distortions
after squish was considered and dropped. Those rows are already damaged, and
most NTIRE ops (`multnoise`, `pixelate`, `lincontrchange`, ...) have no
equivalent in `augment/ops.py`, so the re-applied damage would be neither the
recorded one nor a single application.

### Test 14: shrink x coverage factorial (`tile-factorial`)

**Claim decided:** whether squish's gain on large images comes from the share
of the image shown (coverage, factor f) or from the shrink (factor k), with
every view in the branch's trained presentation (a 200 px window resized to
n). This replaces Test 1 and Test 3 as the decisive design for that claim;
their write-ups stay.

**View `tiles(k, f)`.** For an image of h x w: take the centred rectangle of
area fraction f (sides round(h sqrt f) x round(w sqrt f)); shrink it by k with
squish's downscale kernel (`_DOWNSCALE_INTERP`); cover the result with 200 px
windows, ceil(L / 200) per axis, evenly spaced from 0 to L - 200 (one centred
window when ceil(L / 200) = 1); resize each window to n as `views_grid9`
does. The image score is the mean logit over windows. `tiles(1, 1)` is
`gridall`: the whole frame in native-pixel windows.

**Cells.** k in {1, 2} x f in {0.25, 0.5, 1}, plus `tiles(4, 1)`, plus
`center` and `squish` for reference.

**Rows.** `ntire_val` and `ntire_val_hard` with native short side >= 800 px
(4,068 rows, 1,235 distorted), so that every cell has shrunk-region sides of
at least 200 px and no window is enlarged beyond the trained 200 -> n.
Primary: both splits pooled. Each split is reported alone.

**Statistics** (robust AUC, distorted rows; clean AUC reported):

- dF = mean over k in {1, 2} of [AUC(k, 1) - AUC(k, 0.25)];
- dK = mean over f of [AUC(2, f) - AUC(1, f)];
- C = dF - dK.

**Conditions** (both branches):

- **Coverage:** the interval of dF is above zero and the interval of C is
  above zero.
- **Cleaning:** the interval of dK is above zero and the interval of C is
  below zero.
- **Both:** the intervals of dF and dK are above zero and C's includes zero.
- **Otherwise:** report the three per branch.

**Secondary,** reported, not a condition: gridall's recovered share
R_g = (AUC(gridall) - AUC(center)) / (AUC(squish) - AUC(center)). An interval
above 0.60 supports coverage without resampling. A low R_g is not evidence
for cleaning, because mean pooling over up to ~200 windows may lose signal on
its own (Test 9).

**Known limit.** Within a window, k also sets how much scene the window holds
(k^2 more area at k = 2). The factorial separates image-level coverage from
shrink; it cannot separate shrink from per-window content.

### Test 15: native 512 window for Branch A (`native512-proxy`)

**Claim decided:** whether part of Branch A's crop -> squish gap is a penalty
for enlarging a 200 px window 2.56x to 512 px. Test 8 tested this at 224 px,
where the enlargement is 1.12x, and found no such penalty.

**Arms.** Two proxies, one seed each (20260827), the G1 protocol with Branch
A's backbone (`dinov2regl`, depth 24, `--src-chunk 2`): `crop200@512`
(`--crop-side 200`) and `native512` (`--crop-side 512`), both with
`--nominal-side 512 --crop-clamp`. Everything else as Test 8. Scored with
`score_plan_splits.py` on `ntire_val_hard` and `ntire_val`, centre window.

**Timing first.** 100 timed steps. If one arm's projected time exceeds 7 h,
the test stops and is reported as not run. The protocol is not changed to
fit.

**Condition** (single seed, stated as such):

- **Enlargement penalty present:** robust AUC (native512 - crop200@512) on
  `ntire_val_hard` has an interval above zero. Its size is reported as a
  share of A2's squish - center robust gap, as a description.
- **Absent:** the interval includes zero or lies below it.

Images with a short side below 512 px are clamped (their largest square,
resized up), as the current policy does below 200 px. Their count is
reported.

### Re-analyses (CPU, reported as supplements)

| # | test | what is added |
|---|---|---|
| R1 | 2 | intervals on the level-averaged D per bucket (one draw set across conditions), and the sign-change cell with its interval; the L5 prediction is reported as untestable, because the levels are not matched across op types |
| R2 | 4 | JPEG-containing vs JPEG-free rows, as disjoint groups, gain difference with interval, per branch |
| R3 | 9 | squish - combiner per size bucket, with intervals |
| R4 | 11 | logits z-scored against each model's own reals before any cross-model logit comparison; the raw-logit 5.28 -> 3.94 figure is withdrawn |
| R5 | 12 | the recomputed-AUC count is corrected; E2 and E3 get intervals; E10 is withdrawn |

### Test 6

Test 6's routing variable `d` is not trained to detect absent damage, and the
review reports that an oracle router has no margin on condition 1. A refit
router with a non-inferiority margin, chosen after the condition 1 result,
would be a post-hoc test, so none is registered. **Read 3 of `ntire_test` is
suspended, not spent.** It does not run until Dion decides whether it runs
as registered, under a new registration, or not at all.

### Order

Test 13, then Test 14, then Test 15's timing, then Test 15's arms if the
timing allows. R1-R5 run on CPU alongside.

## Amendment 8 (2026-10-20, after Test 14's verdict, before any number for Test 16 exists)

Test 14 found both a coverage and a shrink effect on NTIRE (verdict BOTH,
both branches). Test 16 asks whether that holds off NTIRE. It reruns Test 14
unchanged on AIGIBench SocialRF. Nothing in Amendments 5-7 changes.

### Test 16: Test 14 on SocialRF (`tile-factorial-socialrf`)

**Views, cells, statistics and conditions:** exactly Test 14's
(`amend5_test14.views_tiles`, `CELLS`, and `amend5_test14_analysis`'s
statistics and verdict rule), frozen Branch A and Branch B.

**Rows.**
- Clean: `ext_aigi_socialrf`, is_distorted = 0.
- Distorted: `ext_aigi_socialrf_deg`, damaged at scoring time by
  `score_plan_splits.degrade_native` (seed 20261008, pool `ntire`, at most 5
  ops), the chain the paper's `_deg` numbers use. is_distorted = 1.
- Filter: the short side of the image as presented (after damage, which can
  include `downscale`) is at least 800 px, as in Test 14. Sizes after damage
  are computed in a CPU pass before any score exists.
- Subsample: 500 per label from each split after filtering, seed 20261019
  (2,000 rows). This keeps Branch A near 1.5 GPU-h.

Strata: label x is_distorted, as in Test 14. Robust AUC = AUC on the `_deg`
rows.

**Conditions:** Test 14's, on both branches: COVERAGE, CLEANING, BOTH or
NEITHER. R_g is reported as in Test 14.

**Known limits.**
- SocialRF's reals are larger than its fakes (median short side 1440 vs
  1024 px), so size is a label cue, and k changes it. The filter does not
  remove this.
- SocialRF's fakes were taken from social platforms and were probably
  re-encoded already, so "clean" means "no added damage".
- The `_deg` chain is our synthesis of the NTIRE families, not NTIRE's own
  damage.

### Order

Test 16 runs after Test 15 on the GPU queue.

## Amendment 9 (2026-10-24, after Test 16's verdict, before any number for Test 17 exists)

Tests 14 and 16 found a shrink effect dK on both branches and both row sets.
Their registration states that k also sets how much scene each window holds
(k^2 more area at k = 2), so dK mixes two things: the shrink averaging away
damage in the pixels (attenuation), and each window seeing more of the scene
(per-window content). Test 17 separates the two with a view that has the
averaged pixels of tiles(2, f) and the per-window scene of tiles(1, f).
Test 1 used the same round trip at squish's ratio (5-18x on large images),
which left mostly blur; Test 17 uses k = 2 only. Nothing in Amendments 5-8
changes.

### Test 17: shrink then enlarge back (`roundtrip`)

**View `rt(k, f)`.** Test 14's centred area-f region, shrunk by k with
squish's downscale kernel (`_DOWNSCALE_INTERP`, as `views_tiles`), enlarged
back to the region's own size with the window kernel (`_UPSCALE_INTERP`),
then covered with tiles(1, f)'s windows (same count, same places), each
resized to n. The image score is the mean logit over windows.

**Cells.** rt(2, f) for f in {0.25, 0.5, 1}. tiles(1, f) and tiles(2, f) are
the logits already scored by Tests 14 and 16, joined on path; they are not
re-scored.

**Rows.** Test 14's rows (NTIRE, short side >= 800, 4,068) and Test 16's rows
(SocialRF, 2,000, `_deg` rows damaged by the same chain and seed), unchanged.
Frozen Branch A and Branch B.

**Statistics** (robust AUC, distorted rows; clean AUC reported):

- dA = mean over f of [AUC(rt(2, f)) - AUC(tiles(1, f))] (attenuation: same
  scene per window, averaged pixels);
- dS = mean over f of [AUC(tiles(2, f)) - AUC(rt(2, f))] (per-window content:
  same averaged pixels, 4x the scene per window);
- dA + dS = dK of Test 14/16.

**Conditions** (per branch, per row set; the NTIRE rows pooled, each split
reported):

- **Attenuation:** dA's interval above zero, dS's not.
- **Content:** dS's interval above zero, dA's not.
- **Both:** both intervals above zero.
- **Neither:** otherwise.

**Known limits.**
- The enlargement back is itself a change no training window had: rt windows
  are softer than tiles(1, f)'s. If that costs AUC, dA understates
  attenuation and dS overstates content. Clean-row dA is reported as the
  size of that cost.
- tiles(2, f) averages about a quarter as many windows as rt(2, f), so dS
  also carries any mean-pooling dilution (Test 9).
- tiles(1, f) and tiles(2, f) logits come from the Test 14/16 runs, with the
  same weights and code, not from the same run as rt(2, f).
- SocialRF's size is a label cue (Test 16's limits apply unchanged).

### Order

Test 17 runs on the GPU queue when Dion confirms the GPU is free. Branch B
first, then Branch A, NTIRE rows before SocialRF rows. About 1.2 GPU-h for B
and 3 GPU-h for A at Test 16's rate (about 1.0 M windows per branch).

## Amendment 10 (2026-11-01, after Test 17's verdict, before any number for Test 18 exists)

Test 13 found that about half of Branch B's damage rise needs squish's
shrink (retained share Q = 0.55), which reads as the shrink reducing the
damage. Test 17 found no such reduction at k = 2 (dA about zero on both
branches and both row sets). Squish shrinks large images 5-18x; Test 17
shrinks 2x. Test 18 asks whether the damage reduction from the shrink
appears only at strong shrinks. It reuses Test 13's damage and Test 14's
tiles. Nothing in Amendments 5-9 changes.

### Test 18: damage before vs after the shrink (`shrinkorder`)

**Rows.** Test 2's / Test 13's clean NTIRE rows (`ntire_val` +
`ntire_val_hard`, is_distorted = 0, never `ntire_test`), cut by native short
side. Row rule, decided from row counts only: use k in {1, 2, 4, 8} with
short side >= 1,600 if that leaves >= 1,000 rows with both labels present;
otherwise k in {1, 2, 4} with short side >= 800. Outcome on the counts: short
side >= 1,600 leaves 679 rows (563 real, 116 fake), under 1,000, so **k in
{1, 2, 4}, short side >= 800: 2,833 rows (1,520 real, 1,313 fake;
2,212 `ntire_val`, 621 `ntire_val_hard`)**, k_max = 4. Frozen Branch A (A2,
n = 512) and Branch B (B2swa, n = 224).

**Damage.** Test 13's three ops at Test 2's heavy level (JPEG q30, blur
sigma 2.0, noise sigma 0.10), with Test 2's code and seeds: noise is seeded
per image by `default_rng([20261019, int(content_sha256[:15], 16)])`, the
same seed in both orders. Damage parameters are fixed in the pixel grid
where the damage is applied, as in Test 13. A native blur of sigma s is
about s/k after the shrink; that reduction is part of what is under test,
so it is deliberately not compensated.

**Views,** per branch, all Test 14's tiles(k, 1): the whole frame, shrunk by
k with INTER_AREA, covered by 200 px windows spread as in Test 14, each
resized to n, mean logit per image.

- clean(k): shrink by k, tile.
- before(op, k): damage at native size, shrink by k, tile.
- after(op, k): shrink by k, damage at the shrunk size, tile.

At k = 1 before and after are the same pixels. The smoke run scores both and
asserts identical logits; the full run scores k = 1 once per op. Coverage is
the whole frame at every k. Per-window content changes with k, identically
in both orders at a given k. Per-window logits are saved beside the mean.
Descriptive only: squish clean / damage before / damage after on the same
rows, joined on path from Test 2's and Test 13's logits (not rescored).

**Statistic,** per branch. Robust AUC = AUC over images damaged by that op
(real and fake both damaged). The damage reduction from the shrink is
R_op(k) = AUC(before(op, k)) - AUC(after(op, k)), and R(k) = mean over the
3 ops of R_op(k) (called S(k) in the design notes). Positive R means
damaging first helped: the shrink reduced the damage. R(1) = 0 by
construction. Reported: R(k) and R_op(k), clean AUC(k), and drop(k) = clean
AUC(k) - damaged AUC for each order. Intervals: 2,000 paired bootstrap draws
over images, 95% percentile, stratified by label (every row is clean),
seed 20260827, `paired_diff`'s index scheme.

**Conditions** (per branch; the pooled R is primary):

- **STRONG-ONLY:** R(2)'s CI includes zero and lies entirely within
  [-0.01, +0.01], AND [R(k_max) - R(2)]'s CI is above zero. The margin 0.01
  is just below the smallest content effect reported (Test 17, Branch B,
  NTIRE dS = +0.0097).
- **EVERYWHERE:** R(2)'s CI is above zero.
- **NONE:** every R(k)'s CI includes zero or is below it, AND
  [R(k_max) - R(2)]'s CI includes zero.
- **UNRESOLVED:** otherwise (including an R(2) CI that includes zero but
  extends past +/-0.01 while the rise is above zero).

Secondary: the same rule on each op's R_op (JPEG, blur, noise), per branch.
The output flags per-op verdicts that disagree with each other or with the
pooled one.

**Known limits.**
- Damage applied after the shrink is relatively stronger on the small view
  (as in Test 13): the same parameters act on fewer pixels of scene.
- Damage applied before the shrink is reduced and also rescaled (for example
  JPEG 8x8 blocks become 2x2 at 4x). R measures both. Test 18 tests whether R
  depends on shrink level, not which of these causes it. Noise is the
  cleanest case (the shrink mostly averages it); JPEG the least.
- The window count falls with k (about 84, 23 and 7 per image at k = 1, 2,
  4), so mean pooling dilutes differently across k (Test 9); it is the same
  in both orders at a given k.
- Rows whose shrunk short side is under 200 px would be clamped; the row
  rule leaves none (the shortest side at k = 4 is 200 px), and the code
  asserts it.
- Single source (clean NTIRE, >= 800 px).

### Budget and order

Projected at Test 17's measured rates (A 96, B 249 windows/s): about
1.56 M windows per branch, 4.5 GPU-h for A and 1.7 GPU-h for B. Before the
full run, about 20 rows (10 per label) are scored per view and the full run
is projected from them. The projection must be <= 8 GPU-h. If it is over,
the registered fallback is a label-stratified random subsample of rows
(seed 20260827), sized to fit 8 GPU-h, fixed and logged before any full
scoring. Branch B first, then Branch A, on the GPU queue
(`queue_a10.sh`), capped at 10 GiB on the shared GPU. The queue logs the md5
of the staged prereg diff and of the staged Test 18 code before scoring
starts.

## Amendment 11 (2026-11-01, during Test 18's run, before any number for Test 19 exists)

Two parts: a record of Dion's decision on Test 6 read 3, and Test 19.
Nothing in Amendments 5-10 changes.

### Test 6 read 3

Test 6 read 3 of `ntire_test`: not run, by Dion's decision (2026-11-01).
Test 6 failed condition 1 on `ntire_val_hard` and cannot pass; the
amendment-7 audit found an oracle router has no margin on condition 1; the
clean-cost comparison it would measure is confounded (checkpoint, round and
run settings, Test 11 R4); `ntire_test` stays reserved for the paper's final
numbers. The extensions have scored no `ntire_test` row.

### Test 19: does enlarging the window hurt, at equal view? (`stretch`)

**Claim decided:** whether enlarging the window hurts when the view is held
fixed. Test 15 found native512 beats crop200@512 by +0.0415 robust AUC on
`ntire_val_hard`, but the 512 window both avoids the 2.56x enlargement and
shows 2.56x more of the scene per side. Test 19 adds the missing cell: a
small window at its own size, with no enlargement and the same view as
crop200@512.

**Input size, decided before any training.** The registered arm is
`crop200@200`. DINOv2's patch is 14, and neither 200 nor 512 is a multiple of
it. The existing pipeline handles 512 by squishing the canonical image to the
tower's fixed `image_size` (`dinov2regl`: 518, `cv2.INTER_AREA`, in
`features.backbones.model_inputs`). At nominal 200 that same mechanism
squishes the window to 518, a 2.59x enlargement, which is exactly what this
arm must not have. So the trainer cannot run 200 unenlarged the same way, and
**the registered substitute applies: n = 196 with window 196 (`crop196@196`)**.
196 = 14 x 14 patches. It runs on `dinov2regl196`, a new registry entry for
the same tower and weights at `image_size` 196 (as `dinov2regl224` is at 224).
A CPU check confirms that a 196 px centre window reaches the tower with no
pixel changed. The view is 196 px instead of 200 px per side, a 2% change.

**Arm.** One proxy, one seed (20260827): Test 15's `crop200@512` arm
(`scripts/amend5_test15.sh`), flag for flag, except `--crop-side 196
--nominal-side 196 --backbone dinov2regl196`. So: the G1 protocol on Branch
A's tower (`dinov2regl` weights, depth 24, `--src-chunk 2`), 1 epoch over 25%
of train, the same optimiser, LR schedule (cosine, warm-up 0.03, SWA from
0.75), augmentation, family weighting and `--crop-clamp`. Scored with
`score_plan_splits.py` on `ntire_val_hard` (primary) and `ntire_val`, centre
window, Test 15's scoring flags. `crop200@512` and `native512` are not
retrained: their per-image logits are Test 15's parquets, joined on path, and
the analysis asserts the three arms have the same row set.

**Rows.** Primary: `ntire_val_hard` distorted rows with native short side
>= 512, where no arm clamps (630 rows). Secondary: all rows, `ntire_val`,
clean rows, and Test 15's size buckets.

**Statistics.** Robust AUC = AUC on the distorted rows.

- E = AUC(crop196@196) - AUC(crop200@512). Same view, enlarged or not.
  Positive means enlarging the window hurts (a stretch penalty).
- V = AUC(native512) - AUC(crop196@196). Both arms unenlarged; the
  difference is the view (and the token count).
- T = AUC(native512) - AUC(crop200@512) = V + E. This is Test 15's effect on
  these rows. The analysis asserts the identity, and that T on all distorted
  rows reproduces Test 15's +0.0415 (`ntire_val_hard`) and +0.0314
  (`ntire_val`) to 4 dp.

Intervals: 2,000 paired bootstrap draws, 95% percentile, stratified by
label x is_distorted, seed 20260827, `paired_diff`'s index scheme. E, V and T
share one draw set (rows in the order of Test 15's parquet).

**Condition on E** (`ntire_val_hard`, distorted, short side >= 512; single
seed, stated as such):

- **PENALTY:** E's CI is above zero.
- **BENEFIT:** E's CI is below zero (enlarging helps, for example through
  more tokens).
- **NONE:** E's CI lies within [-0.01, +0.01] (and so includes zero).
- **UNRESOLVED:** otherwise.

V and T are reported with intervals, and E/T and V/T as descriptions, by size
bucket as in Test 15.

**Known limits.**
- One seed and a proxy, as in Test 15.
- The enlargement also gives the tower about 7x more image tokens of the same
  detail (518 px: 37 x 37 = 1,369 patches; 196 px: 14 x 14 = 196). So E
  measures "same detail, more tokens", not pure resampling.
- Training augmentations whose strength is set in pixels act on the enlarged
  image in crop200@512 and on native pixels in crop196@196. That is part of
  what enlarging changes, and E includes it.
- Two arms' logits come from Test 15's runs (same code path, asserted on
  rows, row counts and T).
- Patch-size handling: 196 instead of 200 (2% less view per side); the tower
  runs at 196 with interpolated position embeddings, as `dinov2regl224` does
  at 224.

### Budget and order

Timing first, as in Test 15: 100 timed steps of this arm. If the projected
training time plus a 0.5 h scoring allowance exceeds 6 GPU-h, the test stops
and is reported as not run. The protocol is not shrunk to fit. Before the
run, a smoke run (a few training steps, a handful of rows scored) goes end to
end through the analysis; its numbers are not results. The run is queued
after Test 18 (`queue_a11.sh`). It starts only after `QUEUE A10 DONE` and
when at least 12 GB of GPU memory is free, with no memory cap. The queue logs
the md5 of the staged prereg diff and of the staged Test 19 code before
training starts.

## Amendment 12 (2026-11-01, during Test 18's run, before any number for Tests 20 or 21 exists)

Tests 13-18 decompose the squish gain on our two frozen DINOv2 branches only.
Amendment 12 asks whether that decomposition holds on two other detectors: our
CLIP ViT-L/14 crop-trained proxies (amendment 4) and B-Free (amendment 3).
Nothing in Amendments 5-11 changes. Dion's decision (2026-11-01) to run both.

Registration file: this text, saved as `outputs/test20/amendment12.md` with
its md5 logged before any Test 20 or Test 21 score. It is appended to
`docs/ntire/paper_prereg.md` (staged) only after `queue_a11` has logged its
own staged-diff md5, so Test 19's logged provenance is unchanged.

**Inclusion rule, fixed now.** If a test's analysis JSON exists by
2026-11-02 18:00 Singapore time, the paper reports it, whatever it shows. If
not, the paper states that the test was registered and had not finished at
submission. The decision depends only on the finish time, never on the result.

### Test 20: the decomposition on CLIP (`clip-decomp`)

**Models.** The three crop-trained CLIP proxies of amendment 4
(`clip_crop_s1`, `_s2`, `_s3`; `clipl`, window 200, nominal 224; checkpoints
in `/scratch/justin_r13/aigcdet/outputs/paper_clip/`), frozen, loaded as
`amend5_test2.OurBranch` loads A2 and B2swa (no SWA). The scorer asserts each
policy is crop, window 200, nominal 224. No training.

**Rows.** Test 16's rows unchanged (`outputs/test16/rows.parquet`: SocialRF
clean and `_deg`, short side >= 800 after damage, 500 per label per split,
2,000 rows; `_deg` damaged by `amend5_test16.presented`).

**Views.** Per image, in one pass: `center`, `squish`, Test 14's seven
tiles(k, f) cells and Test 17's three rt(2, f) cells, with the code of those
tests unchanged. Mean logit per image.

**Statistics, per seed.** Test 16's statistics and verdict rule (Test 14's
code: dF, dK, C; COVERAGE, CLEANING, BOTH, NEITHER) and Test 17's (dA, dS;
ATTENUATION, CONTENT, BOTH, NEITHER), robust AUC on the `_deg` rows. Same
bootstrap as Tests 14 and 17 (2,000 draws, seed 20260827, strata label x
is_distorted).

**Verdict.** For each rule, the verdict that at least two of the three seeds
share; otherwise SEEDS DISAGREE. Descriptive: each AUC averaged over seeds.

**What it decides.** The paper's claims carry to CLIP if the Test 16 rule
gives CLEANING (the label Test 16 gave both DINOv2 branches: the downscale
effect dK above zero and larger than the coverage effect dF) and the Test 17
rule gives CONTENT (more scene per view, no measurable degradation removal).
Any other verdict is reported as a difference between backbones.

**Floor.** If the seed-mean robust AUC of `center` on these rows is below
0.60, the detector is too weak for the decomposition to mean anything: the
numbers are reported and no verdict is read.

**Budget.** Timing on 20 rows (10 per label per split) first, for one seed.
If three seeds project over 6 GPU-h on the RTX 4090, only `clip_crop_s1` runs
and its verdicts stand, marked single-seed. Runs on the GPU queue after
`QUEUE A11 DONE` with at least 12 GB free, no memory cap (`queue_a12.sh`).

**Known limits.** Proxy models (1 epoch, 25% of train), weaker than the full
branches. One held-out set. Test 17's window-count confound applies unchanged.

### Test 21: the shrink x coverage factorial on B-Free (`bfree-tiles`)

**Model.** B-Free (`BFREE_dino2reg4`, authors' checkpoint, ViT-B/14 with
registers at 504 px), loaded by `public_detector_probe.BFree`. Its `--selfcheck`
(the authors' four demo images to 1e-3) must pass on the machine that scores
before any row is scored.

**Window.** B-Free trains on 504 px crops, so its tiles use 504 px windows fed
at 504 px with no resize: tiles504(k, f) is Test 14's tiles(k, f) with window
504 in place of 200 and no resize to n. Same seven cells, same region rule,
same INTER_AREA shrink, same window spacing. For a 504 px input the authors'
five-crop wrapper sees five identical crops, so the scorer runs one forward per
window and asserts, on the smoke rows, that it equals `BFree.logits` to 1e-4.
Reference views: `native` (the authors' five-crop on the whole image, as in
Test 2) and `squish` (504, as in Test 2).

**Rows.** SocialRF clean and `_deg` (damage as in Test 16), short side as
presented >= 2,016 px (so that tiles504(2, 0.25) and tiles504(4, 1) have sides
of at least 504), up to 500 per label per split (seed 20261019, as Test 16). The
row step runs on the RTX 4090 box from the manifest and stores, per row, the
SHA-256 of the presented pixels. The scorer may run on the second box (RTX 4060
Ti, `labbox`); it recomputes and asserts every SHA-256 before scoring, so both
boxes see identical pixels.

**Statistics.** Test 14's code unchanged, with `native` in `center`'s role (it
enters only R_g, which is reported, not read): dF, dK, C and the verdict rule,
robust AUC on the `_deg` rows. Same bootstrap.

**What it decides.** B-Free relies on native-resolution traces, and Test 2
found squish costs it AUC at every degradation level. Prediction: dK not above
zero. If dK's interval is above zero, the downscale effect carries even to a
native-trace detector. If not, B-Free marks where the effect stops, which the
paper states.

**Budget.** Timing on 10 rows first. If the projection is over 8 GPU-h on the
scoring box, a label-stratified random subsample (seed 20260827), sized to fit
and logged before any full scoring, replaces the rows.

**Known limits.** Different window (504 vs 200) and a different detector
family, so a difference from Test 16 cannot be pinned on one cause. Fewer
fakes than reals (SocialRF has few large fakes). Only images >= 2,016 px. Test
16's notes on SocialRF ("clean" is not clean; the size cue) apply.
