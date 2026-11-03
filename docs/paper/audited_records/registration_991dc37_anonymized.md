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
on its own branch off this amendment's commit, `researcher/test<N>-<slug>`. Its
result goes to `docs/ntire/amend5/test<N>_<slug>.md`, with JSON and
per-image parquets beside it.

### Open questions (unresolved when this amendment was written)

1. **Access.** The frozen checkpoints, the proxy checkpoints, the external
   images and the B-Free weights are under `/scratch/anonymous`, which this
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
