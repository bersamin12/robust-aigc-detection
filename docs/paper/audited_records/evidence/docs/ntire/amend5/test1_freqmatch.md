# Test 1: coverage vs cleaning (`freqmatch`)

Amendment 5, Test 1. Branch `researcher/test1-freqmatch`, from `bbb1306`.
Run 2026-10-19. Fit-split numbers: `ntire_val_hard` was used to choose
squish, so every number on it below is a fit-split number.

## 1. Summary & results

**Registered verdict (`ntire_val_hard`): COVERAGE HOLDS, on both branches.**
The registered condition reads: "**Coverage holds:** R < 0.40 on both
branches." with R = (robust AUC freqmatch - centre) / (robust AUC squish -
centre), per branch, on `ntire_val_hard`.

| branch | robust AUC centre | freqmatch | squish | freqmatch - centre [95% CI] | squish - centre [95% CI] | R |
|---|---|---|---|---|---|---|
| A (A2) | 0.8561 | 0.8467 | 0.9482 | -0.0094 [-0.0187, -0.0001] | +0.0920 [+0.0727, +0.1123] | **-0.10** |
| B (B2swa) | 0.8872 | 0.8732 | 0.9599 | -0.0141 [-0.0272, -0.0011] | +0.0727 [+0.0567, +0.0901] | **-0.19** |

R is below 0.40 on both branches. It is also below zero: giving the centre
window squish's frequency content does not recover any of squish's gain, and
costs a little. The secondary split (`ntire_val`) reads the same way (A
R = -0.37, B R = -0.76).

Two qualifications. Section 5 has the detail.

1. **The negative R comes from the largest images.** In the 752+ px bucket,
   freqmatch loses 9-13 points of robust AUC against centre (A 0.8626 to
   0.7740, B 0.8303 to 0.7033). There, squish shrinks the image 1.5x to 8x,
   and the bicubic resize back leaves a window that is mostly blur. The
   crop-trained branches never saw such windows. The registered statistic
   therefore includes an out-of-distribution penalty, not only "cleaning
   without coverage" (caveat C2 below).
2. **Without that bucket the reading does not change.** In the other three
   buckets, freqmatch gains over centre, but the gain is small next to
   squish's: R per bucket is 0.07-0.22 (Table 5.3). That is still below
   0.40 everywhere. Cleaning explains at most about a fifth of the gain in
   any bucket.

## 2. Current state

The paper attributes squish's robust gain over the 200 px centre window to
coverage: squish shows the whole frame. Amendment 5 names a competing
explanation. Squish also shrinks the image, which averages away noise and
JPEG blocking ("cleaning"). Both predict a larger gain on larger images, so
the size gradient alone cannot separate them. Test 4 (damage family) found
the gain spread across all families (supports coverage), with a
JPEG-specific excess on `ntire_val`. Test 3's regression could not be
identified (amendment 6), which left this test to decide.

## 3. Opportunities

- If coverage holds, the paper can keep "the window is the bottleneck" as
  its mechanism, and the input-policy x size framing stands.
- If cleaning explained it, a cheap pre-filter (downscale then upscale)
  would give most of squish's gain inside the existing crop pipeline.
  This test says it does not.

## 4. Testing methodology

### 4.1 Presentation `freqmatch`

Implemented as `views_freqmatch` in this branch's copy of
`scripts/ntire_crop_probe.py` (decision D3):

1. squish to n x n exactly as `views_squish` does (n = 512 for A, 224 for B);
2. resize back to the native h x w with bicubic (`_UPSCALE_INTERP`, the
   crop policy's upscale kernel), whichever way each axis goes;
3. apply the branch's own crop policy (`canonicalise`: 200 px centre
   window, resized to n).

### 4.2 Caveats recorded before the run (open loop L6)

- **C1, extra resample round trip.** Step 2 is a resample the squish arm
  does not have. On images where squish upsamples (short side < n), the
  round trip down again is a mild low-pass of its own.
- **C2, anisotropic band-limit.** Squish shrinks the long axis more than the
  short one. Freqmatch carries that anisotropy into the window, so its
  pixels are blurred more along one axis. Neither branch was trained on
  that.
- **C3, no cleaning when short <= n.** If the short side is at or below n,
  squish upsamples (at least on that axis). Nothing is averaged away there,
  so freqmatch tests "cleaning" only on images larger than n. For A
  (n = 512) that is about half the rows; for B (n = 224) almost all of
  them.

### 4.3 Scoring

Frozen checkpoints, as registered: A2 final
(`unfreeze/ntire2_d24_518/checkpoint.pt`) and B2swa
(`dual/ntire2_dual224/checkpoint.pt`, SWA weights). `ntire_crop_probe.py`
scored `center` and `freqmatch` on `ntire_val_hard` (2,500 images) and
`ntire_val` (10,000), with `--workers 8`, default batch and chunk. Squish
logits are the existing probe parquets
(`outputs/scores/probe_{A2,B2swa}_squish_{split}.parquet`), produced by the
same script and checkpoints.

### 4.4 Reproduction check

The re-scored centre arm was compared with the frozen centre scores
(probe parquet on `ntire_val_hard`; the as-trained parquet on `ntire_val`,
D14).

| branch | split | rescored robust AUC | frozen | rows with abs(logit diff) > 1e-3 | max abs diff |
|---|---|---|---|---|---|
| A | ntire_val_hard | 0.85613 | 0.85615 | 2,225 / 2,500 | 0.269 (median 0.011) |
| B | ntire_val_hard | 0.88722 | 0.88722 | 0 | 0.000 |
| A | ntire_val | 0.96166 | 0.96166 | 0 | 0.000 |
| B | ntire_val | 0.96688 | 0.96687 | 8,572 / 10,000 | 0.619 (median 0.007) |

Two of four reproduce bit for bit. The other two differ by small amounts on
most rows, in a way that leaves robust AUC equal to 4 dp. That pattern is
consistent with bfloat16 forward passes run at a different batch
composition when the frozen files were made. It was not traced further. All
comparisons below use the re-scored centre, so centre and freqmatch come
from the same code path and run.

### 4.5 Statistics

`paired_bootstrap.paired_bootstrap`: 2,000 resamples, stratified by label x
is_distorted, seed 20260827, the same indices for centre, freqmatch and
squish. R is a ratio of the two point differences. The registered rule is on
the point value, so R has no interval.

## 5. Findings

### 5.1 Robust and clean AUC, `ntire_val_hard` (fit split, primary)

| branch | arm | robust AUC | clean AUC |
|---|---|---|---|
| A | centre | 0.8561 | 0.9913 |
| A | freqmatch | 0.8467 | 0.9724 |
| A | squish | 0.9482 | 0.9906 |
| B | centre | 0.8872 | 0.9801 |
| B | freqmatch | 0.8732 | 0.9275 |
| B | squish | 0.9599 | 0.9886 |

Paired differences [95% CI]:

| branch | comparison | robust | clean |
|---|---|---|---|
| A | freqmatch - centre | -0.0094 [-0.0187, -0.0001] | -0.0190 [-0.0250, -0.0134] |
| A | squish - centre | +0.0920 [+0.0727, +0.1123] | -0.0007 [-0.0050, +0.0038] |
| A | freqmatch - squish | -0.1015 [-0.1213, -0.0829] | -0.0183 [-0.0261, -0.0111] |
| B | freqmatch - centre | -0.0141 [-0.0272, -0.0011] | -0.0526 [-0.0638, -0.0413] |
| B | squish - centre | +0.0727 [+0.0567, +0.0901] | +0.0085 [+0.0028, +0.0151] |
| B | freqmatch - squish | -0.0868 [-0.1049, -0.0701] | -0.0611 [-0.0738, -0.0487] |

Freqmatch is below centre on clean rows too (A -1.9, B -5.3 points). A
pure cleaning effect should not hurt clean images. This is the
out-of-distribution penalty of caveat C2.

### 5.2 `ntire_val` (secondary)

| branch | robust: centre / freqmatch / squish | freqmatch - centre | squish - centre | R |
|---|---|---|---|---|
| A | 0.9617 / 0.9529 / 0.9853 | -0.0087 [-0.0111, -0.0065] | +0.0237 [+0.0194, +0.0279] | -0.37 |
| B | 0.9669 / 0.9515 / 0.9871 | -0.0154 [-0.0189, -0.0121] | +0.0202 [+0.0162, +0.0244] | -0.76 |

### 5.3 By native size bucket (descriptive, not registered)

Robust AUC on `ntire_val_hard`. R per bucket = (freqmatch - centre) /
(squish - centre).

| branch | bucket (short side) | n distorted | centre | freqmatch | squish | R |
|---|---|---|---|---|---|---|
| A | 80-336 | 314 | 0.8777 | 0.8887 | 0.9386 | 0.18 |
| A | 352-512 | 331 | 0.8734 | 0.8820 | 0.9632 | 0.10 |
| A | 528-736 | 294 | 0.8091 | 0.8182 | 0.9372 | 0.07 |
| A | 752+ | 311 | 0.8626 | 0.7740 | 0.9688 | -0.83 |
| B | 80-336 | 314 | 0.9195 | 0.9292 | 0.9642 | 0.22 |
| B | 352-512 | 331 | 0.9110 | 0.9145 | 0.9724 | 0.06 |
| B | 528-736 | 294 | 0.8805 | 0.8870 | 0.9602 | 0.08 |
| B | 752+ | 311 | 0.8303 | 0.7033 | 0.9524 | -1.04 |

The largest R in any bucket is 0.22 (B, smallest images). There, per C3,
squish upsamples, so that gain is not "cleaning" in the registered sense
either. Cleaning would predict R rising with size. It falls instead. In the
two middle buckets, where squish shrinks moderately and the window is not
yet degenerate, R is 0.06-0.10.

## 6. Results

- Registered: **COVERAGE HOLDS** (R = -0.10 for A, -0.19 for B, both < 0.40).
- Robust to the out-of-distribution concern: excluding the 752+ bucket,
  R stays at or below 0.22 in every bucket on both branches.
- For the paper: "Matching squish's frequency content inside the centre
  window recovers none of squish's robust gain (R < 0 on both branches;
  at most 0.22 within any size bucket), so the gain is attributed to field
  of view, not to the shrink."

## 7. Future work

- A milder cleaning probe without the anisotropic blur: an isotropic
  downscale of the short side to n, then back up, then the centre window.
  It would remove C2's penalty from the large-image bucket. It is not
  registered and was not run.
- Test 10 (masking) asks the complementary question: under squish, does
  the periphery carry more evidence than the centre?

## 8. What didn't work

- Two of the four centre reproductions are not bit-exact (Section 4.4).
  AUC agrees to 4 dp, so no number here depends on it, but the cause
  (assumed to be bf16 batch composition) is not proven.
- R has no interval. A ratio CI would need a new resampler, which the
  rules forbid. The two differences it is built from have paired CIs.

Numbers from commit `bbb1306` plus this branch's staged files
(`scripts/ntire_crop_probe.py` with `views_freqmatch`,
`scripts/amend5_test1.py`). JSON: `docs/ntire/amend5/test1_freqmatch.json`.
Per-image rows: `/scratch/anonymous/dion_extensions/outputs/test1/`.
