# Test 3: shrink ratio vs field of view

Amendment 5, Test 3. Branch `researcher/test3-shrink-ratio`, from `991dc37`.
Run 2026-10-19. `ntire_val_hard` is a fit split, so every number on it is a
fit-split number.

## 1. Summary & results

**Registered verdict: NOT JUDGED. The registered regression cannot be
fitted, because its design matrix is singular by construction.** The pass
condition reads: "the area-coverage coefficient's CI excludes zero, and the
S_short and S_long coefficients' CIs include zero. In the Figure 1 re-plot,
the two branches' curves line up on FOV and not on S." With the registered
covariates, log area_coverage is an exact linear function of the intercept,
`branch`, log S_short and log S_long. The data confirm this: the design
matrix has rank 5 of 6, R^2 = 1.000000. The three coefficients the
condition names have no unique values, so they have no CIs. Following the
plan's rule ("if a test cannot be run as registered, say so ... and stop"),
the OLS and its bootstrap were **not run**. No substitute model was fitted.

The registered fallback is: "report both, and say the two variables cannot
be separated on this data". That fallback describes this outcome, with one
correction. The cause is the model specification, not a lack of variation
in the data.

**What did run: the Figure 1 re-plot, as registered.** On both splits, the
two branches' gain curves are closer when matched on FOV than when matched
on S:

| split | mean abs gap, matched on FOV (8 bins) | mean abs gap, matched on S (bins in overlap) |
|---|---|---|
| `ntire_val_hard` | 0.024 | 0.057 (5 bins) |
| `ntire_val` | 0.007 | 0.021 (6 bins) |

This is the figure half of the pass condition. The "gap" statistic is an
unregistered, descriptive summary of what the figure shows. Section 5.3
explains what this comparison can and cannot separate.

## 2. Current state

The paper attributes squish's gain to field of view (FOV). The 200 px
centre window covers 39-52% of the short side in training and as little as
5% at test. Amendment 5 raised a competing explanation: squish also shrinks
large images (shrink ratio S = short / n), and shrinking cleans damage.
Within one branch, log FOV = -log S + const, so the two cannot be separated
there. The test proposed two contrasts that might separate them: across
branches (same image and FOV, S differs by 512/224 = 2.29) and across
aspect ratio.

## 3. Opportunities

- The covariates fail on two counts, and amendment 6 can correct both before
  any coefficient is computed. Section 7 lists the options. No coefficient
  has been looked at, so the correction can still be pre-registered
  cleanly.
- Test 1 (`freqmatch`) separates coverage from cleaning by intervention
  rather than by regression. It does not depend on this design. It is
  blocked on researcher’s access.

## 4. Testing methodology

### 4.1 Data

The inputs are the same parquets as Test 4: `center` and `squish` logits of
A2 (n = 512) and B2swa (n = 224). `ntire_val_hard` is primary (2,500
images) and `ntire_val` is secondary (10,000 images). On `ntire_val` the
centre arm is the as-trained score (see Test 4, Section 4.1). No model was
loaded and no image was read. The script refuses `ntire_test`.

### 4.2 Registered quantities

- z-scores: each (branch, arm) logit is standardised with the mean and SD of
  that arm on all of `ntire_val`. Values are in the JSON under `zscore`.
  They match `fusion_center_valfit.json` for the centre arms.
- Signed gain g = (2y - 1)(z_squish - z_center), one row per image per
  branch.
- S_short = short / n, S_long = long / n, area_coverage = 200^2 / (h * w).

### 4.3 Presentation check

`ntire_crop_probe.views_center` calls `canonicalise` with the frozen eval
policy (`mode=crop, crop_side=200, crop_clamp=True`). That takes a
`min(200, h, w)` window at native resolution and upscales it to n. There is
no pre-window downscale at eval. For images with short >= 200, the
realised area coverage therefore equals the registered 200^2 / (h * w)
exactly.

### 4.4 Rank check (added before fitting)

Before fitting, the script builds the registered design matrix
[1, branch, log S_short, log S_long, log area_coverage, is_distorted]. It
reports the matrix rank, the condition number, and the R^2 of log area
regressed on the other columns. The same check is repeated with the
realised clamped window, min(200, short)^2 / (h * w), for information only.

## 5. Findings

### 5.1 The registered model is singular by construction

Substituting the definitions gives:

```
log area = 2 log 200 - log h - log w
         = 2 log 200 - (log S_short + log n) - (log S_long + log n)
         = 2 log 200 - 2 log n_A - 2 (log n_B - log n_A) * branch - log S_short - log S_long
```

`branch` takes two values, so log n is a linear function of it. The regression
of log area on the other columns recovers exactly this identity:

| split | rank / columns | condition number | R^2 (area on others) | coef on log S_short, log S_long | coef on branch | residual SD |
|---|---|---|---|---|---|---|
| `ntire_val_hard` | 5 / 6 | 6.9e15 | 1.000000 | -1.000, -1.000 | +1.653 (= 2 log(512/224)) | 3.1e-15 |
| `ntire_val` | 5 / 6 | 1.5e15 | 1.000000 | -1.000, -1.000 | +1.653 | 2.3e-15 |

The amendment's "Across aspect" argument holds for a model with S_short and
area only: area varies at fixed short side through aspect. With S_long
also in the model, the variation from aspect is absorbed by S_long. The
"Across branches" argument fails for a different reason. On the same image,
log S differs between branches by the constant log(512/224), which the
`branch` term absorbs completely. In both cases the variation the
amendment relied on is spent before area_coverage enters the model.

### 5.2 The realised window does not rescue it

With the clamped window, rows with short < 200 have coverage short/long
instead of 200^2 / (h * w). That breaks exact collinearity: rank 6 of 6,
condition number 107 (`ntire_val_hard`) and 83 (`ntire_val`). But log area
is still predicted at R^2 = 0.992 and 0.989. All of its independent
variation comes from the 118 (`ntire_val_hard`) and 746 (`ntire_val`)
smallest images, the ones where the window is the whole short side. Any
coefficient fitted this way would measure "is the image smaller than 200
px", not FOV. It is not the registered definition either. It was not
fitted.

### 5.3 Figure 1 re-plot: the curves line up on FOV

![gain vs FOV and vs S](test3_shrink-ratio_fig.pdf)

*Figure 1.* Robust AUC gain (squish minus centre) per branch, in 8
equal-count bins of native short side (distorted `ntire_val_hard` rows, the
same bin edges for both branches), with 95% paired CIs. Left: x = centre
window FOV. Right: x = squish shrink ratio S. File:
`docs/ntire/amend5/test3_shrink-ratio_fig.pdf`.

On the FOV axis the two curves overlap within their CIs in every bin, and
both rise as FOV falls. On the S axis, B's curve is shifted right by 2.29x.
At equal S, A's gain is larger than B's. For example, at S ~ 0.9, A
(short 448) gains +0.111, while B (short 208) gains +0.024.

**What this can and cannot show.** Matching on FOV compares the same images
under two models. Matching on S compares A on images 2.29x larger than B's.
"The curves line up on FOV" therefore means the gain follows the image's
native size, not the branch's shrink factor. The test cannot tell which
property of native size matters: FOV, how much damage is averaged away
when a large image is shrunk to 512 vs 224, or other properties that
co-vary with size (content, source, damage chain). The cross-branch
contrast gives mixed evidence on "S is the variable" in its simplest form.
B shrinks every image 2.29x more than A. Its robust AUC gain is smaller
than A's in 7 of 8 bins (bin 7 is the exception), but its mean signed gain
g is larger (+0.091 vs +0.042 z units). The two branches differ in
architecture and input size as well as in S, so neither direction can be
attributed to S.

### 5.4 Aspect gives little leverage

The long/short ratio has p5 = 1.00, p50 = 1.45 and p95 = 1.79 on
`ntire_val_hard` (p50 = 1.35 on `ntire_val`). The corr of log S_short with
log S_long is 0.974 on both splits. Even with a correctly specified model,
the aspect contrast would rest on a narrow range. This matches open
question 2 of amendment 5.

## 6. Results

### 6.1 `ntire_val_hard` (primary, fit split), distorted rows

| branch | bin | n | short p50 | FOV p50 | S p50 | robust AUC centre -> squish | gain [95% CI] |
|---|---|---|---|---|---|---|---|
| A | 0 | 157 | 208 | 0.962 | 0.406 | 0.9088 -> 0.9504 | +0.0417 [+0.0099, +0.0794] |
| A | 1 | 121 | 304 | 0.658 | 0.594 | 0.8576 -> 0.9461 | +0.0886 [+0.0364, +0.1504] |
| A | 2 | 171 | 368 | 0.543 | 0.719 | 0.8830 -> 0.9465 | +0.0636 [+0.0212, +0.1160] |
| A | 3 | 171 | 448 | 0.446 | 0.875 | 0.8512 -> 0.9617 | +0.1105 [+0.0588, +0.1705] |
| A | 4 | 161 | 560 | 0.357 | 1.094 | 0.8317 -> 0.9421 | +0.1104 [+0.0529, +0.1792] |
| A | 5 | 146 | 656 | 0.305 | 1.281 | 0.8150 -> 0.9352 | +0.1203 [+0.0494, +0.1953] |
| A | 6 | 158 | 832 | 0.240 | 1.625 | 0.8553 -> 0.9643 | +0.1091 [+0.0529, +0.1679] |
| A | 7 | 165 | 1360 | 0.147 | 2.656 | 0.8495 -> 0.9731 | +0.1236 [+0.0617, +0.1896] |
| B | 0 | 157 | 208 | 0.962 | 0.929 | 0.9381 -> 0.9621 | +0.0240 [-0.0043, +0.0580] |
| B | 1 | 121 | 304 | 0.658 | 1.357 | 0.9049 -> 0.9716 | +0.0667 [+0.0202, +0.1238] |
| B | 2 | 171 | 368 | 0.543 | 1.643 | 0.9301 -> 0.9792 | +0.0491 [+0.0229, +0.0821] |
| B | 3 | 171 | 448 | 0.446 | 2.000 | 0.8840 -> 0.9633 | +0.0793 [+0.0356, +0.1287] |
| B | 4 | 161 | 560 | 0.357 | 2.500 | 0.8909 -> 0.9598 | +0.0690 [+0.0210, +0.1214] |
| B | 5 | 146 | 656 | 0.305 | 2.929 | 0.8748 -> 0.9576 | +0.0828 [+0.0250, +0.1458] |
| B | 6 | 158 | 832 | 0.240 | 3.714 | 0.8390 -> 0.9333 | +0.0943 [+0.0358, +0.1552] |
| B | 7 | 165 | 1360 | 0.147 | 6.071 | 0.8355 -> 0.9695 | +0.1340 [+0.0705, +0.1997] |

Mean signed gain g over all 2,500 images: A +0.042, B +0.091 (z units).

### 6.2 `ntire_val` (secondary), distorted rows

| branch | bin | n | short p50 | FOV p50 | S p50 | robust AUC centre -> squish | gain [95% CI] |
|---|---|---|---|---|---|---|---|
| A | 0 | 585 | 192 | 1.000 | 0.375 | 0.9805 -> 0.9784 | -0.0021 [-0.0101, +0.0048] |
| A | 1 | 657 | 272 | 0.735 | 0.531 | 0.9713 -> 0.9866 | +0.0153 [+0.0073, +0.0244] |
| A | 2 | 581 | 336 | 0.595 | 0.656 | 0.9726 -> 0.9850 | +0.0124 [+0.0043, +0.0217] |
| A | 3 | 611 | 416 | 0.481 | 0.812 | 0.9633 -> 0.9810 | +0.0176 [+0.0063, +0.0298] |
| A | 4 | 600 | 512 | 0.391 | 1.000 | 0.9575 -> 0.9823 | +0.0247 [+0.0119, +0.0376] |
| A | 5 | 709 | 608 | 0.329 | 1.188 | 0.9548 -> 0.9842 | +0.0294 [+0.0161, +0.0439] |
| A | 6 | 621 | 800 | 0.250 | 1.562 | 0.9645 -> 0.9869 | +0.0224 [+0.0107, +0.0360] |
| A | 7 | 636 | 1392 | 0.144 | 2.719 | 0.9398 -> 0.9967 | +0.0569 [+0.0375, +0.0800] |
| B | 0 | 585 | 192 | 1.000 | 0.857 | 0.9799 -> 0.9829 | +0.0030 [-0.0037, +0.0090] |
| B | 1 | 657 | 272 | 0.735 | 1.214 | 0.9815 -> 0.9896 | +0.0081 [+0.0022, +0.0151] |
| B | 2 | 581 | 336 | 0.595 | 1.500 | 0.9773 -> 0.9861 | +0.0088 [+0.0011, +0.0167] |
| B | 3 | 611 | 416 | 0.481 | 1.857 | 0.9703 -> 0.9784 | +0.0082 [-0.0042, +0.0211] |
| B | 4 | 600 | 512 | 0.391 | 2.286 | 0.9559 -> 0.9844 | +0.0285 [+0.0160, +0.0424] |
| B | 5 | 709 | 608 | 0.329 | 2.714 | 0.9591 -> 0.9865 | +0.0274 [+0.0154, +0.0401] |
| B | 6 | 621 | 800 | 0.250 | 3.571 | 0.9582 -> 0.9873 | +0.0291 [+0.0162, +0.0432] |
| B | 7 | 636 | 1392 | 0.144 | 6.214 | 0.9578 -> 0.9968 | +0.0391 [+0.0219, +0.0611] |

Mean signed gain g over all 10,000 images: A +0.032, B +0.040 (z units).

### 6.3 Verdict table

| condition part | status |
|---|---|
| area-coverage coefficient CI excludes 0 | not computable (coefficient not identified) |
| S_short, S_long coefficient CIs include 0 | not computable (coefficients not identified) |
| Figure 1: curves line up on FOV, not on S | yes, on both splits (descriptive; Section 5.3 limits) |
| registered verdict | **not judged**; Amendment 6 records the model as not identifiable |

## 7. Future work

**Decided 2026-10-19: option 3.** Amendment 6 in `docs/ntire/paper_prereg.md`
(staged on `researcher/amend5`) registers Test 3 as not identifiable on this
design, with the Figure 1 re-plot reported as a description only. Options 1
and 2 are kept below for the record. Neither was run.

The options that were considered:

1. **Reparametrise in identifiable terms.** With `branch` in the model, the
   size information is two-dimensional (log short, log long). One option is
   g ~ branch + log short + log aspect + is_distorted. Here "FOV only"
   predicts a non-zero log short coefficient, and the aspect coefficient
   then tests area coverage against short-side FOV. This separates two
   kinds of FOV. It does not separate FOV from S.
2. **Drop the branch main effect and test S across branches**, by fitting
   g ~ f(log S) and g ~ f(log FOV) with a common curve for both branches,
   and comparing the fits. This is the parametric form of Section 5.3, and
   it has the same limit: matching on S compares different images.
3. **Rely on Test 1 (`freqmatch`)** for the coverage-vs-cleaning question,
   and register Test 3 as "not identifiable; Figure 1 re-plot reported".
   This is the smallest change and is the recommendation.

## 8. What didn't work

- **The registered regression:** singular by construction (Section 5.1).
  The amendment did not foresee that S_long absorbs the aspect variation
  and that `branch` absorbs the constant cross-branch S offset.
- **The realised clamped window:** non-singular, but identified only by the
  images below 200 px (Section 5.2). Not fitted.

## Files

- `scripts/amend5_test3.py`: rank check, z-scores, Figure 1 re-plot. Runs in
  under 30 s, CPU only. Run it from the worktree root with
  `PYTHONPATH=$PWD/src`.
- `docs/ntire/amend5/test3_shrink-ratio.json`: every number above,
  including the singular values and the z-score parameters.
- `docs/ntire/amend5/test3_shrink-ratio_fig.pdf`: Figure 1.

Numbers from commit `991dc3726bf58e1bcf7a8f26270e71205d8c4979`, with the
uncommitted script above.
