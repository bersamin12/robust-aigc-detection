# Test 10: masking test

Amendment 5, Test 10. Branch `dion/test10-masking`, from `bbb1306`.
Run 2026-10-19. Fit-split numbers: `ntire_val_hard` was used to choose
squish, so every number below is a fit-split number.

## 1. Summary & results

**Registered verdict: FAIL, on both branches.** The registered condition
reads: "**Pass:** on distorted rows, robust AUC (mask_centre -
mask_periphery) has a CI above zero on both branches. Masking the periphery
costs more." The difference has a CI **below** zero on both branches.
Masking the centre costs more than masking the periphery.

| branch | squish | mask_centre | mask_periphery | mask_centre - mask_periphery [95% CI] |
|---|---|---|---|---|
| A (A2) | 0.9479 | 0.9250 | 0.9379 | -0.0128 [-0.0210, -0.0042] |
| B (B2swa) | 0.9599 | 0.9421 | 0.9538 | -0.0116 [-0.0201, -0.0034] |

(Robust AUC on distorted rows, n = 1,250.)

What this means for the paper, with the detail in Section 5:

1. **The registered prediction is wrong in its geometry, not in its
   mechanism.** Per unit of area, the middle of the squished frame carries
   more evidence than its outer rim. The periphery mask is a 34 px rim
   (A) or 15 px rim (B). The centre mask is the middle quarter of the frame.
2. **Coverage is still supported, by a supplementary comparison (not
   registered).** Squish with the centre masked has none of the pixels the
   200 px centre window shows on most images. It still beats the centre
   window by +6.9 (A) and +5.5 (B) robust points, with CIs above zero
   (Section 5.3). The evidence outside the window is enough on its own to
   carry most of squish's gain.
3. The paper should not claim "the periphery is where the evidence is". It
   can claim "evidence outside the centre window is enough to beat the
   window".

## 2. Current state

Test 1 found that squish's gain is not explained by cleaning (R < 0 on both
branches): the field of view is the mechanism. Test 10 asks where in the
field of view the evidence sits, by masking 25% of the squished frame in
one of two places.

## 3. Opportunities

- A pass would have let the paper say that the part of the scene the window
  discards is the part the model reads. The fail removes that sentence.
- The supplementary comparison gives the paper a cleaner version of the
  coverage argument than the registered one: remove the window's content
  entirely and squish still wins.

## 4. Testing methodology

### 4.1 Arms

Implemented in this branch's copy of `scripts/ntire_crop_probe.py`
(`views_mask_centre`, `views_mask_periphery`). Both start from `views_squish`
(n x n, n = 512 for A, 224 for B) and fill a region in uint8, before the
backbone's normalisation, with the ImageNet mean (123.675, 116.28, 103.53),
rounded to (124, 116, 104):

| arm | filled region | side of the centred square | masked area |
|---|---|---|---|
| `mask_centre` | inside a centred square | n/2: 256 (A), 112 (B) | 25.00% (both) |
| `mask_periphery` | outside a centred square | round(n * sqrt(0.75)): 443 (A), 194 (B) | 25.14% (A), 24.99% (B) |

A self-check (`outputs/test10/check_t10.py`) confirmed the shapes and the
masked fractions above.

### 4.2 Scoring and statistics

Frozen A2 (final) and B2swa (SWA) on `ntire_val_hard` (2,500 images).
`squish`, `mask_centre` and `mask_periphery` were all scored in one run, so
the unmasked reference comes from the same code. Paired bootstrap: 2,000
resamples, stratified by label x is_distorted, seed 20260827, the same
indices for all arms (`paired_bootstrap.paired_bootstrap`).

### 4.3 Reproduction

The squish re-score matches the frozen probe parquet exactly for B (max
abs logit difference 0). For A it differs by at most 0.240, with robust
AUC 0.9479 against 0.9482 frozen. This is the same small A2 drift seen in
Test 1 §4.4 (assumed bf16 batch composition). The test compares arms from
the same run, so the drift does not enter any difference.

## 5. Findings

### 5.1 Registered comparison, distorted rows

| branch | mask_centre - mask_periphery | squish - mask_periphery (cost of the rim) | squish - mask_centre (cost of the middle) |
|---|---|---|---|
| A | -0.0128 [-0.0210, -0.0042] | +0.0100 [+0.0051, +0.0156] | +0.0229 |
| B | -0.0116 [-0.0201, -0.0034] | +0.0062 [+0.0014, +0.0112] | +0.0178 |

Both masks cost something. The middle costs about twice the rim on both
branches.

### 5.2 Clean rows (reported, not a condition)

| branch | squish | mask_centre | mask_periphery | mask_centre - mask_periphery |
|---|---|---|---|---|
| A | 0.9907 | 0.9842 | 0.9882 | -0.0040 [-0.0073, -0.0009] |
| B | 0.9886 | 0.9836 | 0.9836 | -0.0001 [-0.0039, +0.0042] |

### 5.3 Supplementary (not registered): centre-masked squish vs the centre window

`center` is Test 1's re-scored centre-window arm on the same 2,500 rows,
in the same bootstrap.

| branch | robust: center | mask_centre | mask_centre - center [95% CI] | clean: mask_centre - center |
|---|---|---|---|---|
| A | 0.8561 | 0.9250 | +0.0689 [+0.0485, +0.0896] | -0.0071 [-0.0126, -0.0016] |
| B | 0.8872 | 0.9421 | +0.0549 [+0.0369, +0.0725] | +0.0034 [-0.0032, +0.0104] |

For an image with short side s, the 200 px window covers the central
200/s of each axis. Once s is above about 400 px, all of it falls inside
the masked square (side n/2 of the squished frame), which is most of
`ntire_val_hard`. On those images the two arms see disjoint parts of the
scene. Squish keeps 75% of its robust gain over the window (A +6.9 of
+9.2 points, B +5.5 of +7.3) using only what the window never sees.

### 5.4 Why the registered prediction was the wrong shape

The registration equated "periphery" with "what the centre window
discards". At 25% area, the periphery mask is a thin rim (6.7% of the side
on each edge). The centre window discards far more than that rim: on a
1,000 px image it discards everything outside the central 20% of each
axis. The rim is also where photographic content is sparsest (framing puts
subjects near the middle). So the registered contrast compares the busiest
quarter of the frame with its emptiest quarter. It does not compare
"inside the window" with "outside the window".

## 6. Results

- Registered: **FAIL** on both branches (CI of mask_centre - mask_periphery
  below zero: A -0.0128, B -0.0116).
- Supplementary: centre-masked squish beats the centre window by +6.9 (A)
  and +5.5 (B) robust points, CIs above zero.
- Suggested paper text: "Masking the middle quarter of the squished frame
  costs about twice as much as masking its outer rim, so the evidence is not
  concentrated at the image border. But squish with its middle masked,
  which on most images shows nothing the 200 px window shows, still beats
  the window by 5.5-6.9 robust points."

## 7. Future work

- A window-matched mask: fill exactly the part of the squished frame that
  the 200 px centre window covers (side n * 200/s per axis, per image). This
  is the clean "inside vs outside the window" test. It is not registered.
- Size-bucketed masking deltas would show whether the middle's share falls
  as images grow (the window's share of the frame shrinks).

## 8. What didn't work

- The registered geometry (Section 5.4) does not test the claim it was
  written for. That is reported as is. The supplementary comparison is
  labelled as unregistered.
- A2's squish re-score is not bit-exact against the frozen parquet (Section
  4.3). The cause is not proven.

Numbers from commit `bbb1306` plus this branch's staged files
(`scripts/ntire_crop_probe.py` with the two mask arms,
`scripts/amend5_test10.py`). JSON: `docs/ntire/amend5/test10_masking.json`.
Per-image rows: `/scratch/dion_r13/dion_extensions/outputs/test10/`.
