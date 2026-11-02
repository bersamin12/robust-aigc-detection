# Test 4: squish gain by damage family

Amendment 5, Test 4. Branch `dion/test4-damage-family`, from `991dc37`.
Run 2026-10-19. Fit-split numbers: `ntire_val_hard` was used to choose
squish, so every number on it below is a fit-split number.

## 1. Summary & results

**Registered verdict (primary, `ntire_val_hard`, table (a)): SUPPORTS
COVERAGE, on both branches.** The registered condition reads: "every family
with n >= 100 has a gain whose CI is above zero, and the largest family gain
is less than 2x the smallest." The cleaning condition ("the JPEG and noise
gains each exceed both the blur and the colour gains, with paired CIs of
those differences above zero") is **not met** on either branch.

| branch | families with n >= 100 | all CIs above 0 | min gain | max gain | max / min | coverage | cleaning |
|---|---|---|---|---|---|---|---|
| A (A2) | 7 of 7 | yes | +0.0798 (colour) | +0.1311 (JPEG) | 1.64 | **met** | not met |
| B (B2swa) | 7 of 7 | yes | +0.0563 (colour) | +0.1015 (adversarial) | 1.80 | **met** | not met |

Three qualifications apply to this reading. Sections 5 and 6 give the detail.

1. **JPEG has the largest gain on branch A, on both splits.** On the
   secondary split (`ntire_val`), JPEG's gain exceeds blur's and colour's
   with paired CIs above zero on both branches. Noise's gain does not.
   This is a JPEG-specific component, not general "cleaning". It does not
   meet the registered cleaning condition, which requires noise as well.
2. **On the secondary split the max/min ratio fails** (A 2.54, B 2.02), so
   `ntire_val` reads as mixed. That split is not judged.
3. **Table (a) overlap compresses the ratio.** A distorted row falls in
   2.5 families on average, so each family's gain is pulled toward the
   pooled gain. This makes the "< 2x" condition easier to meet than it
   would be with disjoint families. Table (b), the single-family rows, has
   n = 12-56 per family, which is too few to test the ratio.

## 2. Current state

The paper attributes squish's robust gain over the centre window to
coverage: the whole frame is shown, so evidence outside the 200 px window is
visible. Amendment 5 names a competing explanation, cleaning. Shrinking a
large image averages away JPEG blocking and noise. Both explanations predict
larger gains on larger images, so size alone cannot separate them. They do
predict different patterns across damage types:

- **Coverage** predicts a similar gain for every damage family.
- **Cleaning** predicts larger gains where the shrink removes the damage
  (JPEG, noise) than where it does not (blur, colour).

The existing `paper_analyses.py groups` analysis cuts the distorted rows by
crop, perspective, downscale and photometric. It does not separate
photometric damage into JPEG, noise, blur and colour.

## 3. Opportunities

- Test 1 (`freqmatch`) is the direct test of the same question. It is
  blocked on Justin's access. This test is meant to be read together with
  Test 1.
- The JPEG-specific excess (Section 6.3) predicts that in Test 2 the JPEG
  arm's D rises more steeply with level than the noise arm's D. That is a
  falsifiable prediction for Test 2, stated here before Test 2 runs.

## 4. Testing methodology

### 4.1 Data

| split | role | distorted rows | centre arm | squish arm |
|---|---|---|---|---|
| `ntire_val_hard` | primary (fit split) | 1,250 | `outputs/scores/probe_{A2,B2swa}_center_ntire_val_hard.parquet` | `outputs/scores/probe_{A2,B2swa}_squish_ntire_val_hard.parquet` |
| `ntire_val` | secondary | 5,000 | `outputs/scores/{A2,B2swa}_ntire_val.parquet` (as-trained) | `outputs/scores/probe_{A2,B2swa}_squish_ntire_val.parquet` |

No `center` probe parquet exists for `ntire_val`. The as-trained score uses
the same presentation (`canonicalise`, 200 px centre window). On
`ntire_val_hard` the two parquets agree to corr 0.99997, max |delta logit|
0.27, and give the same robust AUC to 4 dp (`crop_probe.md`). Sanity check:
the centre arm here reproduces the frozen robust AUCs, A 0.8561 and
B 0.8872.

No model was loaded and no image was read. `ntire_test` was not read. The
script asserts `split != "ntire_test"`.

### 4.2 Family map and counting

The family map is the registered one, copied into `FAMILIES` in
`scripts/amend5_test4.py`. Every token in both splits maps to a family, so
the "other" row is empty (21 distinct tokens in `ntire_val_hard`, 19 in
`ntire_val`). `quantization` is counted in JPEG and also reported on its own
row.

- **Table (a), primary:** a row counts in every family its chain contains.
- **Table (b):** rows whose tokens all belong to a single family.

Families per distorted row in `ntire_val_hard`: 1 family in 132 rows,
2 in 488, 3 in 438, 4 in 192.

### 4.3 Statistics

- **Per-family gain:** robust AUC(squish) - AUC(centre) within the family's
  rows. CI from `paper_analyses.paired_diff`: 2,000 resamples stratified by
  label within the family, seed 20260827, 95% percentile.
- **Differences between families (cleaning condition):** the families
  overlap, so their gains are not independent. Each of the 2,000 resamples
  draws once from all distorted rows (stratified by label, seed 20260827,
  the same draw loop as `paired_diff`). Every family's gain is then
  computed on its members within that one resample, and the differences
  are taken. This is `joint_gain_draws` in the script, the only new
  resampling code. Its scheme is identical to `paired_diff`'s; only the
  statistic differs. Dion should review it, given the "reuse the existing
  bootstrap" rule.
- The verdict counts only families with n >= 100 in table (a).

## 5. Findings

### 5.1 Every family gains, on both branches (primary)

![forest plot](test4_damage-family_fig.pdf)

*Figure 1.* Robust AUC gain (squish minus centre) per family, table (a),
`ntire_val_hard`, 95% paired CIs. File:
`docs/ntire/amend5/test4_damage-family_fig.pdf`.

All 7 families have CIs above zero on both branches. Each lower bound is at
least +0.025.

### 5.2 Gains are spread within a factor of 2 (primary)

Branch A ranges from +0.080 (colour) to +0.131 (JPEG), ratio 1.64.
Branch B ranges from +0.056 (colour) to +0.102 (adversarial), ratio 1.80.
Both are under the registered 2x. Branch B is close to the limit.

### 5.3 JPEG is the largest gain on branch A, and noise is not

Branch A's JPEG gain (+0.131) is the largest of the 7 families. Noise's gain
(+0.085) is below blur's (+0.101). Cleaning predicts JPEG and noise above
blur and colour. JPEG fits that prediction; noise does not.

### 5.4 Downscale is in most JPEG rows

633 of 832 JPEG rows in `ntire_val_hard` also contain `downscale` (1,179 of
1,570 in `ntire_val`). Table (a) cannot separate JPEG from the
downscale-plus-JPEG chain. Table (b) has 56 JPEG-only rows, with gains of
+0.180 (A) and +0.130 (B). Both CIs are above zero but wide (lower bounds
+0.047 and +0.034).

### 5.5 Table (b) is at ceiling

The single-family rows are the large images (median short side 640-900 px)
with one light operation. The centre arm already scores AUC 0.93-1.00 on
most of these cells, so the gains are bounded near zero. Table (b) is
reported, but it does not measure the family effect.

## 6. Results

Robust AUC, centre -> squish, and gain [95% CI]. n is total (real / fake).

### 6.1 `ntire_val_hard` (primary, fit split)

**Table (a): a row counts in every family it contains.**

| family | n (real/fake) | short p50 | ops/row | A centre -> squish | A gain [95% CI] | B centre -> squish | B gain [95% CI] |
|---|---|---|---|---|---|---|---|
| JPEG | 832 (417/415) | 496 | 3.00 | 0.8105 -> 0.9416 | +0.1311 [+0.1052, +0.1585] | 0.8608 -> 0.9543 | +0.0935 [+0.0714, +0.1168] |
| blur | 287 (153/134) | 480 | 3.15 | 0.8398 -> 0.9411 | +0.1013 [+0.0610, +0.1431] | 0.9002 -> 0.9723 | +0.0721 [+0.0400, +0.1091] |
| noise | 295 (157/138) | 512 | 2.96 | 0.8554 -> 0.9407 | +0.0853 [+0.0502, +0.1238] | 0.8833 -> 0.9588 | +0.0755 [+0.0406, +0.1129] |
| colour | 300 (153/147) | 528 | 3.10 | 0.8742 -> 0.9540 | +0.0798 [+0.0435, +0.1202] | 0.8961 -> 0.9524 | +0.0563 [+0.0250, +0.0907] |
| downscale | 931 (472/459) | 416 | 2.99 | 0.8562 -> 0.9428 | +0.0866 [+0.0648, +0.1088] | 0.8906 -> 0.9556 | +0.0650 [+0.0459, +0.0835] |
| geometric | 269 (140/129) | 416 | 3.07 | 0.8390 -> 0.9207 | +0.0817 [+0.0422, +0.1234] | 0.8685 -> 0.9497 | +0.0811 [+0.0429, +0.1202] |
| adversarial | 276 (122/154) | 520 | 3.01 | 0.8253 -> 0.9251 | +0.0998 [+0.0545, +0.1499] | 0.8427 -> 0.9442 | +0.1015 [+0.0617, +0.1454] |
| quantization (in JPEG) | 279 (140/139) | 528 | 3.05 | 0.8421 -> 0.9294 | +0.0873 [+0.0491, +0.1298] | 0.8462 -> 0.9516 | +0.1054 [+0.0669, +0.1471] |
| *all distorted rows* | 1250 | | | 0.8561 -> 0.9482 | +0.0920 [+0.0733, +0.1119] | 0.8872 -> 0.9599 | +0.0727 [+0.0567, +0.0899] |

**Table (b): single-family rows only.** Downscale has no single-family rows
on this split.

| family | n (real/fake) | short p50 | ops/row | A centre -> squish | A gain [95% CI] | B centre -> squish | B gain [95% CI] |
|---|---|---|---|---|---|---|---|
| JPEG | 56 (24/32) | 752 | 1.43 | 0.7565 -> 0.9362 | +0.1797 [+0.0468, +0.3216] | 0.8203 -> 0.9505 | +0.1302 [+0.0338, +0.2474] |
| blur | 12 (8/4) | 752 | 1.00 | 1.0000 -> 0.9688 | -0.0312 [-0.1562, +0.0000] | 0.9688 -> 1.0000 | +0.0312 [+0.0000, +0.1250] |
| noise | 15 (9/6) | 896 | 1.00 | 0.9259 -> 1.0000 | +0.0741 [+0.0000, +0.2407] | 0.9815 -> 1.0000 | +0.0185 [+0.0000, +0.0931] |
| colour | 16 (9/7) | 784 | 1.00 | 1.0000 -> 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.0000 -> 1.0000 | +0.0000 [+0.0000, +0.0000] |
| geometric | 20 (10/10) | 640 | 1.00 | 0.9900 -> 1.0000 | +0.0100 [+0.0000, +0.0600] | 0.9800 -> 1.0000 | +0.0200 [+0.0000, +0.0900] |
| adversarial | 13 (6/7) | 768 | 1.00 | 1.0000 -> 0.8571 | -0.1429 [-0.4286, +0.0000] | 0.8333 -> 1.0000 | +0.1667 [+0.0000, +0.4286] |
| quantization (in JPEG) | 23 (12/11) | 736 | 1.74 | 0.8333 -> 0.9167 | +0.0833 [-0.0985, +0.2955] | 0.9015 -> 0.9167 | +0.0152 [-0.1212, +0.1591] |

**Cleaning condition: paired differences of family gains (table (a)).**

| difference | A [95% CI] | B [95% CI] |
|---|---|---|
| JPEG - blur | +0.0298 [-0.0120, +0.0686] | +0.0214 [-0.0178, +0.0573] |
| JPEG - colour | +0.0513 [+0.0150, +0.0882] * | +0.0372 [+0.0027, +0.0701] * |
| noise - blur | -0.0160 [-0.0686, +0.0331] | +0.0033 [-0.0444, +0.0500] |
| noise - colour | +0.0055 [-0.0448, +0.0560] | +0.0192 [-0.0241, +0.0635] |

`*` = CI above zero. Cleaning needs all four starred on a branch; 1 of 4
is starred on each branch.

### 6.2 `ntire_val` (secondary, not judged)

No adversarial or `jpeg_ai` tokens occur in this split.

**Table (a).**

| family | n (real/fake) | short p50 | ops/row | A centre -> squish | A gain [95% CI] | B centre -> squish | B gain [95% CI] |
|---|---|---|---|---|---|---|---|
| JPEG | 1570 (759/811) | 480 | 3.05 | 0.9325 -> 0.9806 | +0.0482 [+0.0373, +0.0594] | 0.9504 -> 0.9830 | +0.0326 [+0.0235, +0.0419] |
| blur | 1225 (645/580) | 480 | 3.04 | 0.9549 -> 0.9845 | +0.0296 [+0.0207, +0.0393] | 0.9697 -> 0.9894 | +0.0197 [+0.0127, +0.0274] |
| noise | 1268 (661/607) | 496 | 3.04 | 0.9637 -> 0.9862 | +0.0225 [+0.0139, +0.0314] | 0.9681 -> 0.9871 | +0.0190 [+0.0116, +0.0268] |
| colour | 3238 (1606/1632) | 480 | 2.97 | 0.9606 -> 0.9843 | +0.0237 [+0.0183, +0.0292] | 0.9646 -> 0.9848 | +0.0202 [+0.0150, +0.0255] |
| downscale | 3815 (1944/1871) | 400 | 2.98 | 0.9642 -> 0.9842 | +0.0200 [+0.0152, +0.0245] | 0.9696 -> 0.9857 | +0.0162 [+0.0119, +0.0206] |
| geometric | 1314 (663/651) | 320 | 3.07 | 0.9632 -> 0.9821 | +0.0189 [+0.0117, +0.0269] | 0.9663 -> 0.9856 | +0.0193 [+0.0122, +0.0279] |
| quantization (in JPEG) | 405 (194/211) | 464 | 3.08 | 0.9474 -> 0.9892 | +0.0418 [+0.0253, +0.0601] | 0.9585 -> 0.9882 | +0.0298 [+0.0161, +0.0455] |
| *all distorted rows* | 5000 | | | 0.9617 -> 0.9853 | +0.0237 [+0.0195, +0.0278] | 0.9669 -> 0.9871 | +0.0202 [+0.0165, +0.0241] |

**Table (b).**

| family | n (real/fake) | short p50 | ops/row | A centre -> squish | A gain [95% CI] | B centre -> squish | B gain [95% CI] |
|---|---|---|---|---|---|---|---|
| JPEG | 70 (29/41) | 768 | 1.09 | 0.9235 -> 0.9958 | +0.0723 [+0.0219, +0.1379] | 0.9504 -> 0.9899 | +0.0395 [+0.0000, +0.0917] |
| blur | 57 (32/25) | 736 | 1.00 | 0.9988 -> 0.9988 | +0.0000 [-0.0050, +0.0050] | 0.9950 -> 0.9988 | +0.0038 [+0.0000, +0.0150] |
| noise | 60 (26/34) | 768 | 1.00 | 0.9751 -> 0.9627 | -0.0124 [-0.0622, +0.0317] | 0.9740 -> 0.9683 | -0.0057 [-0.0464, +0.0317] |
| colour | 235 (99/136) | 768 | 1.40 | 0.9886 -> 0.9976 | +0.0090 [-0.0003, +0.0224] | 0.9773 -> 0.9970 | +0.0197 [+0.0045, +0.0397] |
| downscale | 68 (28/40) | 472 | 1.76 | 0.9795 -> 1.0000 | +0.0205 [+0.0027, +0.0518] | 0.9884 -> 0.9991 | +0.0107 [-0.0018, +0.0339] |
| geometric | 61 (27/34) | 496 | 1.00 | 0.9946 -> 0.9891 | -0.0054 [-0.0327, +0.0153] | 0.9706 -> 0.9935 | +0.0229 [-0.0044, +0.0687] |
| quantization (in JPEG) | 19 (7/12) | 672 | 1.32 | 1.0000 -> 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.0000 -> 1.0000 | +0.0000 [+0.0000, +0.0000] |

**Conditions applied to the secondary split (for reference only).**

| | A | B |
|---|---|---|
| all family CIs above 0 (n >= 100) | yes (6 of 6) | yes (6 of 6) |
| max / min gain | 2.54 (JPEG / geometric) | 2.02 (JPEG / downscale) |
| coverage condition | not met (ratio) | not met (ratio) |
| JPEG - blur | +0.0185 [+0.0064, +0.0316] * | +0.0130 [+0.0021, +0.0236] * |
| JPEG - colour | +0.0245 [+0.0154, +0.0340] * | +0.0125 [+0.0046, +0.0205] * |
| noise - blur | -0.0071 [-0.0185, +0.0044] | -0.0007 [-0.0105, +0.0088] |
| noise - colour | -0.0012 [-0.0094, +0.0075] | -0.0012 [-0.0090, +0.0066] |
| cleaning condition | not met (noise) | not met (noise) |
| reading | mixed | mixed |

### 6.3 Reading

The registered primary reading is **coverage**: every damage family gains,
and the gains lie within a factor of 2. Within that pattern, JPEG has the
largest gain on branch A, on both splits. On `ntire_val` its excess over
blur and colour is separated from zero on both branches. Noise shows no
excess on either split. The result is therefore coverage plus a
JPEG-specific term. It is not the general cleaning effect, which would show
on noise as well. Shrinking an image is known to remove 8x8 block
artefacts; whether this explains the JPEG term is for Test 1 and Test 2's
JPEG arm to decide.

## 7. Future work

- **Test 1** (`freqmatch`, blocked on access) measures the cleaning share
  directly. If R > 0.40 on either branch, revisit Section 6.3.
- **Test 2's JPEG arm vs noise arm:** the prediction stated in Section 3.
- A disjoint attribution, such as a per-image regression of signed gain on
  family indicators, would remove the overlap problem in table (a). It is
  not registered. If it is done, it must be labelled post hoc.

## 8. What didn't work

- **Table (b) cannot test the ratio.** On `ntire_val_hard`, single-family
  rows are 132 of 1,250, with 12-56 per family. They are mostly large
  images near AUC 1.0 under both arms, so the gains are bounded near zero.
- **No `center` probe parquet for `ntire_val`.** The as-trained scores were
  used instead (Section 4.1). They match the probe arm closely but not
  bit for bit.
- **New resampling code:** `joint_gain_draws` was required for the paired
  family differences, because `paired_diff` takes one subset at a time.
  Section 4.3 explains it.

## Files

- `scripts/amend5_test4.py`: the analysis. Run from the worktree root with
  `PYTHONPATH=$PWD/src`. Runtime 24 s, CPU only.
- `docs/ntire/amend5/test4_damage-family.json`: every number above.
- `/scratch/dion_r13/dion_extensions/outputs/test4/test4_damage-family_rows.parquet`:
  the per-image join (6,250 rows, 414 KB). Not in git, because the repo's
  `.gitignore` excludes `*.parquet`. Rerunning the script writes it to
  `docs/ntire/amend5/`.
- `docs/ntire/amend5/test4_damage-family_fig.pdf`: Figure 1.

Numbers from commit `991dc3726bf58e1bcf7a8f26270e71205d8c4979`, with the
uncommitted script above.
