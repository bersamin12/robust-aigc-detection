# Test 2: regime grid with a damage sweep

Amendment 5, Test 2. Branch `researcher/test2-regime-grid`, from `bbb1306`.

## 0. Prediction, written before any Test 2 score exists (open loop L5)

Written 2026-10-19, before the 200-image throughput check and before the
full run. Carried from Test 4 (§3 there), where JPEG had the largest family
gain on Branch A and a JPEG-specific excess on `ntire_val`:

> **The JPEG arm's D rises more steeply with level than the noise arm's.**
> Here D = robust AUC(squish) - AUC(native) on the primary source. "More
> steeply" means: on each of our two branches, D(JPEG, heavy) - D(JPEG,
> none) exceeds D(noise, heavy) - D(noise, none), pooled over size buckets.

This prediction is not a registered pass condition. It is reported next to
the registered result, whichever way it comes out.

The rest of this file is written after the run.

## 1. Summary & results

**Registered verdict: PASS (regime rule holds), on both branches.** The
condition, word for word: "**Pass (regime rule holds):** for both of our
branches, D does not decrease from none -> light -> medium -> heavy in at
least 3 of 4 size buckets, and D changes sign along that axis in at least
one bucket."

- Branch A and Branch B: D is non-decreasing with damage level in all 3
  populated size buckets (the 80-336 bucket is empty, §5.1), and D changes
  sign in the 528-736 bucket on both branches.
- **The sign change is small.** In 528-736, D(none) is -0.0013 [-0.0027,
  -0.0003] on A and -0.0011 [-0.0023, +0.0001] on B. B's interval includes
  zero. The registered rule is judged on point estimates, so the PASS
  stands, but the "crop wins on clean images" side of the crossing is at
  the limit of resolution (§5.2).
- **Secondary (generality): met in part.** B-Free's pooled D rises with
  level (-0.114 -> -0.108 -> -0.088 -> -0.035), but it is non-decreasing in
  only 2 of 3 populated buckets and never crosses zero in the level average.
- **L5 prediction (§0): FAILED on both branches.** The noise arm's D rises
  more than the JPEG arm's: A noise +0.0380 vs JPEG +0.0246; B noise +0.0325
  vs JPEG +0.0203 (none -> heavy, pooled).
- **Synthbuster (secondary, unbucketed):** squish is below `center` at every
  damage condition on both branches (A none -0.030 [-0.037, -0.023], B none
  -0.016 [-0.022, -0.012]). Consistent with the Test 11 clean-cost result.

| model | pooled D: none | light | medium | heavy | buckets non-decreasing | sign change |
|---|---|---|---|---|---|---|
| Branch A (A2) | -0.0011 | +0.0011 | +0.0075 | +0.0285 | 3 / 3 | yes (528-736) |
| Branch B (B2swa) | +0.0043 | +0.0076 | +0.0141 | +0.0271 | 3 / 3 | yes (528-736) |
| B-Free | -0.1139 | -0.1080 | -0.0884 | -0.0346 | 2 / 3 | no |

Level D = mean over the three op types at that level, as registered.

## 2. Current state

Before this test, the paper's regime claim (squish helps under damage,
the window helps on clean large images) rested on the NTIRE distorted
versus clean split, where damage type, damage level and image size are
confounded. Test 2 applies controlled damage to the 6,250 clean NTIRE
images, so level is the only thing that changes along each curve.

## 3. Opportunities

- The regime crossing exists, but its clean side is tiny (about -0.001 AUC).
  The paper can say "squish costs nothing measurable on clean images up to
  736 px and gains with damage", which is better supported than "the window
  wins on clean images".
- The 752+ bucket carries most of the gain (A heavy +0.049, B +0.049); B
  already gains +0.013 at `none` there.
- Noise, not JPEG, gives the steepest rise. This revises the reading carried
  from Test 4 (JPEG had the largest family gain on Branch A).

## 4. Testing methodology

- **Models and presentations (registered):** A2 and B2swa: `center`,
  `grid9`, `squish`; B-Free: `native` (authors' five-crop), `train_fov`,
  `squish`. All six logits are saved per image. The statistic uses `squish`
  vs `center` (our branches) and `squish` vs `native` (B-Free).
- **Damage:** `augment/ops.py`, one op type per arm; levels exactly as in
  the registration table (JPEG q90/q70/q30, blur sigma 0.5/1.0/2.0, noise
  sigma 0.02/0.05/0.10). Noise seeded per image by
  `default_rng([20261019, int(content_sha256[:15], 16)])` (D4).
- **Sources:** primary = `ntire_val` + `ntire_val_hard`, is_distorted = 0,
  6,250 images (3,125 fake). Secondary = SocialRF (2,000) and Synthbuster
  (1,994: 1,000 fake, 994 real) subsampled with seed 20261019, not bucketed
  (D5).
- **Statistic:** D = AUC(squish) - AUC(native) per model x condition x size
  bucket, paired CI from `paper_analyses.paired_diff` (2,000 resamples,
  label strata, seed 20260827). Size buckets on the short side: 80-336,
  352-512, 528-736, 752+.
- **Fidelity check:** the `none` condition reproduces the earlier frozen
  scores: A2 `center` and `squish` exactly; B-Free max |logit diff| <= 0.035.
- **Run:** scoring 2026-10-19 20:47 to 2026-10-20 05:07 in tmux `test2`
  (throughput on 200 images: A 10, B 26, B-Free 6.6 img/s); analysis 05:07
  (`scripts/amend5_test2_analysis.py`, 2 min). `ntire_test` is refused by
  name in both scripts.

## 5. Findings

### 5.1 The 80-336 bucket is empty (deviation)

Clean NTIRE images have a short side of at least 352 px, so the first
bucket holds no rows. "At least 3 of 4 buckets" therefore means all 3
populated buckets. Both branches meet that stricter reading.

### 5.2 Per-bucket D, our branches (level average)

| bucket (n) | model | none | light | medium | heavy |
|---|---|---|---|---|---|
| 352-512 (176) | A | +0.0003 | +0.0013 | +0.0042 | +0.0192 |
| 528-736 (2,694) | A | -0.0013 | -0.0008 | +0.0010 | +0.0082 |
| 752+ (3,380) | A | +0.0016 | +0.0054 | +0.0161 | +0.0491 |
| 352-512 (176) | B | +0.0011 | +0.0016 | +0.0047 | +0.0104 |
| 528-736 (2,694) | B | -0.0011 | -0.0003 | +0.0014 | +0.0069 |
| 752+ (3,380) | B | +0.0134 | +0.0184 | +0.0293 | +0.0486 |

The `none` column is a single condition, so it has a direct paired CI:
528-736 A -0.0013 [-0.0027, -0.0003], B -0.0011 [-0.0023, +0.0001]; 752+
A +0.0016 [-0.0010, +0.0043], B +0.0134 [+0.0097, +0.0173]. The light,
medium and heavy columns average three op-type cells; the analysis did not
compute a CI for the average (each cell's CI is in the json).

### 5.3 D by op type, pooled over size

| model | op | none | light | medium | heavy |
|---|---|---|---|---|---|
| A | JPEG | -0.0011 | +0.0000 | +0.0052 | +0.0235 [+0.0200, +0.0273] |
| A | blur | -0.0011 | -0.0007 | +0.0046 | +0.0251 [+0.0213, +0.0290] |
| A | noise | -0.0011 | +0.0038 | +0.0129 | +0.0369 [+0.0323, +0.0419] |
| B | JPEG | +0.0043 | +0.0060 | +0.0123 | +0.0246 [+0.0209, +0.0285] |
| B | blur | +0.0043 | +0.0051 | +0.0097 | +0.0198 [+0.0164, +0.0230] |
| B | noise | +0.0043 | +0.0117 | +0.0202 | +0.0368 [+0.0322, +0.0416] |

### 5.4 L5 prediction

Pre-stated in §0: D(JPEG, heavy) - D(JPEG, none) > D(noise, heavy) -
D(noise, none) on each branch. Result: A 0.0246 vs 0.0380; B 0.0203 vs
0.0325. **The prediction fails on both branches.** Noise rises most, JPEG
and blur are close to each other.

### 5.5 B-Free (secondary)

Pooled D rises with level (-0.1139 [-0.1228, -0.1055] at none to -0.0346 at
heavy). Per bucket it is non-decreasing in 528-736 and 752+ but not in
352-512 (-0.0433, -0.0442, -0.0474, -0.0357). The rise is driven by the
noise arm: noise heavy D = +0.0416 [+0.0297, +0.0536] pooled, the only B-Free
cell above zero, because B-Free's native AUC falls to 0.666 there. JPEG heavy
-0.060, blur heavy -0.086.

### 5.6 Secondary sources (not bucketed)

| source | model | none | JPEG heavy | blur heavy | noise heavy |
|---|---|---|---|---|---|
| SocialRF | A | +0.0018 [-0.0071, +0.0105] | +0.0313 | +0.0394 | +0.0628 |
| SocialRF | B | +0.0198 [+0.0110, +0.0289] | +0.0463 | +0.0282 | +0.0812 |
| Synthbuster | A | -0.0301 [-0.0368, -0.0234] | -0.0148 | -0.0093 | -0.0166 |
| Synthbuster | B | -0.0163 [-0.0216, -0.0115] | -0.0136 | -0.0131 | -0.0130 |

SocialRF follows the primary pattern. On Synthbuster, squish loses at every
condition on both branches, and heavy damage only narrows the loss.

## 6. Results

Figure: `test2_regime-grid_fig.pdf`, level-averaged D per size bucket, one
panel per model. All numbers: `test2_regime-grid.json`.

- Registered: **PASS** for Branch A and Branch B.
- Secondary: B-Free D rises with level pooled; 2 of 3 buckets
  non-decreasing.
- L5: **fails** on both branches.

## 7. Future work

- A paired CI for the level average (mean over three op types) per bucket
  would show whether the 528-736 sign change on Branch B is resolved. This
  needs `paired_diff` over a stacked statistic, not a new bootstrap; researcher’s
  call.
- `grid9` (ours) and `train_fov` (B-Free) are scored and saved but not used
  by the registered statistic. They can be compared without new GPU time.
- Synthbuster's squish loss at every damage level ties to Test 11 and
  Test 6; read together in the paper's cost section.

## 8. What didn't work

- The L5 prediction (JPEG steepest) was wrong.
- The planned 4-bucket design cannot be used on clean NTIRE images: the
  smallest bucket is empty.

Numbers from commit `bbb1306` with `scripts/amend5_test2.py` and
`scripts/amend5_test2_analysis.py` staged (not yet committed) at run time.
