# Test 9: learned combiner over nine windows

Amendment 5, Test 9. Branch `researcher/test9-mil-combiner`, from `bbb1306`.
Run 2026-10-19. Fit-split numbers: `ntire_val_hard` was used to choose
squish, so every number on it below is a fit-split number.

## 1. Summary & results

**Registered verdict: PASS, on both branches.** The registered condition
reads: "**Pass:** robust AUC (squish - combiner) has a CI above zero on both
branches."

| branch | grid9 (mean of logits) | MIL combiner | squish | squish - combiner [95% CI] |
|---|---|---|---|---|
| A (A2) | 0.8495 | 0.8792 | 0.9482 | +0.0690 [+0.0515, +0.0869] |
| B (B2swa) | 0.9048 | 0.9369 | 0.9599 | +0.0231 [+0.0113, +0.0361] |

(Robust AUC on `ntire_val_hard`, n = 1,250 distorted rows.)

- The learned combiner beats plain averaging of the nine windows by about
  3 robust points on both branches (CIs above zero, Section 5.2). Squish
  still beats the combiner on both branches.
- The registration states the bias: the combiner is fitted on `ntire_val`,
  while squish is zero-shot. Even with that advantage, the combiner recovers
  only 30% (A) and 58% (B) of the gap between grid9 and squish.
- The paper's sentence "nine native windows barely help" should be
  qualified. Averaged, they barely help. Learned weighting helps, but not
  enough to match the whole image.

## 2. Current state

`grid9` takes nine 200 px windows at native resolution and averages their
logits. On `ntire_val_hard` this is no better than the single centre window
for A (0.8495 against 0.8561), and gains +1.8 points for B. The paper reads
this as "the extra windows barely help". The objection is that plain
averaging is a weak estimator: if one window holds the evidence, the mean
dilutes it. A learned combiner is the strongest reasonable form of the
multi-window alternative.

## 3. Opportunities

- A pass keeps squish's advantage over multi-window inference intact, even
  against a combiner trained for it.
- The combiner's +3 points over grid9 is a free improvement for any
  deployment that must keep native windows (for example, very large images
  where squish loses detail). It is not the paper's claim.

## 4. Testing methodology

### 4.1 Features

`scripts/amend5_test9_features.py` runs the probe's `grid9` presentation
(`ntire_crop_probe.views_grid9`, unchanged) through the frozen towers and
saves, per window, the pooled feature vector that the head reads: 1,024-d
for A (one tower), 2,048-d for B (two towers, concatenated). It also saves
the frozen head's logit per window. `ntire_val`: 10,000 images, 87,696
windows. `ntire_val_hard`: 2,500 images, 22,146 windows.

Reproduction: the mean of the saved window logits matches the frozen
`probe_{A2,B2swa}_grid9_ntire_val_hard.parquet` to max abs logit difference
0.072 (A) and 0.145 (B). Robust AUC agrees to 4 dp (A 0.8495, B 0.9048 vs
0.9049 frozen). This is the same bf16 drift seen in Tests 1 and 10.

### 4.2 Combiner

`scripts/amend5_test9.py`: a gated-attention MIL head (Ilse et al. 2018,
eq. 9), hidden 128, dropout 0.1, as registered:

    a_k = softmax_k( w^T (tanh(V h_k) * sigmoid(U h_k)) )
    z   = sum_k a_k h_k
    logit = c^T z + b

Padding windows (images with fewer than nine windows) are masked out of the
softmax. Choices the registration left open were fixed in the script's
docstring before the first fit:

| choice | value |
|---|---|
| input | frozen window features, standardised with the fit windows' mean/std |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| batch | 64 images (bags) |
| epochs | at most 50; early stopping on the 10% split's AUC, patience 5, best epoch kept |
| split | `ntire_val` 90% fit (9,000) / 10% early stopping (1,000), stratified by label x is_distorted, seed 20261019 |

| branch | best epoch | early-stop AUC | epochs run |
|---|---|---|---|
| A | 5 | 0.9965 | 11 |
| B | 4 | 0.9970 | 10 |

### 4.3 Evaluation

The combiner scores `ntire_val_hard`. Squish logits are the frozen probe
parquets (`outputs/scores/probe_{A2,B2swa}_squish_ntire_val_hard.parquet`).
Paired bootstrap: `paired_bootstrap.paired_bootstrap`, 2,000 resamples,
stratified by label x is_distorted, seed 20260827, the same indices for
combiner, squish and grid9.

## 5. Findings

### 5.1 All metrics, `ntire_val_hard`

| branch | arm | AUC | robust AUC | clean AUC |
|---|---|---|---|---|
| A | grid9 | 0.9394 | 0.8495 | 0.9978 |
| A | combiner | 0.9546 | 0.8792 | 0.9967 |
| A | squish | 0.9730 | 0.9482 | 0.9906 |
| B | grid9 | 0.9576 | 0.9048 | 0.9931 |
| B | combiner | 0.9752 | 0.9369 | 0.9959 |
| B | squish | 0.9765 | 0.9599 | 0.9886 |

### 5.2 Paired differences

| branch | comparison | robust | clean |
|---|---|---|---|
| A | squish - combiner | +0.0690 [+0.0515, +0.0869] | -0.0061 [-0.0105, -0.0015] |
| A | grid9 - combiner | -0.0297 [-0.0431, -0.0165] | +0.0011 [-0.0014, +0.0046] |
| B | squish - combiner | +0.0231 [+0.0113, +0.0361] | -0.0073 [-0.0116, -0.0033] |
| B | grid9 - combiner | -0.0320 [-0.0436, -0.0199] | -0.0028 [-0.0060, +0.0002] |

On clean rows the combiner is slightly above squish on both branches (0.6
and 0.7 points). This is the known small clean cost of squish for
crop-trained towers (Table 2 of the paper), not a new effect.

### 5.3 Share of the grid9 -> squish gap recovered

| branch | squish - grid9 | combiner - grid9 | share |
|---|---|---|---|
| A | +0.0987 | +0.0297 | 30% |
| B | +0.0551 | +0.0320 | 58% |

## 6. Results

- Registered: **PASS** (squish - combiner robust CI above zero: A +0.0690,
  B +0.0231).
- Suggested paper text: "A gated-attention combiner over the nine native
  windows, fitted on `ntire_val`, improves on averaging by 3.0-3.2 robust
  points but stays 2.3 (B) to 6.9 (A) points below zero-shot squish."

## 7. Future work

- The combiner's attention weights per window could show whether it
  learns to look at the corners (field of view) or at a single window. The
  weights are computed but not saved or analysed.
- One seed only. The registration does not ask for more. The margin on B
  (+2.3, CI lower bound +1.1) is the one to re-check with more seeds if the
  paper leans on it.

## 8. What didn't work

- The first fit stalled for 10 minutes. The bag loader re-read the whole
  feature array from the `.npz` file for every image. Fixed by reading
  each array once. No result came from the stalled run.
- Window logits reproduce the frozen grid9 parquets to 4 dp in AUC, not
  bit for bit (Section 4.1).

Numbers from commit `bbb1306` plus this branch's staged files
(`scripts/amend5_test9_features.py`, `scripts/amend5_test9.py`). JSON:
`docs/ntire/amend5/test9_mil-combiner.json`. Features and per-image rows:
`/scratch/anonymous/dion_extensions/outputs/test9/`.
