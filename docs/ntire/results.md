# NTIRE 2026 Robust AIGC detection: final result

**`ntire_test` was read once, on 2026-10-04 12:00 UTC, and not again.** The
configuration was frozen on 2026-09-23 by `scripts/ntire_v2.sh`, which trained
both branches, scored each under final and SWA weights, and picked the better
of each on the selection splits (`ntire_val` + `ntire_val_hard`). The read
itself was run by `scripts/ntire_test_read.sh`, which takes the two
checkpoints and their weights tags out of the frozen record
(`fusion_ntire2.json`) rather than from its own arguments, refuses to run if a
test parquet already exists, and re-derives the fusion from the fit-split
parquets before applying it unchanged to the test rows. Nothing in this table
was fitted on `ntire_test`.

Model: two separately fine-tuned `dinov2regl` branches on one RTX 4090,
0.91B parameters total.

* **A** — single tower @518, depth 24, 2 epochs, final weights.
* **B** — dual tower @224, depth 24, 3 epochs, SWA weights.
* **Fusion** — score level, `w·z_A + (1-w)·z_B`, `w = 0.45` fitted on the
  pooled selection splits.

## Our numbers

| split | model | robust AUC | clean AUC | AUC | TPR@1%FPR |
|---|---|---|---|---|---|
| `ntire_test` | A alone | 0.7640 | 0.9886 | 0.9036 | 0.2054 |
| `ntire_test` | B alone | 0.7922 | 0.9666 | 0.8938 | 0.2100 |
| `ntire_test` | **fused, w=0.45** | **0.7931** | **0.9856** | 0.9108 | 0.2892 |
| `ntire_test` | fused, equal weight | 0.7919 | 0.9863 | 0.9113 | 0.2908 |
| `ntire_val_hard` | fused | 0.8868 | 0.9900 | 0.9505 | 0.5096 |
| `ntire_val` | fused | 0.9719 | 0.9965 | 0.9854 | 0.8660 |

Two things worth stating plainly.

**The fitted weight bought 0.0012 robust over equal weighting.** Round one
found the same: weight-fitting pays for lopsided parents and not for balanced
ones. The fusion itself is worth +2.9 robust over branch A and +0.1 over
branch B; what it really buys is clean AUC at B's robustness (+1.9 clean over
B alone).

**Robust AUC falls 9.4 points from `ntire_val_hard` (0.8868) to `ntire_test`
(0.7931) while clean holds (0.9900 -> 0.9856).** The selection split we treated
as the pessimistic one is not: the sealed split's distorted rows are
materially harder. Any future selection on `ntire_val_hard` alone should be
read as optimistic about robustness by roughly this margin.

## Against the published leaderboard

Table 3 of the challenge report (arXiv 2604.11487) gives Open Test and Hidden
Test columns separately. `ntire_test` here is the public (open) test split, so
the Open Test column is the directly comparable one. The challenge's own
ranking is by **average** robust ROC AUC over both splits, which we cannot
compute -- the hidden split is not public.

| team | open clean | open robust | avg robust (rank metric) |
|---|---|---|---|
| MICV (1st) | 0.9978 | 0.9738 | 0.9723 |
| Ant International (2nd) | 0.9973 | 0.9731 | 0.9721 |
| TeleAI-TeleGuard (3rd) | 0.9762 | 0.9215 | 0.9251 |
| INTSIG (4th) | 0.9810 | 0.9090 | 0.9130 |
| vincentlc (5th) | 0.9497 | 0.8633 | 0.8730 |
| UESTC (6th) | 0.9693 | 0.8558 | 0.8679 |
| Reagvis Labs (7th) | 0.9423 | 0.8474 | 0.8603 |
| PSU (8th) | 0.9132 | 0.8334 | 0.8408 |
| Shallow Real (9th) | 0.9954 | 0.8302 | 0.8336 |
| **ours (not a challenge entry)** | **0.9856** | **0.7931** | n/a |

**On clean AUC we sit fourth of ten; on robust AUC we are last, 3.7 points
below the ninth-placed entry.** That is the honest position and it is the
profile the challenge exists to penalise: a detector that is strong on
undistorted images and weak once they are degraded.

The comparison is not like-for-like on compute, which is the one mitigating
fact and should be stated as a caveat rather than a defence:

| team | model | hardware |
|---|---|---|
| MICV | committees of 4 + 2 DINOv3 backbones @512 | 32x A100, ~8 h |
| Ant International | 2x DINOv3-7B, 14B params, full fine-tune | B200 |
| TeleAI-TeleGuard | EVA-CLIP LoRA, pairwise training | 8x A800 |
| INTSIG | 5 detectors, DINOv3-Huge + MetaCLIP2-Giant | 8x H800 |
| ours | 2x dinov2regl, 0.91B params | 1x RTX 4090 |

No parameter or compute cap appears in the report, so these are not
size-matched comparisons and we make no rank claim. The useful reading is
that a 0.91B two-branch model on one consumer GPU reaches clean parity with
the middle of the field and does not reach its robustness.

One convergent finding worth recording: TeleAI-TeleGuard (3rd) trains with
`L_CE + a*L_KL(x, x_hat) + b*L_MSE(f_x, f'_x_hat)`, a = 0.5, b = 0.25 -- clean
and degraded views in the same batch with a KL term on the outputs and an MSE
term on the hidden features. That is the a3 recipe this repo has used since
August, arrived at independently.

## What moved the number, and what did not

Round two (`docs/ntire/round2_scores.md`) is the one change that improved
robustness: +2.1 robust on `ntire_val_hard` fused, from extending the
augmentation pool to the two distortion families the round-one error analysis
ranked weakest. Measured per-distortion on distorted rows only, branch B,
round one -> round two: `jpeg_ai` 0.714 -> 0.797, `jpeg_recompression_comb_jpegai`
0.780 -> 0.849, `downscale` 0.864 -> 0.886, four-op chains 0.773 -> 0.813, every
untargeted token flat within noise. The gains landed where they were
predicted to and nowhere else.

Three method hypotheses were pre-registered against a written gate and none
passed. Each is recorded with its control:

* **Degradation-invariant distillation** (`docs/ntire/did_ablation.md`) --
  refuted. Pulling degraded features toward clean ones cost ~1 robust point;
  the same-window control showed that was the term itself, not the crop
  confound.
* **Degradation-conditioned classification** (`docs/ntire/film_ablation.md`)
  -- null. FiLM on the degradation embedding ties the baseline (0.8032 vs
  0.8040 robust). The block was used, not ignored: its zero-initialised
  projection trains to a non-trivial norm. The shuffled-conditioning control
  is **invalid** and its gate is void -- degraded views are built
  source-major, so rolling the conditioning by one row hands half of them a
  vector computed from another view of the same image, a train-only leak.
  A capacity control would need `cond = 0`, not a permutation.
* **Severity-conditioned fusion** -- ruled out before training anything. The
  optimal fusion weight does move with chain length (clean 0.85 -> four-op
  0.00) and an oracle per-chain weight is worth +2.0 on four-op rows, but
  only +0.43 on pooled robust, under the +0.5 gate. A perfect severity
  predictor could not have cleared it.

The consistent result across all three: the degradation head's prediction is a
real supervised signal, and spending it on the features -- to remove the
degradation or to condition on it -- does not improve robustness. What does is
matching the training distortions to the deployed ones.
