# Read 2's clean loss by native size (saved test scores, no new read)

Clean rows of `ntire_test`. Read 1 = crop fusion, read 2 = squish fusion; per-branch scores are each read's own branch models. Paired 95% CI, 2000 resamples stratified by label.

| short side | n | A read 1 -> 2 | A diff [95% CI] | B read 1 -> 2 | B diff [95% CI] |
|---|---|---|---|---|---|
| 0-736 | 395 | 0.9980 -> 0.9923 | -0.0058 [-0.0116, -0.0011] | 0.9866 -> 0.9751 | -0.0116 [-0.0262, +0.0011] |
| 737-1100 | 489 | 0.9878 -> 0.9828 | -0.0050 [-0.0151, +0.0039] | 0.9762 -> 0.9751 | -0.0011 [-0.0156, +0.0123] |
| >1100 | 366 | 0.9741 -> 0.8799 | -0.0942 [-0.1348, -0.0558] | 0.9315 -> 0.9401 | +0.0086 [-0.0202, +0.0388] |

Mean fused z by native size (higher looks more generated):

| class | read | 0-736 | 737-1100 | >1100 |
|---|---|---|---|---|
| real | 1 | -0.938 | -0.916 | -0.770 |
| real | 2 | -0.954 | -0.955 | -0.990 |
| generated | 1 | +1.018 | +0.790 | +0.671 |
| generated | 2 | +0.664 | +0.582 | +0.383 |

Commit `5ecff8fc3c3070ed0c9b1534666aaf39a5ff1c41`.
