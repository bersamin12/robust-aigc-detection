# G1 -- squish vs crop, proxy protocol, three seeds

Pre-registered in `docs/ntire/paper_prereg.md`. Proxy: one dinov2regl224 tower, depth 24, 1 epoch over 25% of train, bank `ntire_meta`, mean pool, cosine + SWA, final weights. Each seed pair shares its training subset, head init and batch order; only `--canon-mode` differs.

| seed | crop robust | squish robust | diff [95% CI] | crop clean | squish clean | diff [95% CI] |
|---|---|---|---|---|---|---|
| s1 | 0.8037 | 0.8866 | +0.0828 [+0.0621, +0.1047] | 0.9264 | 0.9346 | +0.0082 [-0.0078, +0.0260] |
| s2 | 0.7952 | 0.8881 | +0.0929 [+0.0717, +0.1148] | 0.9266 | 0.9384 | +0.0118 [-0.0041, +0.0291] |
| s3 | 0.7993 | 0.8776 | +0.0783 [+0.0571, +0.1015] | 0.9234 | 0.9342 | +0.0109 [-0.0051, +0.0282] |

Mean robust diff +0.0847 (gate > +0.0100); seeds with CI above zero 3/3 (gate >= 2); mean clean diff +0.0103 (gate >= -0.0050).

**G1 PASSED** -- full training proceeds.

Commit `ec8978e33ec4b0ca096138071f6f71b2e22ce8d9`.

## ntire_val (secondary)

### seed 1, ntire_val
n=10,000 (robust 5,000, clean 5,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9498 [0.9459, 0.9538] | 0.9279 [0.9206, 0.9352] | 0.9683 [0.9639, 0.9725] |
| `squish` | 0.9726 [0.9699, 0.9752] | 0.9648 [0.9602, 0.9691] | 0.9792 [0.9757, 0.9823] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0228 [+0.0190, +0.0265] * | +0.0368 [+0.0305, +0.0436] * | +0.0109 [+0.0066, +0.0152] * |

`*` = the 95% interval on the paired difference excludes zero.

### seed 2, ntire_val
n=10,000 (robust 5,000, clean 5,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9519 [0.9482, 0.9558] | 0.9316 [0.9243, 0.9385] | 0.9691 [0.9650, 0.9733] |
| `squish` | 0.9743 [0.9718, 0.9768] | 0.9663 [0.9619, 0.9704] | 0.9811 [0.9779, 0.9840] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0224 [+0.0188, +0.0260] * | +0.0346 [+0.0285, +0.0412] * | +0.0120 [+0.0079, +0.0161] * |

`*` = the 95% interval on the paired difference excludes zero.

### seed 3, ntire_val
n=10,000 (robust 5,000, clean 5,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9514 [0.9477, 0.9552] | 0.9298 [0.9225, 0.9367] | 0.9693 [0.9655, 0.9734] |
| `squish` | 0.9744 [0.9718, 0.9769] | 0.9666 [0.9622, 0.9707] | 0.9810 [0.9777, 0.9839] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0230 [+0.0194, +0.0267] * | +0.0368 [+0.0303, +0.0435] * | +0.0117 [+0.0074, +0.0157] * |

`*` = the 95% interval on the paired difference excludes zero.

