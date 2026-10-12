# Squish paper -- results

Commit `ec8978e33ec4b0ca096138071f6f71b2e22ce8d9`. Gates as registered in `docs/ntire/paper_prereg.md`. Fusion z-scores and weight are fitted on `ntire_val` alone for both policies; every other split is a readout.

## Gates

- **G2 PASSED**: fused robust AUC on ntire_val_hard, squish - crop = +0.0740 [+0.0580, +0.0907]
- **G3 ext_aigi_socialrf: PASSED**: pooled AUC squish - crop = +0.0142 [+0.0100, +0.0183]
- **G3 ext_aigi_community: PASSED**: pooled AUC squish - crop = +0.0111 [+0.0091, +0.0132]
- **G3 ext_synthbuster: FAILED**: pooled AUC squish - crop = -0.0010 [-0.0016, -0.0005]

## G2 -- ntire_val_hard, fused

### G2 -- fused, ntire_val_hard
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9505 [0.9429, 0.9575] | 0.8856 [0.8676, 0.9025] | 0.9904 [0.9861, 0.9938] |
| `squish` | 0.9806 [0.9765, 0.9845] | 0.9596 [0.9497, 0.9687] | 0.9947 [0.9920, 0.9971] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0301 [+0.0235, +0.0370] * | +0.0740 [+0.0580, +0.0907] * | +0.0043 [+0.0008, +0.0083] * |

`*` = the 95% interval on the paired difference excludes zero.


## Robust AUC by native short side (fused, distorted rows)

| model | 80-336px | 352-512px | 528-736px | 752-4016px |
|---|---|---|---|---|
| `crop` | 0.9113 (n=314) | 0.9031 (n=331) | 0.8597 (n=294) | 0.8651 (n=311) |
| `squish` | 0.9510 (n=314) | 0.9712 (n=331) | 0.9653 (n=294) | 0.9672 (n=311) |

## External benchmarks, fused

`ext_X` = files as distributed; `_jpeg95` = every file re-encoded at JPEG q95 (format equalised); `_deg` = one seeded ntire-pool chain per file on the native image (our synthesis, not the organisers').

| set | n | real/fake | short p50 | >512px | short-side-vs-label AUC | format-vs-label AUC |
|---|---|---|---|---|---|---|
| `ext_aigi_socialrf` | 5994 | 2994/3000 | 1366 | 96% | 0.694 | 0.500 |
| `ext_aigi_community` | 12000 | 6000/6000 | 500 | 42% | 0.916 | 1.000 |
| `ext_synthbuster` | 9990 | 994/8996 | 833 | 70% | 0.519 | 0.555 |

### fused, ext_aigi_socialrf
n=5,994 (robust 0, clean 5,994), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | clean_auc |
|---|---|---|
| `crop` | 0.9633 [0.9591, 0.9674] | 0.9633 [0.9591, 0.9674] |
| `squish` | 0.9776 [0.9737, 0.9813] | 0.9776 [0.9737, 0.9813] |

Paired difference against `crop`:

| model | auc | clean_auc |
|---|---|---|
| `squish` | +0.0142 [+0.0100, +0.0183] * | +0.0142 [+0.0100, +0.0183] * |

`*` = the 95% interval on the paired difference excludes zero.

Resolution-matched (n-weighted AUC within 10 deciles of native short side, 5994 rows): crop 0.9588, squish 0.9759, diff +0.0172 [+0.0099, +0.0250] *

### fused, ext_aigi_socialrf_jpeg95
n=5,994 (robust 0, clean 5,994), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | clean_auc |
|---|---|---|
| `crop` | 0.9653 [0.9611, 0.9692] | 0.9653 [0.9611, 0.9692] |
| `squish` | 0.9778 [0.9740, 0.9815] | 0.9778 [0.9740, 0.9815] |

Paired difference against `crop`:

| model | auc | clean_auc |
|---|---|---|
| `squish` | +0.0126 [+0.0084, +0.0166] * | +0.0126 [+0.0084, +0.0166] * |

`*` = the 95% interval on the paired difference excludes zero.

Resolution-matched (n-weighted AUC within 10 deciles of native short side, 5994 rows): crop 0.9602, squish 0.9762, diff +0.0160 [+0.0089, +0.0238] *

### fused, ext_aigi_socialrf_deg
n=5,994 (robust 5,994, clean 0), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc |
|---|---|---|
| `crop` | 0.8847 [0.8763, 0.8927] | 0.8847 [0.8763, 0.8927] |
| `squish` | 0.9695 [0.9653, 0.9737] | 0.9695 [0.9653, 0.9737] |

Paired difference against `crop`:

| model | auc | robust_auc |
|---|---|---|
| `squish` | +0.0849 [+0.0772, +0.0925] * | +0.0849 [+0.0772, +0.0925] * |

`*` = the 95% interval on the paired difference excludes zero.


### fused, ext_aigi_community
n=12,000 (robust 0, clean 12,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | clean_auc |
|---|---|---|
| `crop` | 0.9809 [0.9787, 0.9830] | 0.9809 [0.9787, 0.9830] |
| `squish` | 0.9920 [0.9905, 0.9934] | 0.9920 [0.9905, 0.9934] |

Paired difference against `crop`:

| model | auc | clean_auc |
|---|---|---|
| `squish` | +0.0111 [+0.0091, +0.0132] * | +0.0111 [+0.0091, +0.0132] * |

`*` = the 95% interval on the paired difference excludes zero.

Resolution-matched (n-weighted AUC within 2 deciles of native short side, 5920 rows): crop 0.9790, squish 0.9839, diff +0.0049 [-0.0018, +0.0119]

### fused, ext_aigi_community_jpeg95
n=12,000 (robust 0, clean 12,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | clean_auc |
|---|---|---|
| `crop` | 0.9802 [0.9780, 0.9823] | 0.9802 [0.9780, 0.9823] |
| `squish` | 0.9920 [0.9905, 0.9933] | 0.9920 [0.9905, 0.9933] |

Paired difference against `crop`:

| model | auc | clean_auc |
|---|---|---|
| `squish` | +0.0118 [+0.0098, +0.0139] * | +0.0118 [+0.0098, +0.0139] * |

`*` = the 95% interval on the paired difference excludes zero.

Resolution-matched (n-weighted AUC within 2 deciles of native short side, 5920 rows): crop 0.9786, squish 0.9834, diff +0.0048 [-0.0022, +0.0120]

### fused, ext_aigi_community_deg
n=12,000 (robust 12,000, clean 0), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc |
|---|---|---|
| `crop` | 0.9059 [0.9008, 0.9108] | 0.9059 [0.9008, 0.9108] |
| `squish` | 0.9759 [0.9735, 0.9784] | 0.9759 [0.9735, 0.9784] |

Paired difference against `crop`:

| model | auc | robust_auc |
|---|---|---|
| `squish` | +0.0699 [+0.0650, +0.0750] * | +0.0699 [+0.0650, +0.0750] * |

`*` = the 95% interval on the paired difference excludes zero.


### fused, ext_synthbuster
n=9,990 (robust 0, clean 9,990), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | clean_auc |
|---|---|---|
| `crop` | 0.9995 [0.9992, 0.9997] | 0.9995 [0.9992, 0.9997] |
| `squish` | 0.9984 [0.9978, 0.9989] | 0.9984 [0.9978, 0.9989] |

Paired difference against `crop`:

| model | auc | clean_auc |
|---|---|---|
| `squish` | -0.0010 [-0.0016, -0.0005] * | -0.0010 [-0.0016, -0.0005] * |

`*` = the 95% interval on the paired difference excludes zero.

Resolution-matched: not computable -- no decile of native short side holds both classes.

### fused, ext_synthbuster_jpeg95
n=9,990 (robust 0, clean 9,990), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | clean_auc |
|---|---|---|
| `crop` | 0.9993 [0.9991, 0.9996] | 0.9993 [0.9991, 0.9996] |
| `squish` | 0.9986 [0.9980, 0.9991] | 0.9986 [0.9980, 0.9991] |

Paired difference against `crop`:

| model | auc | clean_auc |
|---|---|---|
| `squish` | -0.0008 [-0.0014, -0.0002] * | -0.0008 [-0.0014, -0.0002] * |

`*` = the 95% interval on the paired difference excludes zero.

Resolution-matched: not computable -- no decile of native short side holds both classes.

### fused, ext_synthbuster_deg
n=9,990 (robust 9,990, clean 0), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc |
|---|---|---|
| `crop` | 0.9683 [0.9641, 0.9722] | 0.9683 [0.9641, 0.9722] |
| `squish` | 0.9921 [0.9905, 0.9936] | 0.9921 [0.9905, 0.9936] |

Paired difference against `crop`:

| model | auc | robust_auc |
|---|---|---|
| `squish` | +0.0237 [+0.0197, +0.0279] * | +0.0237 [+0.0197, +0.0279] * |

`*` = the 95% interval on the paired difference excludes zero.


## Per branch, ntire_val_hard: crop, crop presented whole at test time, trained under squish

### branch A, ntire_val_hard
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9408 [0.9324, 0.9489] | 0.8561 [0.8350, 0.8755] | 0.9913 [0.9873, 0.9945] |
| `crop_testtime_squish` | 0.9730 [0.9680, 0.9777] | 0.9482 [0.9366, 0.9591] | 0.9906 [0.9870, 0.9937] |
| `trained_squish` | 0.9729 [0.9675, 0.9778] | 0.9425 [0.9302, 0.9539] | 0.9934 [0.9899, 0.9960] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop_testtime_squish` | +0.0322 [+0.0243, +0.0402] * | +0.0920 [+0.0727, +0.1123] * | -0.0007 [-0.0050, +0.0038] |
| `trained_squish` | +0.0321 [+0.0243, +0.0398] * | +0.0864 [+0.0679, +0.1054] * | +0.0020 [-0.0019, +0.0062] |

`*` = the 95% interval on the paired difference excludes zero.


### branch B, ntire_val_hard
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9420 [0.9332, 0.9499] | 0.8873 [0.8694, 0.9036] | 0.9801 [0.9739, 0.9855] |
| `crop_testtime_squish` | 0.9765 [0.9717, 0.9812] | 0.9599 [0.9497, 0.9692] | 0.9886 [0.9845, 0.9924] |
| `trained_squish` | 0.9768 [0.9720, 0.9814] | 0.9589 [0.9489, 0.9685] | 0.9898 [0.9853, 0.9937] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop_testtime_squish` | +0.0345 [+0.0269, +0.0428] * | +0.0726 [+0.0566, +0.0900] * | +0.0085 [+0.0028, +0.0151] * |
| `trained_squish` | +0.0348 [+0.0272, +0.0433] * | +0.0716 [+0.0549, +0.0887] * | +0.0097 [+0.0038, +0.0164] * |

`*` = the 95% interval on the paired difference excludes zero.


Picks (final vs SWA, chosen on ntire_val alone): A -> `outputs/scores/paper/A3sqswa`, B -> `outputs/scores/paper/B3sqswa`, rule -> `mean(robust,clean) on ntire_val; tie -> final`
