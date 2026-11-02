# Amendment 4 secondary -- `clip` crop-trained checkpoints presented whole at test time

Mean crop-arm robust AUC 0.7482.

- **T PASSED**: mean robust diff +0.1003 (> +0.0100), seeds with CI above zero 3/3 (>= 2)
- Mechanism, test-time: gain +0.1680 in 752-4016px against +0.0573 in 80-336px -> PASSED
- Mechanism, trained: gain +0.2061 in 752-4016px against +0.0912 in 80-336px -> PASSED

Commit `9b8b08ebea1988f64f9ae50f5421ab3c602a8e7d`.

### seed 1, crop-trained, test-time squish vs center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.8731 [0.8605, 0.8849] | 0.7493 [0.7225, 0.7748] | 0.9652 [0.9564, 0.9734] |
| `squish` | 0.8887 [0.8757, 0.9012] | 0.8450 [0.8234, 0.8655] | 0.9246 [0.9098, 0.9390] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0156 [+0.0021, +0.0306] * | +0.0957 [+0.0704, +0.1225] * | -0.0406 [-0.0558, -0.0254] * |

`*` = the 95% interval on the paired difference excludes zero.


### seed 2, crop-trained, test-time squish vs center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.8723 [0.8594, 0.8848] | 0.7520 [0.7255, 0.7789] | 0.9622 [0.9526, 0.9708] |
| `squish` | 0.9061 [0.8945, 0.9170] | 0.8635 [0.8431, 0.8832] | 0.9396 [0.9268, 0.9520] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0338 [+0.0203, +0.0480] * | +0.1115 [+0.0866, +0.1364] * | -0.0225 [-0.0367, -0.0085] * |

`*` = the 95% interval on the paired difference excludes zero.


### seed 3, crop-trained, test-time squish vs center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.8762 [0.8635, 0.8882] | 0.7432 [0.7166, 0.7697] | 0.9692 [0.9610, 0.9767] |
| `squish` | 0.8904 [0.8780, 0.9028] | 0.8369 [0.8150, 0.8589] | 0.9329 [0.9194, 0.9463] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0141 [+0.0011, +0.0285] * | +0.0937 [+0.0693, +0.1194] * | -0.0364 [-0.0509, -0.0220] * |

`*` = the 95% interval on the paired difference excludes zero.


