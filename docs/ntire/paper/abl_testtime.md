# Exploratory, not registered -- `abl` crop-trained checkpoints presented whole at test time

Mean crop-arm robust AUC 0.7995.

- **T PASSED**: mean robust diff +0.0314 (> +0.0100), seeds with CI above zero 3/3 (>= 2)
- Mechanism, test-time: gain +0.0755 in 752-4016px against +0.0230 in 80-336px -> PASSED
- Mechanism, trained: gain +0.1147 in 752-4016px against +0.0598 in 80-336px -> PASSED

Commit `ed8f550282f41b5e59359f69d60e78f8ae08cfcc`.

### seed 1, crop-trained, test-time squish vs center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.8727 [0.8589, 0.8858] | 0.8036 [0.7791, 0.8267] | 0.9264 [0.9125, 0.9398] |
| `squish` | 0.8719 [0.8582, 0.8849] | 0.8293 [0.8075, 0.8513] | 0.9123 [0.8970, 0.9274] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | -0.0008 [-0.0150, +0.0137] | +0.0257 [+0.0034, +0.0477] * | -0.0141 [-0.0312, +0.0035] |

`*` = the 95% interval on the paired difference excludes zero.


### seed 2, crop-trained, test-time squish vs center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.8694 [0.8557, 0.8834] | 0.7954 [0.7706, 0.8186] | 0.9266 [0.9120, 0.9406] |
| `squish` | 0.8750 [0.8616, 0.8879] | 0.8337 [0.8120, 0.8561] | 0.9118 [0.8966, 0.9273] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0056 [-0.0086, +0.0201] | +0.0383 [+0.0167, +0.0616] * | -0.0148 [-0.0320, +0.0035] |

`*` = the 95% interval on the paired difference excludes zero.


### seed 3, crop-trained, test-time squish vs center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.8681 [0.8540, 0.8815] | 0.7996 [0.7750, 0.8226] | 0.9233 [0.9094, 0.9375] |
| `squish` | 0.8671 [0.8531, 0.8805] | 0.8299 [0.8076, 0.8516] | 0.9020 [0.8851, 0.9180] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | -0.0010 [-0.0152, +0.0140] | +0.0302 [+0.0088, +0.0523] * | -0.0213 [-0.0402, -0.0028] * |

`*` = the 95% interval on the paired difference excludes zero.


