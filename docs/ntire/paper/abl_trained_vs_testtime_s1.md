### dinov2 s1: squish-trained vs crop-trained under test-time squish (exploratory, not registered)
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `testtime` | 0.8719 [0.8582, 0.8849] | 0.8293 [0.8075, 0.8513] | 0.9123 [0.8970, 0.9274] |
| `trained` | 0.9115 [0.8999, 0.9232] | 0.8866 [0.8678, 0.9046] | 0.9346 [0.9201, 0.9475] |

Paired difference against `testtime`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `trained` | +0.0396 [+0.0304, +0.0490] * | +0.0573 [+0.0419, +0.0733] * | +0.0223 [+0.0118, +0.0335] * |

`*` = the 95% interval on the paired difference excludes zero.
