### clip s2: squish-trained vs crop-trained under test-time squish (exploratory, not registered)
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `testtime` | 0.9061 [0.8945, 0.9170] | 0.8635 [0.8431, 0.8832] | 0.9396 [0.9268, 0.9520] |
| `trained` | 0.9344 [0.9251, 0.9435] | 0.9017 [0.8844, 0.9176] | 0.9650 [0.9552, 0.9740] |

Paired difference against `testtime`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `trained` | +0.0282 [+0.0186, +0.0375] * | +0.0383 [+0.0225, +0.0546] * | +0.0253 [+0.0146, +0.0361] * |

`*` = the 95% interval on the paired difference excludes zero.
