### clip s3: squish-trained vs crop-trained under test-time squish (exploratory, not registered)
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `testtime` | 0.8904 [0.8780, 0.9028] | 0.8369 [0.8150, 0.8589] | 0.9329 [0.9194, 0.9463] |
| `trained` | 0.9283 [0.9187, 0.9378] | 0.8828 [0.8639, 0.9006] | 0.9635 [0.9531, 0.9727] |

Paired difference against `testtime`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `trained` | +0.0379 [+0.0284, +0.0469] * | +0.0459 [+0.0306, +0.0618] * | +0.0306 [+0.0202, +0.0409] * |

`*` = the 95% interval on the paired difference excludes zero.
