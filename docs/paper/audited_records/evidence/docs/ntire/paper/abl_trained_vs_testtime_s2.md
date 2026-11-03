### dinov2 s2: squish-trained vs crop-trained under test-time squish (exploratory, not registered)
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `testtime` | 0.8750 [0.8616, 0.8879] | 0.8337 [0.8120, 0.8561] | 0.9118 [0.8966, 0.9273] |
| `trained` | 0.9139 [0.9028, 0.9249] | 0.8881 [0.8702, 0.9052] | 0.9384 [0.9248, 0.9512] |

Paired difference against `testtime`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `trained` | +0.0389 [+0.0295, +0.0480] * | +0.0544 [+0.0396, +0.0696] * | +0.0267 [+0.0160, +0.0377] * |

`*` = the 95% interval on the paired difference excludes zero.
