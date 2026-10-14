### clip s1: squish-trained vs crop-trained under test-time squish (exploratory, not registered)
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `testtime` | 0.8887 [0.8757, 0.9012] | 0.8450 [0.8234, 0.8655] | 0.9246 [0.9098, 0.9390] |
| `trained` | 0.9313 [0.9217, 0.9409] | 0.8927 [0.8742, 0.9095] | 0.9651 [0.9549, 0.9743] |

Paired difference against `testtime`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `trained` | +0.0426 [+0.0332, +0.0525] * | +0.0477 [+0.0326, +0.0638] * | +0.0404 [+0.0288, +0.0523] * |

`*` = the 95% interval on the paired difference excludes zero.
