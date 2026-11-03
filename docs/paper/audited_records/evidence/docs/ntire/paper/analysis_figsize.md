# Fused robust AUC by native short side (ntire_val_hard, distorted rows)

8 equal-count bins. Crop fusion against the squish-trained fusion; paired 95% CI on the difference, 2000 resamples stratified by label.

| short side | median | n | crop | squish | diff [95% CI] |
|---|---|---|---|---|---|
| 80-256 | 208 | 157 | 0.9359 | 0.9508 | +0.0148 [-0.0148, +0.0492] |
| 272-320 | 304 | 121 | 0.8939 | 0.9533 | +0.0593 [+0.0167, +0.1094] |
| 336-400 | 368 | 171 | 0.9162 | 0.9666 | +0.0504 [+0.0175, +0.0899] |
| 416-496 | 448 | 171 | 0.8773 | 0.9731 | +0.0958 [+0.0550, +0.1458] |
| 512-592 | 560 | 161 | 0.8828 | 0.9642 | +0.0814 [+0.0350, +0.1315] |
| 608-720 | 656 | 146 | 0.8578 | 0.9636 | +0.1059 [+0.0500, +0.1669] |
| 736-976 | 832 | 158 | 0.8662 | 0.9581 | +0.0919 [+0.0414, +0.1428] |
| 992-4016 | 1360 | 165 | 0.8596 | 0.9712 | +0.1116 [+0.0534, +0.1736] |

Figure: `docs/paper/fig_size.pdf`. Commit `5ecff8fc3c3070ed0c9b1534666aaf39a5ff1c41`.
