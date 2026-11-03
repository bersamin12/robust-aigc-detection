Amendment-7 supplement (R3) to Test 9. It changes no verdict: Test 9's registered PASS and its write-up `test9_mil-combiner.md` stand as they are.

# Test 9, R3: squish - combiner per size bucket

## What was added

Test 9 judged robust AUC (squish - combiner) on `ntire_val_hard` pooled over
image size. Amendment 7 (R3) asks for the same difference per size bucket of
the image's short side, with intervals, per branch.
`scripts/amend5_test9_supp_a7.py` reads the per-image rows Test 9 saved
(`outputs/test9/test9_rows_{A,B}.parquet`: combiner, squish and grid9 logits
for all 2,500 images). **Nothing was refitted.** Bootstrap: `paired_diff`'s
index scheme (seed 20260827, 2,000 draws, strata label x is_distorted), which
is also the scheme Test 9 used; each bucket is resampled on its own rows.
The script asserts that its pooled all-row squish - combiner interval equals
a fresh `paired_diff` call, and that its pooled robust squish - combiner
equals Test 9's reported number and interval (A +0.0690 [+0.0515, +0.0869],
B +0.0231 [+0.0113, +0.0361]). Both held. `ntire_test` is refused.

Size = min(width, height) of the image as stored in `ntire_val_hard`. Every
row falls in a bucket (none in the gaps between buckets).

## Robust AUC per bucket, with 95% intervals on the differences

| branch | bucket | distorted n (fake) | grid9 | combiner | squish | squish - combiner | grid9 - combiner |
|---|---|---|---|---|---|---|---|
| A | pooled | 1,250 (625) | 0.8495 | 0.8792 | 0.9482 | +0.0690 [+0.0515, +0.0869] | -0.0297 [-0.0431, -0.0165] |
| A | 80-336 | 314 (153) | 0.8817 | 0.8990 | 0.9386 | +0.0396 [+0.0139, +0.0686] | -0.0173 [-0.0349, +0.0005] |
| A | 352-512 | 331 (172) | 0.8733 | 0.8886 | 0.9632 | +0.0746 [+0.0436, +0.1079] | -0.0153 [-0.0376, +0.0076] |
| A | 528-736 | 294 (161) | 0.8095 | 0.8466 | 0.9372 | +0.0906 [+0.0416, +0.1390] | -0.0371 [-0.0681, -0.0055] |
| A | 752+ | 311 (139) | 0.8435 | 0.8878 | 0.9688 | +0.0810 [+0.0452, +0.1182] | -0.0443 [-0.0782, -0.0134] |
| B | pooled | 1,250 (625) | 0.9048 | 0.9369 | 0.9599 | +0.0231 [+0.0113, +0.0361] | -0.0320 [-0.0436, -0.0199] |
| B | 80-336 | 314 (153) | 0.9391 | 0.9730 | 0.9642 | **-0.0088 [-0.0262, +0.0075]** | -0.0339 [-0.0508, -0.0185] |
| B | 352-512 | 331 (172) | 0.9240 | 0.9499 | 0.9724 | +0.0225 [+0.0064, +0.0417] | -0.0259 [-0.0457, -0.0065] |
| B | 528-736 | 294 (161) | 0.8946 | 0.9240 | 0.9602 | +0.0362 [+0.0044, +0.0673] | -0.0294 [-0.0552, -0.0046] |
| B | 752+ | 311 (139) | 0.8550 | 0.9152 | 0.9524 | +0.0372 [+0.0071, +0.0675] | -0.0601 [-0.0934, -0.0289] |

Clean-row squish - combiner (the clean side; 80-336 has no clean rows):
A 352-512 +0.0000 [+0.0000, +0.0000], 528-736 -0.0005 [-0.0021, +0.0009],
752+ -0.0134 [-0.0215, -0.0048]; B 352-512 -0.0015 [-0.0069, +0.0000],
528-736 -0.0064 [-0.0155, -0.0003], 752+ -0.0103 [-0.0171, -0.0041].

## What the numbers show

- **Branch A: squish beats the combiner in every bucket**, each interval
  above zero. The margin is smallest on the smallest images (+0.040) and
  +0.075 to +0.091 above 352 px.
- **Branch B: squish beats the combiner in the three buckets from 352 px
  up**, intervals above zero (+0.023 to +0.037). In 80-336 the combiner's
  point estimate is above squish (-0.0088) and the interval includes zero:
  no resolved difference there.
- So Test 9's "PASS" holds bucket by bucket except on Branch B's smallest
  images, where it is a tie. The pooled B margin (+0.023) is an average of a
  tie on small images and +0.02 to +0.04 on larger ones.
- B's margin grows with size on point estimates; A's peaks at 528-736.
  The bucket intervals overlap, so neither trend is resolved.
- On clean rows the combiner is slightly ahead of squish on large images
  (752+, both branches), as in Test 9's pooled clean result.

## Limits

- The size is the stored size. In `ntire_val_hard` every image under 352 px
  is distorted, so the 80-336 bucket is a set of images that were shrunk or
  cropped by their recorded distortions. Size and distortion type are
  confounded in this bucket.
- Buckets hold about 300 distorted images each; the intervals are wide
  (about +-0.02 to +-0.05).
- One combiner fit (Test 9's, seed 20261019), on a fit split for squish
  (`ntire_val_hard` was used to choose squish).
- "Resolved" reads each interval on its own; no correction for four buckets
  per branch.

All numbers: `test9_supp_a7.json`. Script: `scripts/amend5_test9_supp_a7.py`.

Base commit: `bbb1306` (Test 9's files and this supplement staged, not committed).
