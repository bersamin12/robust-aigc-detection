**Amendment-7 supplement (re-analysis R4) to Test 11. It changes no verdict.** `test11_clean-cost.md` and `.json` are unchanged.

# Test 11 supplement R4: logits z-scored against each model's own reals

Registered in `docs/ntire/paper_prereg.md`, Amendment 7, "Re-analyses", R4:
"logits z-scored against each model's own reals before any cross-model logit
comparison; the raw-logit 5.28 -> 3.94 figure is withdrawn."

Script `scripts/amend5_test11_supp_a7.py`. Numbers are in `test11_supp_a7.json`.
The script reads the clean Synthbuster scores only, through
`amend5_test11.load_synthbuster`. It never reads `ntire_test`, and it asserts
that no row path contains it.

## 1. Summary

- **The raw-logit figure 5.28 -> 3.94 is withdrawn.** It compared logits of
  two models on different scales. On a common scale (each model's own reals),
  the squish-trained branch-A model puts Firefly (> 1100 px) fakes *further*
  above its reals than the crop-trained model does. The difference is
  +9.0 real-sd [-0.05, +25.2], so the fall in the mean score disappears.
- What survives is relative. Under squish training, the large fakes gain less
  than the small fakes. On branch A the difference of differences is
  -5.1 real-sd [-9.2, -3.2]. On branch B it is -10.7 [-14.2, -8.4].
- At the strict operating point, the large fakes do lose with an interval.
  Branch A TPR@1%FPR on Firefly changes by -0.062 [-0.095, -0.018]. Branch B
  changes by -0.259 [-0.323, -0.179].
- **Test 11's verdict ("cost unexplained", both branches) still holds.** It
  was decided on clean AUC, which does not change under a per-model monotone
  transform. R4 changes only the logit-scale description in Test 11 §5.2 and
  §6.2.

## 2. Z-scored fakes-only comparison (Test 11, analysis 2, redone)

z = (logit - mean of that model's 994 reals) / sd of those reals (ddof = 1).
Both values are re-estimated on every draw. TPR@1%FPR is
`score_plan_splits.tpr_at_fpr`, with its threshold re-derived on each draw's
reals. The bootstrap uses 2,000 draws of `paired_diff`'s index scheme, seed
20260827, stratified by label (all rows are clean). Every statistic is
computed on the same draw.

| branch | bucket (fake short side) | n fake | mean fake z, crop -> squish | squish - crop [95% CI] | TPR@1%, crop -> squish | squish - crop [95% CI] |
|---|---|---|---|---|---|---|
| A | <= 1100 (all but Firefly) | 7999 | 19.73 -> 33.92 | +14.19 [+3.35, +34.30] | 0.982 -> 0.984 | +0.002 [-0.010, +0.014] |
| A | > 1100 (Firefly only) | 997 | 18.17 -> 27.21 | +9.04 [-0.05, +25.17] | 0.980 -> 0.918 | **-0.062 [-0.095, -0.018]** |
| B | <= 1100 | 7999 | 15.00 -> 27.41 | +12.40 [+6.45, +20.34] | 0.961 -> 0.878 | **-0.083 [-0.107, -0.056]** |
| B | > 1100 | 997 | 13.64 -> 15.37 | +1.73 [-2.27, +6.41] | 0.964 -> 0.705 | **-0.259 [-0.323, -0.179]** |

Difference of differences, (> 1100) - (<= 1100), in mean fake z:

| branch | value [95% CI] |
|---|---|
| A | **-5.15 [-9.20, -3.22]** |
| B | **-10.68 [-14.21, -8.36]** |

Index check: the pooled clean AUC difference (squish - crop, all fakes
against all reals) reproduces `paired_diff`'s interval exactly on both
branches. On A it is -0.0009 [-0.0015, -0.0004], and on B -0.0220
[-0.0246, -0.0193].

## 3. Findings

1. Raw logits gave the wrong sign for the level. The squish-trained models'
   reals are much tighter than the crop-trained models' reals, so a lower raw
   mean fake logit (5.28 -> 3.94) is still a larger distance from the reals.
   In mean-z terms, no bucket on either branch shows squish training moving
   the fakes toward the reals. The lower interval bound on A > 1100 is
   -0.05.
2. The mechanism the paper proposes predicts that squish training costs the
   large images more than the small ones. That prediction survives as a
   relative statement: the large bucket gains about 5 (A) and 11 (B) real-sd
   less than the small bucket, and both intervals lie below zero. On this set
   "large" means Firefly (Test 11 §8.1), so this is equally a Firefly effect.
3. At 1% FPR, the large bucket loses on both branches, and the interval
   excludes zero. Test 11 gave branch A as 0.984 -> 0.943 (-0.041), using the
   0.99-quantile threshold. With `score_plan_splits.tpr_at_fpr` it is
   0.980 -> 0.918 (-0.062). The two definitions differ by about one real at
   the threshold. The sign and the conclusion agree.
4. Test 11's verdict is unchanged. The registered condition is about clean
   AUC, and branch A's AUC drops are all under 0.007.

## 4. Does Synthbuster have power for the per-generator AUC comparison?

**Statistically yes. Substantively no.** Branch A's crop-trained AUC is at
least 0.998 on every generator (minimum 0.9980, `dalle2`), and the widest
paired interval is 0.0067 wide (Firefly). A drop of the registered size
(0.020) would have been resolved, and so would anything near the
`ntire_test` loss (-0.094). But the median fake of each generator sits 17.6
to 21.3 real-sd above the crop-trained model's reals. AUC can only fall when
fakes cross into the reals, so a large shift in the fake scores can leave the
AUC near 1.000. The TPR@1%FPR result (finding 3) is an example: a shift that
AUC barely registers. A null per-generator AUC result on Synthbuster therefore
says little about a cost on a set where crop's AUC is 0.974 to begin with,
such as `ntire_test` > 1100 px. Branch B starts lower (minimum 0.9941) and has
wider intervals (up to 0.031), and it is the branch that shows drops.

## 5. What the paper's 0.974 -> 0.880 compares

Confirmed from the fusion configs (`docs/ntire/fusion_ntire2_final.json` and
`docs/ntire/paper/fusion_squish_final.json`), `picks.json` and
`docs/ntire/paper/analysis_groups.md`:

| | read 1 (crop) | read 2 (squish) |
|---|---|---|
| branch A scores | `outputs/scores/A2` | `outputs/scores/paper/A3sqswa` |
| checkpoint | `outputs/unfreeze/ntire2_d24_518/checkpoint.pt` | `outputs/paper/unfreeze/ntire3_d24_512_sq/checkpoint.pt` |
| weights | final ("A2 final", `analysis_groups.md`) | SWA (`picks.json`) |
| training round | 2 (`ntire2`, run name `_518`) | 3 (`ntire3`, run name `_512_sq`) |

The 0.974 -> 0.880 comparison therefore differs in three ways, not only in
input policy: final vs SWA weights, training round 2 vs 3, and the run-name
resolution tag (518 vs 512). Test 11 compares the same pair (A2 vs A3sqswa),
so its Synthbuster differences carry the same confound. On branch B the
weights are SWA in both reads (B2swa vs B3sqswa), but the round still
differs.

## 6. Limits

- A mean of z is driven by the tail of the fakes and by the estimated real
  sd. The intervals are wide and skewed: A <= 1100 runs from +3.3 to +34.3.
  The registration names the mean, so no robust variant was substituted.
- Size and generator cannot be separated on Synthbuster above 1100 px
  (Test 11 §8.1). Every "> 1100" statement here is a Firefly statement.
- TPR rests on the top ~10 of 994 reals.
- No verdict is decided here. The registration gives R4 no condition.

---

Base commit `bbb1306ca202a9d179318def80cead0c66d1cc27` (amendment 6), with
Test 11's staged files and this supplement's script. Seed 20260827, 2,000
draws.
