**Amendment-7 supplement (re-analysis R5) to Test 12. It changes no verdict.** `test12_writing.md` and `.json` are unchanged.

# Test 12 supplement R5: corrected count, intervals on E2 and E3, E10 withdrawn

Registered in `docs/ntire/paper_prereg.md`, Amendment 7, "Re-analyses", R5:
"the recomputed-AUC count is corrected; E2 and E3 get intervals; E10 is
withdrawn."

Script `scripts/amend5_test12_supp_a7.py`. Numbers are in `test12_supp_a7.json`.
`ntire_test` is never read, and each data set asserts this.

## 1. Summary

| item | result |
|---|---|
| count | 66 table rows were scored for TPR. **50 of them** have an AUC printed in `main.tex`, and **76 AUC values** were compared. All 76 match. The claim "All 66 recomputed AUCs match" is wrong in its unit: 16 rows have no printed AUC to check. |
| E2 | Robust TPR gain from squish at test: A **+0.262 [+0.112, +0.534]**, B **+0.378 [+0.202, +0.523]**. Both intervals exclude zero. **Keep**, with the interval and the TPR definition stated. |
| E3 | SocialRF TPR@1%, squish - crop: as distributed **-0.060 [-0.260, +0.068]**, JPEG 95 **-0.079 [-0.279, +0.039]**. Both intervals include zero. **Downgrade**: the crossing of the ROC curves is not established. |
| E10 | **Withdrawn** (section 5). |

## 2. The recomputed-AUC count

Counted from `test12_writing.json` (`tables[*].paper_check`):

| table | rows scored | rows with a printed AUC | AUC values compared |
|---|---|---|---|
| `tab:probe` | 8 | 8 | 16 (robust + clean) |
| `tab:aspect` | 4 | 0 | 0 (`main.tex` prints differences only) |
| `tab:branches` | 6 | 6 | 12 |
| `tab:clip` | 6 | 6 | 12 |
| `tab:external` | 18 | 6 | 6 (as distributed only; `_jpeg95` and `_deg` have no printed AUC) |
| `tab:public` | 24 | 24 | 30 |
| **total** | **66** | **50** | **76, all matched** |

The sentence in `test12_writing.md` §1 and §5.1 should read: "TPR@1%FPR
computed for 66 rows. 50 of them have an AUC printed in `main.tex`, and all 76
printed AUC values reproduce at the printed precision." The 7th table, `tab:groups`,
has a separate TPR block (§5.5) and no AUC check.

## 3. E2: robust TPR@1%FPR, squish at test vs centre window

This uses the distorted `ntire_val_hard` rows of the probe parquets, with the
threshold set on the 625 distorted reals. Strata are label x is_distorted.

| branch | centre TPR | squish TPR | squish - centre [95% CI] | Test 12's value (centre -> squish) |
|---|---|---|---|---|
| A (A2) | 0.138 [0.026, 0.235] | 0.400 [0.282, 0.635] | **+0.262 [+0.112, +0.534]** | 0.158 -> 0.485 |
| B (B2swa) | 0.296 [0.157, 0.370] | 0.674 [0.506, 0.718] | **+0.378 [+0.202, +0.523]** | 0.307 -> 0.674 |

The two TPR definitions differ. `score_plan_splits.tpr_at_fpr` takes the
real ranked at ceil(0.99 n) as the threshold. `aigcdet.eval.metrics.tpr_at_fpr`,
used in Test 12, takes the lowest threshold with FPR <= 1%. On 625 reals they
differ by about one real at the threshold. This moves A's points by
0.020 (centre) and 0.085 (squish), while B's squish TPR is identical under both. **Recommendation:
keep E2**, quoted with the interval and one named definition. The centre-window
TPRs rest on about 6 reals and have wide intervals, so the "3.1x / 2.2x"
ratios in Test 12 §5.2 should not be quoted.

## 4. E3: SocialRF, TPR@1%FPR, squish fusion vs crop fusion

Pooled `fused_z`, 2,994 reals and 3,000 fakes. All rows are clean, so the
strata reduce to label.

| variant | crop TPR | squish TPR | squish - crop [95% CI] | AUC squish - crop [95% CI] | Test 12's value |
|---|---|---|---|---|---|
| as distributed | 0.529 [0.452, 0.569] | 0.469 [0.265, 0.605] | **-0.060 [-0.260, +0.068]** | +0.0142 [+0.0100, +0.0183] | 0.538 vs 0.482 |
| JPEG 95 | 0.557 [0.494, 0.588] | 0.478 [0.276, 0.602] | **-0.079 [-0.279, +0.039]** | +0.0126 [+0.0084, +0.0166] | 0.560 vs 0.490 |

The AUC advantage of squish has an interval above zero. The TPR deficit does
not: both intervals include zero, and the squish TPR interval is very wide
(0.27-0.61). Squish's high-score reals sit in a steep part of the curve. The
review's interval of [-0.250, +0.078] is close to this one but not identical,
which fits a different TPR definition or index scheme. This supplement's
interval is the registered one.

**Recommendation: downgrade E3.** Do not write "At 1% false positive rate it
does not ... The ROC curves cross" as a finding. At most say "At 1% FPR the
two fusions are not separable on SocialRF (TPR difference -0.06, 95% CI
-0.26 to +0.07)". Or drop the qualifier and keep `sec:heldout`'s AUC sentence
as it is.

## 5. E10: withdrawn

E10 (Test 12 §6) proposed adding Test 11's Synthbuster result to `sec:cost`.
It quoted the raw-logit figure "the fakes' mean score falls (5.28 to 3.94)" and
called it "the direction our hypothesis predicts". Amendment 7 withdraws that
figure (R4), because it compares logits of two models on different scales.
On each model's own-real z scale, the level falls in neither branch. Only the
large-vs-small gap and the TPR@1% loss remain
(`test11_supp_a7.md` §2). E10 was also still "pending Dion's Test 11 review".
It is withdrawn as registered. Any replacement text would be a new proposal,
built on R4's z-scored numbers and its power statement, and none is made here.

## 6. Limits

- Each data set's pooled AUC difference reproduces `paired_diff`'s interval
  exactly, so the draws are `paired_diff`'s draws. E2: A +0.0322
  [+0.0244, +0.0403], B +0.0346 [+0.0269, +0.0429]. E3: as listed above.
- TPR at 1% rests on about 6 reals (E2) or about 30 reals (E3) at the
  threshold, which is why its intervals are wide.
- No other Test 12 edit was re-examined. E1, E4 and E6-E9 stand as written,
  pending the reviews Test 12 names.

---

Base commit `bbb1306ca202a9d179318def80cead0c66d1cc27` (amendment 6), with
Test 12's staged files and this supplement's script. Seed 20260827, 2,000
draws.
