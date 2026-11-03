# Test 12: writing fixes

Amendment 5, Test 12 (`docs/ntire/paper_prereg.md`). Script
`scripts/amend5_test12.py`, numbers in `test12_writing.json`, per-row TPR
table in `/scratch/anonymous/dion_extensions/outputs/test12/test12_tpr_rows.parquet`.

## 1. Summary & results

**Registered condition:** none. Test 12 has no pass condition: "No scoring.
The output is a list of proposed edits to `docs/paper/main.tex` with the
numbers that support each one, for review. The edits are not applied."
**Verdict: not applicable (no pass/fail registered).** Every item below is
computed; nothing in `main.tex` is changed (D10).

| item | registered as | result |
|---|---|---|
| 1 | "TPR@1%FPR for every table" | Computed for 66 rows in 7 tables. All 66 recomputed AUCs match `main.tex` at its printed precision. One new finding: on SocialRF as distributed, squish has **lower** TPR@1%FPR than crop (0.482 vs 0.538), although its AUC is higher (+0.014). |
| 2 | "the 416-496 px bin per frozen branch" | The gap holds on each branch alone. Squish at test gains +0.111 [+0.059, +0.170] on A and +0.079 [+0.036, +0.129] on B over the centre window. In this bin the window FOV is 0.403-0.481. |
| 3 | "crop's final-vs-SWA choice redone on `ntire_val` alone" | Same choice as the paper: A final (A2), B SWA (B2swa). Criterion differences are under 0.001, but both intervals exclude zero. |
| 4 | "selection asymmetry for test read 1 is disclosed, not rescored" | Text proposal only (E7). |
| L4 | Test 5 §6.3 text edit and the 70.8% / 72.7% rounding note | Carried as E8 and E9, **pending researcher’s Test 5 review**. |
| - | `sec:cost` text | E10 proposed from Test 11 (NOT MET, cost unexplained), **pending researcher’s Test 11 review**. |

## 2. Current state

`main.tex` prints TPR@1%FPR in only two places: `tab:test` (both
`ntire_test` reads) and one sentence in `sec:diagnosis` (fused, 0.516 to 0.752). Every
other table reports AUC only. The 416-496 px argument in `sec:diagnosis` ("Coverage,
not mismatch") rests on the fused curves of Figure 1. The crop models'
final-vs-SWA choice was made on both validation splits. The squish models'
choice was made on `ntire_val` alone.

## 3. Opportunities

- A TPR column states the operating point directly. That is where AUC and
  deployment can disagree (item 1, SocialRF).
- Per-branch numbers in the 416-496 px bin answer the question of whether
  one branch carries the fused gap.
- Redoing the final-vs-SWA choice on `ntire_val` alone removes one of the
  two selection asymmetries between the policies, or shows that it changes
  nothing.

## 4. Testing methodology

### 4.1 Inputs

Existing per-image parquets only. Nothing is scored.

| table | source |
|---|---|
| `tab:probe` | `outputs/scores/probe_{A2,B2swa}_{center,grid9,fov3,squish}_ntire_val_hard.parquet` |
| `tab:aspect` | `outputs/scores/paper/aspect_{A2,B2swa}_{short_center,letterbox}_ntire_val_hard.parquet` |
| `tab:branches` | `outputs/scores/{A2,B2swa}_ntire_val_hard`, `probe_*_squish`, `paper/{A3sqswa,B3sqswa}` |
| `tab:clip` | `outputs/scores/paper/{abl,clip}_probe_s{1,2,3}_{center,squish}`, `{abl,clip}_squish_s{1,2,3}` (mean over seeds) |
| `tab:external` | fused parquets `fusion_{crop,squish}_<split>_fused.parquet` (column `fused_z`) in researcher’s checkout, `docs/ntire/paper/`; git-ignored, read only |
| `tab:public` | `outputs/scores/paper/public/bfree_<split>.parquet`, columns `logit_{native,train_fov,squish}` |
| `tab:groups` | probe `center` / `squish` parquets; groups from `paper_analyses.distortion_group` |
| bin 416-496 | fused crop / squish parquets (bin edges), probe and squish-trained per-branch parquets |
| final vs SWA | `outputs/scores/{A2,A2swa,B2,B2swa}_{ntire_val,ntire_val_hard}.parquet` |

`tab:fov` has no scores behind it. `tab:test` already prints TPR@1%, and
`ntire_test` is never read (the script asserts this).

### 4.2 Metrics

- TPR@1%FPR is `aigcdet.eval.metrics.tpr_at_fpr`, the project's existing
  definition: the TPR at the lowest threshold whose FPR does not exceed
  0.01, with `drop_intermediate=False`.
- Pooled TPR sets its threshold on all reals. Robust TPR uses the distorted
  reals only, and clean TPR the clean reals only, so each TPR sits next to
  the AUC on the same rows.
- Every row's AUC is recomputed and compared with the value printed in
  `main.tex`, at the number of decimals printed there.
- Item 2 recomputes the Figure 1 bin edges the way `paper_analyses.figsize`
  does: 8 equal-count bins over the distorted `ntire_val_hard` rows. The
  script asserts that bin 4 is 416-496 px with 171 rows, then compares arms
  with `paired_diff` (2,000 resamples, seed 20260827, stratified by label),
  as in `figsize`.
- Item 3 applies the `picks.json` rule: mean of robust and clean AUC on
  `ntire_val`, with a tie going to final. Intervals come from
  `paired_bootstrap.paired_bootstrap` (2,000 resamples, label x is_distorted
  strata, seed 20260827).

### 4.3 Deviation: no intervals on TPR

The amendment requires intervals from `paired_bootstrap.py`, and the plan
forbids writing a new bootstrap. `paired_bootstrap.all_metrics` has no TPR,
so every TPR here is a point estimate. Adding `tpr` to `all_metrics` is a
one-line change, but it touches a shared file whose outputs are already
committed. That is researcher’s call (Section 7).

## 5. Findings

### 5.1 Every AUC reproduces

All 66 recomputed AUCs match `main.tex`. The table rows are tied to the right
parquets, so the TPRs below belong to the rows they sit next to.

### 5.2 TPR@1%FPR, `ntire_val_hard` tables (robust / clean)

| table | row | robust AUC | robust TPR | clean AUC | clean TPR |
|---|---|---|---|---|---|
| probe | A centre | 0.8561 | 0.158 | 0.9912 | 0.861 |
| probe | A grid9 | 0.8495 | 0.138 | 0.9978 | 0.960 |
| probe | A fov3 | 0.8703 | 0.147 | 0.9977 | 0.979 |
| probe | A squish | 0.9482 | 0.485 | 0.9906 | 0.880 |
| probe | B centre | 0.8872 | 0.307 | 0.9801 | 0.786 |
| probe | B grid9 | 0.9049 | 0.354 | 0.9931 | 0.928 |
| probe | B fov3 | 0.9215 | 0.360 | 0.9942 | 0.917 |
| probe | B squish | 0.9599 | 0.674 | 0.9886 | 0.882 |
| aspect | A short-centre | 0.9415 | 0.477 | 0.9903 | 0.874 |
| aspect | A letterbox | 0.9482 | 0.629 | 0.9866 | 0.842 |
| aspect | B short-centre | 0.9575 | 0.638 | 0.9891 | 0.867 |
| aspect | B letterbox | 0.9558 | 0.610 | 0.9851 | 0.810 |
| branches | A crop-trained | 0.8561 | 0.157 | 0.9913 | 0.859 |
| branches | A squish at test | 0.9482 | 0.485 | 0.9906 | 0.880 |
| branches | A squish-trained | 0.9425 | 0.541 | 0.9934 | 0.907 |
| branches | B crop-trained | 0.8873 | 0.310 | 0.9801 | 0.784 |
| branches | B squish at test | 0.9599 | 0.674 | 0.9886 | 0.882 |
| branches | B squish-trained | 0.9589 | 0.661 | 0.9898 | 0.891 |
| clip | DINOv2 crop-trained | 0.7995 | 0.030 | 0.9254 | 0.530 |
| clip | DINOv2 squish at test | 0.8310 | 0.173 | 0.9087 | 0.451 |
| clip | DINOv2 squish-trained | 0.8841 | 0.335 | 0.9358 | 0.604 |
| clip | CLIP crop-trained | 0.7482 | 0.067 | 0.9655 | 0.600 |
| clip | CLIP squish at test | 0.8484 | 0.236 | 0.9324 | 0.551 |
| clip | CLIP squish-trained | 0.8924 | 0.288 | 0.9645 | 0.730 |

Each threshold is set on 625 reals; `tab:clip` rows are means over three
seeds. Crop-trained A has robust TPR 0.158 under its own window and 0.485
under squish (3.1x). B goes from 0.307 to 0.674 (2.2x). The AUC gains (+9.2
and +7.3 points) understate how much the operating point moves. Grid9 lowers
A's robust TPR (0.158 to 0.138) just as it lowers A's robust AUC.

The probe's own centre arm and the as-trained rescore differ by at most
0.003 in TPR (A 0.158 vs 0.157, B 0.307 vs 0.310). The AUCs differ by 0.0001
(the `tab:branches` caption already notes this).

### 5.3 TPR@1%FPR, external sets (fused, pooled)

| set | variant | crop AUC | squish AUC | crop TPR | squish TPR |
|---|---|---|---|---|---|
| SocialRF | as distributed | 0.9633 | 0.9776 | **0.538** | **0.482** |
| SocialRF | JPEG 95 | 0.9653 | 0.9778 | 0.560 | 0.490 |
| SocialRF | degraded (ours) | 0.8847 | 0.9695 | 0.324 | 0.458 |
| CommunityAI | as distributed | 0.9809 | 0.9920 | 0.797 | 0.903 |
| CommunityAI | JPEG 95 | 0.9802 | 0.9920 | 0.781 | 0.902 |
| CommunityAI | degraded (ours) | 0.9059 | 0.9759 | 0.397 | 0.776 |
| Synthbuster | as distributed | 0.9995 | 0.9984 | 0.985 | 0.981 |
| Synthbuster | JPEG 95 | 0.9993 | 0.9986 | 0.986 | 0.983 |
| Synthbuster | degraded (ours) | 0.9683 | 0.9921 | 0.704 | 0.892 |

Thresholds are set on 2,994 (SocialRF), 6,000 (CommunityAI) and 994
(Synthbuster) reals. **On SocialRF, clean and JPEG 95, squish has the higher
AUC but the lower TPR at 1% FPR (-0.056 and -0.070).** The ROC curves cross:
squish ranks better overall, and crop separates better at the strict end.
This qualifies `sec:heldout`'s "On SocialRF, squish wins on the files as
distributed, after JPEG re-encoding, and within size deciles". No interval
exists on these TPR differences (Section 4.3).

### 5.4 TPR@1%FPR, B-Free (`tab:public`)

| split | native | train FOV | squish |
|---|---|---|---|
| hard-val, robust | 0.035 | 0.043 | 0.034 |
| hard-val, clean | 0.570 | 0.373 | 0.315 |
| `ntire_val`, robust | 0.234 | 0.236 | 0.200 |
| `ntire_val`, clean | 0.615 | 0.436 | 0.374 |
| SocialRF | 0.459 | 0.394 | 0.345 |
| SocialRF degraded | 0.285 | 0.296 | 0.270 |
| CommunityAI | 0.460 | 0.231 | 0.168 |
| CommunityAI degraded | 0.113 | 0.105 | 0.074 |
| Synthbuster | 0.980 | 0.878 | 0.780 |
| Synthbuster degraded | 0.421 | 0.388 | 0.430 |

Robust TPR on hard-val is 0.03-0.04 under every presentation. That fits the
paper's caution about B-Free on NTIRE data. On SocialRF degraded, squish's
+6.3 AUC points over native (`sec:public`) come with a lower TPR (0.270 vs 0.285).

### 5.5 Robust TPR@1%FPR within the `tab:groups` groups

| group | n | reals | A centre | A squish | B centre | B squish |
|---|---|---|---|---|---|---|
| photometric only | 241 | 115 | 0.190 | 0.500 | 0.246 | 0.770 |
| downscale, no crop | 740 | 370 | 0.049 | 0.600 | 0.343 | 0.619 |
| crop or aspect crop | 191 | 98* | 0.247 | 0.301 | 0.183 | 0.710 |
| perspective | 78 | 42* | 0.250 | 0.333 | 0.306 | 0.583 |

\* Fewer than 100 reals: a 1% FPR cannot be resolved. These values are TPR at
zero false positives. Squish raises TPR in all eight cells. The largest
change is A on "downscale, no crop", from 0.049 to 0.600.

### 5.6 The 416-496 px bin, per frozen branch

171 distorted `ntire_val_hard` rows (85 fake). The 200 px window covers
200/496 to 200/416 = 0.403-0.481 of the short side.

| branch | centre window | squish at test (gain [95% CI]) | squish-trained (gain [95% CI]) |
|---|---|---|---|
| fused (Figure 1) | 0.8773 | - | 0.9731 (+0.096 [+0.055, +0.146]) |
| A (A2) | 0.8512 | 0.9617 (+0.111 [+0.059, +0.170]) | 0.9676 (+0.116 [+0.067, +0.172]) |
| B (B2swa) | 0.8840 | 0.9633 (+0.079 [+0.036, +0.129]) | 0.9620 (+0.078 [+0.035, +0.129]) |

The fused row reproduces Figure 1 (0.877, 0.973). Both branches show the gap
on their own, and every interval excludes zero. A's gap is larger than its
pooled gain (+0.111 vs +0.092); B's is about the same (+0.079 vs +0.073).
The argument in `sec:diagnosis` does not depend on fusion.

### 5.7 Crop final vs SWA, reselected on `ntire_val`

Criterion: (robust AUC + clean AUC) / 2.

| branch | split | final | SWA | SWA - final, robust [95% CI] | SWA - final, clean [95% CI] | pick |
|---|---|---|---|---|---|---|
| A | `ntire_val` | 0.97922 | 0.97884 | -0.0007 [-0.0010, -0.0004] | -0.0000 [-0.0001, +0.0001] | **final** |
| A | `ntire_val_hard` | 0.92374 | 0.92264 | -0.0021 [-0.0032, -0.0008] | -0.0001 [-0.0005, +0.0002] | final |
| B | `ntire_val` | 0.97846 | 0.97900 | +0.0008 [+0.0005, +0.0011] | +0.0003 [+0.0001, +0.0004] | **SWA** |
| B | `ntire_val_hard` | 0.93266 | 0.93372 | +0.0020 [+0.0008, +0.0033] | +0.0001 [-0.0003, +0.0006] | SWA |

On `ntire_val` alone the choice is unchanged: A2 final, B2swa. Both
splits, pooled or averaged, give the same picks. Redoing the choice changes
no number in the paper.

## 6. Results: proposed edits (not applied)

| # | where in `main.tex` | current | proposed | support |
|---|---|---|---|---|
| E1 | `tab:probe`, `tab:branches` | AUC columns only | Add a "robust TPR@1%" column | §5.2 |
| E2 | `sec:diagnosis`, after "squish gains 9.2 and 7.3" | - | "At 1% false positive rate, squish raises robust TPR from 0.158 to 0.485 on A and from 0.307 to 0.674 on B." | §5.2 |
| E3 | `sec:heldout`, SocialRF sentence | "On SocialRF, squish wins on the files as distributed, after JPEG re-encoding, and within size deciles." | Add: "At 1% false positive rate it does not: TPR 0.482 against 0.538 for crop on the files as distributed, 0.490 against 0.560 after JPEG 95. The ROC curves cross." Optionally add a TPR row to `tab:external`. | §5.3 |
| E4 | `sec:diagnosis`, "Coverage, not mismatch" | "...the crop fusion scores 0.877 against 0.973 for squish." | Add: "Each frozen branch shows the gap on its own: shown the whole image, A gains 11.1 points [5.9, 17.0] and B 7.9 [3.6, 12.9] in this bin." | §5.6 |
| E5 | same paragraph | "the 200-pixel window covers the same share of the image as in training" | "the 200-pixel window covers 40 to 48% of the short side, inside the band of native-scale training windows". Wording **pending researcher’s Test 5 review** (the training band is wider after augmentation). | §5.6, Test 5 |
| E6 | Methods, "Fusion and selection" | "for the crop models we kept the choice made earlier on both validation splits." | "for the crop models we kept the choice made earlier on both validation splits; redoing it on `ntire_val` alone gives the same choice (A final, B averaged)." | §5.7 |
| E7 | `sec:heldout` (second test read) or `tab:test` caption | caption: "Read 1's fusion weight was fitted on both validation splits ($w = 0.45$); read 2's on `ntire_val` alone." | Say why it matters: "Read 1's weight and z-scores saw `ntire_val_hard`, and read 2's did not, so the two reads were not selected the same way. We report this, and do not rescore read 1." | prereg, Test 12 |
| E8 | `sec:diagnosis`, after `tab:fov` | "In training it always covers 39 to 52% of the short side." | Test 5 §6.3: "Before augmentation, the window covers 39 to 52% of the short side. Augmentation widens this in both directions. A pre-window downscale shows the whole short side in 6% of views. A post-window crop shows as little as 20% in 18% of views, at higher magnification. At native pixel scale, no training view shows less than 39%." Caption: "Training is close to a single value" becomes "Before augmentation, training is close to a single value." **Pending researcher’s Test 5 review.** | Test 5 §5.6, §6.3 |
| E9 | `tab:fov` caption | "70.8% fall below it" | Keep 70.8%, which uses the exact bound 200/512 = 0.3906 (1,770 of 2,500 rows have a short side > 512). With the rounded 0.391 the share would be 72.7%. Option: "70.8% fall below the smallest FOV of a native-scale training window (200/512)." **Pending researcher’s Test 5 review.** | Test 5 §6.3 |
| E10 | `sec:cost`, after "...so we cannot say what." | - | Add: "Synthbuster, which has generator labels, does not locate it either. For branch A every generator's clean AUC changes by less than 0.007 under squish training; on Firefly, its only generator above 1100 pixels, the fakes' mean score falls (5.28 to 3.94) and TPR at 1% FPR drops 4.1 points, the direction our hypothesis predicts but a third of the size we registered. Size and generator cannot be separated there: the 997 Firefly fakes are the only fakes above 1100 pixels. Branch B loses clean AUC on Synthbuster instead, most on DALL-E 2 at 1024 pixels (-0.110), below the size cut." **Pending researcher’s Test 11 review.** | Test 11 §1, §5.1, §5.2 |

E1-E4 and E6 rest on numbers computed here. E5, E8 and E9 wait on researcher’s
review of Test 5. E10 waits on researcher’s review of Test 11.

## 7. Future work

- If researcher agrees, add `tpr` to `paired_bootstrap.all_metrics` so TPR
  differences get paired intervals. E3 should not go in as a claim without
  one.
- The DINOv2 crop-trained proxy has robust TPR 0.030. If `tab:clip` gets a
  TPR column, that row will draw attention. It is a correct point estimate
  on 625 reals.

## 8. What didn't work

- TPR intervals: not computed (Section 4.3).
- `tab:groups`: two groups have fewer than 100 reals, so their "TPR@1%FPR"
  is TPR at zero false positives (§5.5).
- `tab:fov` and `tab:test` are not covered. The first has no scores, and the
  second would need `ntire_test`, which only Test 6 may read.

Numbers from commit `bbb1306` (branch `researcher/test12-writing`, script
uncommitted at run time).
