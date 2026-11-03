Amendment-7 supplement (re-analysis R2 to Test 4); it changes no verdict.

# Test 4, R2: JPEG-containing vs JPEG-free rows

Registered in `docs/ntire/paper_prereg.md`, amendment 7, "Re-analyses", R2:
"JPEG-containing vs JPEG-free rows, as disjoint groups, gain difference with
interval, per branch." Test 4 itself (commit `4652e27`) is unchanged and was
not re-run. Script `scripts/amend5_test4_supp_a7.py`, numbers in
`docs/ntire/amend5/test4_supp_a7.json`. Run 2026-10-20, CPU only, 23 s.
`ntire_val_hard` is the fit split for squish, so its numbers are fit-split
numbers.

## Method

- **Rows:** the distorted rows Test 4 uses, loaded with Test 4's own `load()`
  (same parquets; on `ntire_val` the centre arm is the as-trained score,
  plan decision D14).
- **Groups (disjoint):** JPEG = the chain contains any token of Test 4's
  JPEG family (`jpeg`, `jpeg_ai`, `jpeg_recompression_1`,
  `jpeg_recompression_2`, `jpeg_recompression_comb_jpegai`,
  `quantization`), using Test 4's `FAMILIES` and parsed tokens. No-JPEG =
  every other distorted row. The JPEG group is exactly Test 4's table (a)
  JPEG row.
- **Gain:** as in Test 4, robust AUC(squish) - AUC(centre) within the
  group's rows.
- **Interval:** `paired_diff`'s index scheme (seed 20260827,
  `default_rng`, cells from `np.unique(strata)`, per draw concatenate
  `c[rng.integers(0, len(c), len(c))]`), 2,000 draws, 2.5/97.5
  `nanquantile`. Strata: label x is_distorted x group (4 cells; all rows
  are distorted). Both group gains, their difference and the pooled gain
  are computed on the same draw. The script asserts that the pooled gain's
  interval equals `paired_diff(y, centre, squish, strata)`'s exactly; it
  does.
- The script asserts every row is in the named split of the manifest, none
  in `ntire_test`, and that the score parquets' chains match the manifest's.

## Counts

| split | group | n | real / fake | with `downscale` token | short side p50 |
|---|---|---|---|---|---|
| `ntire_val_hard` | JPEG | 832 | 417 / 415 | 633 | 496 |
| `ntire_val_hard` | no-JPEG | 418 | 208 / 210 | 298 | 512 |
| `ntire_val` | JPEG | 1,570 | 759 / 811 | 1,159 | 480 |
| `ntire_val` | no-JPEG | 3,430 | 1,741 / 1,689 | 2,549 | 464 |

## Results

Gain [95% CI]. Difference = JPEG gain - no-JPEG gain, paired on one draw.

### `ntire_val_hard` (primary, fit split)

| branch | JPEG gain | no-JPEG gain | difference | pooled (check) |
|---|---|---|---|---|
| A (A2) | +0.1311 [+0.1047, +0.1578] | +0.0283 [+0.0076, +0.0502] | **+0.1029 [+0.0691, +0.1370]** | +0.0920 [+0.0731, +0.1106] |
| B (B2swa) | +0.0935 [+0.0700, +0.1173] | +0.0400 [+0.0174, +0.0641] | **+0.0535 [+0.0189, +0.0855]** | +0.0727 [+0.0557, +0.0897] |

### `ntire_val` (secondary)

| branch | JPEG gain | no-JPEG gain | difference | pooled (check) |
|---|---|---|---|---|
| A (A2) | +0.0482 [+0.0375, +0.0597] | +0.0153 [+0.0112, +0.0196] | **+0.0328 [+0.0216, +0.0452]** | +0.0237 [+0.0196, +0.0280] |
| B (B2swa) | +0.0326 [+0.0240, +0.0420] | +0.0150 [+0.0109, +0.0193] | **+0.0176 [+0.0074, +0.0276]** | +0.0202 [+0.0165, +0.0243] |

The pooled column reproduces Test 4's "all distorted rows" gains to 4 dp
(its intervals differ slightly because Test 4 stratified by label only).

## Findings

1. **JPEG-containing rows gain more than JPEG-free rows, on both branches
   and both splits, with every difference interval above zero.** On
   `ntire_val_hard` the JPEG-free gain is 22% (A) and 43% (B) of the JPEG
   gain. The ratio JPEG / no-JPEG is 4.6 (A) and 2.3 (B) on
   `ntire_val_hard`, 3.2 (A) and 2.2 (B) on `ntire_val`.
2. **The JPEG-free gain is still above zero everywhere**, with lower bounds
   +0.008 (A) and +0.017 (B) on `ntire_val_hard`, +0.011 on `ntire_val`.
3. **The code review's numbers reproduce.** Points: A +0.131 vs +0.028
   (here +0.1311 vs +0.0283). Difference intervals: the review's
   A [+0.068, +0.138] and B [+0.019, +0.086] are reproduced exactly when the
   strata are label only (A [+0.0676, +0.1376], B [+0.0186, +0.0855], run
   ad hoc, not in the script). With the registered label x group strata they
   are A [+0.0691, +0.1370], B [+0.0189, +0.0855]. Same conclusion.
4. **Test 4 section 4.3's statement that `joint_gain_draws`'s scheme is
   "identical to `paired_diff`'s" is not right for a family.** The draw
   line is copied verbatim, and the seed, label strata and percentile rule
   match. The population differs: `joint_gain_draws` builds its cells from
   **all** 1,250 distorted rows, draws once, then keeps the drawn rows that
   are family members; `paired_diff`, as Test 4 calls it per family, builds
   its cells from the **family's** rows. Consequences, checked numerically
   on `ntire_val_hard` (`test4_resampler_check` in the JSON):
   - For a pseudo-family holding every distorted row, the two give
     bit-identical draws (both branches). The code is the same; only the
     population is not.
   - For the JPEG family the draws are not identical (max |draw difference|
     0.080 for A, 0.058 for B). The family size per resample is fixed at 832
     in `paired_diff` but varies from 763 to 891 (sd 16.9) in
     `joint_gain_draws`, and its real/fake balance is not held fixed.
     The resulting intervals are close (A JPEG gain: [+0.1048, +0.1579] vs
     [+0.1052, +0.1585]; B: [+0.0702, +0.1171] vs [+0.0714, +0.1168]).
   The joint scheme is a valid paired bootstrap for overlapping families
   (it resamples images, and rows in two families move together); it is
   just not the same scheme. Section 4.3 should read "the same draw loop,
   applied to all distorted rows; family sizes vary between draws".
5. **For Test 4's "supports coverage" reading.** The registered verdict
   stands: the coverage condition is defined on table (a) and is met, and R2
   changes no verdict. But the disjoint split makes the qualification in
   Test 4 section 1 much stronger. Section 1 notes that table (a)'s overlap
   compresses the ratio; R2 confirms it. With disjoint groups the gain is
   4.6x larger with JPEG than without on branch A (2.3x on B), and the
   JPEG-free rows carry under half of the JPEG rows' gain on both branches.
   The pattern is "a small gain everywhere, plus a large JPEG-linked term",
   and the JPEG-linked term is most of the gain on the primary split. Under
   a ratio rule applied to these two disjoint groups, coverage would not be
   met on either branch on `ntire_val_hard` (A 4.6, B 2.3). That rule is not
   registered, so it is a description, not a verdict.

## Limits

- **JPEG is confounded with downscale and chain length.** 633 of 832 JPEG
  rows also contain `downscale`, but so do 298 of 418 JPEG-free rows (76% vs
  71%), so downscale alone does not explain the gap. Chain contents
  otherwise differ between groups and are not controlled. This is a
  descriptive split, not an attribution.
- `ntire_val_hard` is the fit split for squish.
- On `ntire_val` the centre arm is the as-trained score, not the probe
  centre arm (Test 4 section 4.1).
- The group intervals here come from the 4-cell joint draw, so they differ
  in the third decimal from Test 4's per-family `paired_diff` intervals
  (JPEG A: [+0.1047, +0.1578] here, [+0.1052, +0.1585] in Test 4).
- A side note on Test 4 section 5.4: its "1,179 of 1,570" JPEG rows "with
  downscale" on `ntire_val` counts the downscale family (`downscale` or
  `pixelate`); with the `downscale` token alone it is 1,159.

## Files

- `scripts/amend5_test4_supp_a7.py`: the re-analysis. Run from the worktree
  root with `PYTHONPATH=src:scripts`.
- `docs/ntire/amend5/test4_supp_a7.json`: every number above.

Base commit `4652e27` (branch `researcher/test4-damage-family`), script
uncommitted at run time.
