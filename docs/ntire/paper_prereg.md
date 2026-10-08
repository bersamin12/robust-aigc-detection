# Pre-registration: training under whole-image canonicalisation ("squish")

Written and committed 2026-10-08, before any squish-trained model exists.
`scripts/paper_pipeline.sh` refuses to start unless this file is committed,
and `scripts/paper_results.py` evaluates the gates below exactly as written.
If the two ever disagree, this file is right and the script has a bug.

## What is already known (and was found on the fit splits)

`docs/ntire/crop_probe.md`, 2026-10-07. The round-two models were trained under
`CanonPolicy(mode="crop", crop_side=200)`: a 200px window at native
resolution. That policy equalises pixel count but leaves field of view
(200 / short side) uncontrolled. In training the field of view is 0.39-0.52,
because every training image is 384-512px. On `ntire_val_hard` 86% of rows
fall outside that band, down to 0.05.

When the frozen crop-trained checkpoints were shown the whole image at test
time instead, fused robust AUC on `ntire_val_hard` rose from 0.8856 to 0.9631.
The z-scores and fusion weight were fitted on `ntire_val` alone for both
numbers. The paired-bootstrap interval on branch B's gain is
[+0.0567, +0.0901]. Three controls argue against a resolution shortcut: the
cue is absent from training, the gain holds within deciles of native size,
and it survives band-limiting to 160px.

That probe chose the policy using `ntire_val_hard`. **So `ntire_val_hard` is
not an unbiased readout of squish.** This study's held-out evidence is the
external benchmarks, plus one declared test read at the end.

## The change under test

Train with `--canon-mode squish`: the whole frame at `nominal_side` square,
aspect discarded (`augment/canonical.py`, `MODE_SQUISH`). Nothing else
changes: same banks, same recipes, same seeds, same schedules, same fusion
procedure.

## Gates

**G1, proxy, decides whether full training happens.** The proxy protocol is
the one every earlier ablation used:

- one dinov2regl224 tower at depth 24;
- 1 epoch over 25% of train, bank `ntire_meta`;
- mean pool, cosine schedule with SWA, final weights;
- scored on `ntire_val_hard` and `ntire_val`.

There are six arms: crop and squish at seeds 20260827, 20260828 and 20260829.
Within a seed, the training subset, head initialisation and batch order are
identical, and only `--canon-mode` differs. A paired bootstrap on
`ntire_val_hard` (2000 resamples, stratified by label × is_distorted, 95%
percentile CI) gives the squish minus crop difference for each seed.

G1 passes if and only if all three hold:

1. The mean over seeds of the robust AUC difference is **> +0.0100**.
2. The CI on the robust difference lies **entirely above zero** for at least
   **2 of 3** seeds.
3. The mean over seeds of the clean AUC difference is **≥ -0.0050**.

If G1 fails, the pipeline stops before full training. The null is written up
as a null: the probe's gain was then a test-time effect that training does not
reproduce, and that finding gets reported.

**G2, full models, NTIRE.** Both branches are retrained under squish exactly
as round two was trained:

- B: dual dinov2regl224 for 3 epochs.
- A: dinov2regl at nominal 512 for 2 epochs.
- Both use bank `ntire2_meta` and seed 20260827.

For each branch, final or SWA weights are picked on `ntire_val` by
mean(robust, clean), with ties going to final. Both the squish fusion and the
crop fusion (round two's A2 + B2swa, unchanged) are re-fitted on `ntire_val`
alone.

G2 passes if the fused robust AUC difference on `ntire_val_hard`
(squish minus crop) is > 0 with a paired 95% CI entirely above zero. Given the
selection caveat above, G2 is necessary but not sufficient evidence.

**G3, external, the held-out test of the policy.** The external sets are
AIGIBench SocialRF, AIGIBench CommunityAI, and Synthbuster with RAISE-1k
reals. They are deduplicated by content digest against every row of
`manifest_ntire`, and nothing was ever trained or selected on them.

G3 is judged on pooled ROC AUC of the fused models on each set's files as
distributed (`ext_<set>`). For each set, G3 passes if squish minus crop is > 0
with a paired 95% CI entirely above zero. **Every set is reported whether it
passes or fails.**

These readings are reported alongside, but are not gates:

- `_jpeg95`: every file re-encoded at JPEG q95. AIGIBench CommunityAI ships
  every real as JPEG and every fake as PNG, so container format alone
  separates its classes.
- `_deg`: one seeded ntire-pool chain per file, applied to the native image.

Short-side-vs-label and format-vs-label AUC are reported for every set.

## Secondary, reported regardless of outcome

- Per branch on `ntire_val_hard`: crop as trained, crop presented whole at
  test time, and trained under squish. Did training add anything beyond the
  test-time change?
- Robust AUC by native short-side bucket for both fusions. Is the size
  gradient gone in a trained model?
- Proxy results on `ntire_val` for each seed.

## The declared second test read

`ntire_test` was read once, on 2026-10-04, with the crop fusion: robust
0.7931, clean 0.9856. One further read is declared here:

- the squish fusion, frozen as above;
- run by `scripts/ntire_test_read2.sh`;
- only after G2, G3 and every table above are written;
- exactly once.

Both reads are reported with this history. There is no third read. Nothing
the read shows can change a choice, because every choice is fixed before it
runs.

## Not done, deliberately

- No sweep of `nominal_side` and no band-limited squish variant in training.
  Those would be post-hoc rescues if squish disappoints.
- No re-picking of the crop baseline. A2 and B2swa were picked in round two on
  pooled `ntire_val` + `ntire_val_hard`, which favours crop, so this
  comparison is conservative.
- No change to the recipe pool, loss or schedule.
