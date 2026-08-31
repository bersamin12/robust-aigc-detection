# Overnight run, 2026-08-30/31

Every number here is `heldout_robust_tpr_at_1pct` (spec §6.4): TPR at 1% FPR,
`val_internal` authentic against `heldout_generator` generated, averaged over
the 19 degraded conditions with `clean` excluded. Probe = the 20k union subset;
"full" = the 138k frozen corpus. **Probe and full numbers are not comparable**
-- the rung ordering inverts between them (see "Recorded, not fixed").

## The headline

**`dinov2regl` under BOTH canonicalisation policies, score-fused with a fitted
weight, scores 0.8714 -- past the barred `dinov3l` reference's 0.8667.**

    dinov2regl:band  a3                      0.7242
    dinov2regl:crop  a3                      0.7858
    band + crop  FUSED  w=[0.45, 0.55]       0.8714    +0.086 over the better parent
    ---
    dinov3l:band a3 (ablation reference)     0.8667

It costs ONE tower: the same 304M weights run under two canonicalisations, not
two backbones. 304M against the 2B cap, not 608M, which leaves room for a
second tower later. DINOv3 is excluded from the shipped bundle by team decision
(see `model_licences.md`), so this is the result that removes the dependency.

## Probe ladders, all eight arms

    arm                      a0      a1=a2     a3     a7_norecon
    band  dinov2l          0.6934   0.7076   0.6565   0.7061
    crop  dinov2l          0.6751   0.7356   0.7355   0.7091
    band  dinov2regl       0.7153   0.7844   0.7242   0.6698
    crop  dinov2regl       0.6717   0.7730   0.7858   0.8022
    band  siglipso400m     0.7729   0.7754   0.7111   0.7639
    crop  siglipso400m     0.6829   0.6627   0.6166   0.6219
    band  convnextv2h      0.4855   0.6172   0.5138   0.5824
    band  eva02l           0.5072   0.5055   0.4036   0.5482
    band  dinov3l (ref)    0.8379   0.8248   0.8667   0.8237

**The canon policy is backbone-specific, not a global choice.** Crop helps the
DINO lineage (+0.079 on dinov2l at a3, +0.062 on dinov2regl) and badly hurts
SigLIP-SO400M (-0.113 at a1). SigLIP is language-supervised on whole-image
semantics at fixed resolution, so a 200px native crop destroys what its
features encode; DINOv2 is self-supervised with heavy crop augmentation and
tolerates the crop while gaining the preserved native pixel grid. An earlier
claim that "crop beats band" generalised from the DINOv2 arm and was wrong.

**`a1` and `a2` are bit-identical in all eight arms** -- same TPR and same AUC
to 16 digits. Whatever flag separates them does nothing at probe scale on any
backbone.

**`eva02l` and `convnextv2h` both failed.** eva02l reaches 0.4036 at a3, below
chance. Neither is a candidate.

## Fitted fusion weights (rung A5)

`fit_fusion_weight` sweeps w on `val_internal` ALONE
(`FIT_SPLITS_WHEN_FITTING_WEIGHT`), then the held-out number is read once.
`equal_weight_control` scores the equal-weight fusion under the SAME
standardisation, so the gap is attributable to the weight and not to the
changed z-score population.

    pair                                  parents        equal   fitted    w
    dinov3l + siglip2l  (ref)         0.9012 / 0.2893   0.8754   0.9222   0.80
    dinov3l + convnextt (ref)         0.9012 / 0.4882   0.8853   0.9236   0.75
    dinov2regl band + siglipso400m    0.7242 / 0.7111   0.8517   0.8534   0.45
    dinov2regl band + crop            0.7242 / 0.7858   0.8702   0.8714   0.45

**Weight-fitting pays for UNEQUAL parents, not balanced ones.** Against a 0.29
partner it is worth +0.047 and the weight goes to 0.80; against a comparable
partner it is worth +0.002 and the weight sits near 0.5. The equal-weight
dilution effect is real, quantified, and specific to lopsided pairs.

**Two policies of one tower beat two different towers** -- 0.8714 vs 0.8534 --
and cost half the weights.

## Off-ladder probes

**`family_experts`.** A GAN-only and a diffusion-only head over the same frozen
features, differing only in training ROWS. The family label is NOT decoration:
they score 41 points apart on `SDwithAdaptor_controlnet` (0.3397 vs 0.7545).
They still both lose to the pooled a3 head (0.9012), as does their best fusion
(0.8843, P(>a3)=0.017). Specialisation is informative and strictly dominated.
The `disagreement` head scores 0.0247 -- far below chance, so `|z_gan - z_diff|`
marks AUTHENTIC images. Dead end, closed cheaply.

**`head_capacity` (new).** `head_hidden` is now plumbed through `RungConfig` ->
`Detector` -> `load_detector`, defaulting to 512 so no existing number moves.

    backbone   gap      256      512     1024     2048    best
    convnextt  0.115   0.4901   0.4882   0.4624   0.4639    256
    siglip2l   0.071   0.2865   0.2893   0.3004   0.2979   1024  (+0.011)
    dinov3l    0.000   0.9007   0.9012   0.8847   0.8905    512

**512 is at or near optimal on all three**, across towers spanning 0.29 to 0.90
in performance and the whole range of `readout_ceiling`'s gap. The best gain
width produces anywhere is +0.011, against fusion's +0.086; two of three
DEGRADE. A larger MLP is not the lever.

`readout_ceiling`'s "the gap opens as the backbone weakens" does NOT translate
into "wider heads help weaker backbones": the weakest tower is the only one
that gains, and marginally.

On convnextt and dinov3l a wider head HURTS while `val_auc` on SEEN generators
stays flat or rises (dinov3l: 0.9977 -> 0.9981 while held-out falls 1.7 points)
-- the overfitting signature, caught only because held-out generators are
absent from training by construction. This also resolves the ConvNeXt capacity
confound in the opposite direction from the one assumed: `convnextv2h`'s 5632-d
bank buys it a 4.8x larger head, which is a handicap on held-out data, not an
unfair advantage.

**TTA entropy-minimising ADAPTATION: ruled out** (`tta_entropy_pilot.json`).
Every arm neutral to harmful, monotone in learning rate; confidence filtering
was WORSE than no filtering (0.9603 vs 0.9627), because lowest-entropy views
are the least degraded and the metric is a robustness metric. Structurally it
also fails three ways: the `P_v` prompt-token variant needs a backbone forward
per view per image (impossible on cached features); per-image adaptation breaks
the cross-image score comparability TPR@1%FPR requires; and "evaluate on the
unaugmented image" yields the `clean` score, which the metric excludes.

**Per-condition decomposition of a3** (mean over degraded = 0.9012, i.e. the
official metric reproduced). Three MILD degradations BEAT clean:

    jpeg_q90 0.9560   blur_s1.0 0.9456   blur_s0.5 0.9431   clean 0.9396
    ...
    messaging_app 0.8433   noise_s0.1 0.8176   (worst)

So a6's `jpeg_95` and `blur_0.3` views sit in a regime with real signal to
gain, not just variance to reduce. The same table falsifies the worry that
resampling views destroy the generator fingerprint: `resize_0.5` at 0.9311 is
the third-best degraded condition.

**Determinism.** a3 retrained under `--force-retrain` reproduced 0.9012 and
val_auc 0.9978353902287388 exactly. The seed controls training; paired
comparisons in the tables are sound.

## Corpus

The union corpus is complete and verified for the first time:

    manifest rows 375,358    train 331,257 | val_internal 37,101 | heldout 7,000
    labels        188,241 real / 187,117 fake  (50.1 / 49.9)
    coco_train2017 39,924 | ntire 149,999 | open_images 25,000
    sid_set 58,755 | wildfake 101,680        total missing: 0

ntire arrived by rsync (62 GB, 149,999 files) after the Kaggle archive failed
to build; `install_ntire.sh` moved it into place as a rename. Both union
manifests were shipped to the pod, which had only the 20k probe manifest --
full-scale Stage A would have failed at the first line without them.

## Incident: pod wedged

Full-scale Stage A (4 shards x 40 workers) died on an OpenBLAS thread storm --
each numpy import spins one BLAS thread per core, and `pthread_create` failed
before an image was read. That is what `7b7bc90` fixed inside
`run_pod_arms.sh`; a script that spawns its own workers must repeat the caps,
because they are environment, not code. The script exited but did NOT reap its
workers, and a relaunch on top of the orphans took the box to a pid limit where
sshd can accept a socket but not fork a session. TCP 42442 still answers; key
exchange resets.

Nothing is lost: banks are `--resume`-able, the failed shard dirs were cleared,
both smoke gates passed (so paths, the `AIGCDET_DATA_ROOT` rebase, the backbone
and the bank writer are confirmed at full-manifest scale), and every result
above is on disk.

Fixes before relaunch: export the thread caps; `WORKERS=16` per shard rather
than 32, since the manifest is 18x the probe's and each forked worker carries
it; and a guard that refuses to launch when an `extract_features` process is
already alive.

## Recorded, not fixed

- **Probe and full numbers are not comparable.** `dinov3l:band` scores a3
  0.8667 on the 20k probe but a1/a2 0.9037 on the 138k frozen corpus -- the
  ordering inverts. Pick the BACKBONE from the probe, re-pick the RUNG at full
  scale.
- **The selection metric rests on two generators**, both wildfake.
- **`a7_norecon` beats a3 on `crop dinov2regl`** (0.8022 vs 0.7858), the only
  arm where FiLM helps. a7 is not in `ELIGIBLE_RUNGS`, so this is a control
  result, but it is the one arm that would justify revisiting that.
