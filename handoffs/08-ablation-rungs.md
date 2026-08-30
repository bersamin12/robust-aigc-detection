# 08 — The ablation rungs, one at a time

**Goal.** Each rung is a self-contained experiment: it adds exactly one
ingredient to its parent and answers one question. This file is the handoff
for all of them — take a section, run it, report the number and what it means.

**The rule that makes any of this valid:** a rung differs from its parent in
**exactly one flag**. `tests/test_rung_ladder.py` fails the build if a pair
differs in more than one. If you find yourself wanting to change a second
thing to make a rung work, that is a new rung, not an edit to this one.

---

## 0. Getting the data — read this first

You cannot run anything without a **feature bank**. A bank is the cached
output of a vision backbone over the whole corpus: every image, every
augmented view, one embedding each. Producing one takes hours of GPU. They are
already produced, so you should never have to.

### The bundle

Kaggle Dataset **`justinbersamin/techjam-aigc-banks`** (private — ask for it to
be shared with your Kaggle username). ~17 GB.

```
banks/dinov3l/          2.9 GB   training bank, the headline ladder ran here
banks/eval_dinov3l/     1.0 GB   the 20-condition evaluation grid
banks/siglip2l/         2.9 GB   A5 fusion partner
banks/eval_siglip2l/    1.0 GB
banks/convnextt/        6.4 GB   the CNN paradigm (dim 2304, not 1024)
banks/eval_convnextt/   2.3 GB
outputs/rungs/           23 MB   trained checkpoints, DINOv3
outputs/rungs_convnextt/ 42 MB   trained checkpoints, ConvNeXt
manifest.parquet         11 MB   REQUIRED
eval_manifest.parquet   3.0 MB   REQUIRED
```

**Minimum to be useful: `dinov3l` + `eval_dinov3l` + both manifests — 3.9 GB.**
That unblocks A0, A1, A2, A3 and A7-norecon. Add `siglip2l` for A5.

### What is inside a bank, and why it is not just a matrix

| file | what it is |
|---|---|
| `feats.npy` | `(n_images, n_views, dim)` float16 — the embeddings |
| `views.parquet` | which augmentation recipe produced each view |
| `meta.parquet` | row → `rel_path`, `label`, `generator`, `source`, `split` |
| `presence.npy` / `severity.npy` | degradation targets for the A2 head |
| `proxies.npy` | low-level confound measurements per row |
| `config.json` | backbone, dim, seed, and `manifest_sha256` |

### The three things that will bite you

**1. Banks are positional, and verified.** A bank does not store image paths
for its rows — it stores row *indices* into the manifest, and a
`manifest_sha256` fingerprint over the ordered `rel_path` column. Use a
different manifest and it refuses to load. **Do not regenerate the manifest.**
Ship the `manifest.parquet` in the bundle and use exactly that one. Every bank
in the bundle carries
`768eeff9713417128fa92fd6b0d2ed8634ebd10b1f923b391cecc4641ced2d00`; the eval
banks carry `0e28afc…` for the eval manifest.

**2. The training banks do not contain the held-out generators.** 131,116 rows
= 117,784 train + 13,332 val_internal. The 7,000 held-out rows live only in
the *eval* bank. That is deliberate — held-out data must be unreachable from
training — but it means row counts will not match the manifest's 138,116 and
that is correct, not corruption.

**3. Preallocated files look finished when they are not.** The writer
preallocates `feats.npy` at full size before writing anything, so file size
proves nothing about completeness. Everything in this bundle is verified
complete; if you ever extract your own, check the resume state, not `ls -la`.

### You do not need a GPU

Stage B is a ~2M parameter head over cached features. `--device cpu` works and
a rung takes minutes. The GPU cost was paid once, during extraction.

### Verify before you trust a number

```python
import json, numpy as np, pandas as pd
d = "banks/dinov3l"
c = json.load(open(f"{d}/config.json")); print(c)
f = np.load(f"{d}/feats.npy", mmap_mode="r")
print(f.shape, f.dtype)                       # (131116, 11, 1024) float16
print("nan:", bool(np.isnan(f[0]).any()))     # must be False
m = pd.read_parquet(f"{d}/meta.parquet")
print(m["split"].value_counts())              # train 117784 / val_internal 13332
```

**`nan` must be False.** A backbone run in the wrong precision produces a bank
of NaN that is the right shape and the right size and completely useless — it
has happened on this project, and row count was the only post-condition being
checked at the time.

---

## How to run any rung

Everything runs on a cached feature bank, so no GPU-heavy backbone work is
involved — minutes, not hours.

```bash
# one rung
python scripts/train_rung.py \
  --config configs/rungs/a3.yaml \
  --bank data/banks/dinov3l \
  --out outputs/rungs --device cuda

# the whole ladder, scored, with the table and heatmap
python -u scripts/run_ablation.py \
  --bank data/banks/dinov3l --eval-bank data/banks/eval_dinov3l \
  --rungs configs/rungs/a{0,1,2,3}.yaml configs/rungs/a7_norecon.yaml \
  --tier ablation \
  --out docs/robustness_table.md \
  --selection docs/selection.json \
  --heatmap docs/robustness_heatmap.png \
  --out-dir outputs/rungs --device cuda
```

**`run_ablation.py` skips any rung that already has a checkpoint.** That is
what makes it resumable, and it is also the trap: if you changed the model
code, pass `--force-retrain` or you will score the old model and not know.

**Always use a fresh `--out-dir` for a new backbone or a new fusion partner.**
Reusing one silently reuses `a5_partner`, which is trained against a specific
bank and a specific feature width.

### The banks on disk

| bank | rows | views | dim | notes |
|---|---|---|---|---|
| `data/banks/dinov3l` | 131,116 | 11 | 1024 | the headline ladder ran here |
| `data/banks/siglip2l` | 131,116 | 11 | 1024 | A5 partner |
| `data/banks/convnextt` | 131,116 | 11 | 2304 | CNN paradigm |
| `data/banks/eval_*` | 25,332 | 20 | — | the 20-condition grid |

None of them carry reconstruction features, which blocks A4 and A7 — see below.

### The selection rule, and why you cannot argue with it

`heldout_robust_tpr_at_1pct` — mean TPR at 1% FPR over the degraded
conditions, val_internal authentic vs heldout_generator generated. It was
fixed before any result existed, and only **A3, A4, A5, A6** are eligible.
A0–A2 and A7 are controls: they can score higher and still not win, and the
script warns loudly when they do. That has now happened twice (A1 on DINOv3,
A7 on ConvNeXt). **Report it as a finding; do not promote the rung.**

---

## A0 — linear probe, clean images only

**Question.** What does the standard published method (UniversalFakeDetect)
get on our data? Everything else is measured against this.

**Flags.** All off. **Parent.** None — it is the floor.

**Measured.** DINOv3 **0.8611** · ConvNeXt **0.4244**

**Status.** Done on both backbones. Re-run only if the bank changes.

---

## A1 — + augmented training views

**Question.** Is showing the model damaged images during training worth
anything?

**Flags.** `use_augmented: true`. **Parent.** A0.

**Measured.** DINOv3 **0.9037** (+4.3 pts) · ConvNeXt **0.4967** (+7.2 pts)

**Status.** Done. This is the single largest win on the ladder, on both
backbones. If you only ever do one thing, do this one.

---

## A2 — + auxiliary degradation loss

**Question.** Does asking the model to *name* the damage make it better at
detecting fakes?

**Flags.** `use_degradation: true`. **Parent.** A1.

**Measured.** DINOv3 **0.9037** · ConvNeXt **0.4967** — bit-identical to A1 on
both.

**Status.** Done, and the answer is a clean **no** — for a structural reason,
not a tuning one. `Detector` has two parameter-disjoint heads over a frozen
feature, so the degradation loss has no path to the classifier's weights. It
was verified by comparing every classifier tensor between the two runs: all
identical. A2 earns its place by feeding calibration, EQI and the dashboard —
not the score.

**Do not "fix" this by wiring the heads together.** That is A7, and it is a
separate rung for a reason.

### Does the head it trains actually work? — measured 2026-08-30

A2's entire justification is that it feeds calibration, EQI and the dashboard.
Nobody had checked whether the head can do that. Scored on `eval_dinov3l`
(25,332 images x 20 conditions = 506,640 views) exactly the way
`degradation_loss` trains it: presence as a per-family binary readout,
severity smooth-L1 **masked to families actually present**, because an absent
family's severity target is meaningless.

**Presence — it works on two families of six.**

| family | base rate | ROC AUC | AP |
|---|---|---|---|
| noise | 0.20 | **0.9726** | 0.9274 |
| jpeg | 0.45 | **0.9197** | 0.9113 |
| crop | 0.10 | 0.8175 | 0.3904 |
| blur | 0.15 | 0.7692 | 0.5231 |
| jitter | 0.10 | 0.6935 | 0.2286 |
| resize | 0.25 | **0.6020** | 0.3569 |

**Severity — read against the trivial "always predict the mean" baseline.**

| family | MAE | mean-baseline | |
|---|---|---|---|
| noise | 0.152 | 0.225 | beats it |
| blur | 0.176 | 0.278 | beats it |
| jpeg | 0.178 | 0.233 | beats it |
| resize | 0.225 | 0.160 | **worse than a constant** |
| jitter | 0.186 | 0.000 | off-scale by 0.19 |
| crop | 0.318 | 0.000 | off-scale by 0.32 |

`jitter_20` and `crop_80` are the only severities of their families in
`EVAL_GRID`, so those targets are constant and the grid cannot measure
severity *tracking* for them at all — only that the head's output sits at the
wrong value.

**Dose-response, mean predicted presence per condition.**

```
noise    clean 0.09 -> 0.73 -> 0.94 -> 0.99          excellent
jpeg     clean 0.17 -> 0.37 -> 0.72 -> 0.83 -> 0.83  good, saturates at q50
blur     clean 0.31 -> 0.35 -> 0.45 -> 0.77          only fires at s2.0
crop     clean 0.26 -> 0.62                          directional
resize   clean 0.38 -> 0.39 -> 0.47                  flat, invisible
jitter   clean 0.41 -> 0.47                          flat
```

Two things to carry forward. **It hallucinates damage on clean images** — 0.41
jitter, 0.38 resize, 0.31 blur where the truth is 0. And **the resize family
is partly unmeasurable by construction** — a property of the pipeline, not of
the head.

Standardisation runs BEFORE the condition (`eval/grid.py:153`), so it cannot
erase a resize that has not happened yet. What it does is cap the CONTENT:
both policies put `band_side`/`crop_side` (200) of real detail inside a
`nominal_side` (512) frame — 0.39 of Nyquist. `resize_0.5` cuts at 0.50,
ABOVE the cap, so it removes almost nothing. `resize_0.25` cuts at 0.25,
below it, so it removes real content. Measured over 80 `coco_crop` images by
`scripts/canon_bandwidth_check.py`, each degraded view against the SAME
canonicalised image:

| condition | cuts at | band PSNR | band AUC | crop PSNR | crop AUC |
|---|---|---|---|---|---|
| `resize_0.5` | 0.50 | 44.6 dB | 0.5958 | 45.1 dB | 0.5664 |
| `resize_0.25` | 0.25 | 33.6 dB | 0.8981 | 33.9 dB | 0.8280 |
| `blur_s2.0` | — | 32.0 dB | 0.9661 | 32.2 dB | 0.9277 |
| `noise_s0.05` | — | 26.4 dB | 1.0000 | 26.4 dB | 1.0000 |

(AUC is orientation-corrected separability of `laplacian_var`, clean vs
degraded, on the same images — roughly the best one pixel statistic could do,
and therefore the floor to read the head against.)

44.6 dB is visually lossless. **`resize_0.5` is very nearly the identity after
standardisation, under EITHER policy.** The head's dose-response says the same
— 0.38 clean, 0.39 at `resize_0.5`, 0.47 at `resize_0.25` — and its pooled
resize AUC of 0.6020 is within noise of what `laplacian_var` alone gets on
`resize_0.5` (0.5958). There is very little there to find.

**An earlier version of this section claimed crop would fix it. It does not:
crop is marginally WORSE on both resize conditions.** The cap is 200/512 under
either policy, so the policy is irrelevant here and only the RATIO matters.
That ratio is load-bearing — `CanonPolicy.__post_init__` requires
`band_side < nominal_side` so step 2 is always an upscale, which is what stops
native resolution being re-recorded as an interpolation signature. **The
pipeline destroys this evidence on purpose**, and the resize channel measures
the remainder. The ladder corroborates it: DINOv3 A0 scores 0.9926 under
`resize_0.5` against 0.9935 clean — a 0.0009 drop, because the condition
barely does anything — and 0.9876 under `resize_0.25`.

**Verdict: passes, unevenly.** Enough to justify building the EQI-on/EQI-off
comparison, because EQI is FITTED — a logistic regression over the head's
outputs (`calibrate/eqi.py`) — so it will learn to weight noise and jpeg and
discount resize and jitter, and the clean-image offset is per-family and
near-constant, which a fitted calibrator absorbs. Expect a modest abstention
gain concentrated on the JPEG and noise conditions and none on resize or
jitter. Not nothing, not transformative.

Reproduce, or run it against any other rung with `use_degradation: true`
(A3, A4, A7 — the head is trained identically in all of them):

```bash
python scripts/degradation_head_report.py \
    --bank data/banks/eval_dinov3l \
    --checkpoint outputs/rungs/a2/checkpoint.pt \
    --out docs/degradation_head_a2_dinov3l.json
```

CPU-only and about a minute; it reads the degradation branch alone, so it
needs no eval manifest, no labels and no classifier.

### The ladder cannot score this rung, and never could

Every metric `run_ablation.py` offers is rank- or threshold-invariant. The
AUCs are rank-based; `tpr_at_1pct` sets its threshold at a *quantile* of
authentic scores, so it is rank-based too; `acc_oracle` picks its threshold
post hoc; and `acc_fixed` uses one fixed threshold that a shrink-toward-0.5
can never cross. A confidence rescaling is invisible to all four.

Worse than invisible, in one case. A degradation-CONDITIONAL rescaling is not
a monotone transform of the score — it depends on per-sample evidence — so it
does reshuffle cross-sample ranks, and AUC would move. It would move DOWN,
because the head de-ranks confident predictions on degraded images and that is
exactly what AUC pays for. **A correctly working degradation head should make
the ladder look worse.** So "wire it into the ablation" is the wrong fix.

Measure it with `eval/metrics.py` instead, where all four already exist and
none are used: `expected_calibration_error` (stratified by condition — the
claim is per-condition, and a pooled number hides it), `brier`,
`risk_coverage` and `accuracy_at_coverage`. And make the comparison EQI-on vs
EQI-off at one fixed rung, not A1 vs A2 — same frozen classifier, one
variable. A ladder rung was never the right vehicle for a module that
deliberately does not touch the classifier.

---

## A3 — + clean/degraded consistency  ← current headline

**Question.** Does forcing the same image to score the same before and after
damage improve robustness?

**Flags.** `use_consistency: true`. **Parent.** A2.

**Measured.** DINOv3 **0.9012** · ConvNeXt **0.4882**

**Status.** Done, and it is the headline model despite A1 scoring 0.0025
higher on DINOv3 — a gap well inside one binomial standard error (0.0036 at
n=7,000), so they are tied, and A1 is not eligible.

**The real argument for A3 is not the catch rate.** It is calibration drift —
the accuracy you lose by fixing a threshold on clean images and applying it to
damaged ones:

| rung | drift |
|---|---|
| A0 | 1.24 pts |
| A1 / A2 | 0.50 pts |
| **A3** | **0.37 pts** |

A 3.4× reduction against the baseline, and completely invisible in AUC. That
is why A3 ships.

---

## A4 — + reconstruction features  🔒 BLOCKED

**Question.** Does a diffusion-reconstruction error signal add anything on top
of A3?

**Flags.** `use_recon: true`. **Parent.** A3.

**Status.** **Never run.** No bank on disk carries reconstruction features.

**To unblock:** run `attach_recon_to_bank` over an existing bank. It needs the
SD 1.5 VAE and LPIPS, which is why it belongs on the local A4500 and not on a
Kaggle session. Budget ~8 h. It *attaches* to a bank rather than rebuilding
one, so it does not invalidate anything already extracted.

**Kill criterion.** If recon does not beat A3, A4 does not ship — and A7 goes
with it, because A7 stacks FiLM on A4 and a poor result could not tell you
which half was to blame. That is why `a7_norecon` exists as a separate rung.

---

## A5 — paradigm-diverse ensemble

**Question.** Do two different vision backbones, fused, beat either alone?

**Flags.** Not a YAML — it is `--fuse-bank` + `--fuse-eval-bank` on top of A3.
Trains an A3 head on the partner bank and fuses the two scores.

**Measured.**

| base | partner | result | base alone |
|---|---|---|---|
| DINOv3 | SigLIP2 | 0.8773 | 0.9012 |
| ConvNeXt | DINOv3 | **0.8860** | 0.4882 |

**Status.** Done, and it is a **negative result for the headline** — fusion has
not beaten a strong base yet. Cause: equal-weight z-score fusion. The SigLIP2
partner reaches val_auc 0.9465 mean-views against A3's 0.9963, so averaging
dilutes the strong signal.

But look at the second row. Fusing a *weak* CNN base with a strong ViT lifts it
from 0.4882 to 0.8860 — fusion transfers almost all of the strong model's
ability. And transformer+CNN (0.8860) beat transformer+transformer (0.8773),
which is the paradigm-diversity hypothesis showing a pulse.

**Open work, and the most promising rung left.** Replace equal weighting with
validation-derived weights, then re-run DINOv3+ConvNeXt. If a weighted fusion
clears 0.9012 it becomes the headline. **Requirement:** derive the weights on
`val_internal` only. Fitting them on held-out rows is leakage and voids the
number.

**Fusion parents must agree** on `n_views`, `conditions` and
`manifest_sha256`. All three eval banks currently do.

---

## A6 — test-time augmentation  🔒 NOT SCORED

**Question.** Does averaging predictions over several augmented views of the
same test image help?

**Status.** Inference-only. It cannot be scored from the cached eval bank,
because the bank stores one fixed set of views per image. `--tta` records its
cost and the tier it applies to, nothing more.

**To unblock:** score it through the live inference path, not the bank. It is
eligible under §6.4, so a real A6 number could take the headline — but it
multiplies inference cost by the view count, and that trade must be reported
alongside any win.

---

## A7-norecon — + FiLM conditioning  ← recently fixed

**Question.** Does letting the degradation estimate rescale the classifier's
hidden state help the *shipping* system?

**Flags.** `use_film: true`. **Parent.** A3.

**Measured.** ConvNeXt **0.5589** (was 0.0296 before the fix) — the strongest
rung on that backbone, at val_auc 0.9900 (was 0.5601).

**Status.** Fixed 2026-08-30 after two attempts, and this history matters if
you touch it:

1. The FiLM projection was randomly initialised, so it perturbed a
   LayerNorm-ed hidden state before any training step. Zeroed it. **Not
   enough** — it still diverged.
2. FiLM's output was never renormalised. `(1 + gamma) * h + beta` is
   unbounded, and the consistency loss is an MSE over exactly that `h`, so
   inflating gamma reduced the loss more cheaply than actually being
   consistent. A LayerNorm after the modulation fixed it.

**Old A7 checkpoints will not load** (`film_norm` is a new parameter) and must
be retrained. A0–A4 are unaffected — only A7 sets `use_film`.

**Outstanding:** re-measure on DINOv3, where A7 was previously unusable. It is
in the chained fusion run.

**Until then, `docs/robustness_table.md`'s DINOv3 A7 row is STALE — do not
cite it.** It reads clean 0.5485 / selection 0.0296, which is the PRE-fix
collapse, and it is the only rung in that table whose checkpoint predates the
fix. Verified 2026-08-30: `outputs/rungs/a7_norecon/checkpoint.pt` contains no
`film_norm` tensor, while `outputs/rungs_convnextt/a7_norecon/checkpoint.pt`
does. Anyone reading that row cold will conclude FiLM destroys the model on
DINOv3; what it actually records is the bug above, on the one backbone that
has not been re-run since.

---

## A7 — FiLM on top of recon  🔒 BLOCKED

Same blocker as A4. Do not attempt until recon features exist and A4 has
passed its kill criterion.

---

## Off-ladder probes — P1–P4

**These are not rungs and cannot take the headline.** A rung differs from its
parent in exactly one FLAG, and `tests/test_rung_ladder.py` enforces it. P1
differs from A3 in its training ROWS; P3 and P4 train nothing at all. §6.4
chooses among A3–A6 and none of these are that. They belong in the table as
probes, beside A1's ineligible win, and a good number here promotes nothing.

Every comparison below is a **paired bootstrap over images** — both scores read
on the same resample — because two heads score the same images and the question
is about the difference. Overlapping marginal CIs are not a test, and on P1 they
gave the wrong answer: the fused head's marginal interval overlaps A3's while
the paired interval on the gap excludes zero. `family_experts.bootstrap_panel`
is the shared implementation; use it rather than `metrics.bootstrap_ci`, which
resamples rows and would treat one image's 19 conditions as independent.

---

## P1 — two generator-family experts, fused

**Question.** Does training one head per generator family and combining them
beat pooling the families into one head?

**Files.** `configs/rungs/a3_gan.yaml`, `configs/rungs/a3_diff.yaml`,
`scripts/family_experts.py`, `docs/family_experts.json`,
`outputs/rungs_family/`.

**Setup.** Two A3 heads over the SAME frozen `dinov3l` bank, differing only in
`train_exclude_generators`: GAN (BigGAN, DF-GAN, GALIP, GigaGAN, starGAN,
styleGAN) against diffusion (adm, ddim, ddpm, imagen, vqdm, SDwithAdaptor_lora,
SDwithAdaptor_lycris). MAE/MAGE/VQVAE/sid_set are excluded from both — neither
family, and putting them in one arm would make the pair differ by more than the
split. Reals stay in both arms (they carry `generator == ""`, so no exclusion
list matches them); halving the negatives per expert would confound "one family"
with "half the data".

**Measured** (`heldout_robust_tpr_at_1pct`, paired against pooled A3):

| head | metric | 95% CI | vs A3 | paired CI on the gap | P(>A3) |
|---|---|---|---|---|---|
| pooled a3 | 0.9012 | [0.8834, 0.9125] | — | — | — |
| a3_gan | 0.6689 | [0.6546, 0.6808] | −0.2323 | [−0.2470, −0.2149] | 0.000 |
| a3_diff | 0.8730 | [0.8589, 0.8871] | −0.0282 | [−0.0418, −0.0101] | 0.001 |
| fused, w_gan=0.40 | 0.8843 | [0.8704, 0.8966] | −0.0169 | [−0.0305, −0.0012] | 0.017 |

**Status.** Done, and a clean negative. Fusion beats either specialist and still
loses to pooling. The weight was swept on `val_internal` alone (unimodal peak at
0.40, so it is well determined, not a flat-curve artefact) and the held-out
number read once.

**The finding that matters is per generator**, TPR@1%FPR against val_internal
authentic:

| head | SDwithAdaptor_controlnet | VQGAN |
|---|---|---|
| a3 | **0.8056** | 0.9969 |
| a3_gan | 0.3397 | 0.9981 |
| a3_diff | 0.7545 | 0.9916 |

Pooled A3 beats the diffusion specialist **on the diffusion-lineage held-out
generator** — 0.8056 against 0.7545 — even though A3 also trained on the GAN
families the specialist was denied. Adding GAN data helps detect a diffusion
generator. That is the hypothesis inverted, and it is the reason not to spend
GPU on a per-family fine-tune.

**Two traps if you re-run this.** `epochs` is 93 and 80, not 30, and that is an
optimisation-budget match rather than a hyperparameter change: PairedSampler
yields `min(len(pos), len(neg)) // (n_src // 2)` batches per epoch, so shrinking
the positive pool shrinks the epoch (pooled 1825 steps × 30 = 54,750; GAN 590 ×
93 = 54,870; diffusion 687 × 80 = 54,960). Left at 30 each expert takes a third
of the gradient steps of the model it is compared against, and a weak score
would mean "undertrained". Second, the 3:1 pool imbalance is NOT a confound —
PairedSampler is class- and generator-balanced by construction, so each
expert's 6–7 families are each drawn 1/6 of the time against 1/17 pooled.

---

## P2 — expert disagreement and ensemble entropy

**Question.** If the two experts do not fuse well, does their DISAGREEMENT carry
a signal — either "this is fake" or "this is from a family I have never seen"?

**Measured.** Three signals, three tasks. T1 is the §6.4 population. T2 is
fakes only: heldout_generator against val_internal, so nothing can be won by
detecting generation. T3 uses pooled A3 as the detector and asks what tells it
when to abstain (258,875 rows, base accuracy 0.9575, threshold −1.553).

| signal | T1 real-vs-fake | T2 novelty AUC | T3 AURC | T3 acc@80% cov |
|---|---|---|---|---|
| \|z_gan − z_diff\| | 0.0247 | 0.3018 | 0.0355 | 0.9596 |
| predictive entropy H(p̄) | 0.1605 | 0.3627 | — | — |
| mutual information (BALD) | 0.1565 | 0.3537 | 0.0153 | 0.9709 |
| a3's own \|logit − thr\| | — | — | **0.0046** | **0.9943** |

**Status.** Dead on all three, and the reasons differ.

T1 was always the wrong question — uncertainty is not a class signal.

**T2 is below 0.5, i.e. inverted, and the cause is the partition, not the
hypothesis.** Held-out families produce LOWER disagreement than seen ones,
because a `val_internal` fake is out-of-family for exactly one expert about half
the time, while `VQGAN` is easy for both (0.9981 / 0.9916) so they agree
confidently. The negative class of T2 is itself half-OOD per expert **by
construction**. A two-expert family split therefore cannot answer the novelty
question at all — do not read 0.3018 as "disagreement does not detect novelty".

T3 is the honest practical refutation: the model's own margin is 8× better by
AURC, and disagreement at 80% coverage (0.9596) is barely above the 0.9575 base
rate. Two members is too few to estimate epistemic uncertainty.

**If you touch the entropy code:** compute in float64. In float32 `1 - 1e-9` IS
`1.0`, so the usual clip is a no-op, a saturated logit gives `1 - p == 0`, and
`0 * log(0)` is NaN. The GAN head's logits reach +65 and 2,581 rows saturate.

---

## P3 — Mahalanobis distance on the frozen features (nothing trained)

**Question.** Ask P2's novelty question without any experts: how far does an
image sit from where the TRAINING features cluster?

**Files.** `scripts/mahalanobis_probe.py`, `docs/mahalanobis_probe.json`.

**Setup.** Gaussians fitted on `split == "train"` clean-view rows only, shared
within-group covariance, ridge 1e-3. Two groupings (by class, by generator
family) and two forms: `MD` = distance to the nearest training Gaussian; `RMD` =
that minus the distance to a single background Gaussian over all training rows
(Ren et al. 2021). Fitting on `val_internal` would make the novelty negatives
close by construction, so it is refused.

**Measured — novelty AUC**, heldout-family fakes against val_internal fakes:

| score | clean | degraded mean |
|---|---|---|
| md_class | 0.4422 | 0.4574 |
| md_family | 0.4659 | 0.4764 |
| rmd_class | 0.4120 | 0.4245 |
| **rmd_family** | **0.8513** | **0.8147** |
| *(P2's expert disagreement)* | — | *0.3018* |

**Measured — §6.4 metric** via distance to the REAL Gaussian: `md_real` 0.0004,
`rmd_real` **0.9005**. Against trained A3's 0.9012 the paired interval is
[−0.0140, +0.0167], P(closed form > A3) = 0.572 — **a tie**.

**Status.** The strongest result in this group, and two things follow.

*The background subtraction is the entire signal.* Plain `md_family` is 0.4764,
i.e. chance; raw distance is dominated by how unusual the PICTURE is. The
per-family grouping is also load-bearing: `rmd_class` (0.4245) is worse than
chance because a fake-class Gaussian pooled over 13 families is too wide to be
far from anything.

*A closed-form discriminant ties the trained headline model.* No gradient
descent, no epochs, no hyperparameters beyond a ridge, minutes of linear algebra.

**Do not call `rmd_real` unsupervised.** It uses the real/fake labels to decide
which rows form the real Gaussian. What is absent is the training loop, not the
labels.

**Confound check, and it comes back clean.** The banks carry `proxies.npy`
(`jpeg_quality`, `laplacian_var`, `noise_floor`). Each proxy scored alone:
0.47–0.56 on novelty, 0.38–0.44 on real-vs-fake — nowhere near 0.8147 or 0.9005.
So this is not sharpness or JPEG history. Three scalars is not a proof of no
confound, but it is the instrument this project has.

**The shippable piece** is `rmd_family` as a lineage-novelty flag: "this image
is from a generator family the detector has never seen." It rides alongside A3
rather than competing with it, costs no extra model, and is the closest thing
here to a novel contribution.

---

## P4 — does the trained head earn its keep? (the readout ceiling)

**Question.** P3's tie admits two opposite readings: (a) these features are so
good that any readout reads them equally well, or (b) they have a CEILING near
0.90 that no readout can pass. Under (a) effort belongs anywhere but the
backbone; under (b) the feature space is the only remaining lever and
fine-tuning is justified. One point cannot separate them.

**Files.** `scripts/readout_ceiling.py`, `docs/readout_ceiling.json`.

**Setup.** The same closed-form-vs-trained comparison at three levels of frozen
feature quality, using the A3 checkpoints already on disk — nothing is
retrained, so this cannot move the ladder's numbers.

**Measured.**

| backbone | dim | trained a3 | closed form | gap | paired 95% CI | verdict |
|---|---|---|---|---|---|---|
| dinov3l | 1024 | 0.9012 | 0.9005 | −0.0007 | [−0.0140, +0.0167] | tie |
| convnextt | 2304 | 0.4882 | 0.3733 | −0.1148 | [−0.1509, −0.0759] | trained wins |
| siglip2l | 1024 | 0.2893 | 0.2178 | −0.0714 | [−0.0940, −0.0469] | trained wins |

The SigLIP2 row reproduces `logs/ablation_siglip2l.log` exactly (0.2893), so the
script agrees with the existing ladder rather than inventing a number.

**Status.** Done. **The gap opens as the backbone weakens**, which kills reading
(b) in its strong form: if the feature space alone decided performance the gap
would be ~0 everywhere, and it is not. Training does real work — on weak
features it is worth 7–11 points with intervals well clear of zero.

**What the DINOv3 tie actually licenses is narrower than it looks.** It says
that on THOSE features the trained head has already extracted everything a
closed-form estimator can — DINOv3 makes this problem linearly easy. It does
NOT establish that 0.90 is a ceiling, and it does NOT show that fine-tuning
would be wasted. Both of those were claimed in conversation before P4 ran, and
P4 is why they should not be repeated: the evidence in hand is about the
READOUT, and fine-tuning changes the FEATURES.

**Note also how much the backbone decides.** Same head, same data, same recipe,
0.9012 against 0.2893 — a 61-point spread from the frozen feature space alone.
Whatever else is true, backbone choice dominates every other lever measured on
this project.

---

## Open work from these probes

**A0 may be understated, and that would inflate every gain above it.** A0 is a
TRAINED linear probe at 0.8611; P3's closed-form Gaussian scores 0.9005, both
fitted on clean view 0 of the same bank. A closed-form estimator beating a
trained probe by 3.9 points suggests A0 is undertrained or over-regularised.
Every rung on the ladder is reported as a gain over A0, so this is worth
settling before the ladder is written up. Cheap: re-run A0 with a longer
schedule or a weaker penalty and see whether it moves.

**Is 0.90 a DINOv3 ceiling?** Untested. Every readout tried so far is linear or
closed-form. A strong nonlinear readout on the frozen bank — deeper MLP, k-NN —
is minutes on cached features and would say whether the features hold more than
the current head extracts. Run this BEFORE any fine-tune: if a nonlinear readout
also lands at ~0.90, the features are the ceiling and fine-tuning becomes the
justified next step; if it clears 0.90, the head is the problem and no backbone
work is needed.

**DINOv2 as a frozen closed-form point.** `dinov2l` is registered
(`facebook/dinov2-large`), weights are cached, and float16 is verified safe for
it (it has no DINOv3 layer-1 overflow — see `backbones.py`). Blocked only on
extraction, and the closed form needs neither a trained head nor the 11-view
augmented bank: clean features for the train rows (117,784 forwards) plus the
eval grid over `val_internal` + `heldout_generator` (20,332 × 20 = 406,640
forwards).

**Run it at 336px, not the registry default.** DINOv2-L is patch14, so 336px
gives 24 × 24 = 576 tokens — exactly DINOv3-L/16's token count at 384px. That
makes the comparison token-matched rather than confounding lineage with
resolution, AND it is the cheap option: `backbones.py` measured 140.4 img/s at
336 against 54.0 at 518, so ~25 min for the train features and ~87 min for the
eval grid, **~1.9h total** against ~5h at 518. The tension to know about is that
`backbones.py` advises dropping to 448 rather than 336 when 518 does not fit,
because DINOv2 was adapted at 518 — so 336 answers "is it the lineage or the
resolution", and 518 answers "which backbone do we ship". Run 336 first.

It tests whether DINOv3's advantage is a DINO-family property or specific to
v3. **It would NOT add a fourth point to P4's trend** — that needs a trained
head, hence the full 11-view bank (~8–9h more).

**Why DINOv3 wins is narrower than "it is a better model."** DINOv3-L/16 and
SigLIP2-L/16 are matched at 384px, patch 16 (576 tokens each), ~300M params,
1024 dims and mean-pooled patch tokens. The only substantive difference is the
pretraining objective — self-supervised image-only against language-supervised
image-text — and the gap is 61 points. SigLIP2 reaches 0.9648 clean val AUC and
0.2893 held-out, i.e. it fits the seen generators and carries no transferable
low-level signature, which is what language alignment would predict: a caption
never mentions resampling artifacts or noise floor, so aligning to captions
discards them. ConvNeXt's 0.4882 is NOT evidence for this argument — at 27.8M
params it confounds capacity with objective.

**Other Apache-2.0, ungated candidates** (licences read from the Hub
2026-08-30, same discipline as `docs/model_licences.md`): `facebook/vit-mae-large`
and `-huge` — a pixel-reconstruction objective, so the most promising untried
one on the argument above; `facebook/convnextv2-large-1k-224`, which would give
A5 a genuinely self-supervised conv paradigm instead of the 27.8M supervised
ConvNeXt-Tiny; `microsoft/beit-large-patch16-224-pt22k`. `timm/eva02_*` is MIT
but distilled from CLIP, so it may inherit the same invariance. Ruled out as
non-commercial: `facebook/hiera-*` and `facebook/ijepa_*` (cc-by-nc-4.0), and
`apple/aimv2-*` (apple-amlr).

**Note on `mahalanobis_probe.py`:** it takes `FeatureBank` objects, so a
single-view DINOv2 feature matrix needs either a bank-shaped wrapper or a small
refactor to accept a plain array.

---

## Reporting

Whatever you run, report:

1. `heldout_robust_tpr_at_1pct` **with its bootstrap CI**.

   **Read a difference off a PAIRED bootstrap, not off two marginal CIs.**
   "The intervals overlap, so it is a tie" is not a test, and it gave the wrong
   answer on P1: the fused head's marginal interval overlaps A3's while the
   paired interval on the gap, [−0.0305, −0.0012], excludes zero. Two heads
   score the same images, so resample the images ONCE and read both heads on
   it — `family_experts.bootstrap_panel` does this and reports the gap with its
   own interval. A tie is a gap whose interval contains zero.
2. `robust_acc_fixed` **and** `robust_acc_oracle`. The difference is
   calibration drift, and it is where A3's actual value showed up.
3. Clean AUC — but never as the headline. It is saturated: every working rung
   on DINOv3 sits in 0.985–0.992, a 0.7pt spread against 4.3pt on the
   selection metric. Selecting on AUC would have called A0 equal to A3, and on
   ConvNeXt it would have called two backbones equivalent that differ by 41
   points at the thing we care about.
