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

## Reporting

Whatever you run, report:

1. `heldout_robust_tpr_at_1pct` **with its bootstrap CI**. A gap smaller than
   the interval is a tie, and should be written as one.
2. `robust_acc_fixed` **and** `robust_acc_oracle`. The difference is
   calibration drift, and it is where A3's actual value showed up.
3. Clean AUC — but never as the headline. It is saturated: every working rung
   on DINOv3 sits in 0.985–0.992, a 0.7pt spread against 4.3pt on the
   selection metric. Selecting on AUC would have called A0 equal to A3, and on
   ConvNeXt it would have called two backbones equivalent that differ by 41
   points at the thing we care about.
