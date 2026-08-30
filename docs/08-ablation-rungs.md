# 08 — The ablation rungs, one at a time

**Goal.** Each rung is a self-contained experiment: it adds exactly one
ingredient to its parent and answers one question. This file is the handoff
for all of them — take a section, run it, report the number and what it means.

**The rule that makes any of this valid:** a rung differs from its parent in
**exactly one flag**. `tests/test_rung_ladder.py` fails the build if a pair
differs in more than one. If you find yourself wanting to change a second
thing to make a rung work, that is a new rung, not an edit to this one.

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
