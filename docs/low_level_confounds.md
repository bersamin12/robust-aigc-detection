# Low-level confounds, and why balancing is the only lever left

Companion to `resolution_shortcut.md`, which documents one confound in
detail. This file states the general problem, sorts the confounds by what
can be done about each, and records the lever we have not yet pulled.

**Summary: these datasets leak the label through low-level image statistics
rather than through content. Some of that leak is an artefact we can destroy;
the rest is baked into pixels, is partly genuine evidence, and can only be
measured, reported, and controlled by balancing.**

## The measurements

| Signal | Separates the classes at | Where |
| --- | --- | --- |
| Resolution (short side) | **72.6%** vs 52.9% majority baseline | our training pool |
| Resolution (short side) | **100%**, in the OPPOSITE direction | organisers' demo benchmark |
| Sharpness (`laplacian_var`) | **68-74%** | our pool, at matched native resolution |

The resolution figure inverts between our pool and the benchmark, which is
what makes it dangerous rather than merely embarrassing: a model that learns
it scores 1.0000 on one and near 0.0000 on the other. See
`resolution_shortcut.md` for the full table and the reason (generators emit at
their native sizes; WildFake's authentic images were stored at 200 or 512).

The sharpness figure is the one that matters here, because it survives the fix
for resolution. Controlling for resolution does not control for sharpness.

## Two kinds of confound

**Killable by normalisation.** Container-level properties that carry no
evidence about how an image was made, only about how it was stored:

- file format
- pixel dimensions
- aspect ratio
- colour mode

These are destroyed at decode time. Nothing downstream can see them.

**Baked into the pixels.** Sharpness, noise floor, and compression history
cannot be normalised away, for a reason worth stating plainly: they are
*partly genuine evidence*. A diffusion model really does produce a different
noise floor than a camera sensor. Stripping that out would remove signal we
are entitled to use along with the bias we are not, and there is no
principled place to draw the line between them.

So they cannot be removed. They can only be:

1. **measured** - which is what the proxies are for;
2. **reported** - a single-feature baseline on each, alongside the headline
   number, the same way `resolution_shortcut.md` is reported;
3. **controlled by balancing** - equalise the class composition along the
   confound so it stops predicting the label, and the model has nothing to
   cheat with.

Resolution is instrumented as a permanent control: a dimensions-only
classifier is registered as a CONTROL, never a baseline, and every headline
number is read against it.

## The lever: balance the composition

Balancing is the right instrument for the second kind, and it is cheaper than
it looks for two reasons.

**The inputs already exist.** `aigcdet.features.proxies.PROXY_NAMES` is
`("jpeg_quality", "laplacian_var", "noise_floor")` - exactly the three
confounds in the second category - and every bank caches them per view.
View 0 is the undegraded view by the bank's own invariant, so
`bank.proxies[:, 0, :]` is each image's NATIVE low-level signature, already on
disk. Auditing how well each one alone predicts the label needs no forward
pass and no new extraction.

**The mechanism already exists, half-built.** `PairedSampler` is already
class-balanced and generator-balanced, and it takes its pool as an `indices`
array. Balancing on a low-level statistic is therefore a FILTER ON ROW
INDICES handed to the trainer - not a data rebuild, not a re-extraction, not
a change to the frozen manifest. Stratify the pool into bins of
`laplacian_var`, drop rows until each bin holds both classes in equal
proportion, and pass the surviving indices as `indices`.

That is why this belongs on the training side of the pipeline even though it
reads like a dataset problem. The build's job is to make the pool large and
diverse enough that balancing has rows to spend; `--max-per-generator`
serves that, since a family capped too hard leaves the balancer nothing to
discard.

## What is not yet done

- ~~The single-feature audit for `laplacian_var`, `noise_floor` and
  `jpeg_quality`, computed from cached proxies.~~ Done; see below.
- The balanced-index filter itself, and the A/B against the unbalanced pool.

Both are free of GPU cost. The audit has now been run (2026-08-30); the
balanced-index filter has not.

---

## Measured, 2026-08-29 and re-measured 2026-08-30, from cached proxies

Computed from a bank's `proxies[:, 0, :]` (the undegraded view) at zero GPU
cost. AUC is orientation-corrected (`max(a, 1-a)`). The middle column is the
counterfactual pool that barring WildFake's authentic bucket would have
produced; the right column is the pool as frozen, re-measured from
`data/banks/siglip2l` after the bar was lifted.

| Signal | post-bar pool (76,116) | **live pool (131,116)** |
| --- | --- | --- |
| `jpeg_quality` | 0.5536 | 0.5532 |
| `laplacian_var` | **0.6871** | **0.6721** |
| `noise_floor` | 0.6582 | 0.6374 |
| short side (manifest) | **0.8692** | **0.5992** |

**Barring WildFake's authentic bucket made the resolution confound worse, and
restoring it fixed that.** Short-side AUC nearly doubled under the bar, 0.599
-> 0.869, because the authentic side collapsed onto a single value: 9,954 of
10,049 authentic images sat at short side 512, and every bucket below 512 was
~100% generated. Roughly 49,000 images, 64% of the pool, fell in a bucket where
"short side < 512" implied "generated" outright.

With the bucket restored, the two dominant buckets carry both classes —
200 -> 40,000 authentic / 11,516 generated, 512 -> 24,954 / 17,069 — and only
**37,482 rows (28.6%)** sit in a single-class bucket, down from 64%. The
authentic side is 55,000 WildFake plus 10,049 SID_Set, so it spans the same
resolution range as the generated side rather than one point of it.

Sampling the SID_Set acquisition confirms more SID would NOT have fixed this on
its own: its `fake` bucket is 1024x1024 for **every** image sampled, its `real`
bucket runs 390-1200 with median 685, 99.1% at or above 512. Both classes
normalise onto 512 and land in the same bucket. Adding authentic SID images
moves the class RATIO inside the 512 bucket; it does not populate the buckets
below it. Two authentic sources, not more of one, is what made the difference.

`laplacian_var` at 0.672 is now the largest remaining confound, and it is the
one that survives canonicalisation — so it, not resolution, is what balancing
has to target.

**What the model actually sees is smaller than that.** `augment.canonical`
already removes the last-step resampling component at zero information cost by
handing the backbone identical pixel dimensions every time. The short-side
figure above is therefore what the dimensions-only CONTROL scores, not what
reaches the classifier. What survives canonicalisation is the intrinsic
band-limit -- a 200px image has a lower Nyquist ceiling than a 512px one, and
no resampling restores it -- and that residue is what `laplacian_var` measures
at 0.672. `canonical.py` already names the remedy: "Addressing it needs
source-balanced sampling."

### Balancing survival

Stratify the pool, then discard the majority class within each stratum until
the classes are equal. Both pools shown, because the difference is the whole
argument for keeping two authentic sources:

| Stratified on | post-bar (of 76,116) | **live (of 131,116)** |
| --- | --- | --- |
| short side (exact) | 19,940 — 26.2% | 57,202 — 43.6% |
| `laplacian_var` (10 quantile bins) | 20,098 — 26.4% | **102,602 — 78.3%** |
| short side x `laplacian_var` | 19,470 — 25.6% | 49,328 — 37.6% |

**Balancing went from expensive to nearly free.** On the post-bar pool the
authentic class bound in every variant — balancing kept essentially ALL
authentic rows and discarded ~56,000 generated ones, costing 74% of the pool.
On the live pool, stratifying on `laplacian_var` keeps **78.3%**: 102,602 rows,
51,301 per class. The constraint was never the generators; it was holding only
10,049 authentic images from one source.

**Generator diversity survives, comfortably.** Balancing on `laplacian_var`
keeps all 17 generated families, every one of them above the 200-row
`MIN_HELDOUT_IMAGES` threshold (post-bar, BigGAN at 170 and starGAN at 161 fell
short). Per-family retention runs 37.6% to 97.6%, against 4.6% to 24.1%
post-bar. The authentic side keeps 43,058 WildFake and 8,243 SID_Set rows, so
balancing does not quietly collapse back onto a single source either.

This makes the balanced-index filter the cheap intervention it was meant to be:
a filter on row indices handed to `PairedSampler`, costing 22% of the pool
rather than 74%, with `laplacian_var` at 0.672 as the signal it removes.

### Consequence for `--max-per-generator`

The cap is close to a no-op for the confound, for a reason worth recording:
WildFake's generated families are ALREADY uniform at ~3,500 rows each, so no
cap at or above 3,500 touches them. The only family a cap binds on is
`sid_set` at 10,079 and rising. A cap of 4,000 therefore leaves generator
diversity completely intact and stops the one pseudo-generator family that
would otherwise dominate -- which is what the README already specifies, now
with evidence behind it rather than intuition.

---

## Update, 2026-08-30: source balancing is not the lever; index balancing still is

`augment/canonical.py` ends its list of residues with "Addressing it needs
source-balanced sampling", and this file's §"The lever" reads as though
balancing on the *source* and balancing on the *index* were the same move.
They are not, and a sweep now separates them. Full method and table in
`docs/dataset_presets.md`.

Splitting the AUCs by source shows the two authentic sources leak through
**different channels**:

| Signal | pooled | wildfake only | sid_set only |
| --- | --- | --- | --- |
| `jpeg_quality` | 0.5532 | 0.5414 | **0.6212** |
| `laplacian_var` | 0.6721 | **0.6944** | 0.5548 |
| `noise_floor` | 0.6374 | 0.6214 | **0.7314** |
| short side | 0.5992 | 0.6525 | 0.5047 |

Every pooled figure sits below the worse of the two singles: each source
dilutes the other. So rebalancing the two against each other only chooses
which channel dominates. Across 36 compositions, capping WildFake drops
`laplacian_var` from 0.641 to 0.604 and raises `noise_floor` from 0.660 to
0.696 — **no mix lowers both**, and the worst-case confound is minimised by
capping nothing at all.

Two revisions follow.

1. **`--max-per-generator` and any source cap should not be justified by the
   confound.** They buy generator-era coverage (SID_Set is the only modern-era
   generated data we hold) and pay confound for it. `configs/datasets/era_forward.yaml`
   makes that trade explicitly; `configs/datasets/max_data.yaml` declines it.
2. **The balanced-index filter is unaffected and still unbuilt.** Stratifying
   `laplacian_var` and discarding within strata breaks a statistic's link to
   the label; it does not shift which source dominates. It remains free of GPU
   cost — it is a filter on the `indices` array `PairedSampler` already takes,
   over a bank that already exists.

One row-level intervention *is* free and is now shipped: 1,308 images sit
below `CANON_BAND_SIDE` (200), and **all 1,308 are generated** — 1,260 BigGAN
at 128px. Canonicalisation cannot raise them to the common ceiling, and index
balancing cannot pair them off, because the authentic class has no support
down there at all. Both presets drop them via `min_short_side: 200`.
