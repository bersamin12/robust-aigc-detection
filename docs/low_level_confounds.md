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

- The single-feature audit for `laplacian_var`, `noise_floor` and
  `jpeg_quality`, computed from cached proxies, reported as controls.
- The balanced-index filter itself, and the A/B against the unbalanced pool.

Both are free of GPU cost. Neither has been run.

---

## Measured, 2026-08-29, from cached proxies

Computed from `data/banks/dinov3l` (`proxies[:, 0, :]`, the undegraded view),
at zero GPU cost. AUC is orientation-corrected (`max(a, 1-a)`).

| Signal | stale pool (131,116) | post-bar pool (76,116) |
| --- | --- | --- |
| `jpeg_quality` | 0.5532 | 0.5536 |
| `laplacian_var` | 0.6721 | **0.6871** |
| `noise_floor` | 0.6374 | 0.6582 |
| short side (manifest) | 0.5992 | **0.8692** |

**Barring WildFake's authentic bucket makes the resolution confound worse, not
better.** Short-side AUC nearly doubles, 0.599 -> 0.869, because the authentic
side collapses onto a single value: post-bar, 9,954 of 10,049 authentic images
sit at short side 512, and every bucket below 512 is ~100% generated
(128 -> 1,262 gen / 0 real; 200 -> 11,516 / 0; 224 -> 7,002 / 0;
256 -> 22,228 / 0; 450 -> 6,158 / 0). Roughly 49,000 images, 64% of the pool,
fall in a bucket where "short side < 512" implies "generated" outright.

Sampling the incoming SID_Set acquisition confirms this does not self-heal:
its `fake` bucket is 1024x1024 for **every** image sampled, its `real` bucket
runs 390-1200 with median 685, 99.1% at or above 512. Both classes therefore
normalise onto 512 and land in the same bucket. Adding authentic images moves
the class RATIO inside the 512 bucket; it does not populate the buckets below
it, so the between-source resolution split survives however much SID_Set
arrives.

**What the model actually sees is smaller than that.** `augment.canonical`
already removes the last-step resampling component at zero information cost by
handing the backbone identical pixel dimensions every time. The short-side
figure above is therefore what the dimensions-only CONTROL scores, not what
reaches the classifier. What survives canonicalisation is the intrinsic
band-limit -- a 200px image has a lower Nyquist ceiling than a 512px one, and
no resampling restores it -- and that residue is what `laplacian_var` measures
at 0.687. `canonical.py` already names the remedy: "Addressing it needs
source-balanced sampling."

### Balancing survival

Stratify the post-bar pool, then discard the majority class within each
stratum until the classes are equal:

| Stratified on | rows kept | of 76,116 | authentic side |
| --- | --- | --- | --- |
| short side (exact) | 19,940 | 26.2% | 9,970 |
| `laplacian_var` (10 quantile bins) | 20,098 | 26.4% | 10,049 |
| short side x `laplacian_var` | 19,470 | 25.6% | 9,735 |

The authentic class binds in every variant: balancing keeps essentially ALL
authentic rows and discards ~56,000 generated ones. Pool size is therefore
governed by how many authentic images we hold, not by any cap on generators.

**Generator diversity survives.** Balancing on `laplacian_var` keeps all 17
generated families, 15 of them above the 200-row `MIN_HELDOUT_IMAGES`
threshold (only BigGAN at 170 and starGAN at 161 fall short, at the current
10,049-image authentic side; both clear it once the SID_Set acquisition lands).
Per-family retention runs 4.6% to 24.1% -- uneven, but nowhere near the
collapse that stratifying on short side alone would cause, because the
low-resolution WildFake families are all in the buckets with no authentic
partner at all.

### Consequence for `--max-per-generator`

The cap is close to a no-op for the confound, for a reason worth recording:
WildFake's generated families are ALREADY uniform at ~3,500 rows each, so no
cap at or above 3,500 touches them. The only family a cap binds on is
`sid_set` at 10,079 and rising. A cap of 4,000 therefore leaves generator
diversity completely intact and stops the one pseudo-generator family that
would otherwise dominate -- which is what the README already specifies, now
with evidence behind it rather than intuition.
