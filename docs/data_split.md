# The complete data split

**Proposed 2026-08-31.** Supersedes the single `heldout_generator` split for
model selection. Written after OV7 scored 0.3998 against the same rung that
scores 0.8300 on `heldout_generator` and 0.9978 val AUC on `val_internal`.

## The principle

Split by **decoder lineage**, whole lineages, never split one across
populations. Lineage — not era, not generator count — is what predicts
transfer:

- `sdxl_self_cond` 0.8279 vs `sdxl_t2i` 0.9895 — same VAE, same era, 0.16 apart
- `flux2_vae` 0.6079 vs `sdxl_vae` 0.9471 — different lineage, same era

`registry.py:351` already encodes this: `HELDOUT_LINEAGE = "flux2_vae"`, and
`heldout_groups()` holds out a lineage whole or not at all.

## The five populations

| population | manifest | rows | role |
| --- | --- | --- | --- |
| `train` | union | 331,257 | fit the head / tower |
| `val_internal` | union | 37,101 | in-distribution health + selection negatives |
| `heldout_generator` (legacy) | union | 7,000 | continuity with historical numbers |
| **`select`** | ov7 | 20,356 | pick rung / epoch / architecture |
| **`sealed`** | ov7 | 3,600 → target 15,000 | unseen-lineage generalisation |

### train — 331,257 rows

| source | reals | fakes | families |
| --- | --- | --- | --- |
| ntire | 48,573 | 86,287 | 42 generators, **1 label** |
| wildfake | 35,962 | 49,239 | 16 |
| sid_set | 26,333 | 26,565 | 1 |
| coco_train2017 | 35,880 | 0 | — |
| open_images | 22,418 | 0 | — |

NTIRE is training-only and always will be. Lineage-disjoint splitting needs
per-image generator labels; NTIRE has none, and k-means could not recover them
(ARI 0.232 on a control of 16 *known* single-generator families). A held-out
split that cannot be verified disjoint is not held out. Training carries no
such requirement — it needs diverse fakes, and 86k across 42 generators is the
most valuable block in the corpus.

### val_internal — 37,101 rows

| source | reals | fakes |
| --- | --- | --- |
| ntire | 5,427 | 9,712 |
| wildfake | 4,038 | 5,441 |
| sid_set | 2,984 | 2,873 |
| coco_train2017 | 4,044 | 0 |
| open_images | 2,582 | 0 |

All 18 of its fake generators also appear in `train`; zero are unseen. This is
a held-out **images** split, not a held-out **generators** split, and it
measures in-distribution fit — which is why it reads 0.9978 while the same
model reads 0.3998 on unseen lineages. It keeps two jobs: supplying the
`label 0` negatives the selection metric uses, and acting as a breakage alarm
(val AUC below ~0.99 means training broke).

### heldout_generator (legacy) — 7,000 rows

`SDwithAdaptor_controlnet` 3,500 + `VQGAN` 3,500, both WildFake, both old, both
in lineages that training already covers. Retained and reported as a legacy
column so the historical ladder (a3 0.8300 … a6 0.8323) stays comparable. It no
longer decides anything.

### select — OV7, four lineages

| lineage | families | fakes | matched reals |
| --- | --- | --- | --- |
| sdxl_vae | sdxl_t2i, sdxl_self_cond, sdxl_img2img | 5,400 | 5,400 |
| sd_vae | sd15_t2i, sd15_img2img | 2,778 | 2,778 |
| movq | kandinsky22_t2i | 1,000 | 1,000 |
| dc_ae | sana1600m_t2i | 1,000 | 1,000 |
| **total** | 7 | **10,178** | **10,178** |

Every fake is paired with a real cropped from the same source image through the
same `jpegtran` path — same crop, same quantisation tables. The negatives
therefore cannot be separated by encoder history, which is the confound
`val_internal`'s cross-source negatives leave open.

**Average the metric per lineage, do not pool.** `sdxl_vae` is 53% of these
fakes; pooling would let one lineage decide selection. Per-lineage then mean,
mirroring what the metric already does across conditions.

### sealed — OV7, three lineages, never scored until the end

| lineage | families | fakes now | target |
| --- | --- | --- | --- |
| flux2_vae | klein4b_t2i, klein4b_ref_image | 1,800 | 2,500 |
| paella_vq | wuerstchen_t2i | 0 | 2,500 |
| cogview_vae | cogview4_t2i | 0 | 2,500 |
| **total** | | **1,800** | **7,500** |

`flux2_vae` is the hardest lineage measured (AUC 0.6079) and is already
`HELDOUT_LINEAGE`. `paella_vq` and `cogview_vae` are the `ov7_lineage2` suite,
not yet generated.

**1,800 negatives is too thin to seal on.** At 1% FPR the threshold is the 99th
percentile order statistic — 18 negatives above the bar. Estimates that far
into the tail need ≥2,500 per lineage, ideally pooled across the three for
~7,500. Negatives may be pooled across sealed lineages: they share the
generation pipeline, so the encoder match is preserved even where the per-image
pairing is not.

## Where new generation should go

Two rules, both from measurements:

1. **Breadth over depth.** `PairedSampler._draw_stratified` picks a family
   uniformly, then an image inside it. The 20,000th image of a family buys zero
   additional gradient. Past ~2,500/family, spend on a **new lineage**.
2. **Open weights train and select; commercial APIs test only.** Open weights
   give a known, verifiable decoder — a prerequisite for guaranteeing
   disjointness. Commercial output (Midjourney, DALL·E, Imagen, Firefly,
   Seedream) is the genuine unknown-decoder case, which is exactly what makes
   it valuable sealed and wasteful in training. You also cannot verify that a
   commercial family does not share a VAE with a training lineage.

Priority order:

| # | what | why |
| --- | --- | --- |
| 1 | `ov7_lineage2` — paella_vq, cogview_vae @ 2,500 | the sealed set is unusable at 1,800 |
| 2 | flux2_vae +700 | brings the hardest lineage to a stable estimate |
| 3 | NTIRE generator inventory (documentation, not labels) | see below |
| 4 | new **training** lineages, 2 families × 2,500 | breadth, only after 3 is known |
| 5 | commercial API, 4–6 services × ~2,000, sealed | true unknown decoder |

Commercial output stays in a **private** dataset — redistribution is usually
forbidden by ToS, consistent with the standing NTIRE rule.

## The open question that gates item 4

Training's lineage diversity is **unmeasured**. NTIRE is 53% of training fakes
under one label with no per-image generator metadata. But assessing lineage
span does not need per-image labels — it needs the *inventory*: which 42 models
NTIRE used. Each maps to a known VAE/decoder family. If the challenge
documents its generator list, we learn whether NTIRE spans 20 lineages or 3
without labelling a single image.

This matters more than any other open item. If those 42 generators are
lineage-narrow, the corpus is far less diverse than "42 generators" implies,
and that explains the flux2 collapse better than the 5.6% draw share does.

## Code changes this implies

1. **Stratify on lineage, not generator.** `sampler.py:145` draws
   `groups[rng.integers(n_groups)]` with groups keyed on `generator`. A lineage
   contributing 8 families gets 8x the gradient of one contributing 1 — and
   lineage is what transfers. Key `_build_groups` on lineage.
2. **Per-lineage metric averaging** for the select population.
3. **OV7 stays its own manifest and its own bank.** It must never be merged
   into `manifest_union.parquet`: banks fingerprint `manifest_sha256` over
   `rel_path` in row order.
4. OV7's internal `train`/`val_internal`/`heldout_generator` split in
   `manifest_ov7.parquet` is overridden — the whole corpus is evaluation-only
   under this plan.

## What this costs

Changing the selection population makes every existing rung number
non-comparable to new ones. The legacy column exists to soften that, but a
comparable table needs the ladder re-run. The old numbers were measuring
in-distribution and near-distribution performance; that is the cost of finding
out they were.

## NTIRE generator inventory (resolved 2026-08-31)

Source: NTIRE 2026 challenge paper, [arXiv:2604.11487](https://arxiv.org/abs/2604.11487),
Tables 6 and 7. The inventory was the open question gating new generation
budget. It is now answered, and it reverses one earlier recommendation.

### What we hold is NTIRE's TRAIN split — 20 open-source generators

> "The training split covers 20 open-source generators in total, with the most
> recent models including Flux Kontext (dev), DeepFloyd-IF, and Ovis-image.
> Most of the top-performing open-source models (e.g. Qwen-Image, HiDream,
> etc.) as well as proprietary generators (Nano Banana, Grok Imagine, etc.)
> were reserved for validation and test splits, with each consecutive split
> containing progressively larger share of state-of-the-art models."

| split | n | models |
| --- | --- | --- |
| **train** | 20 | YOSO PixArt-512, PixArt-α, PixArt-Σ, Kandinsky 2, Kandinsky 3, Kolors, OmniGen, OmniGen 2, SD 1.4, SD 1.5, SD 2.1, SDXL 1.0, SDXL Lightning, SDXL Turbo, Janus Pro 7B, Infinity 2B, Infinity 8B, Ovis Image, DeepFloyd IF, FLUX.1 Kontext Dev |
| validation | 9 | FLUX.1 Kontext Dev, SDXL Turbo, FLUX.1 Dev, Playground v2.5, Lumina Image 2.0, Qwen Image, SD 3 Medium, Ideogram v3 Turbo, ImageGen-4 Fast |
| validation-hard | 7 | Playground v2.5, SDXL Turbo, HiDream, FLUX.1 Schnell, SD 3.5 Large Turbo, Nano Banana, Seedream 4 |
| test (public) | 10 | HiDream, FLUX.1 Schnell, SD 3.5 Large, FLUX Krea, Z-Image Turbo, Nano Banana Pro, FLUX-2 Max, ImageGen-4 Ultra, Seedream 5 Lite, Grok Imagine |
| test (private) | 10 | HiDream, SD 3.5 Large Turbo, FLUX.1 Dev SRPO, Z-Image Turbo, Kandinsky 5, Nano Banana 2, GPT Image 1.5, ImageGen-4 Ultra, Seedream 5 Lite, Grok Imagine |

**42 generators is the union across all splits, not the training inventory.**
Our corpus draws on the train split, so it holds 20 — and the modern SOTA was
deliberately withheld from it.

### Decoder lineage of the 20 training generators

Mapped from architecture, not from released metadata. Rows marked (?) are
inferred from memory of the model's design and should be confirmed before being
relied on.

| lineage | generators | n |
| --- | --- | --- |
| **sd_vae** (KL-f8, 4-ch) | SD 1.4, SD 1.5, SD 2.1, PixArt-α, YOSO PixArt-512 | 5 |
| **sdxl_vae** (KL-f8, 4-ch) | SDXL 1.0, SDXL Lightning, SDXL Turbo, Kolors, PixArt-Σ, OmniGen, OmniGen 2 (?) | 7 |
| **movq** | Kandinsky 2, Kandinsky 3 | 2 |
| **VQ autoregressive** | Janus Pro 7B, Infinity 2B, Infinity 8B | 3 |
| **flux_vae** (16-ch) | FLUX.1 Kontext Dev | 1 |
| **pixel-space** (no latent decoder) | DeepFloyd IF | 1 |
| unresolved | Ovis Image (?) | 1 |

Roughly **6 lineages across 20 generators**, and 12 of 20 use a KL-f8
4-channel VAE. `sd_vae` and `sdxl_vae` are the same architecture with different
weights, so at the architectural level the training pool is 60% one design.

### This explains the flux2 collapse quantitatively

The `ntire` family draws 5.56% of fakes, uniformly within itself. FLUX.1
Kontext Dev is 1 of 20 generators, so FLUX-family decoder output is about

    0.0556 x (1/20) = 0.28% of fake gradient

and FLUX.1's VAE is not FLUX.2's. The OV7 `flux2_vae` lineage — AUC 0.6079,
the worst measured — is the lineage training has almost no exposure to. The
two SD-family lineages, which dominate training, score 0.9471 and 0.9252.

### Consequence: do NOT weight `ntire` at 42

Earlier in this work the plan was to weight `ntire` at 42 in `PairedSampler` to
correct its 5.56% draw share against 53% of rows. The inventory kills it:

- the number is 20, not 42 — 42 is the all-splits union
- more importantly, NTIRE's train split is **lineage-narrow and SD-dominated**.
  Upweighting it pours more KL-f8 gradient into a model that already fails on
  non-KL-f8 decoders. It would deepen the failure it was meant to fix.

The draw-share arithmetic was right; the assumption that NTIRE's rows carried
proportional lineage diversity was wrong.

### Consequence: new generation buys TRAINING breadth

The gate on item 4 is lifted, and the answer is yes. Training needs decoder
lineages that are not KL-f8 4-channel. Underrepresented or single-generator
today: FLUX-family (1), pixel-space (1), MoVQ (2), VQ-autoregressive (3), and
DC-AE, Paella-VQ and CogView absent entirely.

### Still unresolved: per-image labels

The paper does not state that per-image generator labels were released for the
training split, and no NTIRE metadata file exists in our tree. Treat NTIRE as
unlabelled, as before. The inventory above is sufficient to act on without
them — it tells us the lineage *span*, which is what gated the decision.
