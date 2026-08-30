# 03 — AI-OV7 generation: plan and state

**Status 2026-08-30.** The generator side of `docs/02` is built and unrun.
`notebooks/kaggle_generate_pairs.ipynb` exists, its pure-Python core is tested,
and no images have been generated yet. Next action is the smoke run.

This document is the handoff. `docs/02` says *what* to build and why; this says
*what was decided*, *what is done*, and *what to do next*.

---

## 1. The one idea

Every fake is generated **from a specific real**, and carries that real's
`ImageID`. Content, pixel dimensions and JPEG encoder are held fixed across the
pair, so what is left between them is the generator.

1 real : 1 fake. No real is used twice. §4's balance follows by construction.

## 2. What is built

`notebooks/kaggle_generate_pairs.ipynb` — 33 cells, self-documenting, runs on
Kaggle / Colab / a workstation. It clones this repo (branch
`feat/robust-aigc-detection`) so section 10's gate uses
`aigcdet.features.proxies` rather than a re-implementation; without the clone it
falls back to a port and says so.

Reads `portrait/*.jpg` + `attribution.csv` from
`scripts/acquire_open_images_portrait.py`. Writes
`<OUT>/open_images_v7/{real,<family>}/<ImageID>.jpg`, `pairs_*.parquet`,
`attribution.csv`, `config_*.json`.

### Methods

| method | conditioned on | prompt | needs |
| --- | --- | --- | --- |
| `self_cond` | the image, via an **all-zero mask** | none | 9-channel inpaint UNet |
| `ref_image` | the image, as reference tokens | none | FLUX.2 klein |
| `inpaint_box` | image + deterministic rectangle | none | 9-channel inpaint UNet |
| `t2i_caption` | a Localized Narrative | yes | narratives on disk |
| `vae_recon` | the image | none | any latent model |

`self_cond` is the primary method. An inpainting UNet takes 9 channels — 4 noisy
latent, **1 mask**, **4 masked-image latent**. An all-zero mask says "nothing is
protected", so the reference image arrives in full and the denoiser's task
becomes *regenerate this*. Prior art: B-Free (CVPR 2025) built 309k fakes over
51k reals this way for this exact bias reason; DRCT (ICML 2024), TwinSynths
(WACV 2025) are the same idea.

`t2i_caption` is kept at ~20% deliberately: `self_cond` isolates the artifact,
`t2i_caption` resembles what an adversary actually does. Score them separately —
`method` and `synthesis` are on every row.

### Default suite — 7 families, 4 models, 4 decoder lineages

| family | model | licence | lineage | share |
| --- | --- | --- | --- | --- |
| `sd21_self_cond` | SD2.1-inpainting | OpenRAIL++-M | `sd_vae` | .25 |
| `sdxl_self_cond` | SDXL-inpainting-0.1 | OpenRAIL++-M | `sdxl_vae` | .25 |
| `flux_schnell_t2i_caption` | FLUX.1-schnell | Apache-2.0 | `flux_vae` | .20 |
| `sd21_inpaint_box` | SD2.1-inpainting | OpenRAIL++-M | `sd_vae` | .15 |
| `sdxl_inpaint_box` | SDXL-inpainting-0.1 | OpenRAIL++-M | `sdxl_vae` | .10 |
| `sd21_vae_recon` | SD2.1 VAE | OpenRAIL++-M | `sd_vae` | .05 |
| `flux2_klein4b_ref_image` | FLUX.2-klein-**4B** | Apache-2.0 | `flux2_vae` | .20 |

A *family* is a (model, method) pair — §3.3's rule. `lineage` is the decoder and
is the key `heldout_groups` must be built from (§3.4).

## 3. Decisions, with the reason

**JPEG tables are copied, not matched.** The reals are `Thumbnail300KURL`
re-encodes. Saving fakes "at a similar quality" leaves a distribution the gate
finds; lifting the real's own 64 integers and its subsampling onto its fake
leaves no distribution at all. Measured locally: `jpeg_quality` AUC **0.5031**
against the 0.5532 baseline. Reals are copied **byte-for-byte** — re-encoding
one to "match" would add a compression generation to the authentic class.

**Nothing is ever resized.** A resample leaves a spectral signature
(`docs/resolution_shortcut.md`). Geometry is centre-**cropped** to a multiple of
8 with aspect preserved; `MAX_SIDE` scales both sides by one factor rather than
clamping each, which would square up a portrait.

**Seeds are content-addressed.** `blake2b(SEED, ImageID)`, never a counter, so a
rerun with a different `N_TOTAL`, shard boundary or dropped file reproduces the
same pixels for the same real.

**bf16 is refused, not worked around.** fp16 has 5 exponent bits to bf16's 8. A
flow-matching DiT in fp16 does not raise — it emits washed-out or NaN-speckled
images that still look like images, the worst failure for a corpus recording
what a generator's output *looks like*. Quantising is refused for a related
reason: a 4-bit model's traces are partly the compute budget's. Kaggle has only
T4 (sm_75) and P100 (sm_60); neither has bf16. Such families are **skipped** and
the shard block rebalances over the rest.

**Outputs are checked on their pixels.** A near-constant frame is a NaN latent
or a safety-checker black image; a frame identical to its real means the
pipeline composited the original pixels back and the family would be copies
labelled "fake". Both raise.

## 4. Licences — two corrections found while building this

* **SDXL-Turbo is `sai-nc-community`, non-commercial.** `docs/02` §2 lists
  "SDXL 1.0 / SDXL-Turbo | CreativeML OpenRAIL++-M | Use." That is right for
  SDXL 1.0 and **wrong for Turbo**. **`docs/02` still needs this fix.**
* **FLUX.2-klein-9B is FLUX Non-Commercial v2.1.** Only **klein-4B** (and
  klein-base-4B) are Apache-2.0 — confirmed on the 4B model card. Covers
  `klein-9b-kv` and every fp8/nvfp4 repacking. Same trap as FLUX.1-dev.

Enforcement, so this cannot recur silently: every registry entry carries
`commercial: bool`; the suite validator refuses a non-commercial model; and
`load()` asks the Hub for the published `license` tag and asserts it matches the
registry. The 9B entry is kept **refused rather than deleted** — an absent entry
reads as an oversight, a refused one as a decision.

## 5. Fleet protocol

Shard *k* owns block *k* of the reals ordered by `blake2b(SEED, ImageID)`.
Blocks are contiguous and disjoint. **Inside its block** the work splits across
the families *that machine* can run, renormalised.

* `SHARD` splits across **identical** machines.
* `FAMILIES` splits across **different** ones (give the bf16 box the klein family).
* **Rules:** `N_SHARDS` identical everywhere; no two people on the same `SHARD`.

Verified by simulation: different shards overlap on 0 reals; each real used
exactly once; a T4 spreads a 1,000-real block over 6 families, a 4060 Ti over 7.

**Cost of rebalancing:** per-family counts depend on the fleet's GPU mix. In a
60-shard sim with 1-in-6 sessions having bf16, klein came out at 2.8%, not 20%.
`share` is a target, not a guarantee. Section 11 prints achieved counts; top up
a thin family with `FAMILIES`.

## 6. Budget (priors — section 12 replaces with measurement)

| | T4-hours | `N_SHARDS` for 1-hour shards |
| --- | --- | --- |
| 60k **with** FLUX.1-schnell | ~357 | 431 |
| 60k **without** it | ~107 | 160 |
| 2,000-image gate run | 14 / 5 | ~14 / ~5 |

Kaggle: 30 GPU-h per account per week. With FLUX ≈ 2.4 weeks for five friends;
without ≈ under one week. FLUX.1-schnell is **70% of the budget for 20% of the
images** — 12B forces CPU offload on a 16 GB card. Decide on measured numbers.

## 7. Next actions, in order

1. `SMOKE = True`, run all cells. Minutes.
2. **Look at section 9.** `self_cond` fakes must be the same scene, visibly
   redrawn. A different scene means `strength` is wrong or the mask is not
   arriving as zeros.
3. **Read section 12** for measured s/img. Raise `SMOKE_PER_FAMILY` to 8 before
   judging FLUX — its first image pays a cold CPU-offload path.
4. Decide on FLUX.1-schnell. Set `N_SHARDS`.
5. `SMOKE = False`, run ~2,000 images.
6. **Read section 10 — the gate.** It is allowed to cancel the task (§5).
7. Only then scale.

## 8. Open items

* `docs/02` §2's SDXL-Turbo row is wrong and unfixed.
* SD2.1-inpainting's `openrail++` tag is asserted from belief, not verified —
  its HF page returned 401 twice. The runtime Hub check will settle it.
* `scripts/acquire_open_images_portrait.py` is on
  `feat/robust-aigc-detection`, not `master`.
* Nothing generated yet. Every throughput number here is a prior.

## 9. Residuals — do not paper over these

* **Compression history.** Reals carry camera → Flickr → thumbnail; fakes carry
  one pass. `FAKE_ENCODE_PASSES = 2` is the lever if section 10 says it shows.
* **One geometry.** AR ≤ 0.7, short side ≥ 400. A detector trained only here
  has seen one scale.
* **`self_cond` is not the threat model.** It is image-conditioned
  regeneration; an adversary prompts from scratch. Hence `t2i_caption` at 20%.

## 10. Downstream

Register in `sources.py`:

```python
"open_images_v7": SourceSpec(
    name="open_images_v7",
    licence="CC BY 2.0 (per-image) — https://creativecommons.org/licenses/by/2.0/ "
            "— attribution required; ship attribution.csv",
    real_buckets=frozenset({"real"}),
    generator_buckets=True,
    exclude_from_training=False,
),
```

Group `heldout_groups` by the `lineage` column — `sd_vae`, `sdxl_vae`,
`flux_vae`, `flux2_vae` — never by name. Holding out `sdxl_*` while `sd21_*`
trains measures a *cousin*; holding out FLUX measures a lineage jump. Both are
worth a number and they are not the same number.

**Split train/val by `image_id`, never by row.** A real and its fake on opposite
sides of the boundary means the model sees one scene under both labels.

## 11. If the gate fails and the save path is not the cause

That is §6's negative result: generated and photographic images differ, on this
source, below what canonicalisation can remove. Write it up. It is a finding,
not a bug to route around.
