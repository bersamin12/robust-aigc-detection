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

> **As built on 2026-08-30 this was false, and fatally so.** The first smoke run
> emitted 6 pairs and **6 of 6 had mismatched dimensions**: the fake is always a
> multiple of 8 (`crop_box`, because the VAE downsamples by 8) while the real was
> copied byte-for-byte at its original size, so `width % 8 == 0` separated the
> classes at ~100% without touching a pixel. §3 records the fix. Nothing
> generated before that fix is usable as a corpus.

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
| `img2img` | the image + noise at `strength` | optional | any latent model |
| `vae_recon` | the image | none | any latent model |

`t2i_caption` (prompt from a Localized Narrative) is **implemented but no longer
in the suite** — see §3. It stays in `METHODS` because the code is correct and
someone with captions for their reals should be able to use it.

`self_cond` is the primary method. An inpainting UNet takes 9 channels — 4 noisy
latent, **1 mask**, **4 masked-image latent**. An all-zero mask says "nothing is
protected", so the reference image arrives in full and the denoiser's task
becomes *regenerate this*. Prior art: B-Free (CVPR 2025) built 309k fakes over
51k reals this way for this exact bias reason; DRCT (ICML 2024), TwinSynths
(WACV 2025) are the same idea.

**Every family in this suite is image-conditioned regeneration.** `self_cond`,
`inpaint_box`, `img2img`, `ref_image` and `vae_recon` all say "here is a
photograph, redraw it". Nothing here models an adversary who types a prompt and
generates from scratch. §3 records why, and §9 records what it costs. Score
methods separately regardless — `method` and `synthesis` are on every row.

### Default suite — 7 families, 4 models, 4 decoder lineages

| family | model | licence | lineage | share |
| --- | --- | --- | --- | --- |
| `sd15_self_cond` | SD1.5-inpainting | OpenRAIL-M | `sd_vae` | .25 |
| `sdxl_self_cond` | SDXL-inpainting-0.1 | OpenRAIL++-M | `sdxl_vae` | .25 |
| `flux_schnell_img2img` | FLUX.1-schnell | Apache-2.0 | `flux_vae` | .20 |
| `sd15_inpaint_box` | SD1.5-inpainting | OpenRAIL-M | `sd_vae` | .15 |
| `sdxl_inpaint_box` | SDXL-inpainting-0.1 | OpenRAIL++-M | `sdxl_vae` | .10 |
| `sd15_vae_recon` | SD1.5 VAE | OpenRAIL-M | `sd_vae` | .05 |
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

**Reals are cropped losslessly, not copied verbatim.** "Copy the real
byte-for-byte" and "the fake is a multiple of 8" cannot both hold: changing a
JPEG's dimensions normally means re-encoding, and re-encoding the real is exactly
what the byte-for-byte rule exists to prevent. Copying verbatim left the crop
applied to the generator's input and never to the emitted real -- the leak above.

`jpegtran -crop` resolves it. A crop on **MCU boundaries** rewrites no DCT
coefficient, so the real gains no compression generation and the pair matches on
geometry. Measured on the six smoke reals: **5 of 6 bit-exact**; the sixth, the
only 4:2:0 file, differs on **0.137%** of pixels, all in the last two rows and
columns, with the interior **bit-exact at max diff 0**. That residue is the
decoder losing its chroma-upsampling context at the new edge, not a
requantisation.

It costs a little geometry. MCU is 16x16 for 4:2:0 and 8x8 for 4:4:4, and the
smoke reals were **mixed** (5 of 6 at 4:4:4), so the crop must align to **16** to
be safe for every file. That trims up to 15 px per axis rather than 7 --
immaterial at ~420x640 -- and the crop **offset** must be MCU-aligned too, which
the old `(w - cw) // 2` was not.

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

**`t2i_caption` is removed, not shrunk.** Localized Narratives cover
**5.60%** of Open Images train — 504,413 captions against 9,011,219 images — and
**6.48%** of the CC BY 2.0 rows that carry a `Thumbnail300KURL`, which is the
pool the harvest actually draws from (measured 2026-08-30 over 82,663 sampled
candidates). So 60,000 portrait reals contain roughly **3,900** with a usable
prompt, and the planned 12,000-row share was never reachable; reaching it would
need ~185,000 reals harvested. Worse, a real without a narrative is not skipped:
the notebook passes `prompt=""` and records `prompt_source="MISSING"`, so ~93.5%
of the family would have been empty-prompt FLUX output labelled as
prompt-conditioned synthesis. The share goes to `flux_schnell_img2img`, which
needs no prompt, keeps the `flux_vae` lineage, and makes all 60,000 reals
eligible. **The narrower alternative was considered and declined:** keep the
family at ~3,000 rows with reals selected *from* the covered set, which would
have preserved a measurable prompt-from-scratch arm. It was dropped for
simplicity, and §9 states the cost.

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

**Measured 2026-08-30 on a Kaggle T4** (first real run, 6 images): `sdxl_self_cond`
**4.32 s/img**, `sdxl_inpaint_box` **4.12 s/img** -- against a 14.0 s prior, so
**3.2-3.4x faster than this table assumes**. If that 0.30x correction carries to
the untested families, 60k falls from ~379 to **~129 T4-hours** and the "2.4
weeks for five friends" conclusion below no longer holds. Only SDXL is measured;
FLUX and klein remain priors.

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

**Findings from the first smoke run, 2026-08-30** (T4, 6 images, `ov7-smoke-sdxl`):

* **`self_cond` passes §7's visual check.** Same scene, visibly redrawn; the
  text degrades the way diffusion degrades text ("Play. laugh. grow." ->
  "Ploy. lough. grow."). The all-zero mask is arriving as zeros. 0.0-0.5% of
  pixels identical to the real.
* **Dimensions leaked, 6 of 6 pairs.** See §1 and §3. This is the blocking one.
* **`OUT_ROOT` is never resolved.** §1's parameter cell assigns
  `os.path.join(HERE, "aigc_pairs")` and then overwrites it with `None` on the
  next line, commented "resolved in section 0" -- but section 0 is an *earlier*
  cell and nothing reassigns it. Every output went to a directory literally
  named `None/`, and the closing instruction to "publish
  /kaggle/working/aigc_pairs as a Dataset" names a path that does not exist.
* **`inpaint_box` containment is loose.** Change concentrates inside the box
  (mean |diff| 33-34 inside vs 3-11 outside) but leaks: **1.4%** of outside
  pixels visibly changed on one image, **21.3%** on the other. At
  `strength=0.99` the unmasked region is not hard-composited. `synthesis="partial"`
  is approximately, not strictly, true. n=2, so the rate is uncharacterised.
* **`inpaint_box` runs with an empty prompt**, which is right for `self_cond`
  (nothing is missing) but leaves the model no guidance for the 40% of frame it
  must invent -- one smoke image filled with pale mush where the subjects were.
* **The gate failed**, and not obviously by chance: `jpeg_quality` AUC
  **0.0000** -- perfect inverse separation -- for *both* families, despite
  `qtables_copied=True` on every row. Joint probability under chance is ~0.2%.
  **Hypothesis:** this is the dimension leak in disguise. The crop offset
  `(w - cw) // 2` was arbitrary, so each fake's fresh 8x8 DCT grid sat at a
  different phase from its real's preserved grid, and copying quantisation
  tables cannot fix a phase difference. §3's MCU-aligned crop should remove it.
  **Unresolved at n=6** -- the gate is specified for 2,000, and re-running it
  before the geometry fix would only reproduce this.

* `docs/02` §2's SDXL-Turbo row is wrong and unfixed.
* **The SD2 checkpoint is gone, and the suite has moved to SD 1.5.** Two
  separate errors were in the old row. First the name: there has **never been an
  SD 2.1 inpainting checkpoint** — Stability shipped 2.0-inpainting and no 2.1
  equivalent, confirmed by the open requests for one on the Hub — so the `sd21`
  key and the "SD2.1" label were wrong regardless. Second, and fatal, the repo is
  not reachable: **401** anonymously on the page and on `resolve/`, and **404**
  from `model_info()` with a **valid token**. A 404 under authentication is not a
  gate anyone can accept terms for. It cost one run 0.32 GPU-hours before the
  registry's own "not on the Hub" assertion stopped it.
  `sd_vae` was that model's lineage alone, and losing a lineage would break §10's
  held-out design, so the three rows now use
  `stable-diffusion-v1-5/stable-diffusion-inpainting`: same lineage, same three
  methods, ungated, licence tag verified against the Hub as
  `creativeml-openrail-m`, and the diffusers docs' own recommended inpainting
  checkpoint. **The cost is real** — 45% of the corpus now records SD 1.5, an
  older and weaker generator than SD 2.0. The `sd21_inpaint` entry is kept and
  marked `unreachable=True` rather than deleted, for the same reason klein-9B is
  kept refused. `black-forest-labs/FLUX.1-schnell` is also 401 (`gated: auto`)
  -- test `resolve/`, not the metadata API, which answers anonymously either way.
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` and
  `black-forest-labs/FLUX.2-klein-4B` both return 206 and need no token, so the
  registry's `gated=True` on klein-4B is wrong.
* **Kaggle's P100 cannot run this notebook at all.** `torch 2.10.0+cu128` ships
  no Pascal (sm_60) kernels, so every CUDA op raises
  `cudaErrorNoKernelImageForDevice`. This is not the bf16 issue -- it is total.
  A shard that lands on a P100 dies after paying the tar extract and the model
  download. Pin the device: `"machine_shape": "NvidiaTeslaT4"` in the kernel
  metadata. `DEVICE_FACTOR`'s `"P100": 1.2` now describes an unusable device.
* `scripts/acquire_open_images_portrait.py` is on
  `feat/robust-aigc-detection`, not `master`.
* Nothing generated yet. Every throughput number here is a prior.

## 9. Residuals — do not paper over these

* **Compression history.** Reals carry camera → Flickr → thumbnail; fakes carry
  one pass. `FAKE_ENCODE_PASSES = 2` is the lever if section 10 says it shows.
* **One geometry.** AR ≤ 0.7, short side ≥ 400. A detector trained only here
  has seen one scale.
* **No family models the threat.** `self_cond` is image-conditioned
  regeneration and so is every other family here; an adversary prompts from
  scratch. `t2i_caption` was the 20% that covered this and it is **gone** (§3),
  so the residual is no longer mitigated — it is simply open. A detector
  trained on this corpus learns *was this regenerated from an existing
  photograph*, which correlates with *is this synthetic* without being it.
  Whether that transfers to prompted generation is now a question this corpus
  **cannot answer about itself**; it needs an external prompted-synthesis eval
  set. Do not report a number from this corpus as if it settled the point.

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
