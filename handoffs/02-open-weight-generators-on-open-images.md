# 02 — Open-weight generators on Open Images V7 ("AI OV7")

**Goal.** Build a licence-clean, modern-era generated class whose *real*
counterpart is Open Images V7 photographs. Every fake in the corpus today
comes from a 2017–2023 generator; this is the task that closes the era gap
with weights we are allowed to use commercially.

**Why it exists.** Published results show detection accuracy falling from ~79%
on 2020–21 generators to ~38% on 2024 ones. Every public dataset that would
close that gap for us is licence-barred (see `docs/dataset_presets.md`), and
the wall is structural: they draw their reals from web scrapes whose images
keep individual copyrights, so no compiler can grant commercial rights
downstream. Generating our own fakes over reals we *can* redistribute is the
way around it.

---

## 1. What already exists

**The reals are being harvested now.** `scripts/acquire_open_images_portrait.py`
is running and will leave:

```
/mnt/berstorage/techjam/open_images/portrait/<ImageID>.jpg   60,000 images
/mnt/berstorage/techjam/open_images/attribution.csv          one row each
```

Filtered to `License == CC BY 2.0` (verified 100% of 66,654 sampled metadata
rows), aspect ratio ≤ 0.7, short side ≥ 400. Measured yield 7.1%, so it walks
~850k candidate rows to find 60k.

**CC BY 2.0 is the whole point.** It permits commercial use *and*
redistribution with attribution — the only vertical-real source audited that
clears both gates. Pexels bars ML datasets outright; Unsplash bars
redistribution. Do not silently substitute either.

**`attribution.csv` is not optional.** CC BY requires attribution and
normalisation strips image metadata. That file is the only surviving record.
Ship it with any derived dataset.

### The one thing to be honest about

These are **thumbnails** (`Thumbnail300KURL`), not originals — 60k originals
would be ~180 GB and we do not have the disk. A thumbnail is a *re-encoded
JPEG*, and this project has already measured that JPEG history leaks the
label (`docs/low_level_confounds.md`). If your generated images are saved as
PNG while every real is a twice-compressed JPEG, a detector can hit high
accuracy on compression alone.

**Mitigation, and it is mandatory:** save generated images through the *same*
JPEG encoder at the *same* quality distribution as the reals, and run §5
before training anything.

---

## 2. Which models you may use

The licence on the *weights* is separate from the organisers' allowance to use
commercial APIs. An API's terms of service can permit commercial use of its
outputs; a model whose weights are released under a non-commercial licence
does not become commercial because you accessed it a different way. Check the
weight licence, every time.

| model | weight licence | verdict |
|---|---|---|
| FLUX.1-schnell | Apache-2.0 | **Use.** The cleanest modern licence available. |
| SDXL 1.0 / SDXL-Turbo | CreativeML OpenRAIL++-M | **Use.** Permits commercial use under use-based restrictions we already satisfy. |
| SD 3.5 (Medium / Large) | Stability Community License | **Use**, below the revenue threshold. Record the threshold in the source docstring. |
| SD 1.5 / 2.1 | CreativeML OpenRAIL-M | **Use.** Older era, but it is the bridge to WildFake's families. |
| Qwen-Image | Apache-2.0 | **Use.** Different lineage from the SD/FLUX family — valuable for held-out. |
| FLUX.1-dev | FLUX.1 Non-Commercial | **Do not use.** This is the one people reach for by default. |
| SANA (NVIDIA) | non-commercial research | **Do not use.** |

Verify each licence yourself at the model page before you download. This table
was assembled 2026-08-30 and licences change.

---

## 3. What to build

### 3.1 Two kinds of fake, not one

The benchmark may contain both, and they are different detection problems.

* **Fully synthetic** — text-to-image from a prompt describing the real image.
  This is what WildFake and most of our current corpus is.
* **Partially synthetic** — inpaint a region of the real photograph, leaving
  most authentic pixels in place. This is the harder class and the one we have
  almost no coverage of. SID_Set's label-2 tampered split is the only other
  licence-clean source of it and we do not ingest it yet.

Target roughly **70% fully synthetic / 30% inpainted**, and record which is
which in a column — a later ablation will want to score them separately.

### 3.2 Prompts

Do not invent captions. Open Images V7 ships **Localized Narratives**, which
are human-written descriptions keyed by `ImageID`. Use those: they describe the
actual image, so the generated counterpart is a genuine pairing rather than an
unrelated photo. Fall back to the class-label list only for images with no
narrative, and mark those rows.

### 3.3 Register the source

Add to `src/aigcdet/data/sources.py`:

```python
"open_images_v7": SourceSpec(
    name="open_images_v7",
    licence="CC BY 2.0 (per-image) — https://creativecommons.org/licenses/by/2.0/",
    real_buckets=frozenset({"portrait"}),
    generator_buckets=True,          # one directory per generator family
    exclude_from_training=False,
),
```

Directory layout must be `<source>/<bucket>/...`, where the bucket is either
`portrait` (the reals) or a generator family name. `build_dataset.py` reads
`rel[1]` as the bucket, so the family name has to be at that depth.

**Name families precisely.** `flux_schnell_t2i` and `flux_schnell_inpaint`
are two families, not one, and `sdxl` alone is not enough — the whole held-out
design depends on family names meaning something.

### 3.4 Held-out design

The current split holds out `SDwithAdaptor_controlnet` and `VQGAN` while their
siblings (`_lora`, `_lycris`, `VQVAE`, `vqdm`) stay in training — same
decoder, so it measures generalising to a *cousin*. `heldout_groups` exists in
`presets.py` to stop that: it takes a list of lists, each inner list a lineage
held out together or not at all.

Group your families by **decoder**, not by name. Everything sharing the SD VAE
is one lineage. FLUX is another. Qwen-Image is another. Then hold out one
whole lineage.

---

## 4. Volume

Match the authentic side. 60,000 reals means ~60,000 fakes for a balanced
corpus, spread across at least five families so the smallest clears
`splits.MIN_HELDOUT_IMAGES`. Roughly 12,000 per family across five families,
or 8,500 across seven.

**Start with 2,000 total and stop.** Run §5 on those 2,000 before generating
the other 58,000. The gate can fail, and it is much cheaper to discover that
after twenty minutes of GPU than after two days.

---

## 5. Acceptance criteria

Run before anything trains:

```bash
python scripts/gate_confounds.py --manifest <your manifest>
python scripts/stratified_auc.py --stratify-by source ...
```

1. **Confound gate.** `jpeg_quality`, `laplacian_var` and `noise_floor` AUC on
   the new source, against the frozen corpus baselines of 0.5532 / 0.6721 /
   0.6374. Materially above those and the rows are teaching compression or
   sharpness. **This is allowed to cancel the task.**
2. **Encoder parity.** Reals and fakes must have indistinguishable JPEG
   quality distributions. If `jpeg_quality` AUC alone is above ~0.60, fix the
   save path before looking at anything else.
3. **Per-source false positive rate.** Report it separately for Open Images,
   WildFake and SID_Set reals. A model that has memorised Open Images shows a
   far lower FPR there while the headline looks fine.
4. **Attribution intact.** `attribution.csv` present, one row per real, no
   blanks in `Author` or `OriginalURL`.

## 6. What a negative result looks like

Any of these is a real finding and should be written up, not worked around:

* The confound gate fails and encoder parity does not fix it — meaning
  generated and photographic images differ at a level our canonicalisation
  cannot remove, on this source.
* A detector trained with AI-OV7 does no better on the organisers' benchmark
  than one without it — meaning modern open-weight fakes do not transfer to
  the commercial generators the benchmark actually uses. That would be an
  argument *for* task 03, and it is worth knowing early.

---
---

# Updates — 2026-08-31, the scale-up

Written against the corpus at 11,978 pairs / nine families / five lineages
(`docs/ai_ov7_generation.md` §10–11). This is the plan for taking it to ~52,000
on rented GPUs, plus the corrections to §2 that planning it turned up.
Everything above this line is the original task and is left unedited.

## U1. §4's volume is not reachable from this harvest

`build_pool` finds **54,624 eligible of 60,000** (the 9% are
encoder-reproducibility drops, itemised in `ai_ov7_generation.md` §2), and one
real makes exactly one fake for exactly one family. 11,978 are spent, leaving
**42,646**. So the ceiling on this harvest is ~54,600 pairs, not §4's 60,000,
and the plan targets **52,000** to keep ~4.8% slack against `select`'s strata
rounding. More than that needs another `acquire_open_images_portrait.py` run at
7.1% yield — a separate job, not a GPU one.

## U2. §2's model table, corrected

Two rows are wrong and one model is missing. `registry.py`'s rule is that
nothing is deleted when it is ruled out, because a deleted entry gets re-added
by the next person with the same idea; that rule was never applied to this
table.

* **SDXL-Turbo is `sai-nc-community`, not OpenRAIL++.** Non-commercial.
  Recorded as refused in `registry.py`; the row here is still wrong.
* **SANA is not non-commercial.** This table says "non-commercial research —
  Do not use." `Efficient-Large-Model/Sana_1600M_1024px_diffusers` is
  **apache-2.0 including its bundled text encoder**, and it is now the `dc_ae`
  lineage in the suite. The row here is wrong.
* **Z-Image-Turbo is missing.** Released 2025-11-26, before this document was
  assembled on 2026-08-30. See U4 — it is a refusal, but an unrecorded one.

## U3. Qwen-Image-2.0 reverses a hardware refusal

`ai_ov7_generation.md` §11 refuses "Qwen-Image at 40.9 GB on hardware." That is
the **20B** original. **Qwen-Image-2.0, released 2026-02-10, is 7B** — a
lighter rebuild that holds #1 on AI Arena for both text-to-image and editing
and scores 88.32 on DPG-Bench against FLUX.1-12B's 83.84, with open weights in
safetensors.

That is the FLUX.1-schnell situation in reverse: a refusal on hardware, not
licence, that new weights lift. At bf16 the 7B transformer is ~14 GB; with
`enable_model_cpu_offload()` — running the text encoder and freeing it before
the transformer loads — the peak fits a 24 GB card.

**It would be the sixth lineage.** The 20B's VAE is Wan2.1-derived (frozen Wan
encoder, fine-tuned image decoder), unrelated to every decoder in the registry.
By §11's own argument — more lineages widened the generated distribution and
dropped `laplacian_var` from 0.5998 to 0.5632 — a sixth genuinely distinct
decoder is the highest-value thing left to add.

Three checks before it is worth booking hardware, in order:

1. **Does 2.0 keep the Wan-derived VAE?** The spec above is off the 20B's card.
   A 7B rebuild is exactly where an autoencoder gets swapped, and if 2.0 turns
   out flux-derived it fails §11's Lumina test and there is no reason to run
   it. This is the whole decision.
2. **Licence at the card, and pin the repo id.** Qwen's image repos are
   date-coded (`Qwen-Image-2512`), not semver. `check_licence()` enforces it at
   load, which is the right failure but not the cheapest one.
3. **`size_multiple`.** Declare it rather than assume 8. §11's smoke run found
   Kandinsky silently rounding to 64 and returning the same image for two
   different requests; the field exists now, so use it.

Published VRAM figures for this family contradict each other across sources —
"40 GB+", "24 GB+" and "~45 GB" all appear for the same 20B — because they
assume different offload and quantization. None are trustworthy; measure the
peak. Quantized figures are not an option regardless (`docs/03` §3).

## U4. Z-Image-Turbo is a refusal, by §11's own test

6B S3-DiT, Apache-2.0, 2025-11-26, ~1 s/image, fits 16 GB. It is the fast
Apache model anyone scaling this corpus would reach for first, and that is
exactly why it needs recording rather than silence.

**Its VAE is Flux-derived.** So it fails the test §11 applied to
Lumina-Image 2.0 — *architecture novelty is not decoder novelty* — and as a
lineage it is a cousin of the held-out `flux2_vae`, not a jump. Training it
would put a relative of the held-out decoder into training, which is §3.4's
mistake with newer weights.

It is worth a `ModelSpec` entry marked refused, alongside Lumina and PixArt-Σ,
with that reason. Reversible if the recon probe (`features/recon.py`, the
instrument that flagged `recon_probe_collision` for SD 1.5) shows FLUX.1's and
FLUX.2's VAEs are far enough apart to count as separate lineages — which is a
measurement nobody has taken.

## U5. The shard grid is already `--n-shards 5`, and it is not a clean prefix

The consumed reals are **not** a contiguous prefix. Two suites have run on the
grid the supplement established (`ai_ov7_generation.md` §11):

| shard | order positions | suite | used | free |
|---|---|---|---:|---:|
| 0 | 0 – 10,924 | `ov7` | 9,978 | ~925 |
| 1 | 10,925 – 21,849 | `ov7_lineage` | 2,000 | ~8,925 |
| 2 | 21,850 – 32,774 | — | 0 | 10,925 |
| 3 | 32,775 – 43,699 | — | 0 | 10,925 |
| 4 | 43,700 – 54,623 | — | 0 | 10,924 |

**Stay on `--n-shards 5`.** Re-gridding to 4 puts a boundary at 13,656, inside
shard 1's block, so a run there re-deals reals the supplement already
generated — a different suite's share dict re-deals every real in a block, and
`_done_ids` is per family, so one scene would land twice on the generated side
against one real.

That is now caught rather than merely documented: `generate_ov7.used_elsewhere`
refuses the run and names the offending reals. Treat it as the backstop, not
the plan.

A suite may only grow on a shard it already owns, or on a fresh one. Growing
`ov7` means running `ov7` on shard 3; growing the supplement means
`ov7_lineage` on shard 4; the Qwen arm needs a shard of its own.

## U6. Target: lineage parity, not proportional growth

§11 established that adding lineages, not volume, is what moved the gate
(`laplacian_var` 0.5998 → 0.5632 on the supplement alone). So the scale-up
buys lineage breadth rather than scaling the existing shares.

| shard | suite | `--total` | new |
|---|---|---:|---:|
| 0 | `ov7` (resume) | 10,925 | ~925 |
| 1 | `ov7_lineage` (resume) | 10,925 | ~8,925 |
| 2 | `ov7_qwen` (new) | 10,000 | 10,000 |
| 3 | `ov7` | 10,000 | 10,000 |
| 4 | `ov7_lineage` | 10,000 | 10,000 |

| family | lineage | existing | total | split |
|---|---|---:|---:|---|
| `kandinsky22_t2i` | movq | 1,000 | 10,463 | trained |
| `sana1600m_t2i` | dc_ae | 1,000 | 10,462 | trained |
| `qwen_image_2_t2i` | wan_vae | 0 | 10,000 | trained |
| `sdxl_t2i` | sdxl_vae | 3,000 | 6,271 | trained |
| `sd15_t2i` | sd_vae | 1,983 | 4,181 | trained |
| `sdxl_self_cond` | sdxl_vae | 1,400 | 2,926 | trained |
| `klein4b_t2i` | flux2_vae | 1,200 | 2,508 | **held out** |
| `sdxl_img2img` | sdxl_vae | 1,000 | 2,090 | trained |
| `sd15_img2img` | sd_vae | 795 | 1,672 | trained |
| `klein4b_ref_image` | flux2_vae | 600 | 1,254 | **held out** |
| | | **11,978** | **51,827** | |

| lineage | pairs | share | |
|---|---:|---:|---|
| `sdxl_vae` | 11,287 | 21.8% | trained |
| `movq` | 10,463 | 20.2% | trained |
| `dc_ae` | 10,462 | 20.2% | trained |
| `wan_vae` | 10,000 | 19.3% | trained (new) |
| `sd_vae` | 5,853 | 11.3% | trained |
| `flux2_vae` | 3,762 | 7.3% | **held out** |

Ten families, six lineages, five trained — a five-point leave-one-lineage-out
rotation. `HELDOUT_LINEAGE` stays `flux2_vae`; U3 and U4 change nothing about
the held-out design, and per §11 which lineage a rung holds out is a rung-level
choice (`RungConfig.train_exclude_generators`).

**This drifts the conditioning mix to 90/10 and that is a decision, not an
oversight.** The supplement families and Qwen are t2i-only by §11's argument
that an image-conditioned fake shares composition with its real and so carries
less of its decoder's fingerprint. Scaling them to parity pushes
fully-synthetic from 76% to ~90%, against §3.1's 70/30 target. Holding 75/25
would need a rebalanced `ov7` share dict on shard 3 — a new suite entry, since
the share dict *is* the deal — trading roughly 4,000 t2i pairs for img2img.
**Do not spend on this until U9 is scored**; nothing currently knows which
mix trains a better detector.

## U7. What has to change in the code

Most of the machinery this plan needs already exists — `SUITES`,
`SUITE_EXTENDS`, `--suite`, `size_multiple`, `call_kwargs`, `companion_ids`.
What is missing:

1. A `ModelSpec` for `qwen_image_2` (+ its `size_multiple`, and
   `companion_ids` if the text encoder lives in a second repo), and a refused
   entry for `zimage_turbo`.
2. A third suite in `SUITES` extending `ov7` + `ov7_lineage`, for the Qwen arm.
3. `enable_model_cpu_offload()` in `load()`. Today it only calls
   `enable_sequential_cpu_offload()`, the slow per-submodule variant; the
   module-level one is what puts a 7B + text encoder inside 24 GB.
4. `--run-families`, subsetting `sel` *after* `select()`. Both boxes must pass
   an identical `--families`/`--suite` or the strata pattern shifts; this
   separates what defines the deal from what a process executes, and is what
   lets one shard span heterogeneous hardware.
5. Remap `path` when a cached pool is loaded with `--portrait-dir`. The parquet
   carries absolute paths and `run_family` reads `row.path` directly.
6. Record `dtype` and `gpu` in the rows jsonl. Neither is written today, so a
   corpus generated across mixed hardware cannot be stratified afterwards.
7. Precompute captions. `caption_pool` rewrites the whole parquet on every log
   tick, so concurrent workers on one path lose captions to last-writer-wins.

## U8. Hardware, rates and cost

One box: **4x RTX 4090, 24 GB, ~$1.70/hr.** 24 GB clears Qwen-2.0 with
model-level offload and klein-4B's 15.6 GB peak; Ada is bf16-native, so nothing
changes dtype against the A4500's existing rows. Two non-obvious constraints:
do not rent in mainland China (33 GB of weights come from HuggingFace, and
`check_licence()` hits the Hub before any weights load), and prefer Ada over
Blackwell unless the torch/diffusers stack is known-good on sm_120. 16 GB cards
are false economy — klein-4B alone does not fit.

A4500 figures are measured (`ai_ov7_generation.md` §6 and §11's smoke run). The
4090 column is those divided by 2.7 — the 3.0x benchmark ratio derated 15%
because 416x640 underutilises the card. **The Qwen row is an estimate against a
model never run here, and it is half the budget**; smoke it before trusting the
schedule.

| family | A4500 s/img | 4090 s/img | new | GPU-h |
|---|---:|---:|---:|---:|
| `qwen_image_2_t2i` | — | ~2.0 est | 10,000 | 5.56 |
| `kandinsky22_t2i` | 2.07 | 0.77 | 9,463 | 2.02 |
| `sana1600m_t2i` | 1.17 | 0.43 | 9,462 | 1.13 |
| `sdxl_t2i` | 2.15 | 0.80 | 3,271 | 0.73 |
| `klein4b_ref_image` | 6.62 | 2.45 | 654 | 0.45 |
| `klein4b_t2i` | 4.03 | 1.49 | 1,308 | 0.54 |
| `sd15_t2i` | 1.79 | 0.66 | 2,198 | 0.40 |
| `sdxl_self_cond` | 2.18 | 0.81 | 1,526 | 0.34 |
| `sdxl_img2img` | 1.67 | 0.62 | 1,090 | 0.19 |
| `sd15_img2img` | 1.43 | 0.53 | 877 | 0.13 |
| | | | **39,849** | **11.49** |

Captioning adds ~0.5 GPU-h at ~0.045 s/image. **~12 GPU-h, ~3 h across four
GPUs, ~$5-8 including setup.** The Qwen arm is the long pole; split it across
two GPUs with `--run-families`.

## U9. Still owed, and now blocking a spending decision

§3.1 says to record the conditioning type "because a later ablation will want
to score them separately." **That ablation has never been run**, and it is not
in `ai_ov7_generation.md` §12 either. So §3.1's 70/30 target is inherited
judgment, and it is in direct tension with §11's t2i-only argument for the
supplement families. Those cannot both be right, and U6 currently resolves the
tension by ignoring 70/30 — it lands at 90/10.

The frozen corpus already carries 6,983 t2i / 1,400 self_cond / 1,795 img2img /
600 ref_image pairs. **Score the conditioning ablation on those before buying
40,000 more at an assumed mix.** It costs no GPU time and it is owed anyway.

Worth noting for whoever runs it: by this corpus's own logic — hold everything
fixed except the generator — `self_cond` is the sharpest arm, not t2i. It holds
content near-identical (pHash 0-2) while generating every pixel. A common
misreading is that img2img leaves real pixels in place; it does not. SDEdit
encodes, noises, denoises and decodes, so every output pixel is decoder output.
That is true of *inpainting*, which composites unmasked regions back, and is
why `inpaint_box` was refused and `self_cond` built instead.

## U10. Running it

Not derivable from the sections above; these are the mechanics a cold start
would otherwise re-discover the hard way.

**On the box, before anything.** `jpegtran` is a hard `ap.error` at startup,
not a soft dependency — it is the only way to crop a real without re-encoding
it. `apt-get install -y libjpeg-turbo-progs`. Without it all four workers die
before a single model loads.

**Stage onto the box:** the 60,000 portrait JPEGs (~5 GB), `attribution.csv`,
`ov7_pool.parquet`, and a **precomputed** `ov7_captions.parquet`. Captions must
be precomputed because `caption_pool` rewrites the whole parquet on every log
tick — four workers on one path lose captions to last-writer-wins. Caption to
four separate paths, merge, then pass the merged file read-only.

**One process per GPU.** Shared `--out` is safe: image ids are disjoint across
shards, so no two workers write the same file. **Separate `--rows-dir` is
not optional** — `run_family` appends one jsonl per family with prompts of
unbounded length, and four processes interleaving those writes will corrupt
lines. Concatenate them at the end for `pairs.parquet`.

**Share `HF_HOME`** or each worker downloads the same 33 GB.

```bash
for k in 2 3 4; do
  case $k in
    2) SUITE=ov7_qwen ;;
    3) SUITE=ov7 ;;
    4) SUITE=ov7_lineage ;;
  esac
  CUDA_VISIBLE_DEVICES=$((k-2)) HF_HOME=/workspace/hf \
  python scripts/generate_ov7.py --suite $SUITE --total 10000 \
      --shard $k --n-shards 5 \
      --pool /workspace/ov7_pool.parquet \
      --portrait-dir /workspace/portrait \
      --captions /workspace/ov7_captions.parquet \
      --out /workspace/raw_ov7_src --rows-dir /workspace/_rows_$k &
done
```

The two resume runs (shard 0 `ov7`, shard 1 `ov7_lineage`) need the existing
`_rows/*.jsonl` and output tree staged first — `_done_ids` checks the jsonl
*and* both files on disk, so it skips what exists and generates only the tail.

**Do not change `--seed`, or any suite's shares.** The strata pattern is a
function of the share dict; a different one re-deals every real in the block.
`used_elsewhere` will refuse the run, but the cheaper path is not to try.

Afterwards: `scripts/gate_confounds.py --n 4000` before anything trains, per
§5. A gate that comes back worse than `ai_ov7_generation.md` §10's 9-family
reading (0.5152 / 0.5632 / 0.5072 / 0.5015) is a result, not a nuisance.
