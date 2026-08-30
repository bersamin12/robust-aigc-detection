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
