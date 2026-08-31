# Task 3 — The commercial-API held-out set: handover

**Branch `feat/commercial-api-image-generation`. As of 2026-08-31.**
**Spent to date: $0.32, all of it wire tests and price probes. No usable image
exists yet.** The brief is `docs/03-commercial-apis-on-open-images.md`; the
per-provider terms, with quotes and URLs, are in `docs/dataset_licences.md`.
This file replaces the earlier `03a` costing, `03b` status report and `03c`
runbook.

Read one paragraph before running anything: **these images are a held-out test
set and must never enter training.** That is the entire point of the task. A
good score here that turns out to be leakage is worse than no score at all.

---

## 1. What this is

A small held-out test set of images from commercial generators, paired to Open
Images V7 reals by `ImageID`, and **never trained on**. It answers one question:
does A3's 0.8952 TPR at 1% FPR generalise to generators we have never seen, or
have we fitted to one generator's fingerprint?

**A low score is the expected result and is the deliverable.** A high one should
be disbelieved and checked for leakage before anyone reports it.

Not to be confused with **task 02** (branch `feat/ai-ov7-generation`), which
generates ~60,000 *training* fakes with free open-weight models. Same reals,
same pairing method, same encoder parity — opposite purpose. Task 02 teaches the
detector; task 3 grades it.

---

## 2. Status

**Built and verified against the live APIs** (one image per provider,
2026-08-30):

| Piece | File | State |
|---|---|---|
| Encoder parity — copies a real's exact JPEG tables onto its fake | `src/aigcdet/data/encoder_parity.py` | Working. `jpeg_quality` and `short_side` both collapse to 0.5000 on real thumbnails |
| The pre-spend gate | `scripts/prove_encoder_parity.py` | Working; scores both geometries |
| Prompt building (narrative + VLM caption, §3.5 enforced) | `scripts/build_prompts.py` | Working. 54 usable prompts from 60 reals |
| The buyer, four providers | `scripts/pilot_commercial_apis.py` | Working end to end. Dry-run by default |
| Real harvest at `--split validation` | `scripts/acquire_open_images_portrait.py` | Working — 15 MB of metadata instead of 2.7 GB |
| Terms, per provider | `docs/dataset_licences.md` | Three of four read at source; Seedream has none |
| Tests | `tests/data/test_encoder_parity.py`, `tests/scripts/test_pilot_commercial_apis.py` | Passing |

**Not built. Whoever picks this up owns these:**

1. **The pilot has not been run.** No generated image exists.
2. **The source is not registered.** There is no entry in
   `src/aigcdet/data/sources.py`, so nothing yet sets
   `exclude_from_training=True` and no test asserts these rows stay out of
   `train`/`val_internal`. That is acceptance criterion §5.1, and it is the one
   that protects the entire result. **Do it before the images exist.**
3. **The eval real half is not harvested** (criterion §5.5). Free — a download,
   not a purchase.
4. **No dedup check** against the training corpus, reals included.
5. **No inpainting arm.** The brief mandates 70/30 text-to-image / inpainted;
   this buy is 100% text-to-image, deliberately (§3.3).

**Assets so far:** `~/techjam_task03/` — 60 reals + `attribution.csv`, 60
narrative prompts, 54 VLM prompts, wire-test images. 29 MB, not in the repo.

---

## 3. The decision: what to buy

**Four providers at 194 reals each — 776 images for $34.90.** OpenAI
`gpt-image-2` · Google Gemini 3.1 Flash Image · Bytedance Seedream 4.5 ·
Ideogram 4.0 Turbo. No cheap-tier substitution.

These are four of the six vendors **NTIRE 2026** (arXiv 2604.11487) holds out of
training and uses only in validation and test — the best available proxy for
what a graded benchmark contains. Their list reshaped ours: Recraft appears in
none of their five held-out splits, Seedream in four, so Recraft was swapped out
and is now the reserve. They train on FLUX.1 and hold out FLUX-2 Max, which
means the field applies the **decoder-weights** test rather than the *lab* test.

Excluded, and why: **Stability** (the API serves SD 3.5, which task 02 puts in
training by name), **Black Forest Labs** (takes a *"perpetual, irrevocable,
worldwide"* licence to inputs and outputs and says it may train on them — a
terms objection, not a lineage one), **Midjourney** (no public API).

### 3.1 Prices — measured, never listed

| Provider | Model | $/image | Verified by |
|---|---|---:|---|
| Ideogram 4.0 Turbo | `V_4_TURBO` @ 1664×2496 | **0.0300** | vendor dashboard |
| Bytedance Seedream 4.5 | `bytedance-seed/seedream-4.5` @ 2K | **0.0400** | `usage.cost` |
| OpenAI gpt-image-2 | `openai/gpt-image-2` @ 1K medium | **0.0415** | `usage.cost` |
| Google Gemini 3.1 Flash Image | `google/gemini-3.1-flash-image` | **0.0684** | `usage.cost` |
| Captioner | `qwen/qwen3-vl-8b-instruct` | **0.000093** | `usage.cost` |

**OpenRouter's listed per-image price is a floor, not the charge.** Grok Imagine
2.0 listed $0.01 and **billed $0.0600**; Seedream 5 Pro listed $0.003 and
**billed $0.0450**; Seedream 4.5 lists `image=None` and bills $0.04. Only a real
call settles a price. (Meta Muse Image is blocked behind an 18+ age
confirmation.)

**The budget.** S$50 at 0.786 = **$39.30**, less the $0.32 already spent =
**$38.98 to spend**.

| | Cost | Running |
|---|---:|---:|
| Harvest ~216 reals → 194 usable prompts, plus VLM captioning | $0.02 | $0.02 |
| Pilot: 50 reals × 4 providers | $9.00 | $9.02 |
| Gate on `jpeg_quality` AUC, both geometries — pick on the number | $0 | $9.02 |
| Continue the same command to 194 reals (resumes, re-buys nothing) | $25.91 | **$34.93** |
| **Contingency left** | | **$4.05 (12%)** |

12% against the 15% retry budget and the **0% provider refusal rate the wire
test actually observed**. If the pilot measures zero again, the contingency
converts to ~22 more reals per family.

For contrast: a *training* corpus matching task 02's 60,000 reals would cost
**~$2,700** at these prices. Buy the held-out set, never the training set.

### 3.2 Why four families rather than three at a higher count

The brief's §3.4 says that when the total is over budget the lever is "fewer
providers at full count". At S$50 that inverts, and the arithmetic is why —
planning against $35.00:

| Config | $/real | Reals | Images | SE on per-provider TPR (p≈0.2 / 0.5) |
|---|---:|---:|---:|---|
| **A — 4×: OpenAI, Google Flash, Seedream, Ideogram** | 0.1799 | **194** | **776** | **2.9 / 3.6 pt** |
| C — 3×, drop Google | 0.1115 | 313 | 939 | 2.3 / 2.8 pt |
| E — 5×, add Recraft V4.1 | 0.2153 | 162 | 810 | 3.1 / 3.9 pt |
| F — 2×: OpenAI, Ideogram | 0.0715 | 489 | 978 | 1.8 / 2.3 pt |

The budget buys 750–980 images whatever the shape, because the four prices sit
within 2.3× of each other. All that changes is how they are split — and **every
split is precise enough**: the per-provider standard error runs 1.8–3.9pt
against the ~40pt effects expected. Precision stopped binding somewhere below
n=200, so the money buys the thing that is actually scarce, which is **decoder
lineages**, not counts.

Config E tempts, and is declined on two non-statistical grounds: Recraft's terms
are unread, and it is in none of NTIRE's held-out splits. Config F is the
reverse error — the tightest error bars in the table on the fewest lineages, a
precise answer to a question nobody asked.

### 3.3 What this buy deliberately does not do

* **No inpainting arm.** The 30% would split each family into cells of ~136 and
  ~58, bill the input image on top of the output at the edit endpoints, and need
  four more adapters written and wire-tested.
* **776 images against the brief's 2,000–5,000.** Below spec, on the §3.2
  argument. State it as a deviation rather than glossing it.
* **The real half is free, so make it large.** The 1% FPR threshold is set on
  reals and shared across all four providers, so its uncertainty comes down for
  the cost of a download rather than a card charge.

> **Do not quietly swap Google onto its Flash Lite tier to buy more images.**
> It looks like +46 reals per family for nothing. It is not: a cheap tier
> produces *different artefacts*, so it benchmarks the wrong thing; the Lite
> price (~$0.0342) is a halved list price that has **never been measured**, and
> the lesson above is that listed prices are floors; and the precision gain is
> 2.9pt → 2.6pt. There *is* one real argument for it — NTIRE's own validation
> split holds out **ImageGen-4 Fast**, so a fast tier is not off-composition. If
> the extra images are wanted, the order is: spend ~$0.04 measuring Lite's true
> billed price, write the NTIRE-Fast rationale down, *then* swap.

---

## 4. Terms, per provider — the operative facts

Full quotes and URLs are in `docs/dataset_licences.md`. What matters
operationally:

| Provider | Verdict | Redistributable? |
|---|---|---|
| **OpenAI** | Clean. Services Agreement §4.1 assigns Output to us; §4.2 commits not to train on submissions | **Yes** |
| **Google** | Clean on licence; **SynthID in every pixel**, no opt-out | **Yes** |
| **Ideogram** | Permitted, and §2.2 is the strongest no-training commitment of the four — but no explicit IP assignment, and §2.3.1(a) makes attribution **mandatory** | **Local-only** |
| **Seedream** | **No terms document exists** for the image models. BytePlus publishes specific terms for its *video* models only; OpenRouter §5.1/§6.1 bind us to upstream terms it does not supply | **Local-only** by precaution |

**Three consequences to carry into the writeup:**

* **Only two of the four families can ever be published.** If the deliverable
  ships a Kaggle Dataset, it is OpenAI + Google and says so on its face.
* **Seedream does not satisfy acceptance criterion §5.4.** The owner reviewed
  this and chose to proceed on 2026-08-30 rather than lose the most
  benchmark-representative generator on the list; the mitigation is local-only,
  never redistributed. That is a legitimate call, but the writeup must state it
  rather than presenting four cleanly-licensed providers.
* **OpenAI's Permitted Exception constrains the deliverable, not the data.** A
  classifier is permitted *"if these models are not distributed or made
  commercially available to third parties."* Check that before publishing model
  weights or an inference bundle.

**Google's SynthID is a measurement problem, not a licence problem.** The Gemini
API docs state *"All generated images include a SynthID watermark"*, and
DeepMind describes it as built to survive cropping, filters and lossy
compression — i.e. exactly the 20-condition grid these images are scored under.
If the detector scores *well* on Google, we cannot tell generation artefacts
from watermark. Keep Google, **flag its row**, and never pool it.

---

## 5. Encoder parity — the gate that clears before any spend

The reals are `Thumbnail300KURL` thumbnails, i.e. **re-encoded JPEGs**.
Generators return PNG. Store the two side by side and *"has JPEG artefacts"*
separates the classes perfectly, without a detector learning anything that
transfers — this project has already measured that leak at `jpeg_quality` AUC
0.5532 (`docs/low_level_confounds.md`).

Rather than re-encoding fakes "at a similar quality", `encoder_parity.py`
**copies the real's 64 quantisation integers verbatim**. Recovering a quality
number from a table means inverting the standard scaling, and that inversion is
lossy — the residual would be a systematic, label-correlated offset in the exact
statistic being gated, i.e. the confound reintroduced inside its own fix.

**Task 02's fix does not transfer.** It clears this gate at AUC 0.5031 by never
resampling: a local model is told to render at the real's exact size. An API
renders into fixed buckets and hands you what it hands you, so something must
close the gap to a 428×639 real. Two candidates, and the pilot settles it on a
number rather than an argument:

* `resample` — keeps the whole frame, pays a resampling signature.
* `crop` — resamples nothing, pays field of view: a 428×639 window out of a
  1024×1536 frame is a fragment of the scene, while the graded benchmark holds
  whole frames.

Both are derived from the same purchased bytes, so the choice costs one buy, not
two. NTIRE's published method is weaker than this — they *"align distributions
of resolutions, aspect ratios, JPEG compression quality factors"*; we match each
fake to its own partner exactly.

---

## 6. The runbook

Paths assume `~/techjam_task03`. Nothing here belongs in the repo.

### Before you start

```bash
cp .env.example .env      # then fill it in; .env is gitignored
source .env
```

| Key | Buys | Where |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenAI, Google, Seedream — and the captioner | <https://openrouter.ai/keys> |
| `IDEOGRAM_API_KEY` | Ideogram only; it is not on OpenRouter | <https://developer.ideogram.ai> |

**Ideogram is pre-paid**, returns HTTP 402 with no balance, and — this cost real
money — **bills the generation before you fetch the image**, so a failed
download is a charge for nothing. `call_ideogram` now sends a real User-Agent
(its CDN 403s Python's default) and records a failed fetch as billed.

Every script is **dry-run by default** and needs no key until `--execute`, so
the whole chain can be rehearsed for free.

### Step 1 — Harvest the reals (free)

```bash
python scripts/acquire_open_images_portrait.py \
    --out ~/techjam_task03 --split validation --target 216 --threads 16
```

216 because 194 usable prompts survive §3.5's name filter at the measured 90%
yield. `--split validation` is not cosmetic: task 02 harvests from `train`, so
this makes the two disjoint by construction and satisfies half of criterion §5.5
for free.

Leaves `portrait/<ImageID>.jpg` and `attribution.csv`. **Keep
`attribution.csv`** — it is the CC BY 2.0 licence receipt for the authentic
half.

### Step 2 — Build the prompts (~$0.02)

```bash
python scripts/build_prompts.py \
    --reals ~/techjam_task03/portrait \
    --out ~/techjam_task03/prompts.csv \
    --source both --n 216 --execute
```

`--source both` hands the captioner the image *and* Open Images' human-spoken
Localized Narrative. NTIRE captions with a vision model alone; we have a human
description of that exact photograph, so this is strictly more information.
`--source narrative` (free) and `--source vlm` (NTIRE's method exactly) exist so
the three can be compared rather than argued about.

`prompts.csv` is the reproducible artifact, **not** the model call — temperature
0 is not fully deterministic, and one refused caption passed on a re-run. Never
regenerate it mid-buy.

Rows refused here by §3.5 (a proper noun in the narrative ⇒ a possibly
identifiable real person) cost nothing and are **not** provider refusals — do
not let them inflate the retry budget.

### Step 3 — Dry-run the buy (free, no key needed)

```bash
python scripts/pilot_commercial_apis.py \
    --reals ~/techjam_task03/portrait --out ~/techjam_task03/pilot --n 50
```

Prints the plan, the per-provider estimate and an example prompt. Sends nothing.

### Step 4 — The pilot: 50 reals × 4 providers ≈ $9.00

```bash
python scripts/pilot_commercial_apis.py \
    --reals ~/techjam_task03/portrait --out ~/techjam_task03/pilot \
    --n 50 --execute --yes-spend
```

Drop `--yes-spend` to be prompted to type `spend` interactively.

**This is the first 50 reals of the full buy, not a separate purchase** — raw
bytes are written first and the run resumes on their existence, so step 6 picks
up where this stopped and re-buys nothing. It answers three questions at once:
the real per-image cost, the per-provider refusal rate, and the geometry
question.

### Step 5 — The gate. Nothing proceeds until this passes

```bash
for g in resample crop; do
  for fam in openai_gpt_image_2 google_gemini_31_flash_image \
             bytedance_seedream_45 ideogram_40_turbo; do
    python scripts/prove_encoder_parity.py \
        --reals ~/techjam_task03/portrait \
        --generated ~/techjam_task03/pilot/raw/$fam \
        --out ~/techjam_task03/gate_${g}_${fam} \
        --geometry $g --n 50
  done
done
```

**Pass condition: pixel-only `jpeg_quality` AUC after parity below 0.60**, per
provider. Read the pixel-only column, not the path-aware one — once both classes
are JPEGs with identical quantisation tables the path-aware figure is 0.5 by
construction, so gating on it gates on a tautology.

Watch Google and Seedream especially: **both return JPEG already**, so their
rows carry an extra compression generation before parity.

If no geometry clears 0.60 for a provider, **stop and say so.** The brief's §6
fallback (an open-weight held-out lineage — GPU time instead of money) is the
designed exit, and a failed gate is a publishable result rather than a setback.

### Step 6 — The rest of the buy: 194 reals × 4 ≈ $25.91 more

```bash
python scripts/pilot_commercial_apis.py \
    --reals ~/techjam_task03/portrait --out ~/techjam_task03/pilot \
    --n 194 --execute --yes-spend
```

Same command, larger `--n`. Record the refusal rates printed by step 4 first.

### Step 7 — Wire it in (not yet built)

1. Register the source in `src/aigcdet/data/sources.py` with
   `exclude_from_training=True`, **one family name per provider** — never a
   single `commercial_api` bucket, or the report cannot say which provider we
   fail on.
2. Add a test asserting no row from this source reaches `train` or
   `val_internal`.
3. Harvest the eval real half from `ImageID`s in no training split and **not**
   among the 216 partner reals — the brief's §2.2 forbids putting a fake and its
   own partner real in the same eval split. Free, so make it several thousand.
4. Run the dedup check against the training corpus, **reals included**.
5. Add to the **eval** manifest only.

---

## 7. What lands on disk

```
~/techjam_task03/
├── portrait/<ImageID>.jpg              the reals, free
├── attribution.csv                     CC BY 2.0 receipt, one row per real
├── prompts.csv                         image_id,narrative,prompt,source,captioner,ts
└── pilot/
    ├── raw/<family>/<ImageID>.{png,jpg}    ← the ONLY artifact that cost money
    ├── parity_resample/<family>/<ImageID>.jpg
    ├── parity_crop/<family>/<ImageID>.jpg
    └── pilot_receipt.jsonl                 per-image cost, ok, refusal reason, prompt
```

**Nothing may ever write into `raw/`.** It holds exactly what the provider
returned; every derived form is reproducible from it for free, so a geometry or
parity bug is re-run offline rather than re-bought. This is also what makes
resume safe.

`pilot_receipt.jsonl` is the audit trail for the money and the refusal rate.
Keep it with the images.

---

## 8. Four bugs the $0.15 wire test caught

Any of these would have corrupted a full run. They are fixed; they are recorded
because they are the shape of what goes wrong here.

1. **Seedream refuses anything under 3,686,400 output pixels.** 1K at 2:3 is
   700,416 and returns HTTP 400. It is asked for 2K, and $0.04 is the 2K price.
2. **Ideogram v4** names the field `text_prompt`, takes multipart/form-data
   rather than JSON, and selects geometry from a resolution enum starting at
   2048×2048.
3. **Ideogram's CDN rejects Python's default User-Agent with HTTP 403.** The
   generation is billed before the fetch, so this was paying for images and
   silently discarding them — one of the two Ideogram charges bought nothing.
4. **`output_format: png` is a request, not a promise.** Google returns JPEG for
   it, and a JPEG saved as `.png` misstates the one property this task measures.
   The extension now follows the returned media type.

---

## 9. Rules that outlive this handover

* **Per-provider, never pooled.** Report TPR at 1% FPR separately for each of
  the four. Pooling hides the provider we are worst at, which is the number that
  matters.
* **Flag Google's row** for SynthID so a reader can discount it without knowing
  the provider's defaults.
* **Score under the full 20-condition degradation grid**, like every other eval
  row. A clean-only number is not comparable to anything else we report.
* **Ideogram and Seedream images stay local.** Never in a shared dataset.
* **A low score is the deliverable.** Report it as the headline caveat next to
  the benchmark number, not in a footnote.

### What to expect from these four

They are four brands over **possibly two architecture families**: OpenAI and
Google are likely LLM-native autoregressive, Seedream and Ideogram diffusion
transformers. Only Ideogram's is verified, because they published weights; the
other three are inference from brand and press.

If per-provider TPRs cluster into those pairs, that is the finding, and it is
more interesting than the average. But this set **cannot** answer *"which
architectures do we fail on"* — a failure cannot be attributed to an
architecture nobody outside those labs can identify. That question belongs to
task 02, where the decoder is knowable, and is worth raising there: its four
models are all latent diffusion with a VAE, missing the autoregressive (Janus,
Infinity) and pixel-space (DeepFloyd IF) families NTIRE trains on.

---

## 10. Open items

* **Seedream's terms.** Registering at BytePlus ModelArk puts the binding terms
  in front of you at signup. If that yields a document, quote it into
  `dataset_licences.md`. Recraft is the fallback if they turn out to bar the use.
* **A premise in the brief was wrong, and the correction is worth keeping.** It
  assumed the reals are *"overwhelmingly photographs of people"* and built a 20%
  retry budget on it. **"Portrait" is an aspect-ratio filter, not a subject
  filter** — measured over the 60-image harvest, 47% mention a person at all and
  only **7% are face-centric**; the rest are food, flowers, animals, machinery.
  Retry budget cut to 15%, and the pilot should cut it further.
* **SDXL-Turbo's licence, for whoever owns task 02.** Its brief listed "SDXL 1.0
  / SDXL-Turbo | CreativeML OpenRAIL++-M | Use". Turbo is `sai-nc-community`,
  **non-commercial**. Found by dionyichia; that brief is not on this branch, so
  the correction has to be carried across by hand.
* **Open weights erode the "unseen" premise over time.** Ideogram 4.0 (released
  2026-06-03, nf4/fp8 quantised, non-commercial model agreement) and FLUX.2-dev
  both ship weights, so they can appear in anyone's training corpus. Do not run
  them locally to save money for *this* purchase — the public releases are
  quantised and do not reproduce the artefacts of the provider's full-precision
  serving stack, which is the thing being benchmarked. But treat a result on an
  open-weight generation as weaker evidence than one on a closed-serving
  generation of the same age.

---

## Sources

Read at the vendor on 2026-08-30 unless marked otherwise.

* **Google** — [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) · [API terms](https://ai.google.dev/gemini-api/terms) · [image generation docs](https://ai.google.dev/gemini-api/docs/image-generation) (the "All generated images include a SynthID watermark" statement — cite this, not the DeepMind page, which carries only the robustness language) · [SynthID](https://deepmind.google/models/synthid/)
* **OpenAI** — [ToU](https://openai.com/policies/row-terms-of-use/) and [Services Agreement](https://openai.com/policies/services-agreement/), both effective 2026-01-01, **retrieved by hand — both 403 to automated fetch**. The API is governed by the Services Agreement, not the consumer ToU. · [image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
* **Ideogram** — [API ToS](https://ideogram.ai/legal/api-tos) (§2.2, §2.3.1(a), §2.3.6(A)) · [pricing](https://ideogram.ai/pricing/?pricing_tab=api) — rates load by script; $0.03 Turbo confirmed at the vendor dashboard · [4.0 weights](https://huggingface.co/collections/ideogram-ai/ideogram-4)
* **Bytedance** — [Seedream 4.5 on OpenRouter](https://openrouter.ai/bytedance-seed/seedream-4.5); Bytedance's own image-model terms **do not exist publicly**
* **Recraft** (reserve) — [API pricing](https://www.recraft.ai/pricing?tab=api): V4.1 raster $0.03535, raster inpaint $0.0440. Terms unread.
* **Black Forest Labs** (excluded) — [FLUX.2](https://bfl.ai/blog/flux-2) · [API service terms](https://bfl.ai/legal/flux-api-service-terms)
* **NTIRE 2026** — arXiv 2604.11487, the held-out composition this provider set matches
