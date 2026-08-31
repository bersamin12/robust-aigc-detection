# 03c — Handover: how to build the commercial-API held-out set

**Branch `feat/commercial-api-image-generation`.** This is the runbook. The
brief is `docs/03-commercial-apis-on-open-images.md`, the costing is
`docs/03a-commercial-api-costing.md`, and the decision record — what was
measured, what it cost, why these four providers — is
`docs/03b-pilot-status-report.md` §7.

Read one paragraph before running anything: **these images are a held-out test
set and must never enter training.** That is the entire point of the task. A
good score here that turns out to be leakage is worse than no score.

---

## 1. What is done, and what is not

**Done and verified against the live APIs** (one image per provider, 2026-08-30):

| Piece | File | State |
|---|---|---|
| Encoder parity — copy a real's exact JPEG tables onto its fake | `src/aigcdet/data/encoder_parity.py` | Working. `jpeg_quality` and `short_side` both collapse to 0.5000 on real thumbnails |
| The pre-spend gate | `scripts/prove_encoder_parity.py` | Working, scores both geometries |
| Prompt building (narrative + VLM caption, §3.5 enforced) | `scripts/build_prompts.py` | Working. 54 usable prompts from 60 reals |
| The buyer, four providers | `scripts/pilot_commercial_apis.py` | Working end-to-end for all four. Dry-run by default |
| Real harvest at `--split validation` | `scripts/acquire_open_images_portrait.py` | Working |
| Terms, per provider | `docs/dataset_licences.md` | Three of four read at source; Seedream has none — see §5 |
| Tests | `tests/data/test_encoder_parity.py`, `tests/scripts/test_pilot_commercial_apis.py` | Passing |

**Not done. Whoever picks this up owns these:**

1. **The pilot has not been run.** $0.32 spent to date, all of it wire tests and
   candidate probes. No usable image exists yet.
2. **The source is not registered.** There is no entry in
   `src/aigcdet/data/sources.py`, so nothing yet sets
   `exclude_from_training=True` and no test asserts these rows stay out of
   `train`/`val_internal`. That is acceptance criterion §5.1 and it is the one
   that protects the whole result. **Do it before the images exist**, not after.
3. **The eval real half is not harvested.** Criterion §5.5 needs reals whose
   `ImageID`s appear in no training split *and* are not the partner reals behind
   these prompts. Free — it is a download, not a purchase.
4. **No dedup check** against the training corpus (criterion §5.5).
5. **No inpainting arm.** The brief mandates 70/30 text-to-image / inpainted;
   this buy is 100% text-to-image, deliberately (`03b` §7.5). State it.

---

## 2. Before you start

```bash
cp .env.example .env      # then fill it in; .env is gitignored
source .env
```

Two keys cover everything:

| Key | Buys | Where |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenAI, Google, Seedream — and the captioner | <https://openrouter.ai/keys> |
| `IDEOGRAM_API_KEY` | Ideogram only; it is not on OpenRouter | <https://developer.ideogram.ai> |

**Ideogram is pre-paid.** It returns HTTP 402 with no balance, and — this cost
real money — **it bills the generation before you fetch the image**, so a failed
download is a charge for nothing. `call_ideogram` now sends a real User-Agent
(its CDN 403s Python's default) and records a failed fetch as billed.

Every script is **dry-run by default** and needs no key until `--execute`. You
can rehearse the entire chain for free.

---

## 3. The pipeline

Paths below assume `~/techjam_task03`. Nothing here belongs in the repo.

### Step 1 — Harvest the reals (free)

```bash
python scripts/acquire_open_images_portrait.py \
    --out ~/techjam_task03 --split validation --target 216 --threads 16
```

216, because 194 usable prompts survive §3.5's name filter at the measured 90%
yield (6 of 60 refused locally on the first harvest). `--split validation` is
not cosmetic: task 02 harvests from `train`, so this makes the two sets disjoint
by construction and satisfies half of criterion §5.5 for free.

Leaves `~/techjam_task03/portrait/<ImageID>.jpg` and `attribution.csv` — one CC
BY 2.0 attribution row per image. **Keep `attribution.csv`.** It is the licence
receipt for the authentic half.

### Step 2 — Build the prompts (~$0.02)

```bash
python scripts/build_prompts.py \
    --reals ~/techjam_task03/portrait \
    --out ~/techjam_task03/prompts.csv \
    --source both --n 216 --execute
```

`--source both` hands the captioner the image *and* Open Images' human-spoken
Localized Narrative. NTIRE 2026 captions with a vision model alone; we have a
human description of that exact photograph, so this is strictly more
information. `--source narrative` (free, no model) and `--source vlm` (NTIRE's
method exactly) exist so the three can be compared rather than argued about.

`prompts.csv` is the reproducible artifact, **not** the model call — temperature
0 is not fully deterministic and one refused caption passed on a re-run. Commit
the file to your run directory and never regenerate it mid-buy.

Rows refused by §3.5 (a proper noun in the narrative ⇒ a possibly identifiable
real person) are dropped here, before any spend, and are **not** provider
refusals — do not let them inflate the retry budget.

### Step 3 — Dry-run the buy (free, no key needed)

```bash
python scripts/pilot_commercial_apis.py \
    --reals ~/techjam_task03/portrait --out ~/techjam_task03/pilot --n 50
```

Prints the plan, the per-provider estimate, and an example prompt. Sends
nothing. Read the estimate before you go further.

### Step 4 — The pilot: 50 reals × 4 providers ≈ $9.00

```bash
python scripts/pilot_commercial_apis.py \
    --reals ~/techjam_task03/portrait --out ~/techjam_task03/pilot \
    --n 50 --execute --yes-spend
```

Drop `--yes-spend` to be prompted to type `spend` interactively.

This is **the first 50 reals of the full buy, not a separate purchase** — raw
bytes are written first and the run resumes on their existence, so step 6 picks
up exactly where this stopped and re-buys nothing.

It answers three questions at once: the real per-image cost (from the
provider's own `usage.cost`, not a price list), the per-provider refusal rate,
and — because both geometries are derived from the same bytes — the geometry
question, without buying twice.

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

Then **pick the geometry on the number, not on argument** (`docs/03` §3.1):

* `resample` keeps the whole frame and pays a resampling signature.
* `crop` resamples nothing and pays field of view — a 428×639 window out of a
  1024×1536 frame is a fragment of the scene, while the graded benchmark holds
  whole frames.

Watch Google and Seedream especially: both return JPEG already, so their rows
carry an extra compression generation before parity.

If no geometry clears 0.60 for a provider, **stop and say so.** The brief's §6
fallback (an open-weight held-out lineage, GPU time instead of money) is the
designed exit, and a failed gate is a publishable result rather than a
setback.

### Step 6 — The rest of the buy: 194 reals × 4 ≈ $25.91 more

```bash
python scripts/pilot_commercial_apis.py \
    --reals ~/techjam_task03/portrait --out ~/techjam_task03/pilot \
    --n 194 --execute --yes-spend
```

Same command, larger `--n`. Feed the refusal rates printed by step 4 into
`docs/03a` §3 first, then commit.

### Step 7 — Wire it in (not yet built)

1. Register the source in `src/aigcdet/data/sources.py` with
   `exclude_from_training=True`, **one family name per provider** — never a
   single `commercial_api` bucket, or the report cannot say which provider we
   fail on.
2. Add a test asserting no row from this source reaches `train` or
   `val_internal`.
3. Harvest the eval real half from `ImageID`s in no training split and not
   among the 216 partner reals (`docs/03` §2.2 — do not hand the scorer a
   matched pair). It is free, so make it several thousand: the 1% FPR threshold
   is shared across all four providers, so its uncertainty comes down for the
   cost of a download rather than a card charge.
4. Run the dedup check against the training corpus, **reals included**.
5. Add to the **eval** manifest only.

---

## 4. What lands on disk

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

## 5. Rules that outlive this handover

**Per-provider, never pooled.** Report TPR at 1% FPR separately for each of the
four. Pooling hides the provider we are worst at, which is the number that
matters.

**Google's rows carry SynthID.** Every Gemini image is watermarked in-pixel, no
opt-out, and DeepMind states it is built to survive crop, rescale and lossy
compression — i.e. the exact 20-condition grid these images are scored under. If
Google scores *well*, we cannot tell generation artefacts from watermark. Flag
its row in the results table so a reader can discount it.

**Two of the four families can never be published.** Ideogram grants no explicit
IP assignment and requires attribution on anything published; Seedream has no
terms document at all for its image models. Both stay **local-only**. If the
deliverable ships a Kaggle Dataset, it is OpenAI + Google and says so on its
face. The Seedream row does not satisfy criterion §5.4 and the writeup must say
that rather than presenting four cleanly-licensed providers.

**OpenAI's Permitted Exception constrains the deliverable, not the data.** A
classifier is permitted *"if these models are not distributed or made
commercially available to third parties."* Check that before publishing model
weights or an inference bundle.

**A low score is the expected result and is the deliverable.** A high one should
be disbelieved and checked for leakage before anyone reports it.

---

## 6. Money, at a glance

Measured against the live APIs on 2026-08-30, not read off a price list —
OpenRouter's listed price is a floor, not the charge (Grok listed $0.01 and
billed $0.06; Seedream 5 Pro listed $0.003 and billed $0.045).

| Provider | Model | $/image |
|---|---|---:|
| Ideogram 4.0 Turbo | `V_4_TURBO` @ 1664×2496 | 0.0300 |
| Bytedance Seedream 4.5 | `bytedance-seed/seedream-4.5` @ 2K | 0.0400 |
| OpenAI gpt-image-2 | `openai/gpt-image-2` @ 1K medium | 0.0415 |
| Google Gemini 3.1 Flash Image | `google/gemini-3.1-flash-image` | 0.0684 |
| Captioner | `qwen/qwen3-vl-8b-instruct` | 0.000093 |

**The budget: S$50 = $39.30, less $0.32 already spent = $38.98.**
194 reals × 4 providers = **776 images for $34.93**, leaving $4.05 (12%) for
retries against a 15% budget and a 0% observed refusal rate. `03b` §7 is the
argument for that shape — four lineages at n=194 beats three at n=313, because
the per-provider standard error is 2.9pt either way and lineages are the scarce
thing.

Two traps this buy is priced around:

* **Seedream refuses anything under 3,686,400 output pixels.** 1K at 2:3 is
  700,416 and returns HTTP 400. It is asked for 2K; $0.04 is the 2K price.
* **`output_format: png` is a request, not a promise.** Google returns JPEG for
  it, and a JPEG saved as `.png` misstates the one property this task measures.
  The extension follows the returned media type.

Do **not** quietly swap Google onto its Lite tier to buy more images. `03a` §3.4
forbids letting "cheapest" win by default — a cheap tier produces *different
artefacts*, so it benchmarks the wrong thing — the Lite price has never been
measured, and the precision gain is 2.9pt → 2.6pt. `03b` §7.4 sets out the one
condition under which the swap is defensible.
