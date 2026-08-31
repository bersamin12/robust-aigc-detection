# 03b — Task 03 status: what is built, what it cost, what is next

**As of 2026-08-30.** Branch `feat/encoder-parity-pilot`, 12 commits, 43 tests
passing. **Spent to date: $0.32. The pilot has not been run.**

**The buy is decided (§7): S$50, four providers — OpenAI `gpt-image-2`, Google
Gemini 3.1 Flash Image, Bytedance Seedream 4.5, Ideogram 4.0 Turbo — at 194
reals each, 776 images, $34.91.**

`docs/03-commercial-apis-on-open-images.md` is the brief,
`docs/03a-commercial-api-costing.md` is the costing, and
**`docs/03c-handover.md` is the runbook — the exact commands, in order.** This
file is the record: the state of the work against the brief, what was measured,
and the decisions still open.

---

## 1. What this task is, and what it is not

A small **held-out test set** of images from commercial generators, paired to the
same Open Images reals as task 02, and **never trained on**. It answers one
question: does A3's 0.8952 TPR at 1% FPR generalise to generators we have never
seen, or have we fitted to one generator's fingerprint?

A low score is the expected result and the deliverable — §1 of the brief is
explicit about that.

**Not to be confused with task 02** (`feat/ai-ov7-generation`), which generates
~60,000 *training* fakes with free open-weight models. Same reals, same pairing
method, same encoder parity — opposite purpose. Task 02 teaches the detector;
task 03 grades it.

---

## 2. What was built

| Component | Purpose |
|---|---|
| `src/aigcdet/data/encoder_parity.py` | Copies a real's exact JPEG quantisation tables, subsampling, progressive flag and pixel dimensions onto its paired fake |
| `scripts/prove_encoder_parity.py` | The brief's §3.1 gate. Scores each low-level proxy before/after parity, under either geometry |
| `scripts/build_prompts.py` | Localized Narratives + VLM captioning → generation prompts, with §3.5 enforcement |
| `scripts/pilot_commercial_apis.py` | The buyer. Dry-run by default, saves raw bytes first, derives both geometries |
| `scripts/acquire_open_images_portrait.py` | `--split validation` added: 15 MB of metadata instead of 2.7 GB |
| `docs/03`, `docs/03a`, `docs/dataset_licences.md` | Brief, costing, per-provider terms |

### The core idea: encoder parity

The reals are `Thumbnail300KURL` thumbnails — **re-encoded JPEGs**. Generators
return PNG. Store the two side by side and *"has JPEG artefacts"* separates the
classes perfectly, without a detector learning anything that transfers. This
project has already measured that leak at `jpeg_quality` AUC 0.5532
(`docs/low_level_confounds.md`).

Rather than re-encoding fakes "at a similar quality", the module **copies the
real's 64 quantisation integers verbatim**. Recovering a quality number from a
table means inverting the standard scaling, and that inversion is lossy — the
residual would be a systematic, label-correlated offset in the exact statistic
being gated, i.e. the confound reintroduced inside its own fix.

Verified on real Open Images thumbnails: `jpeg_quality` (path-aware) and
`short_side` both collapse to **exactly 0.5000**.

---

## 3. Findings that changed the plan

**NTIRE 2026 validates the design.** Their challenge (arXiv 2604.11487) holds
every proprietary API model out of training and uses them only in val/test, and
states they *"align distributions of resolutions, aspect ratios, JPEG
compression quality factors, and other statistics to those of the real subset."*
That one sentence is the entire published method. Ours is stronger: they match
distributions, we match each fake to its own partner exactly.

**Their provider list reshaped ours.** Recraft appears in none of their five
held-out splits; Seedream appears in four. Swapped.

**They train on FLUX.1 and hold out FLUX-2 Max**, so the field applies the
*decoder weights* test rather than the *lab* test. That settles the open
question in `03a` §1.1 and makes FLUX.2 eligible.

**Task 02's fix does not transfer.** Its plan (branch `feat/ai-ov7-generation`)
clears this gate at AUC 0.5031 by never resampling — a local model is told to
render at the real's exact size. An API renders into fixed buckets and hands you
what it hands you, so something must close the gap to a 428×639 real. Both
geometries are now bought at once and compared.

**A licence error in task 02's brief, still unfixed and now off this branch.**
It listed "SDXL 1.0 / SDXL-Turbo | CreativeML OpenRAIL++-M | Use". Turbo is
`sai-nc-community`, **non-commercial**. Found by dionyichia while building
task 02; carried in §6 as an open item because the document it belongs to is no
longer here.

**A premise in `docs/03` was wrong.** The brief assumed the reals are
*"overwhelmingly photographs of people"* and built a 20% retry budget on it.
**"Portrait" is an aspect-ratio filter, not a subject filter** — measured over
the 60-image harvest, 47% mention a person at all and **7% are face-centric**.
The rest are food, flowers, animals, machinery. Retry budget cut to 15%.

---

## 4. Costs — measured, not estimated

| Provider | Model | $/img | Verified by |
|---|---|---:|---|
| Ideogram 4.0 Turbo | `V_4_TURBO` @ 1664×2496 | **0.0300** | vendor dashboard |
| Google 3.1 Flash **Lite** | `google/gemini-3.1-flash-lite-image` | ~0.0342 | list (½ of Flash) |
| Seedream 4.5 | `bytedance-seed/seedream-4.5` @ 2K | **0.0400** | `usage.cost` |
| OpenAI gpt-image-2 | `openai/gpt-image-2` @ 1K medium | **0.0415** | `usage.cost` |
| Google 3.1 Flash | `google/gemini-3.1-flash-image` | **0.0684** | `usage.cost` |
| Captioner | `qwen/qwen3-vl-8b-instruct` | **0.000093** | `usage.cost` |

**Rejected after probing.** Grok Imagine 2.0 listed at $0.01, **billed
$0.0600**. Seedream 5 Pro listed at $0.003, **billed $0.0450**. Meta Muse Image
is blocked behind an 18+ age confirmation. The lesson worth carrying:
**OpenRouter's listed per-image price is a floor, not the charge** — Seedream
4.5 lists `image=None` and bills $0.04. Only a real call settles a price.

### What the money buys

Each real yields one image per provider, so 500 reals → 2,000 images.

| Config | $/real | Reals | Images | Total |
|---|---:|---:|---:|---:|
| 4 providers, Google Flash | 0.1799 | 500 | 2,000 | **$90** (+15% = $103) |
| 4 providers, Google **Lite** | 0.1457 | 500 | 2,000 | **$73** ≈ S$93 |
| 4 providers, Google Lite | 0.1457 | 269 | **1,078** | **$39** ≈ **S$50** |
| Pilot (next step) | 0.1799 | 50 | 200 | **$9** |

SGD at 0.786 (2026-08-28). **Spent to date $0.32**: $0.15 wire tests, $0.06
Ideogram, $0.006 captioning, $0.105 candidate probes.

For contrast, the number §1 of the brief asks to re-derive: a *training* corpus
matching task 02's 60,000 reals costs **$2,700** at these measured prices. The
brief's conclusion holds — buy the held-out set, never the training set.

---

## 5. Verified working

All four providers, end to end, one image each:

| Provider | Raw output | Parity → 428×639 | Tables match real |
|---|---|---|---|
| OpenAI | 1024×1536 PNG | ✓ both geometries | ✓ |
| Google | 848×1264 **JPEG** | ✓ both geometries | ✓ |
| Seedream | 1570×2352 JPEG | ✓ both geometries | ✓ |
| Ideogram | 1664×2496 PNG | ✓ both geometries | ✓ |

**Four bugs the $0.15 wire test caught**, any of which would have corrupted a
full run:

1. Seedream refuses anything under 3,686,400 output pixels — 1K at 2:3 is
   700,416 and returns HTTP 400. It is asked for 2K.
2. Ideogram v4 names the field `text_prompt`, takes multipart/form-data rather
   than JSON, and selects geometry from a resolution enum starting at 2048×2048.
3. **Ideogram's CDN rejects Python's default User-Agent with HTTP 403.** The
   generation is billed before the fetch, so this was paying for images and
   silently discarding them — one of the two Ideogram charges bought nothing.
4. `output_format: png` is a request, not a promise. Google returns JPEG for it,
   and a JPEG saved as `.png` misstates the one property this task measures.

Two providers (Google, Seedream) return JPEG already, so their rows carry an
extra compression generation before parity. Worth watching in the per-provider
gate results.

**Assets:** `~/techjam_task03/` — 60 reals + `attribution.csv`, 60 narrative
prompts, 54 VLM prompts, wire-test images. 29 MB. Not in the repo.

---

## 6. Open items

**Seedream has no published terms.** BytePlus publishes specific terms for its
*video* models only, and OpenRouter §5.1/§6.1 bind us to upstream model terms
rather than supplying them. The owner chose to proceed (2026-08-30); recorded in
`dataset_licences.md` as a deliberate call, with Seedream images marked
**local-only, never redistributed**, since redistribution is the term we cannot
check. This row does not satisfy the brief's §5.4 and the writeup must say so.

**The OpenAI artifact condition.** Services Agreement §3.3(e) permits classifiers
under a "Permitted Exception" — *provided they are "not distributed or made
commercially available to third parties."* That lands on the deliverable, not
the data. Check it before publishing model weights or an inference bundle.

**SDXL-Turbo's licence, for whoever owns task 02.** Its brief listed
"SDXL 1.0 / SDXL-Turbo | CreativeML OpenRAIL++-M | Use". Turbo is
`sai-nc-community`, **non-commercial**. That brief no longer lives on this branch,
so the correction has to be carried across by hand.

**Temperature 0 is not fully deterministic.** One refused caption passed on a
re-run. `prompts.csv` is the reproducible artifact, not the model call.

---

## 7. The S$50 answer — which generators to buy

**Recommendation: the same four families, at 194 reals each, and no cheap-tier
substitution.** OpenAI `gpt-image-2` · Google Gemini 3.1 Flash Image · Bytedance
Seedream 4.5 · Ideogram 4.0 Turbo. **776 images for $34.91**, leaving $4.07 of
the S$50 as retry contingency.

### 7.1 What S$50 actually is

S$50 at 0.786 is **$39.30**, and $0.32 of it is already spent on wire tests and
candidate probes, so the buy has **$38.98**. §4's S$50 row (269 reals, 1,078
images, $39) spends every cent of that, ignores the spend to date, and reaches
its count by moving Google onto its Lite tier. All three are corrected below.

**The pilot is not an extra cost — it is the first 50 reals of the buy.**
`pilot_commercial_apis.py` writes the provider's raw bytes first, derives both
geometries offline, and resumes on the raw file's existence. So the $9 pilot is
a prefix of the full run rather than a separate purchase, and the only money at
risk in it is the ~$2–4 belonging to a provider we then drop. That makes the
pilot a cheap option to hold, not a tax on a small budget.

### 7.2 The licence filter, applied to the constraint as stated

"Only licensed/public data" bites in two places, and they are different rules.

**The reals.** Settled, and already correct: Open Images V7 is CC BY 2.0, the
only one of the three vertical-real sources audited in `dataset_licences.md`
that permits **both** commercial use and redistribution. Pexels bars collection
for ML; Unsplash bars redistribution. Nothing to change here.

**The fakes.** The organisers' rule names *datasets* — "only public/licensed
datasets (e.g., WildFake, CIFAKE, SID_Set)" — and `docs/03` §1 records that the
organisers permit commercial APIs. Images we generate ourselves through a paid
API are not an ingested dataset, so that rule is satisfied by construction for
all four providers. What the four fail differently is **our own** acceptance
criterion §5.4, which wants a licence verdict per provider with a quote and a
URL:

| Provider | §5.4 verdict | Redistributable? |
|---|---|---|
| OpenAI | Clean — Output assigned (SA §4.1), no training on submissions (§4.2) | **Yes** |
| Google | Clean on licence; **SynthID in every pixel**, no opt-out | **Yes** |
| Ideogram | Clean, and the strongest no-training commitment of the four (§2.2); but no IP assignment | **Local-only**; anything published carries the §2.3.1(a) attribution |
| Seedream | **Fails** — no terms document exists for the image models | **Local-only** by precaution |

Two consequences worth stating plainly:

* **Seedream's problem is ours, not the competition's.** It does not breach the
  licensed/public rule; it breaches a criterion we wrote for ourselves. The
  owner's 2026-08-30 decision to proceed stands and the mitigation (local-only,
  never redistributed) is the right one. Keep it: dropping the generator that
  appears in four of NTIRE's five held-out splits, in order to satisfy our own
  paperwork, moves the eval set away from the composition it exists to
  approximate.
* **Only two of the four families could ever be published.** If the deliverable
  includes shipping this held-out set as a Kaggle Dataset, that dataset is
  OpenAI + Google and says so on its face; Ideogram and Seedream stay on disk.
  This changes nothing about what to buy — only about what can be handed over —
  but decide it before generating rather than after.

### 7.3 Why four families, and not three at a higher count

The brief's §3.4 says that when the total is over budget the lever is "fewer
providers at full count, not fewer images across all of them", because halving
everywhere "gives noisy estimates for every provider". At S$50 that reasoning
inverts, and the arithmetic is why. Planning against $35.00 of the $38.98:

| Config | $/real | Reals | Images | Spend | SE on per-provider TPR (p≈0.2 / 0.5) |
|---|---:|---:|---:|---:|---|
| **A — 4×: OpenAI, Google Flash, Seedream, Ideogram** | 0.1799 | **194** | **776** | **$34.90** | **2.9 / 3.6 pt** |
| B — 4×, Google **Lite** swapped in | 0.1457 | 240 | 960 | $34.98 | 2.6 / 3.2 pt |
| C — 3×, drop Google | 0.1115 | 313 | 939 | $34.91 | 2.3 / 2.8 pt |
| D — 3×, drop Seedream | 0.1399 | 250 | 750 | $34.98 | 2.5 / 3.2 pt |
| E — 5×, add Recraft V4.1 | 0.2153 | 162 | 810 | $34.87 | 3.1 / 3.9 pt |
| F — 2×: OpenAI, Ideogram | 0.0715 | 489 | 978 | $34.98 | 1.8 / 2.3 pt |

The budget buys 750–980 images whatever the shape, because the four prices sit
within 2.3× of each other. All that changes is how they are split — and **every
split is precise enough.** The per-provider standard error runs 1.8–3.9pt
against the ~40pt effects §1 expects to see. Precision stopped being the binding
constraint somewhere below n=200, so the money should buy the thing that is
actually scarce here, which is **decoder lineages**, not counts. That is
config A.

Config E — a fifth family at n=162 — is the one that tempts, and it is declined
on two grounds, neither statistical: Recraft's terms are still unread (§5.4
again, and a new one rather than an inherited one), and it appears in none of
NTIRE's five held-out splits, which is precisely why `docs/03` §2.1 swapped it
out for Seedream. It stays the reserve it already is. Config F is the reverse
error: it buys the tightest error bars in the table on the fewest lineages,
which is a precise answer to a question nobody asked.

### 7.4 Do not take the Google Lite discount by default

§4's S$50 row reaches 1,078 images by moving Google from Flash Image ($0.0684,
**measured** via `usage.cost`) to Flash **Lite** (~$0.0342, a list price halved,
never measured). That buys 46 more reals per family. Three reasons it is not
free:

1. `03a` §3.4 is explicit that the quality tier is not a pure cost knob — the
   cheap tier "produces *different artefacts*", so benchmarking on it measures
   cheap-tier artefacts rather than what the graded benchmark contains. The mid
   tier is the written default; a departure is argued, not defaulted into.
2. The price is **unmeasured**, and this task's own most expensive lesson is
   that OpenRouter's listed price is a floor: Grok Imagine listed $0.01 and
   billed $0.06; Seedream 5 Pro listed $0.003 and billed $0.045.
3. The gain is 2.9pt → 2.6pt of standard error. That is nothing.

There *is* a real argument for Lite, and it is not the price: NTIRE's own
validation split holds out **ImageGen-4 Fast**, so a fast/lite tier is not
off-composition for a held-out set. If the extra 184 images are wanted, the
order is — spend ~$0.04 measuring Lite's true billed price with one call, write
the NTIRE-Fast rationale into `03a` §3, then swap. Not the reverse.

### 7.5 What this buy deliberately does not do

* **No inpainting arm.** `docs/03` §2 mandates 70/30 text-to-image / inpainted.
  At this budget the 30% would split each family into cells of ~136 and ~58,
  bill the input image on top of the output at the edit endpoints, and need four
  more adapters written and wire-tested. The buy is 100% text-to-image; the
  writeup states the deviation.
* **776 images against the brief's 2,000–5,000.** Below spec, on the §7.3
  precision argument, and to be stated as a deviation rather than glossed.
* **The real half is free, so make it large.** The 1% FPR threshold is set on
  reals and shared across all four providers, so its uncertainty comes down for
  the cost of a download rather than a card charge. Harvest a few thousand Open
  Images **validation** photographs, with `ImageID`s disjoint from every
  training split (criterion §5.5) *and* from the 216 partner reals behind these
  prompts (§2.2).

### 7.6 The buy, in order

| Step | Cost | Running |
|---|---:|---:|
| Harvest ~216 reals → 194 usable prompts (90% yield: 6 of 60 refused locally by §3.5) + VLM captioning | $0.02 | $0.02 |
| Pilot: 50 reals × 4 providers | $9.00 | $9.02 |
| Gate on `jpeg_quality` AUC < 0.60, both geometries, pick on the number | $0 | $9.02 |
| Continue the same command to 194 reals (resumes, re-buys nothing) | $25.91 | **$34.93** |
| **Contingency left of the $38.98** | | **$4.05 (12%)** |

12% against the 15% retry budget carried in `03a` §3 — and against the 0%
provider refusal rate the wire test actually observed. If the pilot's measured
refusal rate comes in at zero again, the contingency converts to ~22 more reals
per family.

---

## 8. What to do next

1. **The spend is decided: S$50, config A of §7.3** — four providers at 194
   reals each, 776 images, $34.91, no cheap-tier substitution. Below the brief's
   2,000 and the writeup must say so; §7.3 is the argument for why that costs
   less than it looks.
2. **Harvest ~216 reals** from Open Images validation, `ImageID`s disjoint from
   every training split and from the eval real half (§7.5).
3. **Run the pilot** (~$9, 50 reals × 4 providers) — the first 50 reals of that
   194, not a separate purchase. Measures the refusal rate and settles the
   geometry question.
4. **Score both geometries** and pick on the number, per §3.1:
   ```bash
   for g in resample crop; do
     python scripts/prove_encoder_parity.py --reals ~/techjam_task03/portrait \
       --generated ~/techjam_task03/pilot/raw/<family> \
       --out ~/techjam_task03/gate_$g --geometry $g
   done
   ```
5. **Feed the measured refusal rate into `03a` §3**, then the go/no-go.
6. **Buy the remaining 144 reals** with the same command — it resumes and
   re-buys nothing.
7. **Decide the redistribution question** (§7.2): a published held-out set is
   OpenAI + Google only. Ideogram and Seedream stay local.

### What to expect from these four

They are **four brands over possibly two architecture families**: OpenAI and
Google are likely LLM-native autoregressive, Seedream and Ideogram diffusion
transformers. Only Ideogram's is verified, because they published open weights;
the other three are inference from brand and press.

If per-provider TPRs cluster into those pairs, that is the finding, and it is
more interesting than the average. §6 of the brief already says the per-provider
spread matters more than the mean.

This set can answer *"how badly do we do on current commercial generators."* It
**cannot** answer *"which architectures do we fail on"*, because a failure cannot
be attributed to an architecture nobody outside those labs can identify. That
question belongs to task 02, where the decoder is knowable — and worth raising
there: its four models are all latent diffusion with a VAE, missing the
autoregressive (Janus, Infinity) and pixel-space (DeepFloyd IF) families NTIRE
trains on.
