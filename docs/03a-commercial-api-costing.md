# 03a — Commercial APIs: terms and costing, for go/no-go

The tables `docs/03-commercial-apis-on-open-images.md` §3.3 and §3.4 ask for,
before anything is generated. Prices first costed 2026-08-30 and **re-verified
at the vendor the same day**; rows still marked *unconfirmed* could not be read
off a vendor page (the page paywalls, 403s, or loads its prices by script) and
must be confirmed before a card is charged. Every price here can move.

**Headline: cost is not the blocker.** The held-out set the brief describes
costs roughly **$114**, and about **$285** at the top of its 5,000-image range.
§1.1's price argument is sound but it is an argument against a *training*
corpus — see §3. The two things that should actually gate this task are
SynthID and encoder parity (§4); both are now written into the brief's §3.

---

## 1. Lineage — which providers are even eligible

The brief's premise is that these generators are **unseen**. That is a claim
about the decoder, not the product name, and task 02 approves FLUX.1-schnell,
SDXL, SD 3.5, SD 1.5/2.1 and Qwen-Image *for training*. Any API built on those
decoders is therefore not held out for us, whatever it costs.

| Provider / model | Decoder lineage | Eligible as held-out? |
|---|---|---|
| OpenAI `gpt-image-2` | OpenAI, proprietary | **Yes** |
| Google Gemini 3.1 Flash Image / Imagen | Google, proprietary | **Yes** — but see §4.1 |
| Ideogram 4.0 | Ideogram, proprietary | **Yes** |
| Recraft V4.1 | Recraft, proprietary | **Yes** |
| Bytedance Seedream 4.5 | Bytedance, proprietary | **Yes** |
| Black Forest Labs FLUX.2 [pro/flex/max] | FLUX, but see below | **Unresolved — do not buy until §1.1 is decided.** Not the clean "no" an earlier draft claimed. |
| Stability API (SD 3.5) | Stable Diffusion | **No.** Same reason. |
| Midjourney | Midjourney, proprietary | No public API. Excluded on availability, not terms. |

Stability is the clean exclusion: the API serves SD 3.5, which task 02 puts in
training by name. FLUX.2 is not, and the difference is instructive.

### 1.1 FLUX.2 is the case that shows the lineage test needs stating

The criterion above is *decoder lineage, not product name*. Applied honestly,
it does not exclude FLUX.2 on the evidence available:

* FLUX.1-schnell is a 12B rectified-flow transformer over a 16-channel VAE.
* FLUX.2 is **32B**, pairs the transformer with a **Mistral-3 24B** vision-
  language encoder, and — the part that matters — its latent space was
  *"re-trained from scratch to achieve better learnability and higher image
  quality at the same time"* ([BFL, FLUX.2](https://bfl.ai/blog/flux-2)). A
  from-scratch VAE is a different decoder.

Same lab, same architecture family, plausibly overlapping training data and
recipe — all real reasons for caution. But "shares a decoder with something in
our training set" is not established, and that is the claim `heldout_groups`
actually rests on. Decide explicitly whether exclusion here is on **lab**,
**architecture family**, or **decoder weights**, record the choice, and apply
it to every provider. Under the third test FLUX.2 is eligible; under the first
it is not. The project has not picked one.

Related, and it cuts the same way: **FLUX.2-dev ships open weights** (32B,
FLUX Non-Commercial License for the model, though *"Generated outputs can be
used for personal, scientific, and commercial purposes"*), so if FLUX.2 is
ruled eligible it costs GPU time rather than money — with the quantisation
caveat in the brief's §2.

## 2. Brief §3.3 — Output terms, per provider

Four questions, per the brief: commercial use, redistribution, watermarking
(§4.1), and whether the provider trains on what we send it.

| Provider | Commercial use of output | Redistribution / put in a dataset | Competing-model clause | Verdict |
|---|---|---|---|---|
| **OpenAI** | Granted — OpenAI assigns its right, title and interest in Output to the user | Not restricted by the ToU | Yes: may not use Output to develop models that compete with OpenAI. A detector is not a competing *generator*, but record the reading. | **Use.** ⚠️ `openai.com/policies` returns 403 to automated fetch — a human must open it and paste the two clauses verbatim into `dataset_licences.md`. |
| **Google (Gemini API)** | Google does not claim ownership of generated content | No explicit bar found in the API terms | Yes: *"You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)."* | **Conditional** — on §4.1, not on the licence. |
| **Ideogram** | Permitted via the Developer API agreement; ownership language is weaker than OpenAI's — it grants no explicit assignment of output IP | Not barred, but **disclosure is mandatory, not encouraged**: §2.3.1(a) requires you *"identify on the Developer Application that any User Output generated…was created by the Ideogram AI Model"*. §2.3.6(H) separately encourages disclosing AI origin. Good news elsewhere: §2.2 — *"the Company agrees that it shall not use any User Input or User Output to train the Ideogram AI Model"* | Yes: §2.3.6(A) bars using User Input or User Output *"to develop any product, service, or technology that competes with the Company, the Ideogram AI Model, Ideogram API, or any of the Company's products"* | **Use, local-only until someone confirms the ownership position**, and carry the §2.3.1(a) attribution on anything published. The absence of an assignment clause is not the same as a grant. |
| **Recraft** | Per the API terms | Needs checking | Needs checking | **Check before use.** Priced and lineage-clean; terms not yet read. |
| **Bytedance Seedream** | Per provider (fal / OpenRouter resell it; the reseller's terms may differ from Bytedance's) | Needs checking | Needs checking | **Check before use.** A reseller adds a second set of terms. |
| **Black Forest Labs** | Outputs usable commercially; but BFL takes a *"perpetual, irrevocable, worldwide"* licence to inputs and outputs and states it may train on them | Barred from being used as synthetic training data for a model of "substantially similar functionality" | — | **Terms are the binding objection, not lineage.** §1.1 does not sustain the lineage exclusion; the input/output licence and the training-data bar stand on their own. |

Nothing above goes into `docs/dataset_licences.md` until it carries a quote and
a URL, the way every other source in that file does.

**The training-on-submissions column is the one that separates the field.**
Ideogram commits in writing not to train on our inputs or outputs; BFL takes a
perpetual licence to both and says it may. Google and OpenAI sit between and
need reading. This decides whether the Localized Narrative prompts leak, and
the brief's §3.3 now asks for it explicitly.

## 3. Brief §3.4 — Cost

Baseline: the brief's minimum shape — 4 providers × 500 images, 70% text-to-image
/ 30% inpainted, at 1K resolution. **At the mid quality tier, not the cheapest**
— see the tier note below; that choice is now the largest single lever in this
table.

| Provider | Model / tier | $/image | Images | Subtotal | Price source |
|---|---|---:|---:|---:|---|
| OpenAI | `gpt-image-2`, 1K 2:3 medium | 0.0415 | 500 | $20.75 | **measured** — OpenRouter `usage.cost` |
| Google | Gemini 3.1 Flash Image, 1K 2:3 | 0.0684 | 500 | $34.20 | **measured** — same |
| Bytedance | Seedream 4.5, **2K** 2:3 | 0.0400 | 500 | $20.00 | **measured** — same |
| Ideogram | 4.0 Turbo, 1664x2496 | 0.0300 | 500 | $15.00 | **measured** — vendor dashboard |
| | | | **2,000** | **$89.95** | |
| Retry / refusal budget | +15%, see below | | | **+$13.49** | |
| | | | | **≈ $103** | |

**Every price above is now measured against the live API, 2026-08-30**, from a
$0.22 wire test — not a price list. Two corrections that came out of it:

* **OpenAI is half what the calculator implied.** $0.0415 at 1024x1536 medium
  against the $0.080 this table carried. It is the second-cheapest provider,
  not the dearest.
* **Seedream must be asked for 2K.** It refuses anything under 3,686,400 output
  pixels, and 1K at 2:3 is 700,416 — HTTP 400. The $0.040 is the 2K price.

Ideogram reports no cost through its API; $0.03/image is read off the vendor
dashboard, and it confirmed something worth stating: **a generation that fails
to download is still billed.** Ideogram returns a CDN URL rather than bytes, and
that CDN rejected our first fetch — so one of the two charges bought an image
that was thrown away. `call_ideogram` now records a failed fetch as billed.

Cheaper same-lineage substitutions, if the total needs to come down without
dropping a provider: **Google batch tier** at $0.0335 (halves Google's line to
$16.75), or **Gemini 3.1 Flash Lite Image** at $0.0336 standard / $0.0168 batch.
Both keep Google's SynthID problem (§4.1) intact.

At the top of the brief's range (5,000 images, 1,250 per provider): **≈ $225
before retries, ≈ $259 with them.** The pilot itself is **≈ $9**.

**The retry budget is cut from 20% to 15%, and it may fall further.** See below:
the premise behind 20% does not survive contact with the harvest.

**The quality tier is a 35× lever and it is not a pure cost knob.** `gpt-image-2`
at 1024×1024 is **$0.006 low / $0.053 medium / $0.211 high**. Buying the low
tier would take OpenAI's line to $3 and the whole baseline under $80 — and would
measure low-tier artefacts, which is not what the graded benchmark contains. The
mid tier is the defensible default; anything else has to be argued for in
writing. An earlier draft of this file costed OpenAI at $0.030, which matches no
tier that exists.

**Inpainting is billed separately, and the brief mandates 30% of it.** Recraft
charges $0.0440 for raster inpainting against $0.03535 for text-to-image, and
edit endpoints elsewhere bill the input image on top of the output. Costing one
price per provider under-counts the buy.

**The retry budget's premise was wrong, and the number comes down.** This
table used to argue: the reals are filtered to portrait aspect, therefore they
are *"overwhelmingly photographs of people"*, therefore §3.5-compliant prompts
draw refusals, therefore 20%.

The first step does not hold. **"Portrait" is an aspect-ratio filter, not a
subject filter** — `width/height <= 0.7`, which says nothing about content.
Measured over a 60-image harvest (2026-08-30): **47% mention a person at all and
only 7% are face-centric.** The rest are food, flowers, animals, machinery,
fingernails. And in the wire test all four providers refused nothing.

15% is carried as a margin rather than a prediction. The §3.2 pilot still
replaces it with a measurement; the point is that the measurement is now likely
to come in *under* the budget rather than over it.

Separately, 6 of 60 prompts were refused **locally** by §3.5 before any spend —
including a studio portrait of a man in military uniform wearing an Iron Cross.
That is the rule working, and it is not a provider refusal: it costs nothing and
must not be counted into the retry budget.

**For contrast, the number §1.1 of the brief asks to re-derive:** a balanced
training corpus matching task 02's 60,000 reals costs 60,000 × ~$0.038 ≈
**$2,280**, and that is before retries. The brief's conclusion holds exactly as
written — buy the held-out set, never the training set.

## 4. What should actually gate this — now folded into the brief's §3

### 4.1 Google's SynthID is a watermark in the pixels

The Gemini API image-generation docs are unambiguous: **"All generated images
include a SynthID watermark."** No opt-out is offered. DeepMind's SynthID page
adds that the mark is embedded the moment content is created, is
*"imperceptible to humans"*, and is *"designed to stand up to modifications
like cropping, adding filters, changing frame rates, or lossy compression"* —
which is precisely the 20-condition grid these images would be scored under.

(Cite the API doc, not the DeepMind marketing page: only the former actually
states universal application.)

That is a synthetic, provider-specific signal sitting in one half of one
provider's rows. If the detector scores *well* on Google and poorly on the
rest, we cannot tell whether it found generation artefacts or found SynthID —
and §1 already says a good number here is the one to disbelieve. Options, in
order of preference:

1. Keep Google, and report its number **separately and flagged** — never
   pooled. The brief forbids pooling (§5.2) and now requires the flag (§5.7).
2. Run a SynthID-presence check on a sample and quote the detection rate beside
   the TPR, so a reader can discount it.
3. Drop Google for a fifth lineage-clean provider (Recraft, Seedream).

Recommendation: (1) plus (2). Dropping the provider closest to the graded
benchmark's likely composition costs more than the caveat does.

### 4.2 Encoder parity — inherited from 02, and worse here

Task 02 §1 is unambiguous: the Open Images reals are `Thumbnail300KURL`
thumbnails, i.e. **re-encoded JPEGs**, and this project has already measured
JPEG history leaking the label (`docs/low_level_confounds.md`: `jpeg_quality`
AUC 0.5532 pooled). 02 makes encoder parity a hard gate — *"if `jpeg_quality`
AUC alone is above ~0.60, fix the save path before looking at anything else."*

It is harder here: commercial APIs return PNG or WebP at their own
resolutions, and unlike local weights you cannot change what they hand you.
Every purchased image must go through the **same JPEG encoder at the same
quality distribution as the reals** before it is stored, and that path has to
exist and be tested *before* the money is spent — a mis-encoded local generation
is regenerated for free, a mis-encoded purchased one is bought twice. This is
now the brief's §3.1, ahead of everything else.

## 5. Go / no-go

**Go, at the $114 baseline, conditional on four things landing first:**

1. Encoder-parity save path written and proven on task 02's free images
   (brief §3.1).
2. `jpeg_quality` AUC on the 200-image pilot below ~0.60 (02 §5.2's own gate,
   now brief §5.6). The pilot also replaces the guessed 20% refusal rate.
3. ~~Ideogram's price confirmed at the vendor~~ ✓ done, and ~~the OpenAI and
   Ideogram terms pasted into `dataset_licences.md`~~ ✓ done. **What remains:
   Seedream has no published terms for its image models** — see
   `dataset_licences.md`. Register at BytePlus direct to obtain them, or fall
   back to Recraft.
4. The quality tier decided and written down. Defaulting to "cheapest" silently
   changes what the benchmark measures.

Provider set: **OpenAI, Google (flagged for SynthID), Ideogram, Seedream** —
four of the six vendors NTIRE 2026 holds out, which is the composition the
brief's §2.1 argues for. Recraft is the reserve. Not Stability — the API serves
SD 3.5, which task 02 puts in training by name. Not BFL either, but on its
input/output licence and training-data bar (§2), *not* on the lineage argument,
which §1.1 does not sustain.

If the pilot fails gate 2, the brief's §6 fallback applies: an open-weight
held-out lineage answers a weaker version of the same question for GPU time
instead of money — and that fallback is stronger than it looks, since Ideogram
4.0 and FLUX.2-dev both ship weights. Read the model licences first: both are
non-commercial model agreements even where the generated outputs are not.

## 6. Open weights change the shopping list, but not the way you'd hope

Two of the eligible providers now publish weights for the generation being
bought:

* **Ideogram 4.0** — released 2026-06-03, 9.3B single-stream DiT, nf4 and fp8
  quantised weights on Hugging Face under the *Ideogram Non-Commercial Model
  Agreement*; inference code Apache-2.0.
* **FLUX.2-dev** — 32B, *FLUX Non-Commercial License*, outputs unrestricted.

The temptation is to run these locally and save the money. Two reasons not to,
for *this* purchase: the public releases are quantised, and quantised local
inference does not reproduce the artefacts of the provider's full-precision
serving stack — which is the thing being benchmarked. And the model licences
are non-commercial, which the API terms are not.

The cost of the open-weight releases runs the other way: a public-weights
generator can appear in anyone's training corpus, so its value as an *unseen*
lineage decays. Record the release date next to the provider, and treat a
result on an open-weight generation as weaker evidence than one on a
closed-serving generation of the same age.

## Sources

Read at the vendor on 2026-08-30 unless marked otherwise.

- **Google** — [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) ✓ · [Gemini API terms](https://ai.google.dev/gemini-api/terms) ✓ · [Gemini image generation docs](https://ai.google.dev/gemini-api/docs/image-generation) ✓ (the "All generated images include a SynthID watermark" statement) · [SynthID (DeepMind)](https://deepmind.google/models/synthid/) — robustness language only; does *not* state universal application
- **OpenAI** — [API pricing](https://developers.openai.com/api/docs/pricing) ✓ (confirms no per-image figure is published) · [image generation guide](https://developers.openai.com/api/docs/guides/image-generation) · per-image tiers via [costgoat calculator](https://costgoat.com/pricing/openai-images) — *aggregator* · [OpenAI ToU](https://openai.com/policies/row-terms-of-use/) and [Services Agreement](https://openai.com/policies/services-agreement/) both **403 to automated fetch — a human must open these and paste the output-assignment and competing-model clauses verbatim**
- **Ideogram** — [API ToS](https://ideogram.ai/legal/api-tos) ✓ (§2.2, §2.3.1(a), §2.3.6(A), §2.3.6(H)) · [pricing tab](https://ideogram.ai/pricing/?pricing_tab=api) — rates load by script, **not confirmed**; $0.03/$0.06/$0.10 Turbo/Default/Quality corroborated at aggregators only · [Ideogram 4.0 weights](https://huggingface.co/collections/ideogram-ai/ideogram-4)
- **Recraft** — [API pricing](https://www.recraft.ai/pricing?tab=api) ✓ (V4.1 raster $0.03535, V4.1 Pro $0.21210, raster inpaint $0.0440) · [Recraft API](https://www.recraft.ai/api) — terms still unread
- **Black Forest Labs** — [FLUX.2 announcement](https://bfl.ai/blog/flux-2) ✓ (retrained-from-scratch VAE, Mistral-3 24B encoder) · [FLUX.2-dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-dev) ✓ · [pricing](https://docs.bfl.ml/quick_start/pricing) · [FLUX API service terms](https://bfl.ai/legal/flux-api-service-terms)
- **Bytedance** — [Seedream 4.5 on OpenRouter](https://openrouter.ai/bytedance-seed/seedream-4.5) ✓ ($0.04/image, 1K/2K/4K); Bytedance's own terms behind the reseller still unread
