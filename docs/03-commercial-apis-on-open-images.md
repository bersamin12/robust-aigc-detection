# 03 — Commercial APIs on Open Images V7

**Goal.** A small, never-trained-on **held-out set** of images from commercial
generators, over the same Open Images reals as task 02.

**This is not training data.** That is the whole design. Read §1 before
costing anything, because the instinct is to buy volume and volume is the
wrong thing to buy here.

---

## 1. Why this is a held-out set and not a training set

The organisers permit commercial APIs, and the temptation is to generate a
large commercial-API training corpus. Three reasons not to:

1. **Price.** Commercial image APIs are priced per image at rates that make a
   balanced training corpus (~60k images) cost orders of magnitude more than
   a hackathon budget. Earlier costing in this project concluded exactly that.
   Re-derive the number before you commit (§3.2) — do not trust a remembered
   figure.
2. **It destroys the measurement.** Commercial generators are the closest
   proxy we have for what the graded benchmark actually contains. Train on
   them and we lose the only honest estimate of how we do on unseen
   commercial output. The number would go up and mean less.
3. **We will be bad at them, and we need to know by how much.** Literature
   puts mean accuracy on DALL·E 3 around 31%. Our A3 reaches 0.8952 TPR at 1%
   FPR on the organisers' benchmark, which is a strong result *for that
   benchmark*. A commercial held-out set tells us whether that generalises or
   whether we have fitted to one generator's fingerprint.

**Accept in advance that the score will be low.** A poor number here is the
deliverable. A good number here, if we never trained on these generators, is
the single most valuable result the project could produce — and should be
double-checked for leakage before anyone believes it.

---

## 2. What to generate

**Size: 2,000–5,000 images total.** Enough for a stable TPR at 1% FPR (at
n=2,000 the binomial standard error is ~0.7pt, which is smaller than the
effects we care about), small enough to afford.

**Spread across at least four providers**, because the point is generator
diversity, not image count. Roughly 500–1,000 each. Prioritise families whose
*decoder lineage* is absent from training — that is what makes them unseen.
Two products from the same underlying model count as one family.

### 2.1 Which providers — the reference is NTIRE 2026

We are not the first to build this. The NTIRE 2026 challenge (arXiv 2604.11487)
holds every proprietary API model out of training and uses it only in
validation and test — the same design as this task. Their held-out
composition is the best available proxy for what a graded benchmark contains,
so it is the shopping list to match:

| Split | Proprietary models held out (their † = via API) |
|---|---|
| Train | **none** — 20 open-source generators only |
| Validation | Ideogram v3 Turbo, ImageGen-4 Fast |
| Validation (hard) | Nano Banana, Seedream 4 |
| Test (public) | Nano Banana Pro, FLUX-2 Max, ImageGen-4 Ultra, Seedream 5 Lite, Grok Imagine |
| Test (private) | Nano Banana 2, GPT Image 1.5, ImageGen-4 Ultra, Seedream 5 Lite, Grok Imagine |

Six vendors: **Google, OpenAI, Bytedance (Seedream), Ideogram, xAI (Grok),
Black Forest Labs**. Their split sizes are 2.5K–10K at 1:1 real:fake across
~10 generators, which is where §2's 2,000–5,000 figure sits.

Read two things off that table:

* **Recraft appears nowhere**, while **Seedream appears in four of the five
  held-out splits** and costs $0.04/image. A provider set of OpenAI, Google,
  Ideogram and Seedream matches the reference composition; substituting
  Recraft moves away from it for no gain.
* **FLUX.2 is held out while FLUX.1 is trained on.** NTIRE puts FLUX.1
  Kontext Dev, Dev and Schnell in training and FLUX-2 Max in the test set. So
  the field's flagship benchmark applies the *decoder weights* test, not the
  *lab* test — which is the choice this section asks you to make explicitly.
  Under that test FLUX.2 is eligible for us too.

### 2.2 Do not put a fake and its own partner real in the same eval split

NTIRE pairs generated to real images for *training*, and deliberately breaks
the pairing for validation and test: *"we use only unique images without its
paired counterpart to avoid potential advantage from selecting between multiple
similar images."*

Do the same. Use the pairing to **generate** and to apply §3.1's parity — that
is what makes per-row encoder matching possible at all — then draw the eval
set's real half from *different* photographs. You keep the parity and you do
not hand the scorer a matched pair.

**"Same lineage" needs a stated test, because the obvious cases are not the
hard ones.** A successor model from the same lab can share the product name and
the architecture family while shipping a latent space retrained from scratch
and a different text encoder — which is a different decoder by any measure that
matters to a detector. Decide up front whether a provider is excluded on
**lab**, on **architecture family**, or on **decoder weights**; the three give
different answers for the same model. Write the verdict *and its evidence* per
provider into `docs/dataset_licences.md` alongside the licence, so the
`heldout_groups` assignment can be defended later rather than reconstructed.

**Open weights are not automatically the cheaper path here.** Where a
provider has released weights for the same generation, running them locally
looks free — but public weights are usually quantised releases, and quantised
local inference does not reproduce the artefacts of the provider's
full-precision serving stack. For a *benchmark* the served output is the thing
being measured, so pay for the API. Note the open-weight release anyway: it
erodes the "unseen" premise over time, since anyone else's training corpus can
now contain that generator.

**Same pairing method as task 02.** Use Open Images Localized Narratives as
prompts so each generated image is a counterpart to a specific real. Record
`ImageID` alongside every generated file so the pairing survives.

**Both kinds again.** Fully synthetic and, where the API supports it,
inpainted-over-real. §3.1 of handoff 02 explains why.

---

## 3. Before you spend anything

Five gates, in this order. The first two are the ones that protect money: a
mis-encoded local generation is regenerated for free, a mis-encoded purchased
one is bought twice.

### 3.1 Encoder parity — the gate that must clear first

Task 02 §1 established that the Open Images reals are `Thumbnail300KURL`
thumbnails, i.e. **re-encoded JPEGs**, and this project has already measured
JPEG history leaking the label (`docs/low_level_confounds.md`: `jpeg_quality`
AUC 0.5532 pooled). Task 02 makes encoder parity a hard gate — *"if
`jpeg_quality` AUC alone is above ~0.60, fix the save path before looking at
anything else."*

It is harder here than there. Commercial APIs return PNG or WebP at their own
resolutions, and unlike local weights you cannot change what they hand you.
Every purchased image must go through the **same JPEG encoder at the same
quality distribution as the reals** before it is stored.

**That save path has to exist, and be proven on task 02's free images, before a
card is charged.** If it does not clear, nothing below matters.

**The task 02 solution does not transfer, and this is the part to think about.**
Task 02's generation plan (on branch `feat/ai-ov7-generation`) clears this
gate by never resampling at all — a local model is told to render at the real's exact dimensions, and the
geometry is then centre-*cropped* to a multiple of 8. Measured `jpeg_quality`
AUC **0.5031**. That works because the generator's output size is a parameter.

An API's is not. Providers render into fixed buckets — 1024×1024, 1024×1536 —
and hand you what they hand you, so *something* has to close the gap to a
~427×640 real. Three options, and the pilot in §3.2 is what settles it:

1. **Downscale** the API output onto the real's size. Simple, and it leaves the
   resampling signature `docs/resolution_shortcut.md` describes.
2. **Crop** a real-sized window out of the API output. No resample, so it
   inherits task 02's cleanliness — but the eval set is then crops while the
   graded benchmark holds whole frames.
3. **Downscale both halves** equally. Rejected: re-encoding a real to match a
   fake *"would add a compression generation to the authentic class"*, which
   damages the half we know is clean.

Measure 1 and 2 on the pilot's images and pick on the number, not on argument.

### 3.2 Pilot before the bulk buy

Buy **~50 images per provider first**, not the full count. The pilot pays for
itself twice over:

* It **measures the retry/refusal rate** instead of guessing it. Task 02's
  harvest is filtered to portrait aspect and short side ≥ 400 — overwhelmingly
  photographs of people — so the Localized Narratives that pair with them
  describe people, while §3.5 bars prompting for identifiable individuals.
  Portrait prompts draw refusals at rates generic prompts do not, and some
  providers bill refused generations.
* It **supplies the sample to run §3.1's gate against.** A 200-image pilot set
  across four providers is enough to score `jpeg_quality` AUC and confirm the
  save path holds on real API output rather than on local generations.

Re-cost from the pilot's measured refusal rate, then commit to the bulk buy.

### 3.3 Check the output terms, per provider

Four questions, and providers answer them differently:

* Do the terms grant commercial use of generated output?
* Do they permit **redistribution** — putting the images in a dataset others
  can download?
* Does the output carry a **watermark or provenance signal**? This is not a
  licence question but it belongs in the same pass, because the answer can
  disqualify a provider's *numbers* rather than its images. Google's SynthID
  is the live case: *"All generated images include a SynthID watermark"*,
  in-pixel and built to survive crop, rescale and re-encode — which is exactly
  the 20-condition grid these images get scored under. A provider-specific
  synthetic signal in one provider's rows is a confound; §5.2's no-pooling rule
  contains it, but only if the flag is recorded here.
* Does the provider **train on submitted inputs and outputs**? This decides
  whether the Localized Narrative prompts leak. Providers differ sharply —
  some commit in writing not to, others take a perpetual licence to both.

Some providers grant commercial use and not redistribution. A provider that
bars redistribution can still be used, but the resulting images must stay local
and never enter a shared Kaggle Dataset. Record all four answers per provider
in `docs/dataset_licences.md` **before** generating, with a quote and a URL, the
way every other source in that file is recorded.

### 3.4 Cost it properly, in writing

Produce a table before spending: provider, model, **quality tier**,
**generation mode**, per-image price, image count, subtotal, and the terms
verdict from §3.3. Two columns there are easy to omit and both change the
total:

* **Quality tier is not a pure cost knob.** The spread within a single provider
  can be more than an order of magnitude, and the cheap tier does not just cost
  less — it produces *different artefacts*. Benchmarking a detector on a
  provider's cheapest tier measures cheap-tier artefacts, not what the graded
  benchmark contains. Pick a tier deliberately, default to the mid/standard
  tier, and write the rationale into the table rather than letting "cheapest"
  win by default.
* **Generation mode has its own price.** §2 mandates 70/30 text-to-image /
  inpainted. Inpainting and image-to-image are billed at different rates from
  text-to-image at several providers, and edit endpoints bill the input image
  on top. One price per provider under-counts the buy.

Include the retry budget from §3.2's measured rate — failed and refused
generations are still billed by some providers.

Bring that table back for a go/no-go rather than starting generation. If the
total lands above what the team has agreed to spend, the lever is **fewer
providers at full count**, not fewer images across all of them — halving the
count everywhere gives noisy estimates for every provider instead of a solid
estimate for some.

### 3.5 Content

Open Images portraits are photographs of real people. Do not prompt for
identifiable individuals, and do not use inpainting to place real people in
fabricated situations. Generic prompts derived from the narrative
("a person in a red jacket on a beach") are the intent; reproducing a specific
identifiable person is not, and several providers' terms bar it outright.

## 4. Where it goes

These rows are held out, so they **must not** be reachable by training.

* Register the source with `exclude_from_training=True` in
  `src/aigcdet/data/sources.py`. That flag is spec §4.1(2) and is enforced in
  `build_dataset.py` — it is the mechanism, not a convention.
* Give each provider its own generator family name. They are the held-out
  families; do not merge them into one `commercial_api` bucket, or the report
  cannot say which provider we fail on.
* Add them to the **eval** manifest only, alongside the organisers' benchmark.

## 5. Acceptance criteria

1. `exclude_from_training=True` set, and a test asserting no row from this
   source appears in any `train` or `val_internal` split.
2. Per-provider TPR at 1% FPR reported separately, never pooled — pooling
   hides the provider we are worst at, which is the number that matters.
3. Scored under the full 20-condition degradation grid, like every other eval
   row. A clean-only number here is not comparable to anything else we report.
4. Licence verdict per provider recorded in `docs/dataset_licences.md` with a
   quote and URL, covering all four questions in §3.3 — commercial use,
   redistribution, watermarking, and whether the provider trains on submissions.
5. A deduplication check against the training corpus. If any generated image
   is a near-duplicate of a training image, the held-out claim is void.
   **This covers the reals too, not only the fakes.** Task 02 trains on 60,000
   Open Images photographs; if this task's eval reals are drawn from that same
   pool, the false-positive rate is measured partly on images the model has
   already seen, and the headline flatters itself. Draw the eval real half from
   `ImageID`s that appear in no training split, and assert it the same way
   criterion 1 asserts the generated side.
6. Encoder parity demonstrated, not assumed: `jpeg_quality` AUC on the §3.2
   pilot below ~0.60, reported in the writeup next to the TPR. A provider whose
   images went through a different save path than the reals is measuring its
   encoder, not its generator.
7. Any provider whose output carries a watermark or provenance signal flagged
   in the results table, so a reader can discount its number without having to
   know the provider's defaults.

## 6. What a negative result looks like

* **Low scores across all providers** — the expected outcome, and it
  quantifies the era/lineage gap. Report it as the headline caveat next to
  the benchmark number, not in a footnote.
* **One provider far worse than the rest** — more useful than the average.
  Name it and characterise what it does differently.
* **Costing comes back unaffordable** at any useful size, or §3.1's encoder
  gate will not clear. Then say so and stop; the fallback is task 02's
  open-weight held-out lineage, which costs GPU time instead of money and
  answers a weaker version of the same question. That fallback is stronger than
  it was when 02 was written — several frontier labs now ship open weights for
  their current generation, so a held-out lineage no longer means a stale one.
  Check the weight licence before using it: open weights are frequently
  released under non-commercial model agreements even when the *generated
  outputs* are unrestricted.
