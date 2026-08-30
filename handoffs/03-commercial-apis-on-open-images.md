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

**Same pairing method as task 02.** Use Open Images Localized Narratives as
prompts so each generated image is a counterpart to a specific real. Record
`ImageID` alongside every generated file so the pairing survives.

**Both kinds again.** Fully synthetic and, where the API supports it,
inpainted-over-real. §3.1 of handoff 02 explains why.

---

## 3. Before you spend anything

### 3.1 Check the output terms, per provider

Two separate questions, and providers answer them differently:

* Do the terms grant commercial use of generated output?
* Do they permit **redistribution** — putting the images in a dataset others
  can download?

Some providers grant the first and not the second. A provider that bars
redistribution can still be used, but the resulting images must stay local and
never enter a shared Kaggle Dataset. Record the answer per provider in
`docs/dataset_licences.md` **before** generating, with a quote and a URL, the
way every other source in that file is recorded.

### 3.2 Cost it properly, in writing

Produce a table before spending: provider, per-image price, image count,
subtotal, and the terms verdict from §3.1. Include the retry budget — failed
and refused generations are still billed by some providers, and prompt
refusals on photographs of people are common enough to matter at this scale.

Bring that table back for a go/no-go rather than starting generation. If the
total lands above what the team has agreed to spend, the lever is **fewer
providers at full count**, not fewer images across all of them — halving the
count everywhere gives noisy estimates for every provider instead of a solid
estimate for some.

### 3.3 Content

Open Images portraits are photographs of real people. Do not prompt for
identifiable individuals, and do not use inpainting to place real people in
fabricated situations. Generic prompts derived from the narrative
("a person in a red jacket on a beach") are the intent; reproducing a specific
identifiable person is not, and several providers' terms bar it outright.

---

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
   quote and URL.
5. A deduplication check against the training corpus. If any generated image
   is a near-duplicate of a training image, the held-out claim is void.

## 6. What a negative result looks like

* **Low scores across all providers** — the expected outcome, and it
  quantifies the era/lineage gap. Report it as the headline caveat next to
  the benchmark number, not in a footnote.
* **One provider far worse than the rest** — more useful than the average.
  Name it and characterise what it does differently.
* **Costing comes back unaffordable** at any useful size. Then say so and stop;
  the fallback is task 02's open-weight held-out lineage, which costs GPU time
  instead of money and answers a weaker version of the same question.
