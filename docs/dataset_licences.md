# Dataset licences

Recorded per spec §4.5, same discipline as `docs/model_licences.md`: the licence is
read at its source and quoted, not inferred from a secondary summary. Checked
2026-08-28; the competition-rule reading revised 2026-08-29 (see WildFake below).

Every row of the frozen manifest carries its source's licence string from
`aigcdet.data.sources.LICENCES`, and `scripts/acquire_data.py` writes a
`LICENCES.json` receipt into each acquisition directory at download time.

| Source | Licence | Where it was read | Commercial use? |
| --- | --- | --- | --- |
| SID_Set | CC BY 4.0 | https://huggingface.co/datasets/saberzl/SID_Set — licensing section on the dataset card | Yes, with attribution |
| WildFake — generated buckets | Apache-2.0 | ModelScope hub metadata, `https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake` → `"License":"Apache License 2.0"` | Yes — the authors' own images, covered by the compilation licence |
| WildFake — `real/` bucket | Apache-2.0 on the compilation; organiser-listed | Same ModelScope record; the organisers' rules slide (29 Aug) names WildFake as an approved dataset | Under the competition rules, yes. The re-published subsets' own terms (table below) are non-commercial — anyone taking this beyond the competition must read them |
| COCO val2017 | CC BY 4.0 (images under Flickr terms) | https://cocodataset.org/#termsofuse | Yes, with attribution |
| COCO train2017 | CC BY 4.0 (images under Flickr terms) | https://cocodataset.org/#termsofuse | Yes, with attribution |
| DALL·E Advanced (demo half) | Competition brief terms | TikTok TechJam Track 5 brief | Benchmark only; never trained on (spec §4.1) |

## SID_Set

The dataset card states the work incorporates material from COCO, OpenImages V7 and
Flickr30k, and commits to complying with those datasets' CC BY 4.0 terms, "including
providing appropriate attribution to the original creators and ensuring that the
derived portions remain available under those terms." CC BY 4.0 permits commercial
use. The binding obligation on us is attribution, satisfied by this file plus the
dataset list in the README and the Devpost writeup.

## WildFake — read this before quoting "Apache-2.0" anywhere

The ModelScope hub records the dataset as Apache License 2.0. That is the only
licence declaration that exists: **there is no LICENSE file** in the authors'
GitHub repository (`hy-zpg/AIGC-Image-Detection-Dataset`), and the dataset card
carries no terms-of-use section. The arXiv paper (2402.11843) is CC BY 4.0, but that
covers the paper, not the data.

The caveat that matters: WildFake is a **compilation**. Its generated images are the
authors' own, but its authentic images are re-published from other datasets, and an
Apache-2.0 label on the compilation does not relicense them. The real subsets this
project downloaded are exactly the ones with restrictive upstream terms. They are in
the frozen manifest (55,000 images, all six subsets below), and this table is the
record of what that means:

| Subset | Upstream | Upstream terms (not verified in-session — see below) |
| --- | --- | --- |
| `real_ffhq` | FFHQ (NVIDIA) | CC BY-NC-SA 4.0 — **non-commercial**, share-alike |
| `real_celebahq` | CelebA-HQ / CelebA | Research use only, **non-commercial** |
| `real_afhq` | AFHQ (NVIDIA) | CC BY-NC 4.0 — **non-commercial** |
| `real_imagenet` | ImageNet | Non-commercial research use |
| `real_church` | LSUN | Research use |
| `real_laion5b` | LAION-5B | CC BY 4.0 covers the *metadata*; the images are web-scraped and each carries its own rights |

**Verification status.** The Apache-2.0 declaration and the absence of any other
WildFake licence statement were verified directly in-session. The upstream terms in
the table above were **not** re-read from each originating dataset's own licence page
in this session; they are the widely-documented terms for those datasets and are
recorded here so the constraint is visible, not so it can be cited. Anyone taking this
project commercial must read each one at source first.

**The generated buckets are a compilation too.** The paper (arXiv 2402.11843) says the
GAN and "Others" families were gathered from "official GitHub repositories and model
cards from Hugging Face — when these repositories include generated samples, we
directly extract fake images from there", and that "user-created images from
open-source platforms such as Civitai" were sourced for the Stable-Diffusion-adaptor
families. Only the diffusion families the authors ran themselves are unambiguously
their own. Recorded for the same reason as the table above.

**How the competition rule was read, twice.**

*28 August.* The webinar Q&A said "non-commercial datasets cannot be used". We read
that against the upstream terms above and barred WildFake's `real/` bucket at scan
time (`SourceSpec.restricted_buckets`), replacing it with more SID_Set. The cost was a
single-sourced authentic half — the wrong shape for the sharpness finding in
`docs/resolution_shortcut.md`.

*29 August.* The organisers' rules slide reads: "Data: only public/licensed datasets
(e.g., WildFake, CIFAKE, SID_Set)." WildFake and SID_Set are named as examples of
approved datasets, without qualification. That settles the reading: the rule bars
datasets whose **own** declared licence is non-commercial (GenImage's CC BY-NC-SA,
Community Forensics' research-only clause), not the upstream provenance of the sets the
organisers themselves listed — a test SID_Set (derived from Flickr30k, itself
non-commercial research only) would fail just as WildFake does. The bar was lifted, and
the frozen manifest of 29 Aug 00:32, against which every feature bank is fingerprinted,
includes the bucket: 65,049 authentic images from two sources.

**What stays enforced.** `aigcdet.data.sources.SourceSpec.restricted_buckets` and
`restriction` remain, and `scripts/build_dataset.py` still drops a restricted bucket at
scan time and records counts and reasons in `docs/splits.json`; no registered source
uses them now, and `tests/data/test_sources.py` pins that, because a restriction
reappearing would make the next rebuild disagree with every bank on disk.

**Consequence to state in the Devpost writeup:** the training corpus is the two
organiser-listed datasets. Its authentic side is 85% WildFake's re-published subsets,
whose upstream terms are non-commercial; that is permitted by the competition's rules as
the organisers stated them, and is a constraint on any use beyond the competition.

## Vertical-real sources audited for a 9:16 stream (2026-08-30)

The corpus is 0.3% at 9:16 and COCO train2017 holds only 207 images within
+-0.03 of it (10,111 portrait at all), so a TikTok-aspect stream needs an
authentic source that is natively vertical. Generated images can be produced at
any aspect for free; authentic ones cannot, so this is the binding constraint —
and centre-cropping landscape reals to 9:16 while generating fakes natively
would build a "was this cropped?" detector, a content confound none of the
three proxies can see.

| Source | Declared terms | Verdict |
| --- | --- | --- |
| [Pexels](https://help.pexels.com/hc/en-us/articles/27292485713945-AI-and-ML-FAQ) | Photos are free for commercial use, but the AI/ML FAQ separately prohibits collecting content at scale "to train, fine-tune, evaluate, or develop ML/AI models or datasets" without explicit permission; content also may not be redistributed on a standalone basis | **Barred.** The photo licence is permissive and irrelevant: the ML prohibition is a separate term and it names this exact use. |
| [Unsplash](https://github.com/unsplash/datasets/blob/master/TERMS.md) | Lite (25k, nature-themed) permits commercial ML training for internal business purposes; Full is non-commercial only; both state the dataset "cannot be used to redistribute the images contained within" | **Barred.** The no-redistribution clause blocks publishing a normalised corpus as a Kaggle Dataset, which is how the fleet gets its data. Lite is also 25k nature photos — the wrong content domain, and too small once filtered to portrait. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | Images CC BY 2.0, annotations CC BY 4.0 (Google) | **Passes.** CC BY 2.0 permits commercial use AND redistribution with attribution — the only one of the three that allows republishing. 9.2M images, so even a small portrait fraction clears the volume needed. Precedent: SID_Set is itself derived from COCO, OpenImages V7 and Flickr30k, is CC BY 4.0, and is organiser-listed. Google disclaims per-image licence verification, which is the same posture COCO carries toward Flickr terms and which this project already accepted for COCO. |

**Not yet measured:** Open Images' aspect-ratio distribution. The volume is
clearly sufficient in principle; the portrait and near-9:16 yield should be
counted from the metadata before any download is planned.

## COCO train2017 as a training source — a rule reversed, and the control that replaces it

Registered 2026-08-30 as `coco_train2017`, used by
`configs/datasets/coco_crop.yaml` and by nothing else.

**The licence is not the issue.** COCO is CC BY 4.0 and the obligation is
attribution, satisfied here and in the README. The issue is a rule this
project wrote for itself.

`src/aigcdet/data/wildfake.py` (`_COCO_FORBIDDEN`) bars COCO-derived reals
from training **entirely** — not merely deduplicated against the benchmark —
and says why:

> train2017/test2017/val2017 are one photographic distribution, so training on
> any of them would let the demo-set score measure distribution memorisation
> instead of generalisation.

The organisers' scored benchmark's authentic half **is** COCO val2017. That
argument is sound and has not been retracted.

**What was traded for what.** The alternative on offer was WildFake's
authentic half: 55,000 images, 40,000 of them re-published from FFHQ,
CelebA-HQ, AFHQ, ImageNet and LSUN, whose upstream terms are non-commercial
(table above), and every one of those 40,000 stored at exactly short side 200.
So the choice was between an authentic class that is licence-encumbered and
band-limited, and one that is permissively licensed and photographic but
overlaps the benchmark. The `coco_crop` stream takes the second; the frozen
stream keeps the first; both exist, and they are compared.

**What replaces the rule.** A registry entry cannot detect memorisation, so
nothing is claimed for it. The control is a measurement:

```bash
python scripts/stratified_auc.py --stratify-by source \
    --checkpoint <rung> --bank <bank> --manifest data/manifest_coco_crop.parquet
```

It fixes one threshold at 1% false positive rate over **all** authentic rows —
the same operating point spec §6.4's selection rule uses — and then reports
that threshold's false positive rate separately for `coco_train2017`,
`wildfake` (LAION) and `sid_set` reals. A model reading generation artefacts
has roughly the same rate on all three. A model reading "is this a COCO
photograph" has a far lower rate on COCO than on the other two, while the
benchmark looks excellent. **The spread between those rates is the number that
must be published beside any headline from this stream**, and a headline
quoted without it is not interpretable.

Two things were deliberately NOT relaxed:

- `coco_val2017` keeps `exclude_from_training=True`. Spec §4.1(2) forbids
  training on the scored benchmark itself, and that is untouched.
  `tests/data/test_sources.py` asserts the two halves as a PAIR, so a future
  change that makes them agree in either direction fails.
- The `wildfake.py` COCO markers stay exactly as they are. They bar
  WildFake's own re-published COCO copy, which would now duplicate ours.

**The dedupe guard does not help here and is not being relied on.** COCO
train2017 and val2017 are disjoint image sets, so `find_leaks` at Hamming
distance 4 catches essentially nothing. The risk is distribution memorisation,
which a perceptual hash cannot measure.

## Commercial generator APIs (task 03) — terms, per provider

`docs/03-commercial-apis-on-open-images.md` §3.3 asks four questions of each
provider before a card is charged, and §5.4 makes the answers an acceptance
criterion. Checked 2026-08-30. **Rows marked ⚠ are not yet verified at source
and must not be cited until they are** — the discipline at the top of this file
applies here exactly as it does to the datasets.

| Provider | Commercial use of output | Redistribution | Watermark? | Trains on our submissions? |
| --- | --- | --- | --- | --- |
| **Google** (Gemini API) | Yes — Google disclaims ownership | No explicit bar found | **Yes — SynthID, no opt-out** | ⚠ not read |
| **OpenAI** (`gpt-image-2`) | **Yes — Output assigned to Customer** (Services Agreement §4.1) | No explicit bar; but see the Permitted Exception below | ⚠ not read (C2PA metadata suspected; normalisation strips it) | **No** — §4.2 |
| **Ideogram** (4.0) | Permitted; no explicit IP assignment | Permitted, **but attribution is mandatory** | ⚠ not read | **No — committed in writing** |
| **Bytedance** (Seedream) | **No terms document found for the image models** — see below | unknown | unknown | unknown |

### Google — Gemini API

Read at <https://ai.google.dev/gemini-api/terms> and
<https://ai.google.dev/gemini-api/docs/image-generation>, 2026-08-30.

Ownership: **"Google won't claim ownership over that content."**

Competing models: **"You may not use the Services to develop models that
compete with the Services (e.g., Gemini API or Google AI Studio)."** A detector
is not a competing *generator*; the reading is recorded rather than assumed.

Watermarking, and this is the row that matters for the results table:
**"All generated images include a SynthID watermark."** No opt-out is offered.
DeepMind's page adds that the mark is embedded at creation, is *"imperceptible
to humans"*, and is *"designed to stand up to modifications like cropping,
adding filters, changing frame rates, or lossy compression"* — i.e. it is built
to survive the same 20-condition grid these images are scored under. Cite the
API doc, not the DeepMind page: only the former states universal application.

Consequence: Google's rows carry a provider-specific synthetic signal that has
nothing to do with generation artefacts. Brief §5.2 forbids pooling and §5.7
requires the flag; both exist for this.

### Ideogram — Developer API agreement

Read at <https://ideogram.ai/legal/api-tos>, 2026-08-30.

Does **not** train on us — §2.2: **"the Company agrees that it shall not use
any User Input or User Output to train the Ideogram AI Model"**, with a narrow
exception for flagged policy violations. This is the strongest commitment of
the four and worth having on record.

Competing products — §2.3.6(A) bars using User Input or User Output **"to
develop any product, service, or technology that competes with the Company,
the Ideogram AI Model, Ideogram API, or any of the Company's products"**.

Attribution is an **obligation, not a courtesy** — §2.3.1(a) requires you
**"identify on the Developer Application that any User Output generated…was
created by the Ideogram AI Model"**. §2.3.6(H) separately encourages disclosing
AI origin. Anything published from this provider carries that notice.

Ownership: the agreement grants no explicit assignment of output IP, which is
weaker than OpenAI's position. Absence of an assignment is not a grant. Keep
Ideogram rows local until someone confirms the ownership position.

### OpenAI — read at source 2026-08-30 (both pages 403 automated fetch; retrieved by hand)

**Read the right document.** The consumer **Terms of Use** state up front:
*"Our Business Terms govern use of ChatGPT Enterprise, our APIs, and our other
services for businesses and developers."* The **OpenAI Services Agreement**
(effective 2026-01-01) confirms it applies *"only ... to use of OpenAI's APIs"*.
So the API is governed by the **Services Agreement**, and the consumer ToU is
the wrong document to quote here.

That distinction is not cosmetic. The consumer ToU bars *"Automatically or
programmatically extract data or Output"* — read as binding, that would bar this
entire task. It does not apply to the API.

**Ownership — §4.1:** *"As between Customer and OpenAI, to the extent permitted
by applicable law, Customer: (a) retains all ownership rights in Input; and (b)
owns all Output. OpenAI hereby assigns to Customer all OpenAI's right, title,
and interest, if any, in and to Output."*

**Does not train on us — §4.2:** *"OpenAI will only use Customer Content as
necessary to provide Customer with the Services, comply with applicable law,
enforce the OpenAI Policies, and prevent abuse. **OpenAI will not use Customer
Content to develop or improve the Services, unless Customer explicitly agrees to
such use.**"* As strong as Ideogram's §2.2.

**Competing models — §3.3(e), and read the exception with it.** The restriction
is *"except for a Permitted Exception, use Output to develop artificial
intelligence models that compete with OpenAI's products and services"*. §17
defines:

> *"**Permitted Exception**" means Customer using Output to: (a) develop
> artificial intelligence models **primarily intended to categorize, classify,
> or organize data (e.g., embeddings or classifiers), if these models are not
> distributed or made commercially available to third parties**; and (b) fine
> tune or customize models provided as part of OpenAI's fine-tuning or other
> Services set forth on the Pricing Page.*

**A detector is a classifier, so this project sits inside exception (a) — but
the exception carries a condition, and the condition is about the model, not
the data:** the classifier must not be *distributed or made commercially
available to third parties*. Two consequences to carry into the writeup:

* Task 03's images are **eval-only and never trained on** (`docs/03` §1), so the
  weakest form of the restriction barely engages — we are measuring with Output,
  not developing from it. The condition is recorded because the safe reading
  treats evaluation during development as part of development.
* If the deliverable ships model weights or an inference bundle publicly, check
  that against this clause **before** publishing. This is a constraint on the
  *artifact*, and it is separate from whether the images may be redistributed.

**Redistribution:** not barred. Customer owns Output. §10 (No Publicity) bars
using OpenAI's name or logo in marketing material — it does not restrict the
images.

Sources: <https://openai.com/policies/row-terms-of-use/> and
<https://openai.com/policies/services-agreement/>, both effective 2026-01-01,
retrieved manually 2026-08-30 (both return HTTP 403 to automated fetch).

### Bytedance / Seedream — no terms found, and that is disqualifying as it stands

Searched 2026-08-30. BytePlus publishes general ModelArk Terms and Conditions
and **"Specific Terms for the BytePlus Video Generation Model Services"** — but
no equivalent specific terms for the **image** models. Citing the video terms
for Seedream would be reasoning by analogy, which is exactly what the discipline
at the top of this file forbids.

**The reseller does not help.** OpenRouter's terms punt upstream rather than
covering it: §6.1 — *"Your ownership rights in the Output are set forth in the
Model Terms for each Model you use"* — and §5.1 — *"By accessing or using any
Model through the Service, you agree ... to comply with the applicable terms for
each Model."* So buying through OpenRouter binds us to Bytedance terms we cannot
read, which is worse than buying direct, not better. OpenRouter also adds its own
bar on *"developing a competing service"* (§7) and states *"Where possible,
OpenRouter has opted out of model training with the Models it uses."*

**Decision, 2026-08-30: buy Seedream anyway.** The project owner reviewed the
above and chose to proceed without a terms document rather than lose the
provider. That is a legitimate call — Seedream appears in four of NTIRE 2026's
five held-out splits and is the single most benchmark-representative generator
on the list, and dropping it for Recraft moves the eval set away from the
composition the whole task is trying to approximate.

Recorded rather than resolved, because the finding stands even though the
decision overrides it: **this row does not satisfy `docs/03` §5.4**, and the
writeup must say so rather than presenting four cleanly-licensed providers.

**One mitigation, and it costs nothing.** Redistribution is the term we cannot
check, so treat it as unpermitted: **Seedream rows stay local and never enter a
shared Kaggle Dataset.** `docs/03` §3.3 already carries this pattern for a
provider that grants commercial use but bars redistribution — the same handling
applies to one whose position is simply unknown. Commercial *use* of the output
is not the exposure here; publishing the images is.

**Still worth ten minutes if anyone has them:** registering at BytePlus ModelArk
puts the binding terms in front of you at signup. If that yields a document,
quote it here and delete this note. Recraft remains the fallback if the terms
turn out to bar the use.

### Not purchased, and why

* **Stability API** — serves SD 3.5, which task 02 puts in *training* by name.
  Not held out for us at any price.
* **Black Forest Labs** — outputs are usable commercially, but BFL takes a
  *"perpetual, irrevocable, worldwide"* licence to inputs **and** outputs and
  states it may train on them, and bars output from being used as synthetic
  training data for a model of *"substantially similar functionality"*. Excluded
  on those terms, **not** on lineage: FLUX.2's latent space was retrained from
  scratch, so the lineage argument does not survive contact with it.
* **Recraft** — reserve only. Lineage-clean and slightly cheaper than Seedream,
  but appears in none of NTIRE 2026's held-out splits. Terms published but not
  yet read; read them if Seedream's cannot be obtained.

## Receipts on disk

`data/raw/LICENCES.json` and `data/demo/LICENCES.json` were written during acquisition,
before these confirmations, and still hold the "confirm before use" placeholder text
for `sid_set` and `wildfake`. They must be regenerated from `sources.LICENCES` after
acquisition finishes and **before** `scripts/build_dataset.py` runs, because
`build_dataset` copies the receipt text into per-row provenance in the frozen manifest —
and the manifest is frozen once.
