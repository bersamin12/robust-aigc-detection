# Dataset licences

Recorded per spec §4.5, same discipline as `docs/model_licences.md`: the licence is
read at its source and quoted, not inferred from a secondary summary. Checked
2026-08-28.

Every row of the frozen manifest carries its source's licence string from
`aigcdet.data.sources.LICENCES`, and `scripts/acquire_data.py` writes a
`LICENCES.json` receipt into each acquisition directory at download time.

| Source | Licence | Where it was read | Commercial use? |
| --- | --- | --- | --- |
| SID_Set | CC BY 4.0 | https://huggingface.co/datasets/saberzl/SID_Set — licensing section on the dataset card | Yes, with attribution |
| WildFake — generated buckets | Apache-2.0 | ModelScope hub metadata, `https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake` → `"License":"Apache License 2.0"` | Yes — the authors' own images, covered by the compilation licence |
| WildFake — `real/` bucket | **Barred; not used** | Upstream terms of the re-published subsets (table below) | **No.** Dropped at scan time under the 28 Aug rule |
| COCO val2017 | CC BY 4.0 (images under Flickr terms) | https://cocodataset.org/#termsofuse | Yes, with attribution |
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
Apache-2.0 label on the compilation does not relicense them. The real subsets this project downloaded are exactly the ones with restrictive
upstream terms. **None of them reach the manifest** — they are listed here as the
record of what was acquired and then barred, not as a description of the training
corpus:

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

**What this file used to argue, and why it no longer holds.** This section previously
argued that the hackathon is a research prototype rather than a commercial deployment,
that non-commercial research use is permitted by every licence above, and that we
publish only code, small trained heads and metrics — never redistributed images. That
argument was about what the licences permit. It was overtaken on 28 August by a rule
about what the competition permits.

**The 28 Aug webinar Q&A: "Non-commercial datasets cannot be used."** That is a
competition rule, not a licence interpretation, and it has no research-use exemption to
appeal to. Every WildFake authentic subset in the table above is either explicitly
non-commercial (FFHQ, CelebA-HQ, AFHQ, ImageNet, LSUN) or carries per-image rights we
cannot establish (LAION-5B, whose CC BY 4.0 covers the metadata and not the scraped
images). So the whole bucket is barred.

**What that cost, and what it did not.** WildFake's authentic bucket was 55,000 images
— 42% of the training pool and 85% of its authentic half. WildFake's *generated*
buckets are the authors' own work and are unaffected; they remain the entire source of
generator diversity, and both `heldout_generator` and the LOTO rung still work. The gap
on the authentic side is filled with more SID_Set (CC BY 4.0), which means every
authentic image in the corpus now comes from a single commercially-usable source.

We could absorb this because **no rung had been trained yet**: Stage A caches
frozen-backbone features, so the fix is a manifest rebuild and a re-extraction rather
than a model being retracted.

**How the rule is enforced, rather than remembered.**
`aigcdet.data.sources.SourceSpec.restricted_buckets` declares the bar and `restriction`
records why; `scripts/build_dataset.py` drops those rows at scan time, *before
normalisation*, so no normalised copy is ever written, and it records both the counts
and the reasons in `docs/splits.json`. A restriction naming a bucket the source does not
declare raises — a `"reals"` typo would otherwise bar nothing while every image it was
meant to name flowed through.

**Consequence to state in the Devpost writeup:** the authentic half of the training
corpus is single-sourced (SID_Set, CC BY 4.0) as a direct result of this rule. That is a
real limitation — source diversity on the authentic side is exactly what the sharpness
finding in `docs/resolution_shortcut.md` says we most need — and §5.5 asks for this kind
of reflection.

## Receipts on disk

`data/raw/LICENCES.json` and `data/demo/LICENCES.json` were written during acquisition,
before these confirmations, and still hold the "confirm before use" placeholder text
for `sid_set` and `wildfake`. They must be regenerated from `sources.LICENCES` after
acquisition finishes and **before** `scripts/build_dataset.py` runs, because
`build_dataset` copies the receipt text into per-row provenance in the frozen manifest —
and the manifest is frozen once.
