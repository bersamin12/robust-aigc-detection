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

## Receipts on disk

`data/raw/LICENCES.json` and `data/demo/LICENCES.json` were written during acquisition,
before these confirmations, and still hold the "confirm before use" placeholder text
for `sid_set` and `wildfake`. They must be regenerated from `sources.LICENCES` after
acquisition finishes and **before** `scripts/build_dataset.py` runs, because
`build_dataset` copies the receipt text into per-row provenance in the frozen manifest —
and the manifest is frozen once.
