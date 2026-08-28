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
| WildFake (compilation) | Apache-2.0 | ModelScope hub metadata, `https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake` → `"License":"Apache License 2.0"` | Yes for the compilation — but see the caveat below |
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
project actually downloads are exactly the ones with restrictive upstream terms:

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

**Why this is nonetheless fine for this submission.** The hackathon is a research
prototype, not a commercial deployment, and non-commercial research use is permitted by
every licence above. What we publish is *code, weights of small trained heads, and
metrics* — never redistributed images. Nothing in `data/` is committed (`.gitignore`
excludes all of `data/`) and no Kaggle Dataset built from these images should be made
public without revisiting this table.

**Consequence to state in the Devpost writeup:** the training data is usable for this
prototype, and a production deployment would need either a differently-licensed
authentic corpus or per-subset clearance. That is a real limitation of the approach,
and §5.5 asks for exactly this kind of reflection.

## Receipts on disk

`data/raw/LICENCES.json` and `data/demo/LICENCES.json` were written during acquisition,
before these confirmations, and still hold the "confirm before use" placeholder text
for `sid_set` and `wildfake`. They must be regenerated from `sources.LICENCES` after
acquisition finishes and **before** `scripts/build_dataset.py` runs, because
`build_dataset` copies the receipt text into per-row provenance in the frozen manifest —
and the manifest is frozen once.
