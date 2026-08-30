# Corpus presets: which dataset, as a file rather than a remembered flag

`configs/rungs/*.yaml` already puts the MODEL side of every ablation under
version control. The DATA side was a command line someone typed once. The
frozen manifest of 29 Aug was built with a particular cap, a particular pair
of held-out families and no source balancing at all, and none of that is
recoverable from the manifest itself.

A preset is that discipline one stage earlier:

```
python scripts/build_dataset.py --preset configs/datasets/max_data.yaml \
    --raw <staged root> --out data/normalized_max_data \
    --demo-dir data/demo --manifest data/manifest_max_data.parquet
```

Its `name` and `note` are copied into `docs/splits.json`, so a bank found on
disk six months from now can be traced back to the composition it came from.

Three presets ship. `max_data` and `era_forward` were chosen from the
measurements below, not from intuition — and one of those measurements
overturned the design this work started with. `coco_crop` is a separate
experiment stream with its own standardisation and augmentation policies; it
has its own section at the end.

Both operate on `data/raw` as it stands. Neither downloads anything, and
neither needs the corpus assembled from more than one tree: WildFake's
authentic half is nested one level deeper than its generated buckets
(`wildfake/real/<subset>/`), which `_scan`'s recursive glob handles and
`classify` reads as bucket `real` regardless of depth.

---

## What is actually wrong with the corpus

Measured 2026-08-30 from the frozen manifest and the cached view-0 proxies of
`data/banks/siglip2l` (view 0 is the undegraded view by the bank's own
invariant, so this is each image's native low-level signature and costs no GPU
time). AUC is orientation-corrected, `max(a, 1-a)`.

| Signal | pooled (131,116) | wildfake only (110,988) | sid_set only (20,128) |
| --- | --- | --- | --- |
| `jpeg_quality` | 0.5532 | 0.5414 | **0.6212** |
| `laplacian_var` | 0.6721 | **0.6944** | 0.5548 |
| `noise_floor` | 0.6374 | 0.6214 | **0.7314** |
| short side | 0.5992 | 0.6525 | 0.5047 |

**Neither source is clean, and they are not dirty in the same way.** WildFake
leaks the label through sharpness; SID_Set leaks it through its noise floor
and JPEG history. Every pooled figure sits *below* the worse of the two
singles, because each source dilutes the other's leak.

Two things follow, and the second is the one that mattered.

**Resolution is not the residual problem inside SID_Set.** Its short-side AUC
is 0.5047 — chance. Both of its classes sit at or above 512 (reals: median
512, 0.9% below; fakes: 512 for every image), and `data.normalize` caps the
short side at 512, so both classes land on the same value and canonicalisation
starts from a common ceiling. WildFake is where the resolution story lives:
72.7% of its reals and 87.5% of its fakes are natively below 512.

**Balancing the two sources against each other does not help.** This was the
original plan and the sweep killed it.

## The sweep that killed source balancing

36 compositions, varying the cap on WildFake's authentic rows (15k–55k) and on
each WildFake generated family (1,000–3,500), all with the sub-band floor on,
all with SID_Set's full on-disk 29,318 + 29,439 rows. Scored on the worst of
the four confound AUCs. The five best and the five worst:

| wf reals | wf per family | rows | `jpeg` | `lap` | `noise` | short | **worst** | modern-era share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 55,000 | 3,500 | 168,437 | 0.5706 | 0.6411 | 0.6596 | 0.5566 | **0.6596** | 35.0% |
| 40,000 | 3,500 | 153,437 | 0.5707 | 0.6427 | 0.6622 | 0.5174 | 0.6622 | 35.0% |
| 55,000 | 3,000 | 160,997 | 0.5736 | 0.6366 | 0.6626 | 0.5652 | 0.6626 | 38.4% |
| 30,000 | 3,500 | 143,437 | 0.5711 | 0.6439 | 0.6642 | 0.5198 | 0.6642 | 35.0% |
| 25,000 | 3,500 | 138,437 | 0.5712 | 0.6448 | 0.6653 | 0.5431 | 0.6653 | 35.0% |
| … | | | | | | | | |
| 25,000 | 1,000 | 99,757 | 0.5949 | 0.6006 | 0.6908 | 0.5531 | 0.6908 | 64.8% |
| 30,000 | 1,000 | 104,757 | 0.5947 | 0.6011 | 0.6915 | 0.5729 | 0.6915 | 64.8% |
| 20,000 | 1,000 | 94,757 | 0.5959 | 0.6022 | 0.6932 | 0.5289 | 0.6932 | 64.8% |
| 15,000 | 1,000 | 89,757 | 0.5962 | 0.6043 | 0.6956 | 0.5010 | 0.6956 | 64.8% |

**The optimum is to cap nothing.** The worst confound rises monotonically with
every cap, because thinning WildFake raises SID_Set's share and SID_Set's leak
(`noise_floor` 0.731 within-source) is the worse of the two. Trading one
channel for the other is all a two-source mix can do: `laplacian_var` falls
from 0.641 to 0.604 across the table while `noise_floor` rises from 0.660 to
0.696. **There is no mix that lowers both.**

This is why `max_data` caps nothing, and it is the reason the
`max_real_per_source` knob exists anyway — the negative result is only
readable because the knob could express the alternative.

`docs/low_level_confounds.md` argued for balancing on the *index* side
(stratify `laplacian_var`, discard within strata). That argument is untouched
by this: index balancing removes rows to break a statistic's link to the
label, whereas source balancing changes which statistic dominates. They are
different levers and the index one is still worth running — on the CPU, from
a bank that already exists.

## The one free win

`augment.canonical` band-limits every image to `CANON_BAND_SIDE` (200) and
then upscales, which equalises the band for everything **at or above** that
ceiling. Below it, nothing can be equalised and nothing restored — the
docstring calls this residue irreducible.

In the frozen manifest **1,308 images sit below 200px and every one of them is
generated**: 1,260 BigGAN at exactly 128px, plus 48 SD-adaptor images. A
permanent label leak with no authentic counterpart, which index balancing
cannot touch because there is nothing on the other side of the class to
balance against.

`min_short_side: 200` drops them. It costs 0.9% of the corpus and both shipped
presets set it; `tests/data/test_presets.py` pins it to `CANON_BAND_SIDE`, so
a preset cannot quietly drift off the band the number means.

## The other axis: generator era

The organisers' scored benchmark is DALL·E 3. Our generated side is 85%
WildFake, whose 17 families are all 2021–2023 — GANs, DDPM/DDIM/ADM, the
SD-1.x adaptors, VQGAN/VQVAE, MAE/MAGE, imagen, GigaGAN. SID_Set is the only
modern-era (SD3.5 / Flux) generated data in the corpus and it is **15.3% of
the trained fakes**.

That is the trade the two presets bracket: `max_data` takes the lowest
confound and the most rows at 35% modern; `era_forward` pays ~0.023 of worst
confound and 36% of the rows to reach 55.7%.

---

## The two presets

| | frozen (29 Aug) | `max_data` | `era_forward` |
| --- | --- | --- | --- |
| train+val rows | 131,116 | **168,437** | 107,185 |
| real / fake | 65,049 / 66,067 | 84,318 / 84,119 | 54,318 / 52,867 |
| class imbalance | 0.8% | **0.12%** | 1.35% |
| `jpeg_quality` | 0.5532 | 0.5706 | 0.5873 |
| `laplacian_var` | **0.6721** | 0.6411 | **0.6133** |
| `noise_floor` | 0.6374 | **0.6596** | **0.6829** |
| short side | 0.5992 | 0.5566 | 0.5247 |
| **worst confound** | 0.6721 | **0.6596** | 0.6829 |
| modern-era share of fakes | 15.3% | 35.0% | **55.7%** |
| fake families | 17 | 17 (min 2,240) | 17 (min 949) |
| held-out | controlnet, VQGAN | *same* | *same* |

The held-out pair is pinned to the frozen manifest's own draw in both, so
`max_data` vs frozen and `era_forward` vs `max_data` are each a
single-variable comparison: composition, and nothing else.

**Method and its caveat.** These are simulated by applying each preset's rules
to the frozen manifest's rows and recomputing the AUCs from cached proxies.
The 38,629 SID_Set rows that are on disk but were never normalised have no
proxies, so they are stood in for by a seeded resample of the SID_Set rows
that do. That is unbiased for an AUC — the resample is drawn from the same
source — but it understates the variance, and the simulated pool excludes the
7,000 held-out rows. Treat the table as the predicted ordering, not as
measurements of corpora that exist. The real numbers come out of
`docs/splits.json` and a proxy audit once each build finishes.

## Running one

`data/raw` is complete — WildFake's six authentic subsets, its 17 generated
families, all 58,757 SID_Set images, and COCO val2017 — so a preset build
needs nothing staged:

```bash
python scripts/build_dataset.py --preset configs/datasets/max_data.yaml \
    --raw data/raw \
    --out data/normalized_max_data \
    --demo-dir data/demo \
    --manifest data/manifest_max_data.parquet
```

Note that this picks up the SID_Set images acquired after the freeze
automatically: `_scan` reads the tree, not the frozen manifest, and 29,318 +
29,439 images are sitting there. That is the whole of `max_data`'s row gain.

Then Stage A and the ladder as usual, against the new manifest. **Each preset
needs its own bank**: a bank indexes its manifest positionally, and
`assert_fusion_parents` matches on `manifest_sha256`, so banks from different
presets can never be fused with each other.

Rough costs per preset: ~22 GB for the normalised tree, ~2.7 GB for a
1024-dim bank at 11 views, plus the extraction itself.

---

## `coco_crop` — a second stream, not a third point on the same curve

`max_data` and `era_forward` differ from the frozen corpus in composition
alone. `coco_crop` changes composition **and** two pipeline policies, so it is
not comparable to them row for row — it is a separate stream that asks whether
a photographic real class and native-detail standardisation beat what we have.

| | frozen | `coco_crop` |
| --- | --- | --- |
| authentic | 55,000 WildFake + 10,049 SID | 46,800 COCO + 15,000 LAION + 29,318 SID |
| generated | 62,988 WildFake + 10,079 SID | 61,680 WildFake + 29,438 SID |
| authentic sources | 2 | 3 |
| standardisation | band-limit 200 → upscale 512 | random 200×200 crop → upscale 512 |
| geometric aug | none | dihedral-8, per view |
| total | 138,116 | ~182,200 |

### The three changes

**1. COCO train2017 replaces WildFake's five non-commercial subsets.** This
reverses a rule the project states explicitly. The full argument, what was
traded for what, and the control that replaces the rule are in
`docs/dataset_licences.md`. The short version: `coco_train2017` is registered
as a training source, `coco_val2017` is not and never will be, and every
headline from this stream must be quoted with

```bash
python scripts/stratified_auc.py --stratify-by source ...
```

beside it — the false positive rate per authentic source at one threshold. A
model reading "is this a COCO photograph" shows a far lower rate on COCO than
on LAION and SID while the benchmark looks excellent.

**2. Standardisation becomes a random crop.** `CanonPolicy(mode="crop")`
takes a 200×200 window at NATIVE resolution and upscales it to 512, replacing
the band-limit-then-upscale of `mode="band"`. Step 2 is unchanged, so every
image still arrives at one size through one kernel.

What this buys: the band-limit is a box filter, and `augment/canonical.py`
records its residue — after canonicalisation the natively-200 arm is sharper
than the band-limited-from-512 arm in 99.5% of pairs. A crop removes no detail
at all from the pixels it keeps, so a generator's high-frequency signature
survives inside the window.

What it costs, and this is not a small thing: **field of view now correlates
with native resolution.** A 200px WildFake image contributes its whole frame;
a 429px COCO photograph contributes a detail. That is a *content* confound
substituted for a *spectral* one, and the degradation proxies cannot see it —
`eval/controls.py:content_blind_auc` is the instrument that can, and it is
worth running.

`min_short_side: 200` in the preset and `crop_side: 200` in the policy are the
same number, and must be: `canonicalise` raises rather than upscaling an image
to reach a window it cannot fill, so the preset is what stops those rows ever
being normalised.

**3. Dihedral augmentation, per view.** A random flip and 90-degree rotation
on each of the 11 views, after standardisation and before the recipe.

90-degree multiples specifically, and that is the whole design: `np.rot90` and
`np.fliplr` are index permutations, so `laplacian_var` is bit-exactly
unchanged and `noise_floor` agrees to one float32 ULP. An arbitrary-angle
rotation resamples, which attenuates exactly the channel that is the corpus's
largest confound. (`jpeg_quality` moves by 0.66 points out of 100 under the
transposing half — pinned by a test, and two orders of magnitude below that
estimator's own error.)

Per view rather than per image because Stage A caches features once: whatever
a view holds is what the head sees for every epoch of Stage B, so 11 views buy
11 orientations at no extra extraction cost where a per-image transform buys
one.

**The consequence to state rather than discover.** The A3 consistency loss
compares a clean view against a degraded one, and with per-view crops and
orientations those two views now also differ geometrically. A3 in this stream
asks for invariance to degradation AND to re-cropping AND to orientation. That
is a stronger objective and arguably the right one for a detector — a flipped
fake is still fake — but it is **not the same quantity** A3 measures in the
band-mode stream. The two A3 numbers must not be read against each other.

### Two asymmetries, deliberate and tested

The training bank wants diversity; the evaluation bank and the served path
want a controlled, repeatable comparison.

- **Eval bank: centre crop, no dihedral.** The grid measures how far a score
  falls under `jpeg_q30`. If the window or the orientation also moved between
  conditions, that measurement would be confounded with "a different picture".
  Averaging a score over crops or orientations is a real technique and it is
  A6's, applied at inference to the eval set and the served path alike.
- **Inference: centre crop, no dihedral.** Serving must return the same score
  for the same file twice, and an eval number is a prediction about serving.

### Why a policy object and not a flag

`CanonPolicy` is written into the feature bank's config, and `BankWriter`
treats every unrecognised config key as must-match. That buys three refusals
for free: resuming a bank under a changed policy, merging shards built under
different ones, and fusing an A5 pair whose pixels were never comparable.
Without it a crop bank and a band bank are indistinguishable on disk — same
dtype, same width, same row count — and the failure is silent. `export_bundle`
carries the policy for the same reason: a head trained on crops and served
band-limited images is being shown a distribution it has never seen, and both
policies hand the backbone the same size.

### Running it

```bash
python scripts/stage_corpus_root.py --out data/raw_coco_crop \
    --source wildfake=data/raw/wildfake \
    --source sid_set=data/raw/sid_set \
    --source coco_train2017=/mnt/berstorage/coco/train2017

python scripts/build_dataset.py --preset configs/datasets/coco_crop.yaml \
    --raw data/raw_coco_crop --out data/normalized_coco_crop \
    --demo-dir data/demo --manifest data/manifest_coco_crop.parquet \
    --docs-dir docs/coco_crop

python scripts/extract_features.py --manifest data/manifest_coco_crop.parquet \
    --backbone siglip2l --out data/banks/coco_crop_siglip2l \
    --split train,val_internal --canon-mode crop --crop-side 200 --geometric
```

`--docs-dir` is not optional: `docs/splits.json` and `docs/data_audit.md` are
the frozen stream's provenance and the default would overwrite them with a
different corpus's, silently. Staging is separate because `data/raw` is the
frozen stream's corpus and `max_data` describes it as "every image on disk" —
`_scan` reads the tree, not the manifest, so a source dropped in there changes
what that preset means.

`extract_eval_bank` takes the same `--canon-mode`/`--crop-side` and must be
given the same values as the training bank.

The eval manifest joins this stream's training manifest with the organisers'
benchmark:

```bash
python scripts/build_eval_manifest.py \
    --manifest data/manifest_coco_crop.parquet \
    --benchmark-manifest data/demo/benchmark_manifest_rebased.parquet \
    --out data/eval_manifest_coco_crop.parquet
```

**`_rebased`, and not the original, on purpose.** `data/demo/benchmark_manifest.parquet`
still carries absolute paths under the pre-split root
`/mnt/berstorage/techjam/track5/data/demo/`, and `read_manifest` rebases from
`AIGCDET_DATA_ROOT` — one environment variable, which cannot serve two
manifests rooted at different trees at once. Regenerating the benchmark
manifest would fix the paths and change its fingerprint, orphaning
`data/eval_manifest.parquet` and every eval bank built against it, including
the frozen stream's. So a rebased COPY is written instead and the original is
left exactly as it is. The digests are recomputed by `build_eval_manifest`
either way, so the copy is equivalent to the original in everything that is
checked.

---

### The three things to check before believing a number from it

**Gate 1 — CPU, before any GPU time. The one that can cancel the run.**
`scripts/gate_confounds.py` decodes a stratified sample and runs it through
the pipeline's real front half — `canonicalise` under the proposed policy,
then `dihedral`, then `proxy_vector` — reporting orientation-corrected AUC per
proxy, pooled and per source. It measures view 0's ACTUAL window, drawn from
the same per-view key extraction would use, not a centre crop that merely
resembles it.

```bash
python scripts/gate_confounds.py \
    --manifest data/manifest_coco_crop.parquet \
    --canon-mode crop --crop-side 200 --geometric --max-auc 0.70
```

`--max-auc` is what makes it a gate and not a report: it exits non-zero and
says why.

**Measured on the built corpus, 2026-08-30, n=6000 proportional.** The same
182,150 rows through the two standardisation policies, so this isolates the
policy from the corpus:

| Signal | band | **crop + dihedral** | Δ |
| --- | --- | --- | --- |
| `jpeg_quality` | 0.5703 | **0.5135** | −0.057 |
| `laplacian_var` | **0.7038** | **0.6510** | **−0.053** |
| `noise_floor` | 0.6748 | 0.6774 | +0.003 |
| **worst** | **0.7038** | **0.6774** | −0.026 |
| within `wildfake`: `laplacian_var` | 0.8008 | **0.6549** | **−0.146** |
| within `sid_set`: `noise_floor` | 0.7333 | 0.7341 | +0.001 |

Read that top-to-bottom, because it is the argument for the whole stream:

- **Under band mode this corpus FAILS the gate.** 0.7038 is worse than the
  frozen corpus's 0.6721, and above a `--max-auc 0.70`. Swapping WildFake's
  soft 200px authentic mass for photographs makes the sharpness shortcut
  easier, exactly as predicted before it was built.
- **Crop standardisation is what rescues it**, and by the predicted mechanism:
  it removes no detail from the pixels it keeps, so the gap between a
  200px-native image and a 512px-native one shrinks. The effect is largest
  exactly where the prediction said it would be — inside WildFake, whose
  authentic side is now all 512px LAION and whose generated side is mostly
  256px, `laplacian_var` falls 0.146.
- **It nearly erases the JPEG-history confound** the pre-normalisation audit
  flagged. 61,715 of the 91,032 authentic images are JPEG-sourced and none of
  the generated ones are, so this one mattered: a 200px window of native
  pixels upscaled 2.56x destroys the 8x8 quantisation grid the statistic
  reads.
- **`noise_floor` is untouched by the policy** (+0.003) and is now the worst
  channel at 0.6774, driven by SID_Set's own 0.7341. Standardisation cannot
  help there; that is a source-composition question, and
  `docs/low_level_confounds.md` records why no two-source mix lowers both
  channels at once.

One number this table does not license: `short_side` reads 0.6505 pooled and
**0.9357 within WildFake**, because dropping the five 200px authentic subsets
left that source's real class at 512 and its generated class at 256. Crop
standardisation hands the backbone a 200x200 window from every image alike, so
that is what a dimensions-only CONTROL would score and NOT what reaches the
head — the same distinction `docs/low_level_confounds.md` draws for the frozen
corpus. It does make Gate 2 mandatory rather than advisable.

**Gate 2 — the content confound Gate 1 is blind to.** Three pixel statistics
cannot see that a 200px crop of a 429px photograph is a detail while a 200px
crop of a 200px image is a whole frame. `eval/controls.py:content_blind_auc`
and `metadata_control` can; both need a bank, so they run after extraction and
before the headline. Crop makes every image the same size by construction, so
`metadata_control` must sit at chance — anything else means a size cue
survived a transform designed to remove it.

**Gate 3 — the memorisation control, and the one that is not optional.**

```bash
python scripts/stratified_auc.py --stratify-by source \
    --checkpoint <rung> --bank data/banks/coco_crop_<backbone> \
    --manifest data/manifest_coco_crop.parquet
```

One threshold at 1% false positive rate over all authentic rows — the same
operating point spec §6.4's selection rule uses — applied unchanged to each
authentic source. A model reading generation artefacts has roughly the same
rate on COCO, LAION and SID_Set reals. A model reading "is this a COCO
photograph" has a far lower rate on COCO while the benchmark, whose real half
IS COCO val2017, looks excellent. **The spread between those rates is the
number to publish beside any headline from this stream**, and a headline
quoted without it is not interpretable. See `docs/dataset_licences.md` for why
the rule this replaces existed.

## Datasets considered and rejected

The rule this project reads the competition by (`docs/dataset_licences.md`):
the organisers bar datasets whose **own declared licence** is non-commercial,
not the upstream provenance of the sets they themselves listed.

| Dataset | Declared licence | Verdict |
| --- | --- | --- |
| [OpenFake v2](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | `license: cc-by-nc-4.0` in the Hub metadata; the card body says CC-BY-SA-4.0 with proprietary subsets non-commercial | **Barred.** The machine-readable tag is the declaration, and it is non-commercial — the same bar as GenImage. The tag contradicting the body is a second reason: an ambiguous licence is not a "public/licensed dataset". Otherwise ideal: 2.5M rows, ~80 generators including Flux/SD3.5/GPT-Image/nano-banana, real side from Pexels/DOCCI/LAION/Reddit, plus a 36,240-row in-the-wild Reddit test split. |
| [Defactify / MS-COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) | none declared | **Barred.** 96k COCO-matched images from SD2.1/SDXL/SD3/DALL·E 3/MJv6 — exactly the era gap — but no licence tag at all fails the rule outright. |
| [NTIRE 2026 Robust AIGen Detection](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train) | "research and educational use only", plus an explicit no-redistribution clause (challenge terms §5; nothing is declared on the HF card or in the CVPRW paper) | **Conditional.** Two clauses, and only one of them actually binds. *Research-only* is non-commercial in substance — the same bar as Community Forensics and GenImage — and is the ground on which this is a judgement call rather than a clear pass. *No redistribution* was previously read here as an outright bar because the fleet publishes its normalised corpus as a Kaggle Dataset; that reading was wrong. Kaggle Datasets are created private (kaggle_fleet_runbook.md), and private hosting for the uploader's own use is storage, not distribution — no differently from an S3 bucket. The step that *would* distribute is sharing the Dataset with the four other fleet accounts. The fix costs nothing and is the pattern the runbook already uses for DINOv3: **each teammate accepts the NTIRE terms under their own HuggingFace account**, so the share is between five licensed users rather than a hand-off to unlicensed third parties. Do not make an NTIRE-derived Dataset public. Painful, because it is the best-matched dataset found: 108,750 real + 185,750 generated from 42 generators released 2022-2026, real half filtered from ~12M CC12M/CommonPool/RedCaps images, 36 transformations, and resolution / aspect-ratio / JPEG-quality distributions aligned between the halves BY CONSTRUCTION rather than repaired downstream. |
| Community Forensics | research-only | **Barred** (already recorded in `dataset_licences.md`). |
| GenImage | CC BY-NC-SA | **Barred** (already recorded). |
| CIFAKE | MIT, and named on the organisers' rules slide | **Permitted, not built.** 120k images at 32×32. Both classes sit at one resolution so it adds no resolution leak, but every image is far below `CANON_BAND_SIDE`, i.e. entirely inside the band the canonicaliser cannot reach. Worth a targeted stress run; not worth a box. |

**Why the wall is structural, not bureaucratic.** Every barred dataset that
would close the generator-era gap draws its authentic half from web-scraped
image-text corpora — OpenFake v2 from Pexels/LAION/Reddit, NTIRE 2026 from
CC12M/CommonPool/RedCaps. Those images stay under their individual copyrights,
so the compilers *cannot* grant commercial rights they never held; research-only
is the strongest licence they are able to offer. Asking them to relicense is
therefore not a paperwork request and will not succeed.

The exception is worth naming because it points at the way out: **Defactify's
authentic half is MS COCO (CC BY 4.0) and its generated half is the authors'
own**, so it is the one barred dataset whose licence *could* be made permissive
by its owners. That is also the recipe for building one ourselves — licence-clean
reals plus our own generations — and it is what `coco_crop` already does for the
real half.

The practical conclusion: **the licence wall means the corpus is the two
organiser-listed sources we already hold.** That is why both presets are
recompositions of on-disk data and neither downloads anything.
