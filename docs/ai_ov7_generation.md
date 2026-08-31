# AI-OV7 — open-weight generators over Open Images V7

What was built, what was measured, and what is still owed. Companion to
`handoffs/02-open-weight-generators-on-open-images.md`, which set the task.

Code: `src/aigcdet/generate/`, `scripts/generate_ov7.py`,
`configs/datasets/ov7.yaml`. Tests: `tests/generate/`.

---

## 1. Why this exists

Every fake in `manifest_union.parquet` comes from a 2017–2023 generator, and
published results put detection at ~79% on 2020–21 generators against ~38% on
2024 ones. Every public dataset that would close that gap is licence-barred,
structurally: their reals are web scrapes whose images keep individual
copyrights, so no compiler can grant commercial rights downstream. Generating
our own fakes over reals we may redistribute is the way around it.

It also attacks the confound `README.md` lists as unresolved. Sharpness alone
predicts the label at 68.5% on the frozen corpus. Here each fake is generated
**from one real**, at that real's dimensions and through that real's JPEG
encoder, so content, geometry and compression history are held fixed across the
pair and what is left is the generator.

## 2. The reals

60,000 Open Images V7 thumbnails at
`/mnt/berstorage/techjam/open_images/portrait/`, filtered at harvest to
`License == CC BY 2.0`, aspect ≤ 0.7 and short side ≥ 400.

* **CC BY 2.0 is the whole point**: commercial use *and* redistribution with
  attribution. `build_pool` asserts it rather than filtering on it — a row
  under another licence means the harvest changed, and the run should stop.
* `attribution.csv` is complete: 60,000 rows, zero blank `Author` or
  `OriginalURL`. It is copied into the corpus root; CC BY requires it and
  normalisation strips image metadata.
* **All portrait, none square.** Generated sizes run 0.625–0.714 aspect,
  median 416×640, max 448×640. Zero square and zero landscape outputs.

**Eligibility: 54,624 of 60,000 (91.0%).** The 9% dropped are all encoder
reproducibility, measured not assumed:

| reason | n |
|---|---|
| RGB with a chroma layout PIL cannot name (`get_sampling` → −1) | 4,884 |
| grayscale (one quantisation table, no chroma) | 448 |
| CMYK | 44 |

Both classes of drop would put a difference between real and fake that has
nothing to do with the generator: a grayscale real against an RGB fake is a
total colour leak, and PIL reads subsampling −1 on write as "use the default",
quietly giving the fake 4:2:0 against a real that is something else. With a
60,000 pool and a ~10,000 target, 9% is affordable.

## 3. The two bugs this rebuild exists to fix

The previous attempt (branch `feat/ai-ov7-generation`, smoke-run 2026-08-30)
produced 6 pairs and **6 of 6 leaked the label**:

1. **Dimension leak.** Fakes came out a multiple of 8 — the VAE downsamples by
   that — while reals were copied byte-for-byte at their native size.
   `width % 8 == 0` separated the classes at ~100%.
2. **DCT phase mismatch.** `jpeg_quality` AUC came back **0.0000** — perfect
   inverse separation — despite quantisation tables being copied correctly,
   because `(w - cw) // 2` put each fake's fresh 8×8 grid at a different phase
   from its real's preserved one.

Both are one arithmetic expression in `geometry.crop_box`, which now returns a
box whose **offset and size** are both multiples of 16, and the real reaches it
through `jpegtran -crop` (rewrites no DCT coefficient, no PIL fallback by
design). `encode.assert_parity` runs on **every** pair, not a sample: it is the
check that would have caught both at the first image rather than at the gate.

A third was found during this rebuild and is worth recording because it is the
same shape: PIL **de-zigzags** quantisation tables on read, so comparing them
against a zigzag baseline scrambles the quality estimate silently — it read
q=50 as 42.9. Pinned by round-tripping known qualities through PIL in
`tests/generate/test_encode.py`.

## 4. The suite

| family | model | licence | lineage | method | share |
|---|---|---|---|---|---|
| `sdxl_t2i` | SDXL base 1.0 | openrail++ | `sdxl_vae` | t2i | .30 |
| `sd15_t2i` | SD 1.5 | creativeml-openrail-m | `sd_vae` | t2i | .20 |
| `sdxl_self_cond` | SDXL inpainting 0.1 | openrail++ | `sdxl_vae` | zero-mask regen | .14 |
| `sdxl_img2img` | SDXL base 1.0 | openrail++ | `sdxl_vae` | SDEdit @ 0.75 | .10 |
| `sd15_img2img` | SD 1.5 | creativeml-openrail-m | `sd_vae` | SDEdit @ 0.75 | .08 |
| `klein4b_t2i` | FLUX.2-klein-4B | **apache-2.0** | `flux2_vae` | t2i | .12 |
| `klein4b_ref_image` | FLUX.2-klein-4B | **apache-2.0** | `flux2_vae` | reference | .06 |

Every tag verified against the Hub card, and `run.check_licence()` re-asserts
it at load — a relicensing upstream fails the run rather than the audit.

Fully synthetic (t2i + self_cond) 0.76, image-conditioned 0.24.
**Held out: the whole `flux2_vae` lineage** — a different decoder *and* a
different architecture (flow-matching DiT against UNets), so it measures a
lineage jump rather than `docs/02` §3.4's cousin. Two lineages remain in
training, which `validate_suite` enforces as a minimum.

### Refused, and recorded rather than deleted

* **SDXL-Turbo** — `docs/02` §2 lists it as OpenRAIL++. Its card says
  `sai-nc-community`: **non-commercial**. That doc row is wrong. It is the fast
  SDXL variant anyone would reach for first, so the entry stays, refused.
* **FLUX.1-dev**, **FLUX.2-klein-9B** — non-commercial. Only the *4B* of the
  klein family is Apache.
* **FLUX.1-schnell** — Apache, but 12B is 23.8 GB at bf16 and this card has 20.
  Refused on hardware, not licence; fp8 would put the compute budget into the
  traces. Restore it on a 40 GB card.
* **Kandinsky 2.2** (MoVQ, apache-2.0) — the natural fourth lineage, left in
  the registry but out of the suite: it needs a separate prior pipeline that
  did not fit this session.

## 5. Prompts

`docs/02` §3.2 says to use Localized Narratives. They cover 5.60% of the Open
Images train split and **6.48% of the CC BY 2.0 thumbnail rows** — ~3,900 of
60,000. Building the t2i arm on them would either shrink it to a fifth or fill
the rest with an empty prompt, which is how the old `inpaint_box` arm ended up
generating on no conditioning at all.

Captions instead come from **Florence-2-large (MIT)**, `<DETAILED_CAPTION>`,
cached to `data/ov7_captions.parquet`. Greedy: beam search at width 3 measured
4× slower (0.46 vs 0.117 s/image) for text that is a prompt, not ground truth.

The `microsoft/Florence-2-large` upload predates native transformers support
and loads with the vision tower's conv norms mismatched (ckpt 1024 vs model
512); `ignore_mismatched_sizes` would "fix" that by randomly reinitialising
them and captioning from a partly-untrained backbone. Use
`florence-community/Florence-2-large`, the official re-upload of the same
weights, still MIT.

## 6. Measured throughput (RTX A4500, bf16, at ~416×640)

| family | s/image |
|---|---|
| `sd15_img2img` | 1.43 |
| `sdxl_img2img` | 1.67 |
| `sd15_t2i` | 1.79 |
| `sdxl_t2i` | 2.15 |
| `sdxl_self_cond` | 2.18 |
| `klein4b_t2i` | 4.03 |
| `klein4b_ref_image` | 6.62 |

FLUX.2-klein-4B: 14.9 GB resident, 15.6 GB peak — fits 20 GB at bf16 with no
quantisation. Captioning 0.117 s/image. ~10,000 pairs ≈ 7 h.

## 7. Things a later reader needs to know

* **SD 1.5's VAE is the exact checkpoint `features/recon.py` probes with.** So
  `sd15_*` fakes reconstruct at near-zero error and the A4/A7 recon rungs
  separate them for free. That is the recon feature working as designed, but it
  does not generalise off that lineage — report recon per lineage or the number
  means nothing. Flagged as `recon_probe_collision` in the registry.
* **`self_cond` pairs sit at pHash Hamming 0–2**, inside `dedupe`'s
  `max_distance=4`. They survive only because `find_leaks` is run against the
  *demo* set and never within the corpus. Anyone adding within-corpus dedupe
  will silently delete that entire arm.
* **This is a separate stream.** Feature banks fingerprint `manifest_sha256`
  over `rel_path` in row order, so these rows must never be inserted into
  `manifest_union.parquet` — it would orphan every bank on disk.
* **`assign_splits` now takes a `group_key`.** Drawn per row, a real and its
  own fake straddle the val boundary ~18% of the time. Held-out membership
  propagates to the group, so the real paired with a held-out fake is held out
  too. `group_key=None` preserves the original RNG stream exactly.
* **`select` is prefix-stable.** Families are dealt in repeating strata over
  the hash order, not contiguous slices, so `--total 10000` resumes a
  `--total 2000` run instead of reassigning it, and any prefix is balanced
  across all seven families.

## 8. First gate reading (226 pairs, `klein4b_*` only, band mode)

Taken mid-run on the FLUX.2 families alone — the most modern generator in the
suite and the held-out lineage, so the hardest arm to keep clean, but *not* the
full suite. The `--n 2000` gate across all seven families is what §9.1 owes.

| proxy | AI-OV7 | frozen corpus |
|---|---|---|
| `jpeg_quality` | 0.5653 | 0.5532 |
| `laplacian_var` | **0.6513** | 0.6721 |
| `noise_floor` | **0.5919** | 0.6374 |
| `short_side` | **0.5000** | 0.5992 |

**`short_side` is exactly 0.5000.** That is the §3 dimension leak gone: the
previous attempt separated the classes at ~100% on geometry alone, and this is
chance. `jpeg_quality` at 0.5653 is above the frozen 0.5532 but far below the
~0.60 line `docs/02` §5.2 draws, so encoder parity is holding — against the
previous attempt's 0.0000, perfect inverse separation.

Sharpness and noise both come in **below** the frozen baselines, which is the
confound reduction the pairing was built to produce. Read it as an early
signal, not a result: one lineage, 226 pairs.

## 9. SD 1.5 emits pure-black frames at ~0.8%, and the guard catches them

Measured over the full run: **22 of 2,800 SD 1.5 generations came back with
pixel std exactly 0.00** — a completely black frame — 17 of 2,000 in
`sd15_t2i` and 5 of 800 in `sd15_img2img`. FLUX.2-klein produced none in 1,800,
and SDXL none.

This is SD 1.5's known half-precision VAE instability: a NaN in the decoder
comes out as a flat frame rather than an error. `run.check()`'s `MIN_STD = 2.0`
rejects them, and `run_family` deletes **both** files of the pair, so nothing
half-written survives. Audited at the end of the SD 1.5 arm: **0 orphaned
files, 0 rows whose files are missing.** The families land at 1,983/2,000 and
795/800 — 99.2% and 99.4%.

**It was deliberately not fixed mid-run.** Decoding that VAE in fp32 would
likely remove it, but switching dtype partway through means some `sd15_*` fakes
are decoded at bf16 and the rest at fp32 — a *within-family* forensic
difference, which is a worse problem than 0.8% attrition in a corpus whose
entire purpose is holding everything but the generator fixed. If the arm is
ever regenerated, decode SD 1.5's VAE in fp32 for the whole family or none of
it.

## 9a. Final gate, full suite (n=4,000, band mode)

The reading §8 owed. All seven families, 4,000 images sampled proportionally
from `manifest_ov7.parquet`.

| proxy | AI-OV7 | frozen corpus |
|---|---|---|
| `jpeg_quality` | **0.5038** | 0.5532 |
| `laplacian_var` | **0.5998** | 0.6721 |
| `noise_floor` | **0.5229** | 0.6374 |
| `short_side` | **0.5022** | 0.5992 |

**Every proxy comes in below the frozen baseline, and two are at chance.**

`jpeg_quality` 0.5038 is encoder parity holding across the whole suite --
better than §8's 0.5653, which was `klein4b_*` alone, and against the previous
attempt's 0.0000. `short_side` 0.5022 is the dimension control: the §3 leak
separated the classes at ~100% on geometry alone and it is now chance.

`laplacian_var` 0.5998 is the residual and the honest number. It is 0.072
below the frozen corpus but it is not chance: at matched content, matched size
and matched compression history, these generators still produce measurably
smoother images than the photographs they were made from. That is a property
of the generators rather than of how the corpus was assembled, which is what
the corpus was built to be able to say. It is the floor a head has to beat,
not a leak that has been removed.

## 10. Corpus as frozen

11,978 pairs / 23,956 rows over **nine families and five decoder lineages**,
from 11,978 distinct reals -- each used by exactly one family, so the arms are
disjoint rather than stacked. 8.5 GPU-hours on one RTX A4500.

| | pairs | rows |
|---|---|---|
| `train` | 9,133 | 18,266 |
| `val_internal` | 1,045 | 2,090 |
| `heldout_generator` (`flux2_vae`) | 1,800 | 3,600 |

| lineage | decoder | pairs |
|---|---|---|
| `sdxl_vae` | AutoencoderKL 4ch | 5,400 |
| `sd_vae` | AutoencoderKL 4ch | 2,778 |
| `flux2_vae` | AutoencoderKLFlux2 32ch, 2x2-patched | 1,800 (held out) |
| `movq` | VQModel, vector-quantised | 1,000 |
| `dc_ae` | AutoencoderDC 32x, 32ch | 1,000 |

Every split is exactly 50/50 by label, which is the `pair_split_by_stem` draw
working: groups of two move together, so balance is a consequence rather than
a sampling choice.

**Pair survival is 100%.** `build_dataset` dropped 0 rows below the short-side
floor and 0 as demo near-duplicates -- so the `self_cond` arm, which sits at
pHash Hamming 0-2 from its reals, survived intact, as §7 predicted it would as
long as `find_leaks` is only ever run against the demo set.

Audited on the raw tree before the build: **0 orphaned files, 0 rows whose
files are missing, 0 reals used twice**, and encoder parity re-asserted on a
random 60 pairs with 0 failures. Normalised size ~8.5 GB; raw JPEG pairs
~1.8 GB. Zero square and zero landscape rows across all 23,956; sizes run
400x560 to 448x640.

### Gate, all nine families (n=4,000, band mode)

| proxy | 7 families | **9 families** | frozen corpus |
|---|---|---|---|
| `jpeg_quality` | 0.5038 | 0.5152 | 0.5532 |
| `laplacian_var` | 0.5998 | **0.5632** | 0.6721 |
| `noise_floor` | 0.5229 | **0.5072** | 0.6374 |
| `short_side` | 0.5022 | **0.5015** | 0.5992 |

**The supplement improved the corpus, and most on the statistic that needed
it.** Sharpness -- the worst confound both here and on the frozen corpus --
fell from 0.5998 to 0.5632, and the noise floor to 0.5072, effectively chance.
The mechanism is the obvious one and worth stating because it is the general
argument for more lineages: a single sharpness threshold separates two classes
well only while the generated class is narrow. MoVQ and DC-AE decode unlike
SDXL and SD 1.5, so the generated distribution widened and the threshold lost
purchase. `jpeg_quality` moved the other way, 0.5038 to 0.5152, still far
under the 0.5532 baseline and close enough to chance to be unremarkable.

## 11. The lineage supplement (`--suite ov7_lineage`)

Three decoder lineages give exactly one leave-one-lineage-out rotation, so
"generalises to an unseen decoder" is a point estimate. The supplement adds
two more, over reals the first run never touched, so it becomes a
distribution over five.

| family | model | licence | lineage | decoder class |
|---|---|---|---|---|
| `kandinsky22_t2i` | Kandinsky 2.2 | apache-2.0 | `movq` | `VQModel` -- vector-quantised |
| `sana1600m_t2i` | Sana 1.6B | apache-2.0 | `dc_ae` | `AutoencoderDC` -- 32x, 32 latent ch |

Both are t2i only. An image-conditioned fake shares composition with its real
and so carries less of its decoder's fingerprint; for an arm whose only job is
to represent a decoder, that dilutes the measurement.

These are TRAINING lineages. `HELDOUT_LINEAGE` stays `flux2_vae`; which
lineage a rung actually holds out is a rung-level choice
(`RungConfig.train_exclude_generators`), and adding these is what makes
rotating it possible.

### Correction, 2026-08-31: `flux2_vae` is not AutoencoderKL 16ch

The lineage table in §10 recorded the held-out decoder as "AutoencoderKL
16ch". That is FLUX.**1**'s signature. Both checkpoints were already in this
box's HF cache, so the check was two `cat`s against `vae/config.json`:

| | FLUX.1-dev (= Lumina's, = Z-Image's) | FLUX.2-klein-4B |
|---|---|---|
| `_class_name` | `AutoencoderKL` | **`AutoencoderKLFlux2`** |
| `latent_channels` | 16 | **32** |
| `patch_size` | absent | **[2, 2]** -> 128 effective |
| normalisation | GroupNorm only | **BatchNorm**, `eps` 1e-4 |
| `scaling_factor` | 0.3611 | **absent** |
| `use_quant_conv` | false | true |

The one field they share, `block_out_channels` [128, 256, 512, 512], is the
stock LDM encoder trunk that SD 1.5 and SDXL carry too, so it holds no lineage
information.

**What this does and does not settle.** `registry.LINEAGE_COUSINS` says the
two are "both AutoencoderKL at 16 latent channels and nobody has run them
against each other". The first half is false and came from the table cell
above; the second half stands. Different architectures are not proof of
different FINGERPRINTS -- a detector trained on one may still transfer to the
other, and that is an empirical question about artefacts, not about configs.
So the cousin risk `zimage_turbo` accepts is **smaller than the docstring's
premise implied and not retired**. `features/recon.py` on the two VAEs is
still the thing that retires it. Recorded here rather than downgraded in code,
because the warning firing one time too many costs nothing and the warning not
firing costs the held-out rung.

**Lumina-Image 2.0 was rejected, and the reason is the point of the exercise.**
Apache-2.0, 5.2 GB, a brand-new architecture -- and its `vae/config.json` is
`AutoencoderKL`, 16 latent channels, `scaling_factor` 0.3611, and its
`_name_or_path` is literally `black-forest-labs/FLUX.1-dev`. It IS FLUX.1's
VAE, so it is `flux1_vae` -- an existing lineage, adding no rotation point. That is
`docs/02` §3.4's mistake with newer weights: **architecture novelty is not
decoder novelty**, and only the second is what the held-out rung measures.
PixArt-Σ is out for the same reason (SDXL's VAE); SD 3.5, Stable Cascade and
HunyuanDiT on licence; Qwen-Image at 40.9 GB on hardware.

### Three things the smoke run found

1. **Sana resizes by default, and it would have been the worst confound in the
   corpus.** `use_resolution_binning` defaults True: the request is mapped to
   the nearest 1024-based aspect bin, generated there, and resized back
   (`pipeline_sana.py`, `resize_and_crop_tensor`). Measured on this corpus's
   geometry, a 432x640 request is generated at **1216x832 -- landscape, for a
   portrait request** -- and squashed down, leaving var-Laplacian **2083.8
   against 555.9** native. Nearly 4x sharper, on the generated class only, in
   a corpus whose worst remaining confound is sharpness at 0.5998. Turned off
   in `ModelSpec.call_kwargs`, which makes 32-divisibility a hard error
   instead.
2. **Kandinsky silently rounds the requested size up to a multiple of 64.**
   432x640 and 416x640 both came back 448x640 -- the same image twice.
   `ModelSpec.size_multiple` now declares each model's granularity,
   `run.generate` asks at it and crops back to the real's exact box, and the
   frozen suite is pinned at 8 so its sizes cannot move.
3. **`check_licence` was verifying half the weights.** Kandinsky's combined
   pipeline pulls its prior from a second repo that the registry never named.
   `ModelSpec.companion_ids` closes that; both Kandinsky repos are apache-2.0.

### Running it

The supplement MUST run on a shard clear of the reals the first suite used.
The family deal is a repeating stratum pattern built from the suite's shares
(`pool.select`), so a different suite re-deals every real in the block: one
that is `sdxl_t2i` today can come out `sana1600m_t2i` tomorrow, and nothing
downstream objects because `_done_ids` is per family. The corpus would then
hold one scene twice on the generated side against one real.

The ov7 suite consumed order positions 0-9,999 of 54,624 eligible.
`--shard 1 --n-shards 5` starts at 10,925. `generate_ov7.used_elsewhere`
enforces this rather than trusting it, and refuses the run naming the
offending reals.

```bash
python scripts/generate_ov7.py --suite ov7_lineage --total 2000 \
    --shard 1 --n-shards 5 --out data/raw_ov7_src \
    --captions data/ov7_captions.parquet
```

Run 2026-08-31: **2,000 of 2,000, zero failures.** Kandinsky 2.07 s/image,
Sana **1.17 s/image** -- the fastest family in the corpus, once resolution
binning is off and it generates at the real's own size rather than at
1216x832. Neither produced a single near-constant frame, against SD 1.5's
0.79%.

## 12. Still owed

1. Per-source FPR, reported separately for Open Images, WildFake and SID_Set.
   Needs a trained checkpoint -- `scripts/stratified_auc.py --stratify-by
   source` takes one -- so it belongs to the Stage B session, not this one.
2. The end-to-end question: does a head trained with AI-OV7 beat one without it
   on the organisers' benchmark? If not, that is `docs/02` §6's second negative
   result and an argument for task 03 — worth knowing early either way.

## 13. Two more supplements, and the harness for running them (2026-08-31)

§11 added two lineages and argued that breadth, not volume, is what moved the
gate. Two more supplements follow that argument. Nothing below has generated a
pair yet: this is the registry and the harness, not a corpus.

| suite | families | lineage | decoder |
|---|---|---|---|
| `ov7_lineage2` | `wuerstchen_t2i` | `paella_vq` | `PaellaVQModel`, 4 latent ch, ~42x spatial |
| | `cogview4_t2i` | `cogview_vae` | `AutoencoderKL` 16ch, `scaling_factor` 1.0 |
| `ov7_lineage3` | `zimage_t2i` | `flux1_vae` | `AutoencoderKL` 16ch, 0.3611 — **FLUX.1-dev's** |

Twelve families and eight decoder lineages once all four suites have run;
seven trained against one held out, so leave-one-lineage-out becomes a
seven-point rotation. `HELDOUT_LINEAGE` stays `flux2_vae`.

### The third supplement is a bet, and it is labelled as one

`Tongyi-MAI/Z-Image-Turbo` is 6B, apache-2.0, 9 steps at guidance 0.0, about
1 s/image — the cheapest lineage breadth on offer. Its decoder is not new.
`vae/config.json` carries `_name_or_path: "flux-dev"`, so this is FLUX.1-dev's
VAE by the config's own provenance rather than by inference from
`scaling_factor`, which is how §11 judged Lumina-Image 2.0 and how `shuttle3`
was judged.

So `flux1_vae` is now a **training** lineage while `flux2_vae` is the held-out
rung, and the distance between those two decoders has never been measured.
Both are `AutoencoderKL` at 16 latent channels. If they are close, the
held-out rung measures a cousin rather than an unseen decoder — `docs/02`
§3.4's mistake, reached from the other side.

That is recorded in the code rather than only here. `registry.LINEAGE_COUSINS`
names the pair as an unmeasured claim, and `validate_suite` warns on every run
that trains a cousin of the held-out lineage. It warns rather than raising
because it is a decision; the three suites without `zimage_t2i` do not warn,
which is what keeps the warning meaningful. **Any held-out number produced
while `ov7_lineage3` is in the corpus carries this caveat.**

The instrument that retires it is `features/recon.py` — the same probe that
flagged `recon_probe_collision` for SD 1.5 — run on FLUX.1's VAE against
FLUX.2's. It costs no generation time and it also rules on `shuttle3` and
`lumina2`, both refused on the same unmeasured question.

### Refused, and recorded rather than deleted

* **Qwen-Image-2.0** — no `ModelSpec`, because there are no weights. Queried
  across the whole Hub on 2026-08-31: `Qwen/Qwen-Image-2.0`, `-2602`, `-2601`
  and `Qwen-Image-VAE-2.0` all 401, and the third-party re-uploads that trail
  any Qwen image release stop at `2512`. Its VAE would have been a genuine new
  lineage — Qwen published a dedicated report (arXiv 2605.13565) for a
  purpose-built autoencoder benchmarked *against* Wan2.2 — so it returns the
  day the weights land.
* **`Qwen/Qwen-Image-2512`** — refused on hardware, and it settles §11's
  parenthetical. §11 wrote "Qwen-Image at 40.9 GB on hardware"; the transformer
  shard index totals **40.86 GB** at bf16, so that figure was exact. Its
  `vae/config.json` is `AutoencoderKLQwenImage` with `latents_mean` identical
  to four decimals across all sixteen channels to `Wan2.1-T2V-1.3B`'s VAE —
  the same frozen encoder, which is why the lineage key is `wan_vae` and why
  adding any Wan2.1 model would build a cousin of it rather than a lineage.
* **`Wan-AI/Wan2.2-TI2V-5B`** — refused as a video model. Its decoder *is*
  genuinely new (`base_dim` 160 against Wan2.1's 96, `decoder_base_dim` 256,
  `in_channels` 12, residual), so `wan22_vae` is a separate key on evidence,
  and it fits a 24 GB card. But a still from it is one frame out of a
  temporally-causal VAE trained on H.264 video, and motion blur and codec
  artefacts would enter the generated class only, in a corpus whose worst
  remaining confound is sharpness. That is the shape of §11's Sana
  resolution-binning near-miss.

### What changed in the harness

* `ModelSpec.offload_mode` — "model" moves one component at a time and needs
  only the largest to fit; "sequential" moves submodules and is 10-30x slower.
  Choosing per model is what puts `cogview4_6b` on a 20 GB card at a sane rate.
* `--run-families` on `generate_ov7.py` — subsets the selection *after*
  `select()`. Every box passes an identical `--suite`/`--total`/`--families`,
  so all of them compute the same deal and each executes a slice. That is the
  axis for splitting one shard across heterogeneous hardware; `--families`
  changes the deal and is for smoke runs only.
* `pool.rebase_paths()` — a cached `ov7_pool.parquet` carries absolute paths
  from the box that built it, and `run_family` opens `row.path` directly. It
  now re-points them and fails on five files rather than at row 40 of a
  six-hour run.
* `dtype` and `gpu` on every row. Neither was written before, so the frozen
  11,978-pair corpus cannot be stratified by hardware after the fact; from
  here on it can.
* `scripts/caption_ov7.py` — captions in parts, then merges. `caption_pool`
  rewrites the whole parquet on every log tick, so four workers on one path
  lose captions to last-writer-wins, and the reals whose captions were lost
  then fail generation on an empty prompt. Precompute, merge, then read.
* `notebooks/ov7_scale_up.ipynb` — the four-GPU driver, ported from
  `kaggle_generate_pairs.ipynb` on `feat/ai-ov7-generation` **without its
  inline logic**. That notebook's own `crop_box` and encoder handling are the
  ones §3 records as leaking the label on 6 of 6 pairs; every cell here drives
  the tested package instead.
