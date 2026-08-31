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
  traces. It is also gated now. **Superseded 2026-08-31:** `zimage_turbo`
  reaches the same `flux1_vae` lineage at ~15.4 GB and is ungated — see §11b.
* **Kandinsky 2.2** (MoVQ, apache-2.0) — was out of the suite because it needs
  a separate prior pipeline. **Resolved:** it is `ov7_lineage`'s first arm and
  has generated 2,500 pairs. `companion_ids` on the `ModelSpec` is what that
  separate prior cost us — before it existed, `check_licence` verified only
  `hf_id`, so half the weights that made an image were licence-checked and
  half were not.

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

**Lumina-Image 2.0 was rejected, and the reason is the point of the exercise.**
Apache-2.0, 5.2 GB, a brand-new architecture -- and its `vae/config.json` is
`AutoencoderKL`, 16 latent channels, `scaling_factor` 0.3611, and its
`_name_or_path` is literally `black-forest-labs/FLUX.1-dev`. It IS FLUX.1's
VAE, so it is `flux1_vae` -- an existing lineage, adding no rotation point.
That is `docs/02` §3.4's mistake with newer weights: **architecture novelty is
not decoder novelty**, and only the second is what the held-out rung measures.
PixArt-Σ is out for the same reason (SDXL's VAE); SD 3.5, Stable Cascade and
HunyuanDiT on licence; Qwen-Image at 40.9 GB on hardware.

### Correction, 2026-08-31: Lumina is NOT a cousin of the held-out lineage

The paragraph above originally read "as a held-out lineage it is a cousin of
`flux2_vae`", and the lineage table in §10 recorded `flux2_vae` as
"AutoencoderKL 16ch". Both were wrong. The check is one `cat` against the two
`vae/config.json` files already in this box's HF cache:

| | FLUX.1-dev (= Lumina's VAE) | FLUX.2-klein-4B |
|---|---|---|
| `_class_name` | `AutoencoderKL` | **`AutoencoderKLFlux2`** |
| `latent_channels` | 16 | **32** |
| `patch_size` | absent | **[2, 2]** -- 128 effective channels |
| normalisation | GroupNorm only | **BatchNorm** (`batch_norm_eps` 1e-4) |
| `scaling_factor` | 0.3611 | **absent** |
| quant convs | `use_quant_conv: false` | `true` |

The one thing they share, `block_out_channels` [128, 256, 512, 512], is the
stock LDM encoder trunk -- SD 1.5 and SDXL carry it too -- so it holds no
lineage information. **FLUX.1 and FLUX.2 are separate decoders.** That is
exactly the measurement `docs/02` U4 said was owed, and it settles it without
the recon probe.

What this reverses: every refusal that rested on *cousinhood with the held-out
lineage* -- Lumina here, Z-Image-Turbo in `docs/02` U4. Those models are
`flux1_vae`, and `flux1_vae` is a training lineage, so using them does not leak
`flux2_vae`.

What it does not reverse: the **redundancy** verdict. They add exposure to a
lineage that already exists, not a sixth rotation point. That is still worth
having -- `flux1_vae` currently carries 0.28% of the gradient -- but it is a
different argument from the one originally written here, and it comes with a
caveat: once `flux1_vae` is well represented, the `flux2_vae` held-out number
is a *near*-transfer result (sibling lab, sibling design) and must be reported
against a rung that excludes `flux1_vae`, or the two cannot be told apart.

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
python scripts/generate_ov7.py --suite ov7_lineage --total 5000 \
    --shard 1 --n-shards 5 --out data/raw_ov7_src \
    --captions data/ov7_captions.parquet
```

Run 2026-08-31, in two passes: 2,000 then 3,000. **5,000 of 5,000, zero
failures**, leaving 2,500 pairs in each family. Kandinsky 2.19 s/image, Sana
**1.18 s/image** -- the fastest family in the corpus, once resolution binning
is off and it generates at the real's own size rather than at 1216x832.
Neither produced a single near-constant frame across all 5,000, against SD
1.5's 0.79%.

The second pass is additive on the first: `--total` is this shard's own
target and `_done_ids` is per family, so re-running with a larger total tops
each family up rather than re-dealing it. `_rows/stats.json` records only the
LAST pass, which is a reporting gap and not a data one -- the per-family
`rows_*.jsonl` are the complete record.

## 11b. The flux1_vae exposure arm (`--suite ov7_lineage3`)

Not a new lineage, and that is the point. `flux1_vae` is a training lineage on
paper carrying 0.28% of the gradient, and **no OV7 family produced it at
all**: every FLUX.1 route was refused, and Z-Image was refused on a premise
that turned out to be false (see the correction above). This arm fixes the
exposure, not the lineage count.

| family | model | licence | lineage | decoder class |
|---|---|---|---|---|
| `zimage_t2i` | Z-Image-Turbo 6B | **apache-2.0**, ungated | `flux1_vae` | `AutoencoderKL` 16ch, scaling 0.3611 |

Its `vae/config.json` names itself: `_name_or_path` is `flux-dev`. So it IS
FLUX.1's VAE. Held out it would measure nothing `flux1_vae` does not already
cover; trained on, it is the cheapest route into the emptiest lineage.
Transformer is 22.9 GiB on disk at fp32, ~11.5 GB resident at bf16, plus a
~3.7 GB text encoder and a 0.16 GB VAE -- ~15.4 GB, which clears a 24 GB 4090
outright and this 20 GB card with model offload. Turbo-distilled: 8 steps,
guidance recorded rather than obeyed.

**It obliges a control.** Once `flux1_vae` is well represented, the
`flux2_vae` held-out number is a NEAR-transfer result -- sibling lab, sibling
design intent -- and must be reported against a rung that excludes
`flux1_vae`. `registry.RUNG_FLUX1_EXCLUDED` names the family to drop; the
difference between the two rungs IS the sibling transfer, and it costs one
rung on cached features.

### Three more candidates, read at the card on 2026-08-31

* **HunyuanImage 2.1** — `AutoencoderKLHunyuanImage`, **64 latent channels**,
  scaling 0.75289, six block levels (32x downsample). Nothing in this corpus
  is near it; the widest we hold is 32 (`dc_ae`, `flux2_vae`). It is the one
  genuinely unrepresented decoder in reach. **Blocked on licence:** the card
  tag is `other` (Tencent community licence), `validate_suite` refuses
  anything not `commercial`, and that licence has not been read at source.
  49.5 GiB. Read it before booking anything.
* **Qwen-Image** — there is no `Qwen-Image-2.0` repo; the id 401s. Qwen's
  image repos are date-coded, and the newest, `Qwen-Image-2512` (apache-2.0),
  ships a `vae/config.json` identical to the original's: `AutoencoderKLQwenImage`,
  `base_dim` 96, `dim_mult` [1,2,4,4], **`z_dim` 16**, and a
  `temperal_downsample` field, which is a video-VAE tell. Own class, latents
  not expanded, Wan-shaped architecture. `docs/02` U3 called it the sixth
  lineage on the strength of a report; the published configs do not support
  that, and the config route is now exhausted. It needs the recon probe.
* **Emu3, Lumina-mGPT** — autoregressive VQ. Autoregressive is a SAMPLER
  difference, not a decoder one, and this document's own rule only counts the
  second. `Emu3VisionVQModel` at `embed_dim` 4 / codebook 32768 is the shape
  `paella_vq` already has. Deferred behind a free test: rotate `movq` against
  `paella_vq` once `ov7_lineage2` has run. If they transfer, VQ is one lineage
  and Emu3 adds little; if they do not, codebook identity is the unit and Emu3
  is a third. Lumina-mGPT carries no licence tag at all, which is its own
  blocker.

## 12. Still owed

1. Per-source FPR, reported separately for Open Images, WildFake and SID_Set.
   Needs a trained checkpoint -- `scripts/stratified_auc.py --stratify-by
   source` takes one -- so it belongs to the Stage B session, not this one.
2. The end-to-end question: does a head trained with AI-OV7 beat one without it
   on the organisers' benchmark? If not, that is `docs/02` §6's second negative
   result and an argument for task 03 — worth knowing early either way.
3. **Generate `ov7_lineage2` and `ov7_lineage3`.** Wuerstchen and CogView4 are
   registered and ungenerated; Z-Image is registered and undownloaded. Until
   `ov7_lineage2` runs, the free movq-vs-paella_vq rotation that would settle
   the Emu3 question cannot be run either.
4. **Read the Tencent community licence.** It is the only thing between this
   corpus and its first genuinely new decoder in months.
5. **Does depth survive an unseen decoder?** Opened 2026-08-31 and running.
   The unfreeze ladder reads +0.1365 at depth 4 -- but against
   `heldout_generator`, whose two families (`SDwithAdaptor_controlnet`,
   `VQGAN`) are lineage-SIBLINGS of training families, over cross-source
   negatives. AI-OV7 is the population without either defect: encoder-matched
   pairs, genuinely unseen decoders. No OV7 bank has ever been built through a
   fine-tuned tower, so `a3`'s **0.3998** is the only number that exists
   there, and it is from a frozen one. The d0/d4 OV7 banks are extracting now.
   If depth does not move that number, the ladder measured confound-fitting
   and the lever is elsewhere.
