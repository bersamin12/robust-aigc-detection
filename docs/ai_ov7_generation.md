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

## 10. Still owed

1. The `docs/02` §5 gate at 2,000 images: `jpeg_quality`, `laplacian_var`,
   `noise_floor` against 0.5532 / 0.6721 / 0.6374, plus a dimensions-only
   control that must score ~0.5. **Allowed to cancel the task.**
2. Per-source FPR, reported separately for Open Images, WildFake and SID_Set.
3. Pair survival count after `build_dataset`, not just row count.
4. The end-to-end question: does a head trained with AI-OV7 beat one without it
   on the organisers' benchmark? If not, that is `docs/02` §6's second negative
   result and an argument for task 03 — worth knowing early either way.
