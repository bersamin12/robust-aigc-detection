# Handover: Tier 2 on a second 4×4090 box

*Written 2026-08-31 03:10 by the agent running box #1. You are running box #2.
Box #1 is saturated for ~4½ h with full-scale Stage A and will not be free
before you finish reading this.*

---

## 1. What this project is, in one screen

A detector for AI-generated images (TikTok TechJam Track 5). Two stages, and
the split matters more than anything else here:

* **Stage A** runs a **frozen** backbone over canonicalised image *views* and
  writes float16 embeddings to a bank (`feats.npy` + `meta.parquet` +
  `config.json`). `backbones.py:560` calls `requires_grad_(False)`, `:558`
  `.eval()`, `:643` `@torch.inference_mode()`. No gradient reaches a backbone
  anywhere in this repo.
* **Stage B** trains a small head on the cached memmap and **never sees an
  image**. It is CPU-cheap and rerunnable in minutes.

Every experiment below is either "extract a new feature block" (Stage A, GPU,
hours) or "train a head on cached blocks" (Stage B, minutes).

### The selection metric — memorise this

```
heldout_robust_tpr_at_1pct
  = mean TPR @ 1% FPR over the 19 DEGRADED conditions (clean EXCLUDED)
    positives : split == heldout_generator, label 1
    negatives : split == val_internal,      label 0
```

`eval/errors.py:heldout_robust_tpr`. Four rungs may be the headline
(`ELIGIBLE_RUNGS = ("a3","a4","a5","a6")`, `errors.py:65`). **a0–a2 and
a7_norecon are ablation controls: a control winning is a finding to report,
never a model to ship.**

`clean` is excluded *by design*. Several probes have "improved" a number by
quietly averaging it back in. Don't.

---

## 2. Non-negotiable discipline

These are the rules that make a number mean anything. They have all been
violated at least once and caught.

1. **A rung differs from its base by exactly ONE flag.**
   `tests/test_rung_ladder.py` enforces it by parametrisation. If your new
   block needs two flags to be interesting, it is two rungs.
2. **Anything fitted is fitted on `val_internal` only.**
   Fusion weights and z-score populations are *two separate fits*, and
   `fuse_scores(fit_splits=...)` has **no default** on purpose. Use
   `FIT_SPLITS_WHEN_FITTING_WEIGHT` (`("val_internal",)`) when a weight is
   being swept. Read the held-out number exactly once, after the knob is fixed.
3. **A lineage holdout is two halves in two files.** The eval manifest promotes
   a family into `heldout_generator`; `RungConfig.train_exclude_generators`
   drops it from training. Do only the first and the number is silently
   inflated — `grid.assert_heldout_not_trained` is the check, run it.
4. **Every rung compared must be trained on identical view coverage** (spec
   §3.3). If you can't afford a full pass, train *both* members of the pair on
   the covered subset and report that reduced-coverage pair beside the
   full-coverage a3. Never compare across coverages.
5. **Off-ladder probes write their own JSON** with `off_ladder: true` and a
   `not_eligible_reason`. They never enter `selection.json`.

---

## 3. What is already measured — do not redo this

### Wave 2 probe ladders (20k, `heldout_robust_tpr_at_1pct`)

| arm | a0 | a1 = a2 | a3 | a7_norecon |
|---|---|---|---|---|
| band dinov2l | 0.6934 | 0.7076 | 0.6565 | 0.7061 |
| crop dinov2l | 0.6751 | 0.7356 | 0.7355 | 0.7091 |
| band dinov2regl | 0.7153 | 0.7844 | 0.7242 | 0.6698 |
| **crop dinov2regl** | 0.6717 | 0.7730 | **0.7858** | 0.8022 |
| band siglipso400m | 0.7729 | 0.7754 | 0.7111 | 0.7639 |
| crop siglipso400m | 0.6829 | 0.6627 | 0.6166 | 0.6219 |
| band convnextv2h | 0.4855 | 0.6172 | 0.5138 | 0.5824 |
| band eva02l | 0.5072 | 0.5055 | 0.4036 | 0.5482 |
| *band dinov3l (BARRED — reference only)* | 0.8379 | 0.8248 | 0.8667 | 0.8237 |

`a1` and `a2` are **bit-identical in all eight arms**. Canon policy is
backbone-specific: crop helps DINO, hurts SigLIP-SO400M (−0.113 at a1).

### Fusion (this is where the wins are)

| pair | primary split | sid_set | wildfake GAN | mean |
|---|---|---|---|---|
| band+crop (one tower, two policies) | 0.8702 | 0.5168 | **0.7482** | 0.7117 |
| **crop+siglip** | **0.9139** | **0.6278** | 0.7198 | **0.7538** |
| band+siglip | 0.8517 | **0.6278** | 0.7141 | 0.7312 |
| 3-way equal | 0.9181 | — | — | — |
| 3-way fitted `w=[0.19,0.33,0.48]` | **0.9236** | — | — | — |

A proper A5 rung run confirmed crop+siglip at **0.9105** fitted (w=0.45/0.55).

### The finding that should shape your priors

**Arm ordering is not stable across held-out families.** crop beats band by
+0.062 on the primary split and *loses* by 0.129 (sid_set) and 0.026 (GAN
family). The primary split is **two generators, both wildfake**
(`SDwithAdaptor_controlnet` 766, `VQGAN` 734) — it cannot resolve small
margins. What *does* transfer is that **fusion beats both its parents on every
split tested**. Trust fusion gains; distrust single-arm rankings.

Use `scripts/second_holdout.py` (already written) to re-check any ranking you
care about against a second family. It is Stage B only, ~2 min per pair.

### Already ruled out — recorded, don't re-run

* **Test-time *adaptation*** (MEMO/TENT: entropy minimisation on prompt tokens
  or the head). Every arm neutral-to-harmful, monotone in learning rate;
  confidence filtering was *worse* than none (0.9603 vs 0.9627), because the
  lowest-entropy views are the least degraded and the metric is a robustness
  metric. Also structurally impossible on cached features. `docs/tta_entropy_pilot.json`.
* **Head capacity.** 512 is at or near optimal on all three banks tried; best
  gain anywhere +0.011 against fusion's +0.086, and two of three degrade.
* **Family experts** (GAN head + diffusion head). Fused 0.8843 vs pooled a3
  0.9012 — strictly dominated, even though the experts score 41 points apart on
  `SDwithAdaptor_controlnet`, i.e. the family label *is* informative and still
  loses.
* **LARE (paper 7.1)** — out of scope, needs a UNet against the 2B cap.

---

## 4. Your job: Tier 2

Five experiments. **Two of them are eligible rungs that have never been run**,
which is the single biggest gap in the programme: of `a3/a4/a5/a6`, only a3 and
a5 have ever produced a number.

### T2-1 · `a4` — does the VAE reconstruction branch help at all? *(eligible rung)*

`configs/rungs/a4.yaml` **already exists** and has never been run. The whole
branch is built: `features/recon.py:recon_features` produces the 12-d vector,
`attach_recon_to_bank` attaches it, `bank.attach_recon` pins `(N, V, 12)`.

This is the cheapest real result available to you. **Do it first.**

* AEROBLADE (paper 6) is the `l1` slot; FIRE (7.2) is `spec_b0..b3` +
  `spec_mid_ratio` / `spec_high_ratio` via `_radial_bands`. Both are already in
  the 12-d vector — you are *measuring* those papers, not implementing them.
* Then `a4 → a7` (`a7.yaml` also exists, never run): does FiLM help the recon
  system? That completes the ladder:
  `a3→a4` VAE branch, `a4→a7` FiLM on recon, `a3→a7_norecon` FiLM without.

### T2-2 · `a4vq` — a second, vector-quantised autoencoder *(new)*

**Do not widen `RECON_DIM`.** `bank.attach_recon` pins `(N, V, 12)` and the
rung test enforces one-flag steps; widening breaks both. Add a *second named
block*:

```
recon.npy      (N, V, 12)  SD 1.5 KL VAE   flag: use_recon      exists
recon_vq.npy   (N, V, 12)  VQ autoencoder  flag: use_recon_vq   new
```

`recon_features()` is autoencoder-agnostic — it takes `vae, lpips_fn` and calls
`_roundtrip`. Any AE with `.encode`/`.decode` reuses it unchanged; only
`load_recon_models` grows a selector.

**Why this is not busywork:** the held-out generators are
`SDwithAdaptor_controlnet` and **`VQGAN`**. A continuous KL VAE has no
structural reason to round-trip a VQ autoencoder's output anomalously, so the
branch as built is arguably blind to half the population we select on.

Files: `features/recon.py`, `features/bank.py` (`attach_recon_vq` mirroring
`attach_recon`), `train/train_head.py` (`RungConfig.use_recon_vq`, `Detector`),
`configs/rungs/a4vq.yaml`, `a4both.yaml`, and the parametrisation in
`tests/test_rung_ladder.py`.

**Update the 2B guard deliberately.** `tests/features/test_backbones.py:109`
hardcodes `+ 84_000_000 + 2_500_000` for one VAE plus LPIPS. Two autoencoders
is ~168M + 15M. Change it on purpose, not by discovering it red.

### T2-3 · `aF` — the frequency branch, **crop only** *(new)*

`baselines/npr.py:npr_feature` is a 4-d descriptor (two magnitudes,
`contrast_h`, `contrast_v`) that detects the periodic cell structure a
transposed-convolution upsampler leaves behind. Add as `freq.npy (N, V, 4)`,
flag `use_freq`, rung `a3 → aF`.

* **Nearly free**: pure numpy on the already-decoded view. No model, no GPU. It
  folds into the same view replay as the recon pass.
* **Only measurable under `crop`.** `band` resamples every image to a nominal
  side, destroying the generator's native pixel grid and substituting the
  resampler's own. The confound probe also shows band *leaks* (pooled 0.6105,
  SID_Set 0.9976) where crop is near chance (0.5081, 0.6316). Band would
  destroy the real signal and supply a fake one. Say so in the rung's comment.
* **Mandatory control.** `proxy_vector` is 3-d (JPEG quality, Laplacian
  variance, noise floor) and **sharpness alone reaches AUC 0.672**. A 4-d
  frequency descriptor could simply relearn it. Train a **frequency-only head,
  no backbone**, and run `scripts/content_blind_probe.py` on it. A high
  frequency-only score is a red flag, not a result.

### T2-4 · `a6` — TTA *(eligible rung, never run)*

`--tta` currently records a cost multiplier and emits no row
(`run_ablation.py:532`, `"scored_here": False`). Everything else exists:
`TTA_VIEWS` (8), `apply_tta_view`, `tta_logit`,
`check_views_avoid_heldout_bands`.

**The module's own warning is the whole job:**

> `tta_logit` returns the MEAN of eight per-view logits… a `T` fitted on
> single-view logits is being applied to a differently-scaled quantity…
> Whoever does that must **REFIT the temperature on TTA-averaged logits over
> `val_internal` and carry it as a separate `T`**, not reuse the single-view one.

The measured case for it: three *mild* degradations beat clean
(`jpeg_q90` 0.9560, `blur_s1.0` 0.9456, `blur_s0.5` 0.9431 vs clean 0.9396),
and a6's views (`jpeg_95`, `blur_0.3`) sit exactly in that regime. The same
table kills the worry that resampling destroys the fingerprint: `resize_0.5` at
0.9311 is the third-best degraded condition.

**Aggregation, in descending order of trust:** (1) per-view weights on the
simplex fitted on `val_internal` only — use `scripts/fuse_simplex.py`, it
already does n-parent simplex sweeps with the null on the lattice; (2) trimmed
mean (no fitting, no leak surface); (3) a logistic stacker on the 8 logits only
if (1) shows large stable asymmetries. **Max/min are ruled out** — they move
TPR and FPR together at a fixed 1% FPR.

**Nothing cached helps**: TTA views compose *on top of* each condition. New
code needed — `extract_eval_bank.py` builds conditions, `eval/tta.py` builds
views, **nothing composes them**. ~4,000 images × 5 conditions × 8 views ≈ 160k
forwards, ~30 min on one 4090. The no-TTA baseline for identical rows is
already in the eval bank, so the comparison is paired for free.

### T2-5 · One designed saving — read this before you extract anything

**Recon and frequency features are backbone-independent.** They depend on the
canonicalised view, not the tower. With the same `manifest_sha256`,
`canon_policy`, seed and `n_views`, **one replay pass produces blocks valid for
every bank.** Compute once, attach three times.

**Assert those four fields match before attaching.** A silent mismatch here is
a bank that trains on another bank's features, and nothing downstream would
catch it.

---

## 5. Box setup

```bash
git clone <repo>; cd robust-aigc-detection
bash scripts/pod_bootstrap.sh          # warms model cache, checks torch/CUDA
```

`python3` is the interpreter (`python` often does not exist on these images —
`pod_bootstrap.sh` resolves it). Verify: `python3 -c "import torch;
print(torch.__version__, torch.cuda.device_count())"` → expect 4.

### Corpus — 375,358 rows

```
train              169,166 real / 162,091 fake
val_internal        19,075 real /  18,026 fake
heldout_generator                    7,000 fake
```

Seven private Kaggle Datasets (five images, one manifests, one benchmark):

```
justinbersamin/techjam-aigc-union-ntire            ntire            ~61 GiB
justinbersamin/techjam-aigc-union-sid-set          sid_set
justinbersamin/techjam-aigc-union-coco-train2017   coco_train2017
justinbersamin/techjam-aigc-union-wildfake         wildfake
justinbersamin/techjam-aigc-union-open-images      open_images
justinbersamin/techjam-aigc-manifests-union        (manifests)
justinbersamin/techjam-aigc-benchmark              (organisers' demo set)
```

`ROOT=/data bash scripts/pull_union.sh` — **largest first is deliberate**:
`kaggle datasets download --unzip` writes the zip, extracts, then deletes it,
so the disk peak is (everything so far) + 2× (the one in flight). Pulling
ntire first caps the peak at ~136 GiB; last would spike to ~188 GiB.

> ⚠️ **NTIRE licence: usable for training locally and in a PRIVATE Kaggle
> Dataset. Do NOT publish an NTIRE-derived Dataset.** Keep every Dataset above
> private.

> ⚠️ **Never paste the HF or Kaggle token into a notebook cell.** This repo is
> public and notebooks are committed with their cell source — *a pasted token
> is a published token.*

Manifests rebase automatically: `read_manifest(path, root=None)` falls back to
`$AIGCDET_DATA_ROOT` (`manifest.py:83`). **Export it or every `open()` fails on
row 0**, because the manifest's own `path` column is box #1's local tree.

### The 2B cap and the two-backbone rule

Spec: *"Final model uses at most two backbones."* `dinov2regl` 304,372,736 +
`siglipso400m` 428,225,600 = **732,598,336**. Note the band/crop pair is **one
backbone loaded once and run twice** — the cap is on backbones, not forward
passes, so the three-arm fusion is legal.

> **`dinov3l` is BARRED from the shipped bundle** — ablation reference only.
> The exclusion is a team decision under a rule that has *not yet been
> supplied*; `docs/model_licences.md` records this as an open item rather than
> inventing a justification. Use it for reference numbers, never in a bundle.

---

## 6. Operational hazards — every one of these actually happened

1. **Thread caps, or the box dies at launch.** Export all five before spawning
   any worker. These are *environment, not code*:
   ```bash
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
   ```
   Without them, 4 shards × 40 workers each spun OpenBLAS to one thread per
   core (64) ≈ 10,000 tasks → `pthread_create failed: Resource temporarily
   unavailable` before a single image was read.
2. **Reap your workers.** That script then exited *without reaping them*, and a
   relaunch landed on ~160 orphans; two minutes later `sshd` could no longer
   fork and the box was unreachable for 75 minutes. Guard every launcher with a
   "refuse if extractions are already running" check.
3. **`pgrep -fc extract_features.py` counts your own shell.** The pattern
   matches any command line containing that text — including the `ssh … pgrep`
   you are inspecting with. It reported 5 running when 4 were live and refused
   a valid launch. Filter on the executable:
   ```bash
   for p in $(pgrep -f "scripts/extract_(features|eval_bank)\.py"); do
     case "$(ps -o comm= -p "$p")" in python*) n=$((n+1));; esac
   done
   ```
4. **Never edit a running bash script.** Bash reads scripts *incrementally*
   from a file offset. Editing `full_scale.sh` mid-run corrupts the tail.
5. **Heredocs through `ssh` mangle f-strings.** Broke twice. Write the script
   locally and `scp` it.
6. **`--shard i/N` blocks are contiguous and N is load-bearing.** Changing N
   redraws every boundary, so existing shard banks **cannot** be resumed
   against a different N — `writer.completed` is a set of `write_idx` *within
   that shard's dataframe*. Silent corruption, not just wasted work.
7. **`--batch-size` above `n_views` (11) is inert.** `extract_bank` loops one
   image at a time and calls `embed(..., prepared["views"], batch_size=...)`;
   `embed` chunks with `range(0, 11, batch_size)` → one forward of 11. Measured
   consequence: GPU at ~61% SM oscillating 30–93%, on **6.7% of memory**, with
   the CPU at 2.8% of 320 cores and zero IO wait. **Run 2–3 extraction
   processes per GPU** (`SHARDS_PER_GPU` in `scripts/full_scale_arm.sh`) — ~1.5×
   for no code change. Batching across images is the real fix (~2–2.5×) and is
   unclaimed work if you want it.
8. **Verify finiteness after every extraction.** On 2026-08-29 a five-hour bank
   came back 131,116 rows of **all NaN** because the only post-condition was
   the row count. `dinov3l` must run **bf16, never fp16** (layer-1 overflow).
   `embed()` now raises on non-finite output; keep the explicit check too.

---

## 7. Verification gates

* Full suite green before any GPU commitment. Baseline **1934 passed, 12
  skipped** (2026-08-31, ~6 min).
* New rungs must extend `tests/test_rung_ladder.py`'s parametrisation.
* `FeatureBank.check_invariants()` plus an explicit finite check after every
  extraction.
* a4/a4vq/aF: the block covers **all 11 views**, or the comparison is invalid.
* a6: assert the TTA temperature is a **separate** fitted `T`, and that
  `check_views_avoid_heldout_bands` still passes.
* aF: frequency-only head + `content_blind_probe.py` **before** any gain is
  believed.
* Winner: `scripts/stratified_auc.py --stratify-by source` before quoting.
* Re-check any ranking with `scripts/second_holdout.py` against a second
  family. See §3 for why.

---

## 8. Suggested order

| # | task | kind | rough cost |
|---|---|---|---|
| 1 | full test suite, `pod_bootstrap.sh`, corpus pull | setup | ~1 h (pull overlaps) |
| 2 | **`a4` + `a7`** — configs exist, just run them | Stage B on an existing bank | **minutes** |
| 3 | replay pass: SD 1.5 recon + VQ recon + freq, probe scale | GPU | ~1 h |
| 4 | `a4vq`, `a4both`, `aF` ladders + frequency-only control | Stage B | ~30 min |
| 5 | `a6` composition code + refit `T` + simplex aggregation | code + GPU | ~2 h |
| 6 | promote whatever won to full scale | GPU | hours |

**Start with #2.** Two eligible rungs whose configs already exist, never run,
one Stage-B command each. If the VAE branch does nothing, you have saved
yourself all of T2-2.

---

## 9. Coordination with box #1

Box #1 owns: full-scale Stage A for `dinov2regl:crop`, `dinov2regl:band`,
`siglipso400m:band`; the full-scale fusions; the three-way at full scale.
**Do not extract those.** Everything in §4 is disjoint from it.

Open items box #1 has *not* done and is not planning to:
* `docs/model_licences.md` still needs the actual DINOv3 rule quoted.
* The **unfreeze depth ladder** (D0–D4, plan §8) is blocked on a trainer that
  reads images — *nothing in `train/` does*. That is a real build, not a flag.
* `kb.unify_mounts` strips the source directory level for the Kaggle `union`
  stream — same latent bug `pull_union.sh` fixed at `8383d14`. That stream has
  never been run, so nothing on disk is wrong. Don't run it without checking.

## 10. Box #2: pull AI-OV7 (2026-08-31, box #1)

**Action: `kaggle datasets download justinbersamin/techjam-aigc-ov7 --unzip`.**
9.2 GB, private, ~2 min on a datacentre link. Everything below is why it is
worth the disk and what the traps are.

### What it is

Our own corpus: 9,978 pairs / 19,956 rows, open-weight generators run over
Open Images V7 portraits. Every fake is generated FROM one real, at that
real's own MCU-aligned crop dimensions and saved through that real's own JPEG
quantisation tables. `docs/ai_ov7_generation.md` is the full build record.

It exists because **every fake in the union corpus is 2017-2023**, and
published results put detection at ~79% on 2020-21 generators against ~38% on
2024 ones. AI-OV7 is the only 2024-25 material we may redistribute.

Its confound gate beats the frozen corpus on all four proxies — `jpeg_quality`
**0.5038** (vs 0.5532), `short_side` **0.5022** (vs 0.5992), `noise_floor`
**0.5229** (vs 0.6374), `laplacian_var` **0.5998** (vs 0.6721). Two are at
chance. It is the cleanest material we have.

### Why box #2 specifically wants it

Tier 2 is `a4` / `a4vq` / `aF` — the reconstruction and frequency branches.
Both are *generator-artefact* branches, and both were designed against
2017-2023 artefacts:

* `a4`/`a4vq` round-trip through an SD 1.5 KL VAE and a VQ autoencoder. AI-OV7
  contains SDXL, SD 1.5 and Klein-4B output, so `sd15_*` is the one family
  where the reconstruction branch is testing its own decoder lineage and
  `klein4b_*` is the one where it certainly is not. That contrast is not
  available anywhere in the union corpus.
* `aF` reads the periodic cell structure of a transposed-convolution
  upsampler. Diffusion decoders from 2024-25 do not all upsample that way, so
  AI-OV7 is where a frequency branch either generalises or is shown to be an
  artefact detector for an era.

### Three traps, all verified on box #1

1. **The root is one level deeper than it looks.** `rel_path` is
   `real/0001800.png` with NO source-level prefix, so the root must be
   `normalized_ov7/open_images_v7`. Measured: 0/50 rel_paths resolve at
   `normalized_ov7`, 50/50 one level down. This is the same shape as the
   `kb.unify_mounts` bug in §9 — check, do not assume.
2. **`--no-subsample` is required, not an optimisation.** The `ablation` tier
   plan carries `subsample={"benchmark": 5000}` and `subsample_manifest`
   raises on a budget naming a split the rows do not contain. AI-OV7 has no
   `benchmark` split, so the tier default aborts every shard.
3. **An eval bank's `meta.parquet` carries no digest column** — it is
   `image_idx, row_id, path, label, generator, source, split, rel_path`. Any
   join keyed on `content_sha256` matches zero rows and reports a clean number
   having excluded nothing. Resolve hashes to `rel_path` against the manifest
   first.

### Two corrections to what is written elsewhere

* `docs/ai_ov7_generation.md` §10 says the held-out family is `flux2_vae`.
  **The frozen manifest disagrees and contains no `flux2_vae` row.** It is the
  klein4b lineage: `klein4b_t2i` 1,200 + `klein4b_ref_image` 600 = 1,800 pairs.
  The manifest is what shipped; the doc line is wrong.
* This corpus is **not** a widening of `open_images` and must not be merged
  into `manifest_union.parquet`. Banks fingerprint `manifest_sha256` over
  `rel_path` in row order (`sources.py:190`), so inserting rows orphans every
  bank on disk. It is its own stream with its own manifest, as `coco_crop` is.

### Contamination, and why the number you can measure is a floor

AI-OV7's reals and the union's `open_images` reals come from the same
60,000-image portrait pool. **71 of the 9,978 AI-OV7 reals are byte-identical
to a union real, and 68 of those sit in the union's train split** — a
union-trained head has seen them. Exclude them.

That is a floor, not a clearance: AI-OV7's reals are MCU-aligned *crops*
re-saved, so a crop of a photograph the union also holds has different bytes
and `content_sha256` cannot see it. `pixel_sha256` is `""` on every row of both
manifests (`manifest.py:74` — frozen with byte digests) and would not catch a
crop anyway. **The photograph-level overlap is unmeasured.** Read a strong
transfer number as evidence and a weak one as conclusive, not the reverse.

### Do not duplicate

Box #1 owns the transfer test for the four full-scale a3 arms:
`scripts/chain_ov7.sh` + `scripts/ov7_transfer.py`, writing
`docs/ov7_transfer.json`. Both readings are implemented there — AI-OV7's own
splits with klein4b held out, and a per-family breakdown. **Nothing is ever
fitted on AI-OV7**; weights arrive from the union fit or are equal, because
there is no second AI-OV7 to recover an honest number from afterwards. Reuse
`ov7_transfer.py` for your rungs rather than writing a second scorer.

> ⚠️ The Dataset is **private**. It needs the user's Kaggle credentials, which
> are never to be pasted into a notebook cell or committed — see §5.
