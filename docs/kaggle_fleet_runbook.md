# Kaggle Fleet Runbook — Stage A

Five people, five Kaggle accounts, one feature bank. Each person extracts a
contiguous fifth of the same frozen manifest into a *shard bank*; one person
merges the five afterwards.

## Which machine runs what, and why

The GPU work does not divide evenly, because two of the jobs are chained and
weight-heavy and the rest are embarrassingly parallel.

| | Runs | Because |
|---|---|---|
| **A4500** (local) | DINOv3 bank → reconstruction features (~8 h) → eval banks | Recon *attaches to an existing bank*, so it cannot start until DINOv3 finishes. It also needs the SD 1.5 VAE and LPIPS — extra gated downloads that do not belong on five teammates' sessions. One machine, no 12-hour session cap, data already local. |
| **Kaggle fleet** (5×) | SigLIP2 bank, then CLIP bank | No chaining, no extra weights, and 131k images split five ways is exactly what five accounts are for. These two backbones are what rung A5 (paradigm-diverse ensemble) needs, and A5 cannot start until both exist. |

So **the fleet's backbone is `siglip2l`, not `dinov3l`** — that is already the
default in `kaggle_stage_a.ipynb`. DINOv3 is the A4500's job.

> **Before five people spend a session on it:** SigLIP2 takes a different
> forward-pass input contract from DINOv3 and CLIP — patchified input plus an
> attention mask and spatial shapes, not `pixel_values=(B,3,H,W)`. That path is
> covered by tests against a stub, but it has never been run against the real
> weights. Someone should do a `SMOKE = True` run and post the result before
> the other four start.

Because every view's pixels depend only on `(seed, row_id, view_idx)` — never
on who extracted the image or when — five shards recombine into a bank that is
bit-identical to one extracted in a single uninterrupted run, provided nobody
changes the parameters except `SHARD_INDEX`.

---

## The notebook is the procedure

**[`notebooks/kaggle_stage_a.ipynb`](../notebooks/kaggle_stage_a.ipynb)** is what
you actually run. It checks the environment, verifies the data against the
frozen manifest, cuts your shard, measures the rate on ~40 images and then does
the real extraction — with the reasoning for each step in the cell above it.

This document is the context the notebook cannot carry: who runs what, which
shard is yours, and what to do when it stops at 2am.

To get it into Kaggle:

1. Download the file from GitHub (**Raw** → save as `kaggle_stage_a.ipynb`), or
   clone the repo.
2. In Kaggle: **Create → New Notebook → File → Import Notebook** → upload it.
   (If your Kaggle build offers a URL field there, the GitHub link works too.)
3. Then follow the steps below.

Do not re-type the cells. The notebook clones this repo at run time and imports
`notebooks/kaggle_bootstrap.py` from it, so the logic that has to be *right* —
the pip plan that does not destroy Kaggle's torch, the mount unification, the
shard arithmetic, the resume check — is versioned and tested rather than living
in a cell someone edited.

---

## What you need before you start

| | |
|---|---|
| Kaggle account | With GPU quota remaining (30 h/week free tier) |
| Access to `techjam-aigc-train` | The owner must share it with your **Kaggle username** |
| A HuggingFace account | Only if your `BACKBONE` is gated — see below |

**You almost certainly do not need an HF token.** SigLIP2 (Apache-2.0) and CLIP
(MIT) are public repos, and the auth cell now says so and moves on. Only DINOv3
is gated, behind Meta's custom licence — and DINOv3 runs on the A4500, not here.

If you do run a gated backbone: acceptance is per account, someone else
accepting does nothing for you, and the failure it produces from a notebook is
an ordinary-looking `401`/`403` on `from_pretrained`.

---

## The roster

Agree these in the team chat **before** anyone runs anything. `N_SHARDS` must be
identical for all five or the shards will not tile the manifest.

| Shard | Images | Publish your bank as (printed as `BANK_DATASET`) |
|------:|-------:|---|
| 0 | 26,224 | `aigcdet-bank-siglip2l-shard0` |
| 1 | 26,223 | `aigcdet-bank-siglip2l-shard1` |
| 2 | 26,223 | `aigcdet-bank-siglip2l-shard2` |
| 3 | 26,223 | `aigcdet-bank-siglip2l-shard3` |
| 4 | 26,223 | `aigcdet-bank-siglip2l-shard4` |

131,116 rows of `train,val_internal`. The remainder goes to the *first* shard,
which is why shard 0 has one extra image. That rule is shared by
`scripts/extract_features.py`, `notebooks/kaggle_bootstrap.py` and
`scripts/extract_eval_bank.py` — they must not diverge, because two rules that
each look correct in isolation put every boundary one row apart at 131,116 rows
over 5 shards, and the resulting one-image overlap is only discovered by
`merge_banks` refusing the lot after five people have each paid for a session.

---

## Steps

1. **Skip this unless your backbone is gated.** For the fleet's `siglip2l` it
   is not, and the auth cell will confirm that. For a gated backbone, accept
   the licence on the model's HuggingFace page using your own account, then
   create a read token.

2. **New Kaggle notebook.** Settings → Accelerator → **GPU T4 ×2** (or P100).
   Settings → Internet → **On**. Without the GPU, extraction runs on CPU and
   will not finish inside a session.

3. **If you needed a token in step 1**, add it as a Secret, never as a cell.
   Add-ons → Secrets, named `HF_TOKEN`. This repo is public and notebooks are
   committed with their cell source, so a pasted token is a published token.

4. **Add data** → `techjam-aigc-train`. It carries `sid_set/`, `wildfake/` and
   `manifest.parquet` in one mount.

5. **Open the imported notebook**, set your `SHARD_INDEX` in cell 0, and run
   every cell with `SMOKE = True`. The smoke run exercises
   everything the real run does — the gate, the shard slice, the backbone, the
   augmentation, the bank writer, the invariant check — on about 40 images, and
   then estimates how long your real shard will take.

6. **Set `SMOKE = False`**, re-run cells 0–5, then run the extraction cell.

7. **Publish your shard.** Save Version → Quick Save, then Output → New
   Dataset, named per the roster above. Post the slug and your shard index in
   the chat.

   Publish only a **complete** shard — the last cell prints `COMPLETE`, or
   tells you how many images remain.

---

## If the session dies

Start a new session, re-run every cell from the top **with the same
parameters**, and the extraction continues where it stopped. You lose at most
`CHECKPOINT_EVERY` (200) images.

Do **not**:

- change `SHARD_INDEX`, `N_SHARDS`, `SEED`, `BACKBONE` or `SPLITS` between
  sessions of the same shard — the resume check will refuse, and it is right to;
- delete a bank directory that refuses to resume — that throws away every image
  already extracted. Extract to a new directory instead;
- skip the verify cell. It costs minutes. Skipping it costs a full run of
  features that do not correspond to the manifest's labels, and nothing
  downstream would notice.

---

## Errors worth knowing before 2am

| What you see | What to do |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` to 8, then 4. Re-run — nothing repeats. |
| `MemoryError`, kernel dies | Lower `WORKERS`. Retryable. |
| `ReadTimeout`, `ConnectionError` | The clone or the model download. Just re-run. |
| `401` / `403` / `gated repo` | **Fatal**, and only possible on a gated backbone. Your account has not accepted the licence, or `HF_TOKEN` is not attached to *this* notebook. On `siglip2l` this means the mirror, not auth. |
| `cannot resume the bank at …` | **Fatal.** A parameter moved between sessions, usually `SHARD_INDEX`. Restore it, or extract to a new directory. |
| `no image Datasets attached` | You attached the benchmark rather than the train Dataset, or the share has not reached you yet. |

The notebook's last cell takes a pasted error string and tells you whether
re-running can possibly help.

---

## After all five shards land

One person runs `notebooks/kaggle_merge_train.ipynb`. It attaches all five
shard Datasets plus the train Dataset, verifies each against the frozen
manifest, merges them in shard order, and trains the heads on the result.

The merge refuses shards that disagree on backbone, dim, view count, seed or
excluded transform families, or whose rows overlap. A refused merge is telling
you two people ran different parameters — far cheaper to learn there than from
an unexplained drop in val AUC three days later.


---

## The evaluation bank (`kaggle_stage_a_eval.ipynb`)

Everything above builds the TRAINING bank. The robustness table is scored from
a separate EVALUATION bank, and rung **A5** fuses two backbones, so it needs one
for *each*. `dinov3l`'s is chained after its training extraction on the A4500;
`siglip2l`'s is this notebook's job.

**Attach three Datasets, not two:**

| Dataset | Carries |
|---|---|
| `techjam-aigc-train` | `sid_set/`, `wildfake/` — the normalised tree |
| `techjam-aigc-benchmark` | `coco_val2017/`, `dalle_advanced/` — the organisers' demo halves |
| `techjam-aigc-eval-manifest` | `eval_manifest.parquet` |

The manifest is published on its own because neither image Dataset carries it,
and because it must be the SAME FILE the local runs use. `manifest_sha256` is
taken over the ordered `rel_path` column, and `eval.fusion.assert_fusion_parents`
refuses to fuse two banks whose fingerprints differ — so a manifest rebuilt in
the session, however correctly, would produce a bank that cannot be fused with
the A4500's. Upload it; do not regenerate it.

**Why this notebook does not call `unify_mounts`.** The eval manifest joins two
trees and is re-rooted onto their common ancestor, so its `rel_path` starts
`normalized/` or `demo/`. Both Datasets mount their CONTENTS at that level
instead of a directory of that name, so `content_root` cannot find either.
The notebook builds a two-link symlink farm explicitly, asserts the link names
against the manifest's own top-level names, and then resolves 200 sampled rows
before any GPU is spent — a farm whose links point at a mount attached under a
different slug lists correctly and resolves to nothing.

Verified locally: the farm yields fingerprint `7015981b…`, identical to
resolving the same manifest against `data/`.

**Size.** 25,332 rows x 20 conditions = 506,640 forwards, and about 1.1 GB of
`feats.npy`. That fits one session and `/kaggle/working`, which is why
`N_SHARDS` defaults to 1. Raise it only if the smoke cell's estimate does not
fit a 9 h GPU session.


---

## The convolutional banks (`notebooks/kaggle_stage_a_cnn.ipynb`)

Not a fleet job. Both conv banks fit one session whole, so this is **one
account, one session, no shards, no merge** — run it twice, changing a single
line.

| BACKBONE | dim | bank | stages pooled |
| --- | --- | --- | --- |
| `convnextt` | 2304 | 6.27 GiB | 3 and 4, mean+std |
| `resnet50` | 4096 | 11.08 GiB | 4 only, mean+std |

Sizes are `kb.fits_in_working` against the real 131,116-row train+val_internal
split; both clear the 20 GiB working quota with the 0.5 GiB reserve.

**Attach one Dataset**: `justinbersamin/techjam-aigc-train`. Same manifest,
same `SEED = 20260827`, same `SPLITS = "train,val_internal"` as the SigLIP2
fleet — those three are what make the result fusable at A5, and changing any of
them produces different view pixels that `assert_fusion_parents` will reject.

**No HuggingFace token.** Both checkpoints are Apache-2.0 and ungated
(`docs/model_licences.md`), so the auth cell reports "not gated" and moves on.

**Why `N_SHARDS = 1` here and 5 for SigLIP2.** A conv tower is ~50x cheaper per
image than SigLIP2-L, so the run is CPU-bound on JPEG decode rather than
GPU-bound; `BATCH_SIZE` is raised to 64 because the GPU is otherwise idle
waiting, and `WORKERS` is what actually sets the pace. The smoke cell still
measures the marginal rate and tells you whether your session will finish — run
it with `SMOKE = True` first and read that number rather than trusting this
paragraph.

**The head is not the same size, and that is a confound in the comparison.**
`train_head` takes `dim_feat=bank.config["dim"]`, so a wider bank silently buys
a bigger head: 923,405 parameters on a 1024-d ViT bank, 1,906,445 on
`convnextt`, 3,282,701 on `resnet50`. A CNN rung that beats a ViT rung has
therefore been handed 2-3.5x the head capacity, and "the conv paradigm helps"
is not the only explanation available. Before claiming the third paradigm
works, re-run the winning ViT rung with `hidden` raised to match the CNN head's
parameter count — that is a CPU-only Stage B run of a few minutes, and it is
the difference between a finding and an artefact.

**Reading the result.** Sharpness alone predicts the label at AUC 0.672 in this
pool (`docs/low_level_confounds.md`), and a tower pooling spatial standard
deviations is the one most likely to lean on it. Treat a clean CNN AUC as
provisional until the balanced-index filter is applied, and compare the
stratified number.

## The crop-vs-band A/B (`kaggle_all_experiments.ipynb`, streams `probe_*`)

Two Kaggle sessions, run **concurrently**, answering one question: does random
cropping beat band-limit standardisation? Set `STREAM = "probe_crop"` in one
and `STREAM = "probe_band"` in the other. Nothing else differs — same probe
manifest, same eval subsample, same seed, same backbone.

**Attach three Datasets to each session:**

| Dataset | Why |
| --- | --- |
| `techjam-aigc-train-coco-crop` | the images (64 GB, shared by both arms) |
| `techjam-aigc-probe-manifests` | the 20,000-row probe manifest |
| `techjam-aigc-eval-manifest-coco-crop` + `techjam-aigc-benchmark` | the eval grid |

**Budget.** ~45 min Stage A (20,000 rows) + ~35 min eval bank (4,000 rows ×
20 conditions) + ~20 min ladder. Both arms in parallel: **~1.5 h to a verdict**,
against ~26 h to run the same comparison at full corpus size.

**Read the result off `heldout_robust_tpr_at_1pct`** in each arm's
`selection.json`, at the same rung. Not clean AUC — the whole point of crop is
what it does to *degraded* generalisation.

### Three things that are held fixed on purpose

- **`geometric=False` on both arms.** Dihedral augmentation needs a square
  input, so it is crop-only. Enabling it on the crop arm tests two changes at
  once and the number cannot say which one moved.
- **The coco_crop corpus, not the frozen one.** 85% of the frozen corpus sits
  at exactly 200 px short side, where a 200×200 crop is nearly the whole frame
  and the two policies collapse into each other. coco_crop's images are
  425–512 px, so the policies genuinely differ.
- **The same probe manifest for both arms**, cut once by
  `scripts/cut_probe_manifest.py` with the `uniform` sampler so it is a scale
  model of the corpus. Re-cutting it per arm with a different seed would make
  the comparison a comparison of two samples.

The probe manifest is **not committed** — `data/` is a symlink to the artefact
tree — so it is reproduced from the script and its seed rather than stored:

```bash
python scripts/cut_probe_manifest.py \
    --manifest data/manifest_coco_crop.parquet \
    --out data/probe/manifest_coco_crop_probe.parquet \
    --budget train=16000 --budget val_internal=4000 \
    --split train,val_internal
```

It must print `manifest_sha256
9a60e22759aa98bd710798ace81ff26dcb7f3bee44fb688334882e87432170b1` and file
sha256 `4fd2d1a7a39f11dd9771439dd6dd36ff175350b880598363d9f10e52436e8188`. A
different digest means a different 20,000 rows, and the arm you run against it
is not comparable with anyone else's — check the parent manifest before
uploading it anywhere.

### What a probe bank is not

A probe manifest fingerprints differently from `manifest_coco_crop.parquet`,
so a probe bank **cannot** verify against the real manifest, merge with a real
shard, resume from one, or fuse against one. Every one of those refusals is
correct. A probe is evidence about a policy; it is never a component of the
shipping system. Do not publish one as a bank Dataset.

Before spending even the 1.5 h, `scripts/gate_confounds.py` answers the cheaper
half of the question on CPU in ~20 minutes: crop's stated justification is that
it preserves native detail instead of box-filtering it away, so if it does not
pull `laplacian_var` back toward the frozen 0.6721 baseline it has failed on its
own terms.

## The full union extraction (`kaggle_all_experiments.ipynb`, stream `union`)

The shipping corpus: 376,744 rows over NTIRE, WildFake, SID_Set, COCO
train2017 and Open Images V7. This is the run the fleet exists for — the
probe and the two ancestor streams are all smaller things that feed it.

**Attach six Datasets, not one.** Five carry images, one carries manifests.

| Dataset | what |
| --- | --- |
| `techjam-aigc-union-coco-train2017` | 18 GB |
| `techjam-aigc-union-ntire` | 62 GB |
| `techjam-aigc-union-open-images` | 9.3 GB |
| `techjam-aigc-union-sid-set` | 23 GB |
| `techjam-aigc-union-wildfake` | 16 GB |
| `techjam-aigc-manifests-union` | both parquets, a few MB |
| `techjam-aigc-benchmark` | the organisers' half of the eval manifest |

The corpus is split per source only because of the per-Dataset size cap.
`DATA_GLOB` is the prefix glob `/kaggle/input/techjam-aigc-union*`, which
matches all five, and `kb.unify_mounts` symlinks them into one root — the same
mechanism the five-teammate shard farm uses. **Attach all five.** A missing
mount does not raise: it removes rows, and the first thing that notices is the
verify gate.

The manifests Dataset is called `techjam-aigc-manifests-union` and *not*
`techjam-aigc-union-manifests` for exactly that reason — the latter matches
the prefix glob, would arrive as a sixth image mount, and `unify_mounts` would
symlink two parquet files into the corpus root. Published by
`scripts/publish_union_manifests.sh`, which prints both `manifest_sha256`
values so a session can check its mount against what was frozen without
re-reading 128 GB.

**Storage.** Private datasets are capped at 200 GB in total for the account,
and the union is 128 GB of it. `techjam-aigc-train` (25.5 GB, the frozen
stream's corpus) was deleted on 2026-08-30 to fit — its banks remain published
as `techjam-aigc-banks` and the corpus is still on local disk, so the frozen
stream's results stand; only re-extracting it on Kaggle now costs a re-upload.
Measured upload rate to Kaggle is 22 MB/s on a single stream (192 Mbps),
consistent across two transfers hours apart, so the 128 GB is ~1 h 40 of
transfer plus ~25 min of archiving — provided the archiver has the disk to
itself. It must: with `build_dataset` reading the same spinning disk the
archiver drops to 7 MB/s and takes the build down with it, and no `ionice`
class fixes that because the contention is seek time, not bandwidth.

**`N_SHARDS` is pinned at 8, and the notebook will not derive it.** The
derivation can only see the 20 GiB working quota, and at dim 1024 the whole
bank is 8.5 GiB — so it would return `N_SHARDS=1`, which fits the quota fine
and takes about 23 hours, twice the session cap. The clock binds here, not the
quota, and a count that cannot see the clock must not win over one chosen
against it. Eight shards is 47,093 rows each:

| backbone | px | per shard | 8 shards |
| --- | --- | --- | --- |
| `dinov2l` | 518 | ~2 h 53 | ~3 h wall on 16 T4s |
| `siglip2l` | 384 | ~1 h 30 | ~2 h wall |

Those come from the A4500's measured 54 views/s at 518 px, halved for a T4 and
doubled for the two in a session. **Projected, not measured** — no `dinov2l`
run has happened on a T4. Run one session with `SMOKE=True` first and read
`kb.session_plan`, which measures the rate on that session's actual GPU and
says whether 8 was right. Raising `N_SHARDS` costs a re-run of the plan cell;
discovering at hour eleven that it was too low costs the session.

Per-shard bank is 1.06 GiB, nowhere near the quota, so the merge afterwards is
eight small files.

**Order.** Run `SMOKE=True` in one session and read the plan before anyone
else starts — that is the cheapest moment to discover the shard count is
wrong. Then eight sessions, `SHARD_INDEX` 0..7, same `BACKBONE`, same
`STREAM`. `merge_banks` refuses shards that disagree on backbone, dim,
n_views, seed **or** `canon_policy`, so a session that ran the wrong stream is
caught at merge rather than folded in.

**`STREAM` is `union` (crop) or `union_band`, and the probe decides which.**
Both are spelled out in the notebook so the winner needs no edit. Delete the
loser from `STREAMS` once the probe reports, so nobody runs it by muscle
memory.

## Publishing a corpus as a Kaggle Dataset

Nothing in this repo automated this and nothing recorded it either, so the two
Datasets the fleet reads were reproducible only from memory. The procedure:

```bash
cat > data/normalized_coco_crop/dataset-metadata.json <<'JSON'
{
  "title": "TechJam Track5 AIGC Train COCO crop",
  "id": "justinbersamin/techjam-aigc-train-coco-crop",
  "licenses": [{"name": "other"}]
}
JSON

set -a; . ~/.kaggle/env; set +a       # never echo this file
kaggle datasets create -p data/normalized_coco_crop -r tar -t
```

**The manifest must be INSIDE the published directory**, named exactly
`manifest.parquet`. `build_dataset --manifest` writes it wherever you point,
and the notebooks glob `{DATASET_SLUG}*/manifest.parquet` — so a tree without
it uploads happily and then fails at cell 12, after the Dataset exists. Copy
it in before publishing:

```bash
cp data/manifest_coco_crop.parquet data/normalized_coco_crop/manifest.parquet
```

**Both flags below are load-bearing and neither is the default.**

- `-r tar`. The CLI's default `--dir-mode` is **`skip`**, which ignores
  subdirectories entirely — it would upload `manifest.parquet` and none of the
  images, and succeed. `tar` rather than `zip` because PNGs are already
  compressed, so zip spends CPU on nothing.
- `-t` (`--keep-tabular`). The CLI converts tabular files to CSV by default.
  `manifest.parquet` would arrive as CSV, `read_manifest` would fail on the
  Kaggle side, and the failure would be an hour into a session.

Datasets are created **private**. To share one with the fleet:

```bash
kaggle datasets metadata -p /tmp/dsmeta <slug>   # then edit, or use the web UI
```

or flip it in the Dataset's settings page. Prefer doing this deliberately:
the tree carries WildFake's re-published subsets, whose upstream terms are
non-commercial (`docs/dataset_licences.md`).

**Updating an existing Dataset** is `kaggle datasets version -p <dir> -m "..."`,
not `create` — `create` on an existing slug fails.

### Attaching the right corpus

`DATASET_SLUG` in cell 2 of the Stage A notebooks selects the stream, and both
globs derive from it. The glob is a **prefix**, so `techjam-aigc-train` also
matches `techjam-aigc-train-coco-crop`. Attaching both at once is the mistake
that costs a session: the manifest of one stream would be read against the
images of both. The attach cell now refuses when more than one manifest
matches, but the cheap fix is to attach only the stream you are running.

The two streams are separate corpora with different fingerprints. A bank built
against one can never be resumed from, merged with, or fused against a bank
built from the other — `manifest_sha256` and, since the crop policy landed,
`canon_policy` in the bank config both enforce it.
