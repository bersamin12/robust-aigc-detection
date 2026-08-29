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
