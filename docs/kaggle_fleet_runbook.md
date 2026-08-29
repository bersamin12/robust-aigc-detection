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
