# Kaggle Fleet Runbook — Stage A

Five people, five Kaggle accounts, one feature bank. Each person extracts a
contiguous fifth of the same frozen manifest into a *shard bank*; one person
merges the five afterwards.

Because every view's pixels depend only on `(seed, row_id, view_idx)` — never
on who extracted the image or when — five shards recombine into a bank that is
bit-identical to one extracted in a single uninterrupted run, provided nobody
changes the parameters except `SHARD_INDEX`.

---

## What you need before you start

| | |
|---|---|
| Kaggle account | With GPU quota remaining (30 h/week free tier) |
| Access to `techjam-aigc-train` | The owner must share it with your **Kaggle username** |
| A HuggingFace account | With the DINOv3 licence accepted **on your own account** |
| An HF read token | Added to your notebook as a Kaggle Secret named `HF_TOKEN` |

DINOv3 licence acceptance is per account. Someone else accepting does nothing
for you, and the failure it produces from a notebook is an ordinary-looking
`401`/`403` on `from_pretrained`.

---

## The roster

Agree these in the team chat **before** anyone runs anything. `N_SHARDS` must be
identical for all five or the shards will not tile the manifest.

| Shard | Images | Publish your bank as |
|------:|-------:|---|
| 0 | 26,224 | `aigcdet-bank-dinov3l-shard0` |
| 1 | 26,223 | `aigcdet-bank-dinov3l-shard1` |
| 2 | 26,223 | `aigcdet-bank-dinov3l-shard2` |
| 3 | 26,223 | `aigcdet-bank-dinov3l-shard3` |
| 4 | 26,223 | `aigcdet-bank-dinov3l-shard4` |

131,116 rows of `train,val_internal`. The remainder goes to the *first* shard,
which is why shard 0 has one extra image. That rule is shared by
`scripts/extract_features.py`, `notebooks/kaggle_bootstrap.py` and
`scripts/extract_eval_bank.py` — they must not diverge, because two rules that
each look correct in isolation put every boundary one row apart at 131,116 rows
over 5 shards, and the resulting one-image overlap is only discovered by
`merge_banks` refusing the lot after five people have each paid for a session.

---

## Steps

1. **Accept the DINOv3 licence** at
   `facebook/dinov3-vitl16-pretrain-lvd1689m` on your own HuggingFace account,
   then create a read token.

2. **New Kaggle notebook.** Settings → Accelerator → **GPU T4 ×2** (or P100).
   Settings → Internet → **On**. Without the GPU, extraction runs on CPU and
   will not finish inside a session.

3. **Add your token as a Secret**, never as a cell. Add-ons → Secrets, named
   `HF_TOKEN`. This repo is public and notebooks are committed with their cell
   source, so a pasted token is a published token.

4. **Add data** → `techjam-aigc-train`. It carries `sid_set/`, `wildfake/` and
   `manifest.parquet` in one mount.

5. **Paste in `notebooks/kaggle_stage_a.ipynb`**, set your `SHARD_INDEX` in
   cell 0, and run every cell with `SMOKE = True`. The smoke run exercises
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
| `401` / `403` / `gated repo` | **Fatal.** Your account has not accepted the DINOv3 licence, or `HF_TOKEN` is not attached to *this* notebook. |
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
