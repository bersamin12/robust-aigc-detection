# 09. Generating more AI-OV7 on a second box, without regenerating anything

**Owner:** unassigned. **Blocked on:** nothing. Needs a GPU, ~20 GB of VRAM
for the models listed in §4, and the two inputs in §2.

You are adding pairs to a corpus that already exists. The only thing that can
go badly wrong is generating a second fake for a real that already has one:
one scene would then sit twice on the generated side against a single real,
and any content a detector memorises from it arrives with a label prior. This
file is mostly about how the split of work prevents that.

Read `docs/ai_ov7_generation.md` for what the corpus is and why. This is only
how to extend it from somewhere else.

---

## 1. How the work is divided, and why it is safe

Every real has a position in one global order:

    order_key(image_id) = blake2b(f"{SEED}:{image_id}")   # SEED = 20260830

sorted ascending over the **eligible** pool. The order is a property of the
image id and the seed alone — not of the directory listing, the filesystem,
the machine, or how many rows the pool happened to have. Two boxes that build
the pool from the same photographs compute the same order.

A **shard** is a contiguous block of that order (`geometry.shard_block`).
`--shard k --n-shards 5` gives block k. Blocks are disjoint by construction,
so two boxes working different shards cannot collide, and they need no lock,
no coordination and no shared filesystem while they run.

Within a shard, a **suite** deals families by a repeating length-100 stratum
pattern built from the suite's shares. That is what makes a run resumable at a
larger `--total`: raising it extends the same deal instead of reshuffling it.
It is also the reason for the rule in §3.

## 2. What the box needs

| input | size | notes |
|---|---|---|
| the repo, `pip install -e .` | — | branch `feat/robust-aigc-detection` |
| `jpegtran` | — | `apt install libjpeg-turbo-progs`. **Required, not optional** — the only way to crop a real without re-encoding it. `generate_ov7.py` refuses to start without it. |
| the source photographs | 5.2 GB | 60,000 Open Images V7 thumbnails, `License == CC BY 2.0`, aspect ≤ 0.7, short side ≥ 400. Locally `/mnt/berstorage/techjam/open_images/portrait/` plus `attribution.csv` beside it. Pass with `--portrait-dir` / `--attribution`. |
| a GPU | ≥ 20 GB | bf16. Everything below runs unquantised; see §4 for what needs offload. |

**Do not copy `data/ov7_pool.parquet` between boxes** — its `path` column is
absolute. Let the new box rebuild it (it will, automatically, if the file is
absent). Rebuilding is safe and gives the identical order: eligibility is a
deterministic probe of each JPEG, and the order depends only on `image_id` and
the seed. It takes a few minutes for 60,000 files.

`data/ov7_captions.parquet` **is** portable and worth copying (2 MB, 19,000
rows). It is a resumable cache keyed by `image_id`; anything missing is
captioned on the fly with Florence-2-large at ~0.12 s/image.

## 3. The rules that keep the corpus consistent

1. **One shard, one suite.** A different suite's share dict re-deals every
   real in a block. A real that is `sdxl_t2i` under one suite comes out
   `sana1600m_t2i` under another, and `run._done_ids` is per family, so the
   second fake is simply generated. Never run suite B on a shard suite A has
   touched.
2. **Grow a suite by raising `--total` on the shard it already owns.** The
   deal is prefix-stable, so the existing pairs are recognised and skipped —
   you will see `done_before` in the `[done]` line, and the progress counter
   starts at the remainder, not at zero.
3. **Never change `--n-shards`.** Re-gridding to 4 puts a boundary at 13,656,
   inside a block that has already been generated.
4. **Never change `--seed`.** It defines the order and every per-image
   generation seed.
5. A **new** suite goes on a **free** shard.

`generate_ov7.used_elsewhere` enforces 1 and 5: it refuses the run and names
the offending reals before a single model loads. Treat it as the backstop, not
the plan — it can only see rows that are on the same filesystem, so it will
not catch two boxes colliding. §1's disjointness is what protects that case.

## 4. Shard map — as measured, 2026-08-31

54,624 eligible reals of 60,000 probed. 14,983 pairs generated.

| shard | order positions | suite | used | free | families |
|---|---|---:|---:|---:|---|
| 0 | 0 – 10,924 | `ov7` | 9,978 | 947 | the 7 base families |
| 1 | 10,925 – 21,849 | `ov7_lineage` | 5,000 | 5,925 | `kandinsky22_t2i`, `sana1600m_t2i` |
| 2 | 21,850 – 32,774 | — | 0 | 10,925 | **free** |
| 3 | 32,775 – 43,699 | — | 0 | 10,925 | **free** |
| 4 | 43,700 – 54,623 | `ov7_lineage2` | in progress | ~10,919 | `wuerstchen_t2i`, `cogview4_t2i` |

Regenerate this table on the box you are on rather than trusting the numbers
above — §7 has the script; it reads the rows files and is authoritative.

**Shard 4 is being written by the primary box right now.** Do not touch it.

### Suites and what they cost

| suite | families | model | licence | lineage | s/image |
|---|---|---|---|---|---:|
| `ov7` | 7 | SDXL, SD 1.5, FLUX.2-klein-4B | openrail++, creativeml, apache | `sdxl_vae`, `sd_vae`, `flux2_vae` | 1.7 – 7.8 |
| `ov7_lineage` | `kandinsky22_t2i` | Kandinsky 2.2 | apache-2.0 | `movq` | 2.07 |
| | `sana1600m_t2i` | Sana 1.6B | apache-2.0 | `dc_ae` | 1.17 |
| `ov7_lineage2` | `wuerstchen_t2i` | Wuerstchen | mit | `paella_vq` | ~3 |
| | `cogview4_t2i` | CogView4-6B | apache-2.0 | `cogview_vae` | **~39** |
| `ov7_lineage3` | `zimage_t2i` | Z-Image Turbo | apache-2.0 | see registry | unmeasured |

CogView4 is 29 GiB across the repo and runs under **model** CPU offload on a
20 GB card, which is where the ~39 s/image comes from; on a 40 GB card it
should be several times faster. Nothing is quantised anywhere —
`docs/03` §3 refuses it, because a 4-bit model's traces are partly the compute
budget's and this corpus exists to isolate the generator.

## 5. Run it

Pick a free shard from §4. Add `--rows-dir` pointing somewhere local if you
are not writing to a shared filesystem.

```bash
export PATH="$HOME/.local/bin:$PATH"          # jpegtran
python scripts/generate_ov7.py \
    --suite ov7_lineage3 --total 2000 \
    --shard 3 --n-shards 5 \
    --out data/raw_ov7_src \
    --portrait-dir /path/to/portrait \
    --attribution /path/to/attribution.csv \
    --captions data/ov7_captions.parquet
```

Smoke first, into a throwaway root — `--smoke` refuses to write into the real
corpus root, deliberately:

```bash
python scripts/generate_ov7.py --suite ov7_lineage3 --total 6 --smoke \
    --shard 3 --n-shards 5 --out /tmp/ov7_smoke
```

Check three things on the smoke before spending hours: `real.size ==
fake.size` on every pair, `encode.assert_parity` passes (it runs on every pair
already and will abort the run), and the fake is visibly a redrawing of the
same scene rather than a copy or a blank frame.

Resuming after an interruption is just re-running the identical command.

## 6. Sending the work back

A shard's output is self-contained:

```
data/raw_ov7_src/open_images_v7/<family>/<ImageID>.jpg    the fakes
data/raw_ov7_src/open_images_v7/real/<ImageID>.jpg        their cropped reals
data/raw_ov7_src/_rows/rows_<family>.jsonl                one row per pair
```

Ship all three. The rows file is the authority — it carries the seed, steps,
guidance, strength, prompt, prompt source, crop box, and the real's measured
JPEG quality and subsampling, which is what makes a pair auditable rather than
merely present.

Merging is concatenation: copy the image files in, and append the `rows_*.jsonl`
lines for families the target already has (or drop the file in whole for a
family it does not). Because shards are disjoint, no line can conflict. Then
on the primary box:

```bash
python scripts/build_dataset.py --raw data/raw_ov7_src --out data/normalized_ov7 \
    --demo-dir data/demo --manifest data/manifest_ov7.parquet \
    --preset configs/datasets/ov7.yaml --docs-dir docs/ov7 --workers 8 --force
python scripts/gate_confounds.py --manifest data/manifest_ov7.parquet --n 4000
```

`--docs-dir docs/ov7`, never `docs/` — that flag writes the frozen stream's
provenance and the default would overwrite the union stream's, silently.

## 7. Verify before you start, and after you finish

Run this on the box you are about to generate on. It is the authoritative
shard map, and if it disagrees with §4, believe it.

```bash
python - <<'PY'
import glob, json
import pandas as pd
from aigcdet.generate.geometry import order_key, shard_block
SEED = 20260830
pool = pd.read_parquet("data/ov7_pool.parquet")
elig = pool.loc[pool["eligible"]].copy()
elig["order_key"] = [order_key(i, SEED) for i in elig["image_id"]]
elig = elig.sort_values("order_key", kind="mergesort").reset_index(drop=True)
pos = {i: n for n, i in enumerate(elig["image_id"])}
rows = [json.loads(l) for f in sorted(glob.glob("data/raw_ov7_src/_rows/rows_*.jsonl"))
        for l in open(f) if l.strip()]
df = pd.DataFrame(rows); df["pos"] = df.image_id.map(pos)
for s in range(5):
    a, b = shard_block(len(elig), s, 5)
    m = df[(df.pos >= a) & (df.pos < b)]
    print(f"shard {s} [{a},{b})  used={len(m)}  free={b-a-len(m)}  {sorted(m.family.unique())}")
assert df.image_id.is_unique, "a real has more than one fake -- STOP"
print(f"{len(df)} pairs, {df.image_id.nunique()} distinct reals, no duplicates")
PY
```

After a run, audit the tree: 0 orphaned files, 0 rows whose files are missing,
0 reals used twice. `docs/ai_ov7_generation.md` §10 has the script and the
numbers the primary box got.

## 8. What would make this a negative result worth reporting

* **The confound gate regresses.** `scripts/gate_confounds.py --n 4000` against
  0.5532 / 0.6721 / 0.6374 (`jpeg_quality` / `laplacian_var` / `noise_floor`),
  plus a dimensions-only control that must sit at ~0.5. At nine families the
  corpus reads 0.5152 / 0.5632 / 0.5072 / 0.5015. A new arm that pushes any of
  those up has handed the head a shortcut, and the arm is worth less than the
  volume it added. Say so rather than absorbing it into the average.
* **A model that will not hold its size.** Two already do this: Kandinsky
  silently rounds up to a multiple of 64, Sana refuses anything 32 does not
  divide, and Sana's `use_resolution_binning` — on by default — generates at a
  1024-based aspect bin and **resizes back**, which measured 4x sharper than
  native and would have been the strongest confound in the corpus. Any new
  model must be checked for both before it is trusted, and the finding
  recorded in `ModelSpec` rather than in a comment.
* **A lineage that is not one.** Group by DECODER, not by model name. Check
  `vae/config.json` for the class, latent channel count and scaling factor
  before adding anything. Lumina-Image 2.0 and shuttle-3 both look like new
  families and both decode through FLUX.1's VAE; Kolors decodes through
  SDXL's. Architecture novelty is not decoder novelty, and only the second is
  what the held-out rung measures.
