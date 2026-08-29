# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026, Track 5.

A detector that has to survive the internet: JPEG re-encoding, blur, rescaling,
sensor noise, filter-app colour shifts and profile-picture cropping. Clean-data
accuracy is the easy half; this project is built around the half that isn't.

---

## The approach in one paragraph

A **frozen** vision backbone (DINOv3 ViT-L/16, 303M parameters) embeds every
training image under 11 sampled views — one clean, ten degraded by randomly
composed real-world transforms. Those embeddings are cached once to disk
(**Stage A**, GPU-bound). A ~1M-parameter head then trains on the cached
vectors (**Stage B**, CPU), which makes the entire ablation ladder cheap enough
to run honestly: every rung is the same function with different flags, so
comparisons differ only in the thing under test.

Nothing here fine-tunes a backbone. The whole budget goes into *what you train
on top of frozen features*, and into measuring robustness rather than asserting
it.

---

## Repository layout

```
src/aigcdet/
  data/        manifest freezing, dataset registries, image verification
  augment/     the six transform primitives, recipe sampling, canonicalisation
  features/    backbones, feature banks, sharded extraction, merge
  models/      detector heads, losses, paired sampler
  train/       Stage B rung training
  eval/        condition grid, robustness table, controls, fusion, reporting
  baselines/   published baselines + the resolution control
scripts/       CLI entry points (acquire, build, extract, merge, train, report)
notebooks/     Kaggle bootstrap for the five-person Stage A fleet
configs/rungs/ the ablation ladder, one YAML per rung
docs/          licences, the resolution-shortcut finding, the fleet runbook
tests/         1600 tests
```

---

## Setup

Python ≥ 3.11 (held there for Kaggle compatibility).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

`torch` and `transformers` are declared with loose lower bounds on purpose — on
a machine that already has a CUDA-matched torch, they must not be allowed to
fight it. On Kaggle, install with `--no-deps`; see
`notebooks/kaggle_bootstrap.py`, which encodes the pip plan that does not
destroy the session's torch.

The reconstruction branch (§3.3) needs `pip install -e ".[recon]"` and is
imported lazily, so the package stays importable and testable without it.

---

## Scoring a directory of images

The submission's inference entry point. A directory in, a JSON file out, one
object per image:

```bash
python scripts/predict.py --images path/to/images \
       --checkpoint outputs/rungs/a3/checkpoint.pt --out predictions.json
```

```json
[
  {"image_path": "path/to/images/a.png", "pred": 0.9317},
  {"image_path": "path/to/images/b.jpg", "pred": 0.0412}
]
```

`pred` is P(AI-generated) in [0, 1]. The directory is searched recursively,
non-images are ignored, and rows come out sorted so two runs of the same
directory are diffable.

It scores the **clean view only** — the transforms exist to make training
robust to degradation, not to degrade the image you asked about — and it
canonicalises resolution before the backbone sees anything, exactly as the
three other decode sites do. A file that cannot be decoded is named on stderr
and the run exits non-zero with the other scores still written; an empty
directory and a checkpoint whose head needs the reconstruction branch are both
refused by name rather than producing a plausible-looking empty or wrong result.

---

## Reproducing the results

### 1. Acquire and freeze the data

```bash
python scripts/acquire_data.py --dataset sid_set  --out data/raw
python scripts/acquire_data.py --dataset wildfake --out data/raw \
       --generators ddim,ddpm,VQGAN,BigGAN,...
python scripts/acquire_data.py --dataset wildfake_benchmark --benchmark-dir data/demo
python scripts/build_dataset.py --raw data/raw --out data/normalized \
       --demo-dir data/demo --manifest data/normalized/manifest.parquet
python scripts/build_benchmark_manifest.py    # -> data/demo/benchmark_manifest.parquet
```

Every dataset's licence and source URL is recorded **at acquisition time** into
`LICENCES.json` and carried into the manifest per row. `build_dataset` refuses
to fabricate provenance for a source with no recorded entry.

The manifest is **frozen on write**. Feature banks index it positionally, so
re-splitting after banks exist silently misaligns labels against cached
features — `FeatureBank.verify_against_manifest` exists to make that loud.

### 2. Stage A — extract the feature bank (GPU)

Single machine:

```bash
python scripts/extract_features.py --manifest data/normalized/manifest.parquet \
       --backbone dinov3l --out banks/dinov3l --split train,val_internal
```

Across five Kaggle accounts: **[docs/kaggle_fleet_runbook.md](docs/kaggle_fleet_runbook.md)**.

```bash
python scripts/merge_banks.py banks/dinov3l_shard{0,1,2,3,4} --out banks/dinov3l
```

### 3. Stage B — train the ladder (CPU)

```bash
python scripts/train_rung.py --config configs/rungs/a3.yaml \
       --bank banks/dinov3l --manifest data/normalized/manifest.parquet
python scripts/run_ablation.py --bank banks/dinov3l --out outputs/rungs
```

| Rung | What it adds |
|---|---|
| A0 | Linear probe, clean views only (= UniversalFakeDetect) |
| A1 | + augmented training views |
| A2 | + auxiliary degradation-prediction loss |
| A3 | + clean/degraded consistency loss — **headline candidate** |
| A4 | + reconstruction features (kill criterion applies) |
| A7 | + FiLM conditioning |

### 4. Evaluate under the transformation grid

The ablation tier spans `val_internal`, `heldout_generator` and `benchmark`,
which live in two separate frozen manifests — so they are joined first, into a
manifest re-rooted onto their common ancestor with fresh, unique index labels
(the per-view RNG key):

```bash
python scripts/build_eval_manifest.py \
       --manifest data/normalized/manifest.parquet \
       --benchmark-manifest data/demo/benchmark_manifest.parquet \
       --out data/eval_manifest.parquet

python scripts/extract_eval_bank.py --manifest data/eval_manifest.parquet \
       --backbone dinov3l --out banks/eval --tier ablation
```

20 conditions: JPEG q∈{90,70,50,30}, Gaussian blur σ∈{0.5,1.0,2.0}, resize
0.5×/0.25× then upscale, Gaussian noise σ∈{0.02,0.05,0.10}, colour jitter ±20%,
centre crop 80%, plus clean and composed conditions.

---

## Two findings that shaped the design

### Resolution leaks the label — and it leaks *backwards*

29% of the training pool sits at image sizes that are 100% generated. Short side
alone separates the classes at 72.6% accuracy against a 52.9% baseline.

On the organisers' demo benchmark it is worse: COCO val2017 is uniformly
200×200 and DALL·E Advanced runs 346–1746, so the two halves share **no**
resolution stratum under any definition. A classifier that reads only the image
dimensions and never a pixel scores **TPR 1.0000, AUC 1.0000** there.

The two shortcuts point in **opposite directions**. Fit the dimensions-only
model on the training pool, score it on the benchmark, and you get **AUC
0.0006** — near-perfectly inverted. A model leaning on resolution is punished,
not rewarded.

So every decode site canonicalises resolution before any condition transform is
applied (`src/aigcdet/augment/canonical.py`, wired in `features/extract.py`,
`eval/grid.py` and `features/recon.py`), and the dimensions-only classifier is
kept as a registered **control** — never a published baseline — so every
headline number is read against it. Full analysis:
[docs/resolution_shortcut.md](docs/resolution_shortcut.md).

### The benchmark's effective sample size is smaller than it looks

Of the 8,843 DALL·E 3 benchmark images, only **3,719 are distinct**; 5,124 are
byte-identical duplicates (verified by content digest and independently by
`md5sum`). COCO's 4,998 are clean. That is roughly 42% effective sample size on
the generated half, and it widens every confidence interval on that side.

---

## Limitations, and what we would do next

- **A second shortcut is unresolved.** At matched native resolution, image
  sharpness alone predicts the label at 68.5% — real images in this corpus are
  simply sharper than generated ones. Canonicalisation neither causes nor cures
  it. The fix is source-balanced sampling, which we have not done.
- **One backbone.** The design supports DINOv3, SigLIP2 and CLIP with a fusion
  path, but only DINOv3 has been budgeted for.
- **Generalisation is measured, not solved.** `heldout_generator` holds out
  entire generator families, and the LOTO rung holds out entire transform
  families, but a genuinely novel generator remains the open risk.
- **The benchmark is a reference, not a score.** COCO val2017 and DALL·E
  Advanced are excluded from training by source, and `build_dataset` enforces
  the exclusion structurally rather than by convention.

---

## Datasets and licences

| Dataset | Role | Licence |
|---|---|---|
| SID_Set | Training | CC BY 4.0 |
| WildFake | Training | Apache-2.0 (compilation); constituent subsets keep upstream terms, several non-commercial |
| COCO val2017 | Benchmark only | CC BY 4.0, images under Flickr terms |
| DALL·E Advanced | Benchmark only | Competition brief |

WildFake is a **compilation**, and its Apache-2.0 hub metadata does not relicense
its constituents (FFHQ CC BY-NC-SA 4.0, CelebA-HQ research-only, AFHQ CC BY-NC
4.0, ImageNet non-commercial, LSUN research). Full table, with what was verified
and what was not: [docs/dataset_licences.md](docs/dataset_licences.md).
Model weight provenance: [docs/model_licences.md](docs/model_licences.md).

All models are well under the 2B-parameter limit — the largest is DINOv3
ViT-L/16 at 303,129,600 parameters, measured rather than quoted.

---

## Development

Built with subagent-driven development: isolated git worktrees per task, TDD,
and mutation testing as the standard for whether a test earns its place. A test
is not accepted until a deliberate mutation of the code it covers has been shown
to make it fail — validated against unmutated code first, counted as a kill only
on a reported `FAILED`, and with the source restored and sha256-verified
afterwards.

```bash
pytest -q        # 1600 passed, 3 skipped
```
