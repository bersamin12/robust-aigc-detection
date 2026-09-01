# Robust Detection of AI-Generated Images Under Real-World Transformations

**Team AquaForge8 — TikTok TechJam 2026, Track 5.**

A detector that has to survive the internet: JPEG re-encoding, blur,
rescaling, sensor noise, filter-app colour shifts and profile-picture
cropping. Clean-data accuracy is the easy half; this project is built around
the half that isn't — and around measuring generalisation to generator
lineages the model has never seen, rather than asserting it.

---

> Cleaned primary submission repo:
> [TechJam2026-aigc-detection-AquaForge8](https://github.com/bersamin12/TechJam2026-aigc-detection-AquaForge8);
> `master` here carries the identical submission content, and the feature
> branches carry the full experiment history.

## Project overview

### The final model

Two fully fine-tuned **DINOv2 ViT-L/14 with-registers** towers (304M
parameters each) read the same 224px image in parallel; their pooled features
are concatenated into a small MLP head that outputs one logit. Both towers
start from the same pretrained weights and diverge only through the head's
random initialisation over the two halves of the feature vector.

Every image is standardised the same way before either tower sees it: a
uniform **200px crop upscaled once to 224** ("crop-200"). That policy is not
incidental — it is the result of a confound audit
([docs/resolution_shortcut.md](docs/resolution_shortcut.md),
[docs/low_level_confounds.md](docs/low_level_confounds.md)): any
canonicalisation that makes the amount of resampling a function of native
resolution leaks the label through resolution alone (upscale factor by itself
scores AUC 0.54 with no image content at all). Training pairs each clean
image with degraded views composed from six real-world transform primitives
(JPEG, blur, resize, noise, colour, crop), so the decision boundary is
learned on the transforms the model will meet, not on laboratory pixels.

### The evaluation design

The train/val split is by **decoder lineage**, never random: a generator
appears in validation only if its VAE/decoder family is absent from training
([docs/data_split.md](docs/data_split.md), `scripts/build_plan_manifest.py`).
A validation number here is a zero-shot number. The corpus: WildFake, NTIRE,
COCO and Open Images, plus ~55k fake/real pairs we generated ourselves with
open-weight generators on Open Images scenes ("OV7"). SID_Set was used in
earlier experiments but excluded from the final corpus — it has no per-model
labels, so it cannot be shown to be lineage-disjoint from training.

| split | rows | what it measures |
| --- | --- | --- |
| `train` | 350,663 (~1:1) | fit |
| `val` | 56,100 | zero-shot: lineages held out whole (VQ-VAE/MAGE/MAE, FLUX.2, NTIRE-val's unseen generators) |
| `test_transfer` | 32,196 | unseen lineages on our own OV7 generations + commercial APIs |
| `demo` | 13,841 | COCO val2017 reals vs DALL·E fakes — clean out-of-corpus check |

### Results (dual-tower @224, epoch 1)

| split | AUC | TPR @ 1% FPR |
| --- | --- | --- |
| val (zero-shot lineages) | 0.9654 | 0.7348 |
| test_transfer | 0.9963 | 0.9825 |
| demo | 0.9989 | 0.9758 |

The single-tower fine-tuned baseline (same recipe, one tower) reads 0.9177 /
0.9897 / 0.9956 AUC on the same three splits; the second tower is worth ~5
points of zero-shot val AUC. Ensembling the two models (both a 0.5-mean and a
weight fitted on val) was measured and did not beat the dual-tower model
alone on transfer, so the submission ships one model and one forward pass
per image.

### How we got here (the research trail, all in this repo)

The final recipe was chosen by an ablation ladder run on **frozen** features
first: a backbone embeds every image under 11 sampled views, embeddings are
cached to disk once (`src/aigcdet/features/`), and a ~1M-parameter head
trains on the cache in minutes on CPU (`scripts/run_ablation.py`,
`configs/rungs/`). That made it cheap to answer honestly, with everything
else held fixed:

- **Backbone**: DINOv2-with-registers beat SigLIP2, EVA-02, ConvNeXt and
  DINOv3 under fine-tuning (training loss mis-ranks; transfer AUC decides) —
  `docs/backbone_probe_table.md`.
- **Canonicalisation**: crop beats resize-to-band on both towers tested, same
  sign and margin — `docs/crop_vs_band_ablation.md`.
- **Robustness**: every candidate is scored on a 20-condition degradation
  grid, not on clean pixels — `src/aigcdet/eval/grid.py`,
  `docs/robustness_table*.md`.
- **Fusion**: all 15 combinations of 4 scoring arms were fused and scored;
  fusion helps in-domain and *hurts* transfer, so it was dropped.
- **Confounds**: sharpness/noise/JPEG-history imbalances between real and
  fake sources are measured and controlled (`scripts/audit_confounds.py`,
  `scripts/gate_confounds.py`) so the model cannot pass by reading the
  label off preprocessing history.

---

## Setup

Python ≥ 3.11, a CUDA GPU for inference (CPU works, ~10x slower).

```bash
git clone https://github.com/bersamin12/robust-aigc-detection.git
cd robust-aigc-detection
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # numpy, torch, transformers, timm, ...
pip install huggingface_hub      # only for auto-downloading the released weights
```

The first run downloads the DINOv2 backbone from the Hugging Face Hub
(ungated, no token needed) and the released checkpoint (~1.2 GB).

## Scoring a directory of images

```bash
python predict.py --images /path/to/images --out predictions.json
```

Output is a JSON list, one object per image found (searched recursively):

```json
[
  {"image_path": "/path/to/images/cat.jpg", "pred": 0.0132},
  {"image_path": "/path/to/images/dalle.png", "pred": 0.9871}
]
```

`pred` is the model's confidence in [0, 1] that the image is AI-generated.
Useful flags: `--checkpoint <file>` to score a local checkpoint instead of
downloading, `--tta` for 8-view test-time augmentation, `--skip-bad` to warn
and continue past an undecodable file instead of failing loudly, `--device
cpu` if there is no GPU.

---

## Reproducing the results

**1. Acquire the data** (~400 GB). Sources, licences and download commands:
[docs/dataset_licences.md](docs/dataset_licences.md) and
`scripts/acquire_data.py`. The OV7 pairs are generated, not downloaded:
`scripts/generate_ov7.py` runs open-weight generators spanning six decoder
lineages (SD 1.5, SDXL, Kandinsky's MoVQ, Sana's DC-AE, FLUX.1, FLUX.2) over
Open Images scenes, pairing every fake with a real crop of the same source
image so content cannot separate the classes.

**2. Build the lineage-split manifest.**

```bash
python scripts/build_plan_manifest.py   # -> data/manifest_plan.parquet
```

**3. Train the dual-tower model** (4×24 GB GPUs; both towers fully unfrozen
is ~608M trainable parameters, which is what the 224px input buys room for):

```bash
bash scripts/run_exp2_dual.sh           # wraps scripts/train_dual.py (DDP)
```

The launcher's environment variables are documented inline; the defaults are
the shipped configuration (crop-200 → 224, full unfreeze of both towers,
cosine schedule, SWA tail). Each epoch writes an atomic checkpoint;
`checkpoint_ep1.pt` is the released model.

**4. Score the splits** (this reproduces the results table):

```bash
python scripts/score_plan_splits.py --ckpt outputs/dual/dual_d24/checkpoint_ep1.pt \
    --splits val,test_transfer,demo --out-prefix outputs/dual_ep1
```

**5. Export the slim inference checkpoint** (what `predict.py` downloads):

```bash
python scripts/export_finetuned.py --ckpt outputs/dual/dual_d24/checkpoint_ep1.pt \
    --out dual_d24_ep1.pt --verify some_image_dir/
```

The frozen-feature ablation ladder that selected this recipe is reproducible
separately: `scripts/build_train_bank.py` → `scripts/extract_features.py` →
`scripts/run_ablation.py --config configs/rungs/<rung>.yaml`; the test suite
(`PYTHONPATH=src python -m pytest tests -m "not gpu"`, ~1400 tests) runs
without a GPU or any downloaded model.

---

## Limitations, and what we would do with more time

- **One epoch.** The released weights are epoch 1 of a 3-epoch schedule;
  later epochs were still training at submission time. A companion experiment
  at 518px input (14× the tokens) was lost to a cloud-pod failure mid-run.
  Both are pure compute limitations, not design ones.
- **`pred` is a raw sigmoid, not a calibrated probability.** The repo
  contains a full conditional-calibration path (temperature fitted per
  degradation condition, `src/aigcdet/calibrate/`) built for the frozen-rung
  models; it was not refit for the fine-tuned model in time. Ranking and
  thresholded decisions are unaffected; "0.9 means 90%" is not yet a claim we
  make.
- **Zero-shot means zero-shot *lineage*, not zero-shot *world*.** Val holds
  out decoder families wholesale, and transfer is measured on generators we
  ran ourselves plus commercial APIs — but the strongest proprietary models
  of the month after submission are, by construction, in nobody's training
  set. Continuous re-benchmarking is the only honest answer.
- **Dataset provenance confounds are mitigated, not eliminated.** We audit
  and control resolution, sharpness, noise and JPEG-history shortcuts
  (docs/), and the crop-200 policy exists precisely to close the largest one.
  A residual "which website did this come from" signal cannot be fully ruled
  out with public corpora.
- **608M parameters, two towers.** Fine for a scoring service, heavy for
  on-device. Given more time we would distil the dual model into a single
  small student on the same degradation-paired data, and refit the
  calibration + selective-abstention policy (both already in the repo) on the
  distilled model.
- **Still images only.** Video (per-frame + temporal consistency) is the
  obvious next surface for short-form platforms.

---

## Team members

| members | 
| --- | 
| Justin | 
| An Xian |
| Dion |
| SiTong |
| Shuen Wei | 

## Licences

Model weights: DINOv2 is Apache-2.0 ([docs/model_licences.md](docs/model_licences.md)).
Dataset terms are catalogued in [docs/dataset_licences.md](docs/dataset_licences.md).
