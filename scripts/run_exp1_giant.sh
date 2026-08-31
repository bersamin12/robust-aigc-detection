#!/usr/bin/env bash
# Experiment 1: dinov2reg GIANT (1.136B) at 518, crop, full unfreeze, 375k.
#
# TARGET: taiwan8 (8x RTX 4090, 24 GiB) as of the 2026-09-01 swap.
# The memory risk here is real and is the
# reason src-chunk defaults to 1: AdamW holds fp32 master weights plus two
# moments, so 1.136B parameters cost ~13.6 GiB of optimiser state before a
# single activation, and the tower sees 1369 tokens at 518. If this OOMs, the
# lever is --depth (fewer trailing blocks trainable), NOT --n-src: n_src is
# what makes this arm comparable with every cached rung.
set -uo pipefail
cd "$(dirname "$0")/.."

NM=${NM:-giant_d40}
DEPTH=${DEPTH:-40}          # ViT-g/14 has 40 blocks; 40 == full unfreeze
EPOCHS=${EPOCHS:-3}
CHUNK=${CHUNK:-1}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
# Decode and canonicalisation are the CPU floor and they do not shrink when the
# tower grows, so the worker count is set from the box, not from a default that
# was right on a 24-core one. One BLAS thread per worker: a fat box otherwise
# multiplies WORKERS x OMP into more threads than it has cores.
WORKERS=${WORKERS:-24}
OMP=${OMP:-1}
# Canonicalisation. 518 is the giant's native side, so CROPSIDE=NOMINAL=518
# with CLAMP=1 takes the native window whenever the image is big enough and
# upscales only the images that are smaller -- one resample, never two.
CROPSIDE=${CROPSIDE:-200}
NOMINAL=${NOMINAL:-518}
CLAMP=${CLAMP:-0}
CLAMP_FLAG=""; [ "$CLAMP" = "1" ] && CLAMP_FLAG="--crop-clamp"
BANK=${BANK:-data/banks/full_crop_dinov2regl}
ROOT=${ROOT:-/workspace/data/union}
PY=${PY:-/venv/main/bin/python}
# Schedule is ON here even though the library default is off: these two
# arms are new shipping candidates, not rungs of the constant-LR ladder,
# so they are comparable with each other and NOT with D0..D24.
SCHED=${SCHED:-cosine}
WARMUP=${WARMUP:-0.03}
MINLR=${MINLR:-0.01}
SWA=${SWA:-1}
SWA_START=${SWA_START:-0.75}
SWA_FLAG=""; [ "$SWA" = "1" ] && SWA_FLAG="--swa"
LOG=logs/${NM}.log
mkdir -p logs outputs/unfreeze

# --- preflight ------------------------------------------------------------
# Every one of these has cost this project a run. A bank whose rel_paths do not
# resolve fails hours in, at the first decode, with the tower already loaded.
$PY - "$BANK" "$ROOT" <<'PY' || exit 1
import os, sys
from aigcdet.features.bank import FeatureBank
bank_dir, root = sys.argv[1], sys.argv[2]
if not os.path.isdir(bank_dir):
    sys.exit(f"REFUSING: no bank at {bank_dir}")
b = FeatureBank.open(bank_dir)
n_train = int((b.meta["split"].to_numpy() == "train").sum())
if n_train == 0:
    sys.exit(f"REFUSING: {bank_dir} has no train rows")
# Mirror `LiveViewSampler` (train/finetune.py:204) EXACTLY. It resolves
# `rel_path` against the root when present and takes `path` as already
# absolute otherwise, which is what a bank built with --path-map carries:
# rewriting paths across machines makes any single root a lie.
col = "rel_path" if "rel_path" in b.meta.columns else "path"
if col not in b.meta.columns:
    sys.exit("REFUSING: bank has neither a rel_path nor a path column")
absolute = col == "path"
paths = b.meta[col].to_numpy()
resolve = (lambda p: p) if absolute else (lambda p: os.path.join(root, p))
miss = [p for p in paths[:400] if not os.path.exists(resolve(str(p)))]
if miss:
    sys.exit(f"REFUSING: {len(miss)}/400 sampled rows do not resolve "
             f"({col}, absolute={absolute}, root={root}), e.g. {miss[:2]}")
print(f"preflight OK: {len(b.meta):,} rows, {n_train:,} train, "
      f"corpus resolves via {col} (absolute={absolute})")
PY

echo "[$(date +%H:%M:%S)] $NM depth=$DEPTH epochs=$EPOCHS gpus=$GPUS chunk=$CHUNK workers=$WORKERS omp=$OMP sched=$SCHED swa=$SWA crop=$CROPSIDE nominal=$NOMINAL clamp=$CLAMP" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] NOTE: prints nothing until epoch 1 completes. Silence is normal." | tee -a "$LOG"

OMP_NUM_THREADS=$OMP $PY -m torch.distributed.run \
  --nproc_per_node="$GPUS" --master_port=29585 \
  scripts/train_unfreeze.py \
  --bank "$BANK" --root "$ROOT" \
  --backbone dinov2regg --depth "$DEPTH" --name "$NM" \
  --out-dir outputs/unfreeze --epochs "$EPOCHS" \
  --canon-mode crop --crop-side "$CROPSIDE" \
  --nominal-side "$NOMINAL" $CLAMP_FLAG \
  --device cuda --workers "$WORKERS" --src-chunk "$CHUNK" --resume \
  --lr-schedule "$SCHED" --warmup-frac "$WARMUP" \
  --min-lr-frac "$MINLR" $SWA_FLAG --swa-start-frac "$SWA_START" \
  >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] exit=$? -> outputs/unfreeze/$NM/checkpoint.pt" | tee -a "$LOG"
