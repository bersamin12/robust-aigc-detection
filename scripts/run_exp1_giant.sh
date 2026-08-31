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
col = "rel_path" if "rel_path" in b.meta.columns else "path"
if col != "rel_path":
    sys.exit("REFUSING: bank has no rel_path column, so it cannot be rebased "
             "onto this box's corpus root")
paths = b.meta[col].to_numpy()
miss = [p for p in paths[:400] if not os.path.exists(os.path.join(root, str(p)))]
if miss:
    sys.exit(f"REFUSING: {len(miss)}/400 sampled rows do not resolve under "
             f"{root}, e.g. {miss[:2]}")
print(f"preflight OK: {len(b.meta):,} rows, {n_train:,} train, corpus resolves")
PY

echo "[$(date +%H:%M:%S)] $NM depth=$DEPTH epochs=$EPOCHS gpus=$GPUS chunk=$CHUNK" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] NOTE: prints nothing until epoch 1 completes. Silence is normal." | tee -a "$LOG"

OMP_NUM_THREADS=4 $PY -m torch.distributed.run \
  --nproc_per_node="$GPUS" --master_port=29585 \
  scripts/train_unfreeze.py \
  --bank "$BANK" --root "$ROOT" \
  --backbone dinov2regg --depth "$DEPTH" --name "$NM" \
  --out-dir outputs/unfreeze --epochs "$EPOCHS" \
  --canon-mode crop --crop-side 200 \
  --device cuda --workers 24 --src-chunk "$CHUNK" --resume \
  --lr-schedule "$SCHED" --warmup-frac "$WARMUP" \
  --min-lr-frac "$MINLR" $SWA_FLAG --swa-start-frac "$SWA_START" \
  >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] exit=$? -> outputs/unfreeze/$NM/checkpoint.pt" | tee -a "$LOG"
