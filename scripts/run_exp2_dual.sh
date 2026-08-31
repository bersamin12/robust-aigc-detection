#!/usr/bin/env bash
# Experiment 2: TWO dinov2regl towers at 224 in parallel into one MLP head,
# crop, both fully fine-tuned, 375k.
#
# Sized for 8x RTX 4090 (24 GiB). 2 x 304M = 608M trainable, so ~7.3 GiB of
# AdamW state; 224px is 262 tokens against 518's 1374, which is what makes two
# towers fit where one at 518 barely does.
#
# Both towers start from the SAME pretrained weights and see the SAME pixels.
# They diverge only because the head's first layer has different random weights
# over the two halves of the concatenated vector. If they turn out to stay too
# similar to be worth two towers, --perturb-tower2 is the stronger symmetry
# break -- but it makes tower 2 no longer the pretrained model, so it is a
# deliberate second experiment, not a knob to reach for silently.
set -uo pipefail
cd "$(dirname "$0")/.."

NM=${NM:-dual_d24}
DEPTH=${DEPTH:-24}          # ViT-L/14 has 24 blocks; 24 == full unfreeze, per tower
EPOCHS=${EPOCHS:-3}
CHUNK=${CHUNK:-2}
GPUS=${GPUS:-8}
PERTURB=${PERTURB:-0.0}
BANK=${BANK:-data/banks/full_crop_dinov2regl}
ROOT=${ROOT:-/workspace/data/union}
PY=${PY:-/venv/main/bin/python}
LOG=logs/${NM}.log
mkdir -p logs outputs/dual

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
if "rel_path" not in b.meta.columns:
    sys.exit("REFUSING: bank has no rel_path column")
paths = b.meta["rel_path"].to_numpy()
miss = [p for p in paths[:400] if not os.path.exists(os.path.join(root, str(p)))]
if miss:
    sys.exit(f"REFUSING: {len(miss)}/400 sampled rows do not resolve under "
             f"{root}, e.g. {miss[:2]}")
print(f"preflight OK: {len(b.meta):,} rows, {n_train:,} train, corpus resolves")
PY

echo "[$(date +%H:%M:%S)] $NM 2x dinov2regl224 depth=$DEPTH epochs=$EPOCHS gpus=$GPUS" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] NOTE: prints nothing until epoch 1 completes. Silence is normal." | tee -a "$LOG"

OMP_NUM_THREADS=4 $PY -m torch.distributed.run \
  --nproc_per_node="$GPUS" --master_port=29586 \
  scripts/train_dual.py \
  --bank "$BANK" --root "$ROOT" \
  --backbone dinov2regl224 --depth "$DEPTH" --name "$NM" \
  --perturb-tower2 "$PERTURB" \
  --out-dir outputs/dual --epochs "$EPOCHS" \
  --canon-mode crop --crop-side 200 \
  --device cuda --workers 24 --src-chunk "$CHUNK" --resume \
  >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] exit=$? -> outputs/dual/$NM/checkpoint.pt" | tee -a "$LOG"
