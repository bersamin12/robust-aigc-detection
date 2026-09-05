#!/bin/bash
# The NTIRE stream on ONE RTX 4090: two dinov2regl branches, trained one after
# the other, scored on the organisers' sets, fused on val, read once on test.
#
#   STAGE=smoke  bash scripts/ntire_one_gpu.sh     # 2% of train, 1 epoch each
#   STAGE=B      bash scripts/ntire_one_gpu.sh     # dual @224, 3 epochs
#   STAGE=A      bash scripts/ntire_one_gpu.sh     # single @518, 2 epochs
#   STAGE=score  bash scripts/ntire_one_gpu.sh     # both branches on the
#                                                  # selection splits + fusion
#   STAGE=test   bash scripts/ntire_one_gpu.sh     # the ONE read of ntire_test
#
# Every stage is idempotent -- a finished artefact is not rebuilt -- and both
# trainers take --resume, so a killed run picks up at its last epoch.
#
# Memory: bench_d24_c2.json measured 13.5 GiB for the single tower at 518 with
# src_chunk 2 (and that bench used the ineffective checkpointing path, so the
# real run should sit below it). The dual tower at 224 was never benchmarked
# on one card; DUAL_CHUNK=8 is the norway recipe, drop to 4 if nvidia-smi
# shows more than ~20 GiB during the smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-smoke}"
PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
BANK="${BANK:-data/banks/ntire_meta}"
WORKERS="${WORKERS:-14}"
SEED="${SEED:-20260827}"
A_EPOCHS="${A_EPOCHS:-2}"
B_EPOCHS="${B_EPOCHS:-3}"
A_CHUNK="${A_CHUNK:-2}"
DUAL_CHUNK="${DUAL_CHUNK:-8}"
SUBSAMPLE="${SUBSAMPLE:-1.0}"
NAME_A="${NAME_A:-ntire_d24_518}"
NAME_B="${NAME_B:-ntire_dual224}"
SCORES="${SCORES:-outputs/scores}"
FIT_SPLITS="${FIT_SPLITS:-ntire_val,ntire_val_hard}"

export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs "$SCORES"

log() { echo "[$(date +%H:%M:%S)] $*"; }

preflight() {
  [ -x "$PY" ] || { echo "no python at $PY" >&2; exit 1; }
  [ -d "$BANK" ] || { echo "no bank at $BANK; run build_train_bank.py" >&2; exit 1; }
  [ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 1; }
  "$PY" - "$BANK" <<'PYEOF'
import sys, os, pandas as pd, json
bank = sys.argv[1]
cfg = json.load(open(os.path.join(bank, "config.json")))
m = pd.read_parquet(os.path.join(bank, "meta.parquet"))
n = (m.split == "train").sum()
assert n > 0, "bank has no train rows"
miss = [p for p in m.path.sample(min(400, len(m)), random_state=0) if not os.path.exists(p)]
assert not miss, f"{len(miss)} sampled bank paths missing, e.g. {miss[:2]}"
pos, neg = int((m[m.split=='train'].label==1).sum()), int((m[m.split=='train'].label==0).sum())
print(f"bank ok: {n:,} train rows ({pos:,} fake / {neg:,} real), pool={cfg.get('recipe_pool')} "
      f"max_ops={cfg.get('max_ops')} -> {min(pos,neg)//32:,} steps/epoch")
PYEOF
}

common_train=(--bank "$BANK" --root "$EXP" --depth 24 --n-src 64 --m-deg 2
              --workers "$WORKERS" --seed "$SEED" --device cuda
              --canon-mode crop --crop-side 200
              --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01
              --swa --swa-start-frac 0.75 --family-weighting proportional
              --train-subsample-frac "$SUBSAMPLE" --resume)

train_B() {
  local name="$1" epochs="$2"
  if [ -f "outputs/dual/$name/checkpoint.pt" ] && \
     "$PY" -c "import torch,sys; ck=torch.load('outputs/dual/$name/checkpoint.pt',map_location='cpu',weights_only=False); sys.exit(0 if ck['epoch']>=$epochs else 1)"; then
    log "B ($name) already complete"; return
  fi
  log "B: dual dinov2regl224, depth 24, $epochs epoch(s), chunk $DUAL_CHUNK -> logs/$name.log"
  "$PY" -u scripts/train_dual.py "${common_train[@]}" \
    --backbone dinov2regl224 --name "$name" --out-dir outputs/dual \
    --perturb-tower2 0.0 --epochs "$epochs" --src-chunk "$DUAL_CHUNK" \
    --nominal-side 224 2>&1 | tee -a "logs/$name.log"
}

train_A() {
  local name="$1" epochs="$2"
  if [ -f "outputs/unfreeze/$name/checkpoint.pt" ] && \
     "$PY" -c "import torch,sys; ck=torch.load('outputs/unfreeze/$name/checkpoint.pt',map_location='cpu',weights_only=False); sys.exit(0 if ck['epoch']>=$epochs else 1)"; then
    log "A ($name) already complete"; return
  fi
  log "A: single dinov2regl @518, depth 24, $epochs epoch(s), chunk $A_CHUNK -> logs/$name.log"
  "$PY" -u scripts/train_unfreeze.py "${common_train[@]}" \
    --backbone dinov2regl --name "$name" --out-dir outputs/unfreeze \
    --epochs "$epochs" --src-chunk "$A_CHUNK" 2>&1 | tee -a "logs/$name.log"
}

score() {
  local ckpt="$1" prefix="$2" splits="$3" chunk="$4"; shift 4
  for sp in ${splits//,/ }; do
    if [ -f "${prefix}_${sp}.parquet" ]; then log "score: ${prefix}_${sp}.parquet exists"; continue; fi
    "$PY" scripts/score_plan_splits.py --ckpt "$ckpt" --manifest "$MANIFEST" \
      --splits "$sp" --crop-clamp --batch 32 --chunk "$chunk" --workers 12 \
      --out-prefix "$prefix" "$@" 2>&1 | tee -a logs/score_$(basename "$prefix").log
  done
}

case "$STAGE" in
  smoke)
    preflight
    SUBSAMPLE=0.02
    common_train[-2]="$SUBSAMPLE"   # --train-subsample-frac value sits before --resume
    train_B "smoke_dual224" 1
    train_A "smoke_d24_518" 1
    score outputs/dual/smoke_dual224/checkpoint.pt "$SCORES/smokeB" "ntire_val_hard" 16 --limit 500
    score outputs/unfreeze/smoke_d24_518/checkpoint.pt "$SCORES/smokeA" "ntire_val_hard" 8 --limit 500
    "$PY" scripts/fuse_branches.py --a "$SCORES/smokeA" --b "$SCORES/smokeB" \
      --fit-splits ntire_val_hard --out docs/ntire/fusion_smoke.json
    log "smoke chain complete"
    ;;
  B) preflight; train_B "$NAME_B" "$B_EPOCHS" ;;
  A) preflight; train_A "$NAME_A" "$A_EPOCHS" ;;
  score)
    score outputs/dual/$NAME_B/checkpoint.pt "$SCORES/B" "val,$FIT_SPLITS" 16
    score outputs/unfreeze/$NAME_A/checkpoint.pt "$SCORES/A" "val,$FIT_SPLITS" 8
    "$PY" scripts/fuse_branches.py --a "$SCORES/A" --b "$SCORES/B" \
      --fit-splits "$FIT_SPLITS" --eval-splits val --out docs/ntire/fusion_ntire.json
    ;;
  test)
    # The ONE read of the sealed split. Refuses to run twice.
    for p in "$SCORES/A_ntire_test.parquet" "$SCORES/B_ntire_test.parquet"; do
      [ -f "$p" ] && { echo "$p exists: ntire_test has already been read" >&2; exit 1; }
    done
    [ -f docs/ntire/fusion_ntire.json ] || { echo "fit the fusion first (STAGE=score)" >&2; exit 1; }
    score outputs/dual/$NAME_B/checkpoint.pt "$SCORES/B" "ntire_test" 16
    score outputs/unfreeze/$NAME_A/checkpoint.pt "$SCORES/A" "ntire_test" 8
    "$PY" scripts/fuse_branches.py --a "$SCORES/A" --b "$SCORES/B" \
      --fit-splits "$FIT_SPLITS" --eval-splits val --test-split ntire_test \
      --write-fused --out docs/ntire/fusion_ntire_final.json
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
