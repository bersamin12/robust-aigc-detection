#!/bin/bash
# Amendment 4 (docs/ntire/paper_prereg.md): the G1 proxy protocol with a CLIP
# ViT-L/14 tower in place of DINOv2, three seeds per policy, then the
# crop-trained checkpoints presented whole at test time.
#
#   bash scripts/clip_proxy.sh            # ~6-7 GPU-hours
#   SMOKE=1 bash scripts/clip_proxy.sh    # minutes; separate output dirs
#
# Every flag of the training call is the one stage P of paper_pipeline.sh
# used, except --backbone. Fit splits only; ntire_test is never touched.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

SMOKE="${SMOKE:-0}"
PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
OUT_DIR="${OUT_DIR:-/scratch/justin_r13/aigcdet/outputs}"
WORKERS="${WORKERS:-14}"
DEGRADE_SEED=20261008
SEEDS=(20260827 20260828 20260829)
PROXY_BANK=data/banks/ntire_meta
BACKBONE=clipl
if [ "$SMOKE" = 1 ]; then
  TAG=paper_clip_smoke; PROXY_FRAC=0.002; LIMIT="--limit 200"; NBOOT=200
  SC=outputs/scores/paper_clip_smoke; DOCS=docs/ntire/paper_clip_smoke
else
  TAG=paper_clip; PROXY_FRAC=0.25; LIMIT=""; NBOOT=2000
  SC=outputs/scores/paper; DOCS=docs/ntire/paper
fi
RUNS="$OUT_DIR/$TAG"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$RUNS" "$SC" "$DOCS" logs

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "REFUSING: tracked files are modified; commit first so every number has a commit"
  git status --porcelain --untracked-files=no; exit 1
fi
grep -q "^## Amendment 4" docs/ntire/paper_prereg.md || {
  log "REFUSING: amendment 4 is not in the committed pre-registration"; exit 1; }
COMMIT=$(git rev-parse HEAD)
log "clip proxy at $COMMIT (SMOKE=$SMOKE) runs=$RUNS scores=$SC"

wait_gpu() {
  local need="$1" seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    if [ -z "$free" ]; then
      log "nvidia-smi gave no reading; retrying in 5 min"; sleep 300; continue
    fi
    if [ "$free" -ge "$need" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free"; return; }
    else
      [ "$seen" -gt 0 ] || [ $((SECONDS % 1800)) -lt 60 ] && log "waiting for GPU: ${free} MiB free, need $need"
      seen=0
    fi
    sleep 60
  done
}

wait_disk() {
  local free_gb
  while true; do
    free_gb=$(df --output=avail -BG "$RUNS" | tail -1 | tr -dc 0-9)
    [ "$free_gb" -ge "${1:-40}" ] && return
    log "only ${free_gb} GB free under $RUNS, need ${1:-40}; waiting"; sleep 600
  done
}

retry() {   # <label> <cmd...>
  local label="$1" n="${TRIES:-6}" attempt; shift
  for attempt in $(seq 1 "$n"); do
    "$@" && return 0
    log "$label: attempt $attempt of $n failed"; [ "$attempt" -lt "$n" ] && sleep 600
  done
  log "CLIP_PROXY_FAILED at $label"; exit 1
}

epoch_of() {
  [ -f "$1" ] || { echo -1; return; }
  "$PY" -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False,mmap=True)['epoch'])" 2>/dev/null || echo -1
}

score() {   # <ckpt> <prefix> <splits>
  local ck="$1" prefix="$2" splits="$3" sp
  for sp in ${splits//,/ }; do
    [ -f "${prefix}_${sp}.parquet" ] && continue
    wait_gpu 8000
    # shellcheck disable=SC2086
    "$PY" scripts/score_plan_splits.py --ckpt "$ck" --manifest "$MANIFEST" --splits "$sp" \
      --crop-clamp --degrade-seed "$DEGRADE_SEED" --batch 32 --chunk 16 \
      --workers 12 --out-prefix "$prefix" $LIMIT 2>&1 \
      | grep -av "it/s\]" | tee -a "logs/${TAG}_score.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "${prefix}_${sp}.parquet" ] || return 1
  done
}

probe() {   # <ckpt> <seed-index>: the crop checkpoint, center vs whole image
  local ck="$1" i="$2" pre="$SC/clip_probe_s$2"
  [ -f "${pre}_center_ntire_val_hard.parquet" ] && [ -f "${pre}_squish_ntire_val_hard.parquet" ] && return 0
  wait_gpu 9000
  "$PY" -u scripts/ntire_crop_probe.py --ckpt "$ck" --arms center,squish \
    --split ntire_val_hard --out "$DOCS/clip_probe_s$i.json" --parquet-prefix "$pre" \
    --batch 32 --chunk 8 --workers 12 2>&1 | grep -av "it/s\]" | tee -a "logs/${TAG}_probe.log"
  [ "${PIPESTATUS[0]}" -eq 0 ]
}

arm() {   # <policy> <seed-index 1..3>
  local pol="$1" i="$2" seed="${SEEDS[$(( $2 - 1 ))]}" name="clip_${1}_s${2}"
  local ck="$RUNS/$name/checkpoint.pt" canon
  if [ -f "$SC/${name}_ntire_val.parquet" ] && [ -f "$SC/${name}_ntire_val_hard.parquet" ] \
     && { [ "$pol" = squish ] || [ -f "$SC/clip_probe_s${i}_squish_ntire_val_hard.parquet" ]; }; then
    return 0
  fi
  if [ "$pol" = crop ]; then canon="--canon-mode crop --crop-side 200 --nominal-side 224"
  else canon="--canon-mode squish --nominal-side 224"; fi
  if [ "$(epoch_of "$ck")" -lt 1 ]; then
    wait_disk 15; wait_gpu 12000
    log "$name: $BACKBONE depth 24, 1 epoch over $PROXY_FRAC of train, seed $seed, $pol"
    # shellcheck disable=SC2086
    "$PY" -u scripts/train_unfreeze.py --bank "$PROXY_BANK" --root "$EXP" \
      --backbone "$BACKBONE" --depth 24 --name "$name" --out-dir "$RUNS" \
      --epochs 1 --n-src 64 --m-deg 2 --src-chunk 8 --workers "$WORKERS" \
      --seed "$seed" --device cuda $canon \
      --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
      --family-weighting proportional --train-subsample-frac "$PROXY_FRAC" \
      --token-pool mean --resume 2>&1 | grep -av "it/s\]" | tee -a "logs/$name.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  fi
  score "$ck" "$SC/$name" ntire_val_hard,ntire_val || return 1
  if [ "$pol" = crop ]; then probe "$ck" "$i" || return 1; fi
}

for i in 1 2 3; do for pol in crop squish; do
  retry "clip $pol s$i" arm "$pol" "$i"
done; done

"$PY" scripts/paper_results.py g1 --scores "$SC" --prefix clip --backbone "$BACKBONE" \
  --n-boot "$NBOOT" --out "$DOCS/clip_proxy_gate" | tail -12
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || { log "CLIP_PROXY_FAILED at gate (exit $rc)"; exit 1; }
"$PY" scripts/paper_results.py testtime --scores "$SC" --prefix clip \
  --n-boot "$NBOOT" --out "$DOCS/clip_testtime" | head -12
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "CLIP_PROXY_FAILED at testtime"; exit 1; }
echo "$COMMIT $(date -u +%FT%TZ)" > "$DOCS/clip_proxy.done"
log "CLIP_PROXY_DONE (gate exit $rc: 0 passed, 3 failed)"
