#!/bin/bash
# The squish paper, end to end, unattended (plan: i-actually-want-to-giggly-volcano.md).
#
#   OUT_DIR=/scratch/justin_r13/aigcdet/outputs bash scripts/paper_pipeline.sh
#   SMOKE=1 OUT_DIR=... bash scripts/paper_pipeline.sh     # minutes, not days
#
# Stages, each idempotent and marked by $MARK/stage_<id>.done (which records
# the commit it ran at), so a relaunch resumes at the first unfinished one:
#
#   V   the new canonicalise reproduces the probe's squish numbers exactly
#   P   proxy arms: crop and squish, three seeds each (dinov2regl224, 25%)
#   G1  the pre-registered proxy gate; FAIL stops the pipeline here, by design
#   B   branch B (dual dinov2regl224) trained under squish, 3 epochs
#   A   branch A (dinov2regl @512) trained under squish, 2 epochs
#   X   external manifest (waits for the Synthbuster download, up to a cap)
#   S   scoring, final-vs-SWA picks on ntire_val, both fusions fitted on ntire_val
#   G   G2 / G3 and every table, from parquets (paper_results.py full)
#   T   the DECLARED second read of ntire_test (never in SMOKE)
#
# The gates are pre-registered in docs/ntire/paper_prereg.md, which must be
# committed before this starts: the script refuses a dirty tree, so every
# number it writes is attributable to one commit.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

SMOKE="${SMOKE:-0}"
PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
EXT_MANIFEST="${EXT_MANIFEST:-data/manifest_external$( [ "${SMOKE:-0}" = 1 ] && echo _smoke ).parquet}"
EXT_ROOT="${EXT_ROOT:-/scratch/justin_r13/ext}"
SB_LOG="${SB_LOG:-$EXT_ROOT/logs/synthbuster_dl.log}"
SB_WAIT_H="${SB_WAIT_H:-24}"
OUT_DIR="${OUT_DIR:-/scratch/justin_r13/aigcdet/outputs}"
WORKERS="${WORKERS:-14}"
DEGRADE_SEED="${DEGRADE_SEED:-20261008}"
SEEDS=(20260827 20260828 20260829)
PROXY_BANK="${PROXY_BANK:-data/banks/ntire_meta}"
FULL_BANK="${FULL_BANK:-data/banks/ntire2_meta}"
NAME_A="${NAME_A:-ntire3_d24_512_sq}"
NAME_B="${NAME_B:-ntire3_dual224_sq}"

if [ "$SMOKE" = 1 ]; then
  TAG=paper_smoke; PROXY_FRAC=0.002; FULL_FRAC=0.0004; LIMIT="--limit 200"
  A_EPOCHS=1; B_EPOCHS=1; NAME_A="${NAME_A}_smoke"; NAME_B="${NAME_B}_smoke"
else
  TAG=paper; PROXY_FRAC=0.25; FULL_FRAC=1.0; LIMIT=""
  A_EPOCHS=2; B_EPOCHS=3
fi
RUNS="$OUT_DIR/$TAG"                       # checkpoints (on /scratch)
SC="outputs/scores/$TAG"                   # per-row parquets
DOCS="docs/ntire/$TAG"                     # fusion records, gates, tables
MARK="$RUNS/stages"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$RUNS" "$SC" "$DOCS" "$MARK" logs

# --- provenance --------------------------------------------------------------
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "REFUSING: tracked files are modified; commit first so every number has a commit"
  git status --porcelain --untracked-files=no; exit 1
fi
git ls-files --error-unmatch docs/ntire/paper_prereg.md > /dev/null 2>&1 || {
  log "REFUSING: docs/ntire/paper_prereg.md is not committed; the gates must exist before the numbers"; exit 1; }
COMMIT=$(git rev-parse HEAD)
log "paper pipeline at $COMMIT (SMOKE=$SMOKE) runs=$RUNS scores=$SC docs=$DOCS"

done_stage() { [ -f "$MARK/stage_$1.done" ]; }
mark() { echo "$COMMIT $(date -u +%FT%TZ)" > "$MARK/stage_$1.done"; log "PAPER_STAGE $1 DONE"; }

wait_gpu() {
  local need="$1" seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    if [ -z "$free" ]; then
      log "nvidia-smi gave no reading (driver/library mismatch?); retrying in 5 min"; sleep 300; continue
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

retry() {   # <label> <cmd...> : up to $TRIES attempts (default 6), 10 min apart
  local label="$1" n="${TRIES:-6}" attempt; shift
  for attempt in $(seq 1 "$n"); do
    "$@" && return 0
    log "$label: attempt $attempt of $n failed"; [ "$attempt" -lt "$n" ] && sleep 600
  done
  log "PAPER_PIPELINE_FAILED at $label"; exit 1
}

epoch_of() {  # <ckpt> -> epoch or -1
  [ -f "$1" ] || { echo -1; return; }
  "$PY" -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False,mmap=True)['epoch'])" 2>/dev/null || echo -1
}

score() {   # <ckpt> <prefix> <splits> <chunk> [manifest] [extra...]
  local ck="$1" prefix="$2" splits="$3" chunk="$4" man="${5:-$MANIFEST}"; shift 5 || shift $#
  local sp
  for sp in ${splits//,/ }; do
    [ -f "${prefix}_${sp}.parquet" ] && continue
    wait_gpu 8000
    # shellcheck disable=SC2086
    "$PY" scripts/score_plan_splits.py --ckpt "$ck" --manifest "$man" --splits "$sp" \
      --crop-clamp --degrade-seed "$DEGRADE_SEED" --batch 32 --chunk "$chunk" \
      --workers 12 --out-prefix "$prefix" $LIMIT "$@" 2>&1 \
      | grep -av "it/s\]" | tee -a "logs/${TAG}_score.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "${prefix}_${sp}.parquet" ] || return 1
  done
}

# --- V: the new code path reproduces the probe ------------------------------
stage_V() {
  local br ck flag want got
  for br in A B; do
    if [ "$br" = A ]; then ck=/scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt; flag=""; want=docs/ntire/crop_probe_A2.json
    else ck=/scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt; flag="--swa"; want=docs/ntire/crop_probe_B2swa.json; fi
    wait_gpu 9000
    # shellcheck disable=SC2086
    "$PY" -u scripts/ntire_crop_probe.py --ckpt "$ck" $flag --arms squish \
      --out "$RUNS/verify_$br.json" --batch 32 --chunk 8 --workers 12 2>&1 | grep -av "it/s\]" || return 1
    "$PY" - "$want" "$RUNS/verify_$br.json" <<'PYEOF' || return 1
import json, sys
w = json.load(open(sys.argv[1]))["arms"]["squish"]
g = json.load(open(sys.argv[2]))["arms"]["squish"]
for k in ("robust_auc", "clean_auc"):
    d = abs(w[k] - g[k])
    print(f"verify {k}: probe {w[k]:.6f} new {g[k]:.6f} |diff| {d:.2e}")
    if d > 5e-4:
        sys.exit(f"REFUSING: the trained-squish code path does not reproduce the probe ({k})")
PYEOF
  done
}

# --- P: proxy arms -----------------------------------------------------------
proxy_arm() {   # <policy> <seed-index 1..3>
  local pol="$1" i="$2" seed="${SEEDS[$(( $2 - 1 ))]}" name="abl_${1}_s${2}"
  local ck="$RUNS/$name/checkpoint.pt" canon
  [ -f "$SC/${name}_metrics.json" ] && [ -f "$SC/${name}_ntire_val.parquet" ] && return 0
  if [ "$pol" = crop ]; then canon="--canon-mode crop --crop-side 200 --nominal-side 224"
  else canon="--canon-mode squish --nominal-side 224"; fi
  [ -f "$ck" ] || [ ! -f "$RUNS/$name/checkpoint_slim.pt" ] || ck="$RUNS/$name/checkpoint_slim.pt"
  if [ "$(epoch_of "$ck")" -lt 1 ]; then
    wait_disk 15; wait_gpu 12000
    log "$name: dinov2regl224 depth 24, 1 epoch over $PROXY_FRAC of train, seed $seed, $pol"
    # shellcheck disable=SC2086
    "$PY" -u scripts/train_unfreeze.py --bank "$PROXY_BANK" --root "$EXP" \
      --backbone dinov2regl224 --depth 24 --name "$name" --out-dir "$RUNS" \
      --epochs 1 --n-src 64 --m-deg 2 --src-chunk 8 --workers "$WORKERS" \
      --seed "$seed" --device cuda $canon \
      --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
      --family-weighting proportional --train-subsample-frac "$PROXY_FRAC" \
      --token-pool mean --resume 2>&1 | grep -av "it/s\]" | tee -a "logs/$name.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
    ck="$RUNS/$name/checkpoint.pt"
  fi
  score "$ck" "$SC/$name" ntire_val_hard,ntire_val 16 "$MANIFEST" || return 1
  if [ "$ck" = "$RUNS/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ck" > /dev/null && rm -f "$ck" "$RUNS/$name"/checkpoint_ep*.pt
  fi
}

stage_P() {
  local i pol
  for i in 1 2 3; do for pol in crop squish; do
    retry "proxy $pol s$i" proxy_arm "$pol" "$i"
  done; done
  "$PY" scripts/metrics_table.py "$SC" --pattern 'abl_*_metrics.json' \
    --out "$DOCS/squish_proxy.md" > /dev/null
}

# --- B / A: full training ----------------------------------------------------
full_common() {
  echo --bank "$FULL_BANK" --root "$EXP" --depth 24 --n-src 64 --m-deg 2 \
    --workers "$WORKERS" --seed "${SEEDS[0]}" --device cuda \
    --canon-mode squish \
    --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 \
    --swa --swa-start-frac 0.75 --family-weighting proportional \
    --token-pool mean --train-subsample-frac "$FULL_FRAC" --resume
}

train_B() {
  [ "$(epoch_of "$RUNS/dual/$NAME_B/checkpoint.pt")" -ge "$B_EPOCHS" ] && return 0
  wait_disk 60; wait_gpu 17000
  # shellcheck disable=SC2046
  "$PY" -u scripts/train_dual.py $(full_common) --backbone dinov2regl224 \
    --name "$NAME_B" --out-dir "$RUNS/dual" --perturb-tower2 0.0 \
    --epochs "$B_EPOCHS" --src-chunk 8 --nominal-side 224 2>&1 \
    | grep -av "it/s\]" | tee -a "logs/$NAME_B.log"
  [ "${PIPESTATUS[0]}" -eq 0 ]
}

train_A() {
  [ "$(epoch_of "$RUNS/unfreeze/$NAME_A/checkpoint.pt")" -ge "$A_EPOCHS" ] && return 0
  wait_disk 40; wait_gpu 13000
  # shellcheck disable=SC2046
  "$PY" -u scripts/train_unfreeze.py $(full_common) --backbone dinov2regl \
    --name "$NAME_A" --out-dir "$RUNS/unfreeze" \
    --epochs "$A_EPOCHS" --src-chunk 2 --nominal-side 512 2>&1 \
    | grep -av "it/s\]" | tee -a "logs/$NAME_A.log"
  [ "${PIPESTATUS[0]}" -eq 0 ]
}

start_watchers() {
  [ "$SMOKE" = 1 ] && return 0
  nohup bash scripts/slim_watch.sh "$RUNS/dual/$NAME_B" "$RUNS/unfreeze/$NAME_A" \
    > "logs/${TAG}_slim_watch.log" 2>&1 &
  SLIM_PID=$!
  PIPELINE_LOG=/dev/null OUT="$SC/epochs" nohup bash scripts/epoch_watch.sh \
    "$RUNS/dual/$NAME_B:16" "$RUNS/unfreeze/$NAME_A:4" > "logs/${TAG}_epoch_watch.log" 2>&1 &
  EPOCH_PID=$!
  log "watchers: slim pid $SLIM_PID, epoch pid $EPOCH_PID"
}
stop_watchers() {
  for p in ${SLIM_PID:-} ${EPOCH_PID:-}; do kill "$p" 2>/dev/null || true; done
}

# --- X: external manifest ----------------------------------------------------
stage_X() {
  local waited=0 cap=$((SB_WAIT_H * 3600)) req=""
  if [ "$SMOKE" != 1 ]; then
    while ! grep -q SYNTHBUSTER_DONE "$SB_LOG" 2>/dev/null; do
      if [ "$waited" -ge "$cap" ]; then
        log "Synthbuster not ready after ${SB_WAIT_H} h; building without it (reported as missing)"; break
      fi
      [ $((waited % 7200)) -eq 0 ] && log "waiting for Synthbuster download ($(du -sh "$EXT_ROOT/synthbuster" 2>/dev/null | cut -f1))"
      sleep 600; waited=$((waited + 600))
    done
    grep -q SYNTHBUSTER_DONE "$SB_LOG" 2>/dev/null && req="--require-all"
  fi
  # shellcheck disable=SC2086
  "$PY" scripts/build_external_manifest.py --ext-root "$EXT_ROOT" $req \
    --out "$EXT_MANIFEST" --provenance "$DOCS/external_sets.json" 2>&1 | tee -a "logs/${TAG}_ext.log"
  [ "${PIPESTATUS[0]}" -eq 0 ]
}

ext_splits() {
  "$PY" -c "import pandas as pd;print(','.join(sorted(pd.read_parquet('$EXT_MANIFEST',columns=['split']).split.unique())))"
}
ext_clean_splits() {
  "$PY" -c "import pandas as pd;print(','.join(s for s in sorted(pd.read_parquet('$EXT_MANIFEST',columns=['split']).split.unique()) if not s.endswith(('_deg','_jpeg95'))))"
}

# --- S: scoring, picks, fusions ---------------------------------------------
pick() {   # <final prefix> <swa prefix> -> the better on ntire_val (tie: final)
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
def obj(p):
    s = json.load(open(f"{p}_metrics.json"))["splits"]["ntire_val"]
    return (s["robust_auc"] + s["clean_auc"]) / 2
f, w = sys.argv[1], sys.argv[2]
print(w if obj(w) > obj(f) else f)
PYEOF
}

stage_S() {
  local ckA="$RUNS/unfreeze/$NAME_A/checkpoint.pt" ckB="$RUNS/dual/$NAME_B/checkpoint.pt"
  local EXT EXTC PA PB
  EXT=$(ext_splits) && EXTC=$(ext_clean_splits) || return 1
  # selection candidates: final and SWA of each branch, on the fit split + hard-val
  score "$ckB" "$SC/B3sq"    ntire_val,ntire_val_hard 16 "$MANIFEST"        || return 1
  score "$ckB" "$SC/B3sqswa" ntire_val,ntire_val_hard 16 "$MANIFEST" --swa  || return 1
  score "$ckA" "$SC/A3sq"    ntire_val,ntire_val_hard 8  "$MANIFEST"        || return 1
  score "$ckA" "$SC/A3sqswa" ntire_val,ntire_val_hard 8  "$MANIFEST" --swa  || return 1
  PA=$(pick "$SC/A3sq" "$SC/A3sqswa") && PB=$(pick "$SC/B3sq" "$SC/B3sqswa") || return 1
  "$PY" -c "import json;json.dump({'A':'$PA','B':'$PB','rule':'mean(robust,clean) on ntire_val; tie -> final'},open('$DOCS/picks.json','w'),indent=2)"
  log "picks on ntire_val: A=$PA B=$PB"
  local flagA="" flagB=""
  [[ "$PA" == *swa ]] && flagA="--swa"; [[ "$PB" == *swa ]] && flagB="--swa"
  # the picked squish models on every external split
  # shellcheck disable=SC2086
  score "$ckA" "$PA" "$EXT" 8  "$EXT_MANIFEST" $flagA || return 1
  # shellcheck disable=SC2086
  score "$ckB" "$PB" "$EXT" 16 "$EXT_MANIFEST" $flagB || return 1
  # the crop baselines (round two, as frozen) on the same splits. Their
  # ntire_val / hard-val scores are COPIED into this run's score dir rather
  # than the external splits being written beside them, so a smoke run's
  # --limit parquets can never be mistaken for, and skipped over by, the real
  # run's.
  local b sp_
  for b in A2 B2swa; do
    for sp_ in ntire_val ntire_val_hard; do
      [ -f "$SC/${b}_${sp_}.parquet" ] || cp "outputs/scores/${b}_${sp_}.parquet" "$SC/"
    done
    [ -f "$SC/${b}_metrics.json" ] || cp "outputs/scores/${b}_metrics.json" "$SC/"
  done
  score /scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt \
    "$SC/A2" "$EXT" 8 "$EXT_MANIFEST" || return 1
  score /scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt \
    "$SC/B2swa" "$EXT" 16 "$EXT_MANIFEST" --swa || return 1
  # crop checkpoints presented whole at test time, clean external splits (secondary)
  local sp
  [ "$SMOKE" = 1 ] && EXTC=""        # the probe has no --limit; real run only
  for sp in ${EXTC//,/ }; do
    [ -f "$SC/probe_A2_squish_${sp}.parquet" ] || {
      wait_gpu 9000
      "$PY" scripts/ntire_crop_probe.py --ckpt /scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt \
        --manifest "$EXT_MANIFEST" --split "$sp" --arms squish --out "$DOCS/probe_A2_$sp.json" \
        --parquet-prefix "$SC/probe_A2" --batch 32 --chunk 8 2>&1 | grep -av "it/s\]" ; }
    [ -f "$SC/probe_B2swa_squish_${sp}.parquet" ] || {
      wait_gpu 9000
      "$PY" scripts/ntire_crop_probe.py --ckpt /scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt --swa \
        --manifest "$EXT_MANIFEST" --split "$sp" --arms squish --out "$DOCS/probe_B2swa_$sp.json" \
        --parquet-prefix "$SC/probe_B2swa" --batch 32 --chunk 16 2>&1 | grep -av "it/s\]" ; }
  done
  # both fusions fitted on ntire_val ALONE, read out everywhere else
  "$PY" scripts/fuse_branches.py --a "$SC/A2" --b "$SC/B2swa" \
    --fit-splits ntire_val --eval-splits "ntire_val_hard,$EXT" --write-fused \
    --out "$DOCS/fusion_crop.json" | tee -a "logs/${TAG}_fuse.log" || return 1
  "$PY" scripts/fuse_branches.py --a "$PA" --b "$PB" \
    --fit-splits ntire_val --eval-splits "ntire_val_hard,$EXT" --write-fused \
    --out "$DOCS/fusion_squish.json" | tee -a "logs/${TAG}_fuse.log" || return 1
}

# --- run ---------------------------------------------------------------------
trap stop_watchers EXIT

if [ "$SMOKE" != 1 ]; then
  done_stage V || { retry V stage_V; mark V; }
fi
done_stage P || { stage_P; mark P; }

if ! done_stage G1; then
  "$PY" scripts/paper_results.py g1 --scores "$SC" --out "$DOCS/squish_proxy_gate" \
    $( [ "$SMOKE" = 1 ] && echo --n-boot 200 ) | tail -25
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 3 ] && [ "$SMOKE" != 1 ]; then
    mark G1; log "PAPER_PIPELINE_STOPPED_G1 -- the proxy gate failed; see $DOCS/squish_proxy_gate.md"; exit 0
  elif [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
    log "PAPER_PIPELINE_FAILED at G1 (exit $rc)"; exit 1
  fi
  mark G1
fi
if grep -q '"passed": false' "$DOCS/squish_proxy_gate.json" 2>/dev/null && [ "$SMOKE" != 1 ]; then
  log "PAPER_PIPELINE_STOPPED_G1 (recorded earlier)"; exit 0
fi

start_watchers
# Training gets 20 attempts, not 6: the card is shared with an inference
# server (root's ninfer-serve) that reloads without warning, the trainer
# checkpoints only at epoch ends, and each attempt waits for a clean GPU
# first. Twenty crashes is a real problem worth stopping for; six is not.
done_stage B || { TRIES=20 retry "train B" train_B; mark B; }
done_stage A || { TRIES=20 retry "train A" train_A; mark A; }
stop_watchers

done_stage X || { retry X stage_X; mark X; }
done_stage S || { retry S stage_S; mark S; }

if ! done_stage G; then
  "$PY" scripts/paper_results.py full --scores "$SC" --fusion-dir "$DOCS" \
    --picks "$DOCS/picks.json" --ext-provenance "$DOCS/external_sets.json" \
    --ext-manifest "$EXT_MANIFEST" \
    --out "$DOCS/paper_results" $( [ "$SMOKE" = 1 ] && echo --n-boot 200 ) | tail -40
  [ "${PIPESTATUS[0]}" -eq 0 ] || { log "PAPER_PIPELINE_FAILED at G"; exit 1; }
  mark G
fi

if [ "$SMOKE" != 1 ] && ! done_stage T; then
  retry T bash scripts/ntire_test_read2.sh
  mark T
fi
log "PAPER_PIPELINE_DONE"
