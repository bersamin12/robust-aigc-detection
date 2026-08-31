#!/usr/bin/env bash
# d24 -- the whole tower -- on four cards, and a measurement of what that bought.
#
# WHY d24 AT ALL. Every rung measured so far (d0,1,2,4,6,8) unfreezes only the
# top third of a 24-block tower, so the data cannot say whether the frozen
# lower blocks are what keeps the features general. d24 is the endpoint: if
# OV7 is still rising there, depth was never the limit and the 300k run should
# be deep. If OV7 turns over while WildFake keeps climbing, that divergence IS
# the cliff, and d8 is near the boundary.
#
# WHY DISTRIBUTED. `train_unfreeze.py` was single-GPU, so d24 was a ~104 min
# job on one card while three sat idle. The split is per accumulation chunk:
# every rank draws the IDENTICAL batch from the identical seed (so the sampler
# stream is untouched and the rung stays comparable to d0..d8), decodes only
# the sources it will forward, and one flat all-reduce before `opt.step()`
# sums the partial gradients. Verified before this ran: gradient norm
# 1.2840072838 on one card against 1.2840072849 on four, and only the 20
# tensors a d1 rung is allowed to move actually moved.
#
# --src-chunk 4, not 8: per-rank activation memory does not fall with world
# size, since ranks process a chunk at a time exactly as the single-GPU path
# does. Extrapolating the depth-4 bench (4.77 GiB at chunk 8) puts d24 at
# ~22 GiB of a 24.5 GiB card at chunk 8, which is not a margin. The step is
# mathematically identical either way -- chunk size changes only peak memory.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/venv/main/bin/python}
ROOT=${ROOT:-/workspace/data/probe}
TRAIN_BANK=${TRAIN_BANK:-data/banks/probe_crop_dinov2regl_local}
OUT=${OUT:-outputs/unfreeze}
DEPTH=${DEPTH:-24}
NM=${NM:-d24band}
CHUNK=${CHUNK:-2}
EPOCHS=${EPOCHS:-5}
NPROC=${NPROC:-4}
WAIT_FOR=${WAIT_FOR:-"=== D24 COMPLETE"}
WAIT_LOG=${WAIT_LOG:-logs/d24_ddp.log}

WF_MAN=$ROOT/eval_manifest_union_probe.parquet
OV7_MAN=/data/ov7_stage/eval_manifest_ov7_transfer.parquet
OV7_ROOT=/data/ov7_stage/normalized_ov7/open_images_v7

export AIGCDET_DATA_ROOT="$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# 1.35 GiB sat reserved-but-unallocated in the run that OOMed; on a job this
# close to the card that is the difference between fitting and not.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs "$OUT"
L=logs/d24_band.log
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }

if [ -n "$WAIT_FOR" ]; then
  say "waiting for '$WAIT_FOR' in $WAIT_LOG"
  while ! grep -q "$WAIT_FOR" "$WAIT_LOG" 2>/dev/null; do sleep 20; done
  while pgrep -f "scripts/(extract_features|extract_eval_bank|train_rung|train_head)\.py" >/dev/null; do sleep 15; done
fi

T0=$(date +%s)

# --- stage 0: pick the largest chunk that fits, by measuring ----------------
# Activations are the whole budget here and they do NOT shrink with world
# size: every rank forwards a chunk at a time exactly as one card would. So
# the chunk is chosen by measurement, not extrapolation. d24 at chunk 4 came
# back at 22.02 GiB of a 23.52 GiB card, leaving nothing for AdamW's two
# moments (2.42 GiB at 302M trainable) and the gradient buffer, and the run
# died 28 s in. The ceiling below is the bench peak plus that headroom.
CEIL=${CEIL:-16.0}
say "=== stage 0: single-GPU baseline at d$DEPTH, largest chunk under $CEIL GiB"
BASE=nan
for c in $CHUNK 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/bench_finetune.py \
    --depths "$DEPTH" --src-chunk "$c" --steps 6 --device cuda \
    --out "docs/bench_d${DEPTH}_c${c}.json" > "logs/bench_d${DEPTH}_c${c}.log" 2>&1 \
    || { say "  chunk $c: bench failed or OOMed"; continue; }
  read -r SPS PK <<<"$("$PY" -c "
import json
r = json.load(open('docs/bench_d${DEPTH}_c${c}.json'))['rows'][0]
print(r['sec_per_step'], r['peak_gib'])")"
  say "  chunk $c: $SPS s/step, peak $PK GiB"
  if "$PY" -c "import sys; sys.exit(0 if float('$PK') < float('$CEIL') else 1)"; then
    CHUNK=$c; BASE=$SPS
    say "  -> src-chunk $CHUNK, single-GPU baseline $BASE s/step"
    break
  fi
  say "  -> over the ceiling; trying a smaller chunk"
done
if [ "$BASE" = nan ]; then say "no chunk size fits d$DEPTH on this card"; exit 1; fi

# --- stage 1: the distributed run -------------------------------------------
say "=== stage 1: d$DEPTH on $NPROC GPUs, $EPOCHS epochs, src-chunk $CHUNK"
T1=$(date +%s)
"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29581 \
  scripts/train_unfreeze.py \
  --bank "$TRAIN_BANK" --root "$ROOT" --depth "$DEPTH" --name "$NM" \
  --out-dir "$OUT" --epochs "$EPOCHS" --src-chunk "$CHUNK" --canon-mode band \
  > "logs/unfreeze_${NM}.log" 2>&1 || { say "TRAIN FAILED"; tail -30 "logs/unfreeze_${NM}.log" | tee -a "$L"; exit 1; }
T2=$(date +%s)
STEPS=$("$PY" -c "
import numpy as np, pandas as pd
m=pd.read_parquet('$TRAIN_BANK/meta.parquet'); t=m[m.split=='train']
print(min((t.label==1).sum(),(t.label==0).sum())//32*$EPOCHS)")
say "stage 1 done in $(( (T2-T1)/60 )) min over $STEPS steps"
"$PY" - <<PYEOF | tee -a "$L"
base = "$BASE"
wall, steps = $T2 - $T1, $STEPS
per = wall / steps
print(f"  {$NPROC}-GPU: {per:.4f} s/step  ({wall/60:.1f} min for {steps} steps)")
if base != "nan":
    print(f"  1-GPU:   {float(base):.4f} s/step")
    print(f"  SPEEDUP: {float(base)/per:.2f}x  "
          f"(perfect would be {$NPROC}x; the gap is the all-reduce)")
PYEOF

# --- stage 2: this tower's own eval banks ------------------------------------
bank_for () {
  local kind=$1 man root split bank slots=24
  if [ "$kind" = wf ]; then
    man=$WF_MAN; root=$ROOT; split=""; bank=data/banks/eval_unfreeze_${NM}
  else
    man=$OV7_MAN; root=$OV7_ROOT; split="heldout_generator,val_internal"
    bank=data/banks/eval_ov7_${NM}
  fi
  [ -f "$bank/config.json" ] && { say "  $kind already built"; return 0; }
  say "  $kind -> $bank ($slots slots)"
  rm -rf "$bank" "$bank".shard*
  local FF; FF=$(mktemp); local s
  for s in $(seq 0 $((slots-1))); do
    ( CUDA_VISIBLE_DEVICES=$((s % 4)) "$PY" -u scripts/extract_eval_bank.py \
        --manifest "$man" --root "$root" --backbone dinov2regl \
        --out "${bank}.shard${s}" --tier ablation --no-subsample \
        ${split:+--split "$split"} --canon-mode band --crop-side 200 \
        --batch-size 64 --device cuda --tower-checkpoint "$OUT/${NM}/checkpoint.pt" \
        --shard "${s}/${slots}" --resume \
        > "logs/bank_${kind}_${NM}_${s}.log" 2>&1 || echo "$kind shard $s" >> "$FF" ) &
  done
  wait
  [ -s "$FF" ] && { say "FAILED $kind:"; cat "$FF" | tee -a "$L"; return 1; }
  "$PY" -u scripts/merge_banks.py --out "$bank" \
    $(for i in $(seq 0 $((slots-1))); do echo "${bank}.shard${i}"; done) >>"$L" 2>&1
}
bank_for wf  || exit 1
bank_for ov7 || exit 1
say "stage 2 done"

# --- stage 3: this tower alone, on the bank its own policy produced ---------
# NOT the crop ladder. d0..d8 were trained and evaluated under crop, and their
# banks record canon_policy mode=crop; putting a band rung in that table would
# compare two towers whose pixels were never the same pixels. The crop number
# to read this against is d24's 0.7580 on OV7 / 0.9841 on WildFake, and the
# comparison is between two runs, not between two rows of one table.
for kind in ov7 wf; do
  bank=data/banks/eval_ov7_${NM};      outj=docs/selection_ov7_${NM}.json
  [ "$kind" = wf ] && { bank=data/banks/eval_unfreeze_${NM}; outj=docs/selection_wf_${NM}.json; }
  say "scoring $kind: $NM on $bank"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/score_unfreeze_ladder.py \
    --rung "${NM}=$OUT/${NM}/checkpoint.pt:${bank}" \
    --out "$outj" --device cuda 2>&1 | tee -a "$L"
done

say "=== D24 BAND COMPLETE in $(( ($(date +%s)-T0)/60 )) min"
