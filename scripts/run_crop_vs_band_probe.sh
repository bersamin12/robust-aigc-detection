#!/bin/bash
# The union's band bank, and dinov2l's first ladder at scale.
#
# THIS SCRIPT RAN TWO ARMS UNTIL 2026-08-30. THE CROP ARM WAS CUT.
# It was cut on evidence, not to save time. `docs/crop_vs_band_ablation.md`
# (commit 10b7893) ran this exact comparison under TIGHTER control than this
# script offers: one corpus, one fingerprinted 10,000-row manifest, both arms
# refusing to start unless `--expect-manifest-sha256` matched, `--canon-mode`
# as the only variable, dihedral disabled so it stayed single-variable. Band
# won the SS6.4 selection metric 0.1903 to 0.1542 at a3. Re-deriving that here
# on 20,000 rows of a different corpus is not a stronger test, it is a second
# weaker one.
#
# WHY THE UNION WOULD NOT HAVE OVERTURNED IT
# That doc's SS8 diagnostic located the entire deficit in FIELD OF VIEW: at
# full frame coverage the policies tie (dTPR -0.011); below 42% coverage crop
# catches a third as many fakes. Coverage is crop_side/short_side, so the
# question "would the union differ" reduces to arithmetic on
# docs/union/data_audit.md. It does differ -- in crop's DISfavour. By the
# per-source medians the union puts ~67% of generated rows at or below 42%
# coverage (ntire 0.21, sid_set 0.20, both large-format) against ~18% in the
# coco_crop eval. The mechanism predicts a LARGER band margin here, so the arm
# would have spent 2-3 h of A4500 to confirm a verdict with more of the same.
#
# The genuinely open question crop leaves is multi-crop tiling (that doc's
# SS10.1), which recovers field of view without touching the upscale-only
# invariant. A second crop_side=200 arm does not advance it, and this script
# cannot run it.
#
# WHAT SURVIVES THE CUT IS NOT A LEFTOVER
# The band arm was never the control; it is the deliverable. Band is
# `CanonPolicy`'s default and the policy every bank on disk was built under, so
# this is the bank the union would actually ship.
#
# WHY dinov3l, AND NOT dinov2l (CHANGED 2026-08-30)
# This probe's question is now "does the union corpus beat the frozen one",
# and that is a comparison, so the backbone has to be the one there is already
# something to compare AGAINST. There is exactly one: docs/robustness_table.md
# is a dinov3l ladder over a0/a1/a2/a3/a7_norecon -- the same five rungs in
# $RUNGS below, the same §6.4 metric, the same seed 20260827 and boot seed.
# Frozen dinov3l scores 0.8611 / 0.9037 / 0.9037 / 0.9012 / 0.0296 on
# `heldout_robust_tpr_at_1pct`, and that is the bar the union has to clear.
#
# It had been dinov2l, which answered NEITHER question. Not the corpus one --
# no dinov2l ladder exists at any scale, so a union dinov2l number has nothing
# to sit beside. And not the fleet one either: the fleet's backbone is
# siglip2l (kaggle_stage_a.ipynb), with dinov2l a listed option nobody has run.
# Running it here would have changed corpus AND backbone at once and produced a
# number that cannot be attributed to either.
#
# dinov2l's own question -- does the ungated Apache-2.0 stand-in retain
# dinov3l's advantage -- is real and still open, but it is a BACKBONE question
# and belongs on the FROZEN corpus, where dinov3l, siglip2l and convnextt
# already have ladders to rank it against.
#
# The switch also makes this run FASTER, which is the opposite of the usual
# trade. dinov2l is registered at 518px (1369 tokens, a measured 54.0 img/s on
# this card) because `canonicalise` delivers 512 and 518 is the only registered
# size that does not throw pixels away; dinov3l runs at 384. Per that entry the
# same bank projects to ~7.4 h for dinov2l against ~2.9 h for dinov3l, so the
# probe drops from roughly 2.8 h to 1.1 h wall.
#
# Gating is not an objection HERE. dinov3l is gated per account and that is why
# a five-member fleet cannot use it -- but this runs on the A4500, where the
# licence is already accepted. It is a fleet constraint, and this is not the
# fleet.
#
# THE CONTENT-BLIND CONTROL STILL RUNS BOTH POLICIES.
# It is CPU-only, so it costs the GPU arm nothing, and it is exactly the
# measurement the ablation could not make: its SS9.6 records `metadata_control`
# as NOT RUN because only the feature banks came back from Kaggle, never the
# images. Here the images are on local disk. So the one piece of the crop arm
# worth keeping is kept -- it quantifies, for free and on the corpus that
# ships, the content confound crop was suspected of buying its proxy advantage
# with (laplacian_var 0.6345 against band's 0.7508).
#
# WHY IT IS A PROBE AND NOT THE CORPUS
# Standardisation is baked into the features at extraction, so unlike dataset
# composition it cannot be revisited from a cached bank. Extracting the full
# union at 11 views is ~15 h of GPU. 20,000 rows returns the ladder in about
# ninety minutes, and the policy it is extracted under is now decided.
#
# WHAT IS HELD FIXED
# One backbone, one probe manifest, one eval manifest, one seed, five rungs.
# `--geometric` is OFF: dihedral needs a square input, which band mode does not
# produce.
#
# WHY THE PROBE'S IMAGES ARE COPIED TO THE SSD FIRST
# The corpus lives on sda, a spinning disk, and the Kaggle upload of the same
# corpus has to read ~128 GB off it. Measured 2026-08-30: three sequential
# readers on one head do not share, they seek -- adding two archivers beside
# `build_dataset` took average read latency from 222 ms to 319 ms and dropped
# normalisation from 73 img/s to 40. Copying the WHOLE corpus to the SSD would
# be pointless, because the copy pays exactly the read it is trying to avoid.
# The probe's own working set is different: 24,000 images, ~8 GB, read many
# times over ninety minutes by the band arm and a CPU control. Staged on the
# NVMe once,
# those reads leave sda entirely, and the upload can run beside the probe
# instead of behind it -- which is worth about two hours of critical path.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/mnt/berstorage/techjam/experiments/data
MANIFEST=$DATA/manifest_union.parquet
EVAL_MANIFEST=$DATA/eval_manifest_union.parquet
# REBASED, and that is not a detail. `benchmark_manifest.parquet` froze its
# `path` column under an older layout ($ROOT/track5/data/demo/...) which no
# longer holds any images -- track5 still exists as a code directory, so the
# failure is a per-file FileNotFoundError deep inside a digest pass, not a
# missing-directory error anyone would notice early. 0 of 400 sampled paths
# resolve there; 400 of 400 resolve in the rebased copy. `build_eval_manifest`
# has no --root, so the rebased FILE is the rebasing mechanism.
BENCH=$DATA/demo/benchmark_manifest_rebased.parquet
PROBE=$DATA/probe/manifest_union_probe.parquet
EVAL_PROBE=$DATA/probe/eval_manifest_union_probe.parquet
# On the NVMe (/dev/nvme0n1p2), which is a different spindle from $DATA.
SSD=/home/administrator/aigc_probe_ssd
SSD_TRAIN=$SSD/train_root
SSD_EVAL=$SSD/eval_root
STAGED_MARKER=logs/probe_ssd_staged.done
BACKBONE=dinov3l
RUNGS="configs/rungs/a0.yaml configs/rungs/a1.yaml configs/rungs/a2.yaml configs/rungs/a3.yaml configs/rungs/a7_norecon.yaml"
BUILD_PID="${1:-}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 0. preflight, BEFORE the wait --------------------------------------
# Everything below step 1 is gated on a build that takes five hours, so an
# input that is wrong at launch stays undiscovered for five hours and then
# fails in seconds. That is exactly what happened on 2026-08-30: the run
# waited from 19:16 to 23:28 and died immediately on a benchmark manifest
# whose paths had been dead the whole time. Check the cheap things first.
preflight() {
  local bad=0
  for f in "$BENCH" configs/rungs/*.yaml; do
    [ -f "$f" ] || { log "PREFLIGHT: missing $f"; bad=1; }
  done
  python - "$BENCH" <<'PY' || bad=1
import os, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
sample = df["path"].head(200)
missing = [p for p in sample if not os.path.exists(p)]
if missing:
    print(f"PREFLIGHT: {len(missing)}/{len(sample)} benchmark paths do not "
          f"resolve, e.g. {missing[0]}")
    sys.exit(1)
print(f"preflight: benchmark manifest OK ({len(df)} rows, 200 sampled resolve)")
PY
  [ "$bad" = 0 ] || { log "PREFLIGHT FAILED -- not waiting on the build"; exit 1; }
}
preflight

# ---- 1. wait for the manifest to freeze ---------------------------------
if [ -n "$BUILD_PID" ]; then
  log "waiting for build pid $BUILD_PID"
  while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 60; done
fi
[ -f "$MANIFEST" ] || { log "FATAL: $MANIFEST does not exist -- the build did not finish"; exit 1; }
log "manifest frozen: $(python -c "import pandas as pd; d=pd.read_parquet('$MANIFEST'); print(len(d),'rows;',d['split'].value_counts().to_dict())")"

# ---- 2. the union's eval manifest ---------------------------------------
# Built from the frozen manifest plus the organisers' benchmark, which is
# untouched across every stream so numbers stay comparable.
if [ ! -f "$EVAL_MANIFEST" ]; then
  log "building the union eval manifest"
  python scripts/build_eval_manifest.py \
      --manifest "$MANIFEST" \
      --benchmark-manifest "$BENCH" \
      --out "$EVAL_MANIFEST"
fi

# ---- 3. cut both probes -------------------------------------------------
# `uniform`, so the probe is a scale model of the corpus and its verdict
# transfers. The eval probe caps each split it scores; benchmark is kept
# small because the SELECTION metric does not read it -- it is there so the
# report is not missing a column, not because 500 rows measure the benchmark.
if [ ! -f "$PROBE" ]; then
  python scripts/cut_probe_manifest.py --manifest "$MANIFEST" --out "$PROBE" \
      --budget train=16000 --budget val_internal=4000 --split train,val_internal
fi
if [ ! -f "$EVAL_PROBE" ]; then
  python scripts/cut_probe_manifest.py --manifest "$EVAL_MANIFEST" --out "$EVAL_PROBE" \
      --budget val_internal=2000 --budget heldout_generator=1500 --budget benchmark=500
fi

# ---- 3a. stage the probe's images onto the SSD --------------------------
# TWO trees, because the two manifests measure rel_path from different roots:
# a TRAIN manifest's rel_path starts at the corpus top level
# (`sid_set/sid_set/x.png`), an EVAL manifest's starts one level up
# (`normalized_union/...`, `demo/...`) because it spans the corpus and the
# organisers' benchmark. Staging both into one tree would put some images at
# two rel_paths; giving each consumer its own root keeps every rel_path
# byte-identical to the manifest that names it, which is the half of bank
# identity that lives outside the parquet.
NEED_GB=12
FREE_GB=$(df -BG --output=avail "$(dirname "$SSD")" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  log "FATAL: only ${FREE_GB}G free on the SSD, need ~${NEED_GB}G"; exit 1
fi
log "staging the probe's images onto the SSD (${FREE_GB}G free)"
python scripts/stage_manifest_images.py --manifest "$PROBE" \
    --root "$DATA/normalized_union" --dest "$SSD_TRAIN" --mode copy
python scripts/stage_manifest_images.py --manifest "$EVAL_PROBE" \
    --root "$DATA" --dest "$SSD_EVAL" --mode copy
# The upload chain waits on THIS, not on the probe finishing. Staging is the
# last thing here that touches sda; once it is written, the probe reads only
# the NVMe and the archivers can have the spinning disk to themselves.
touch "$STAGED_MARKER"
log "staged; sda released to the upload chain"

# ---- 3b. the content-blind control, on CPU, beside the GPU arms ---------
# Crop's cost is a CONTENT confound, and no proxy in gate_confounds.py can see
# it: a 200x200 window is a whole frame for a 200px WildFake image and a
# detail for an 800px NTIRE photograph, so field of view tracks source, and
# two of the union's sources are authentic-only. This runs the 16x16
# thumbnail control over the CANONICALISED view -- the thing the model
# actually receives -- for both policies on identical rows, because the
# number is only meaningful as a difference. Pure CPU, so it costs no wall
# clock against the GPU arms. `--root` so it reads the SAME staged tree the
# arms read; without it the control would silently measure the original tree.
log "content-blind control (CPU, both policies) in the background"
# 8 workers, not 12: the freed CPU goes to the arm's dataloader below, and
# 12 + 12 oversubscribed 24 cores. This is the slower of the two and it is off
# the critical path -- it only has to finish before the ladder is read.
python -u scripts/content_blind_probe.py --manifest "$PROBE" --root "$SSD_TRAIN" \
    --out docs/content_blind_probe_union.json --workers 8 \
    > logs/content_blind_probe.log 2>&1 &
PID_CONTROL=$!

# ---- 4. the band arm ----------------------------------------------------
# `extract_features` has no --root, so it takes the staged tree from
# $AIGCDET_DATA_ROOT; `extract_eval_bank` takes its own --root because its
# manifest is measured from the other level. Both are set per-command rather
# than exported, so nothing else in this script can pick up the wrong tree.
arm () {
  local NAME=$1 MODE=$2; shift 2
  local EXTRA=("$@")
  local BANK=data/banks/probe_${NAME}_${BACKBONE}
  local EBANK=data/banks/eval_probe_${NAME}_${BACKBONE}
  log "[$NAME] stage A ($MODE)"
  AIGCDET_DATA_ROOT="$SSD_TRAIN" \
  python -u scripts/extract_features.py --manifest "$PROBE" --backbone "$BACKBONE" \
      --out "$BANK" --split train,val_internal --device cuda \
      --batch-size 16 --workers 12 --resume \
      --canon-mode "$MODE" "${EXTRA[@]}"
  # dinov3l OVERFLOWS IN FLOAT16 AND THE FAILURE IS SILENT. On 2026-08-29 a
  # full bank came back 100% NaN and was not noticed, because the only
  # post-condition anything checked was the ROW COUNT -- which a NaN bank
  # satisfies perfectly. `BACKBONES["dinov3l"]` now pins dtype=bfloat16 so it
  # should not recur, but "should not recur" is what was believed last time.
  # Check before spending the eval bank and the ladder on top of it.
  python - "$BANK" <<'PY' || { log "[$NAME] BANK IS NOT FINITE -- stopping"; exit 1; }
import sys, numpy as np
f = np.load(f"{sys.argv[1]}/feats.npy", mmap_mode="r")
n = min(len(f), 4096)
sample = np.asarray(f[np.linspace(0, len(f) - 1, n).astype(int)], dtype=np.float32)
bad = ~np.isfinite(sample)
if bad.any():
    print(f"  {bad.mean() * 100:.1f}% of {n} sampled vectors are NaN/inf")
    sys.exit(1)
print(f"  finite: {n} sampled vectors, |x| max {np.abs(sample).max():.2f}")
PY
  log "[$NAME] eval bank ($MODE)"
  python -u scripts/extract_eval_bank.py --manifest "$EVAL_PROBE" --backbone "$BACKBONE" \
      --out "$EBANK" --tier ablation --device cuda --root "$SSD_EVAL" \
      --batch-size 16 --checkpoint-every 200 --resume --no-subsample \
      --canon-mode "$MODE" "${EXTRA[@]}"
  log "[$NAME] ladder"
  python -u scripts/run_ablation.py --bank "$BANK" --eval-bank "$EBANK" \
      --rungs $RUNGS --tier ablation --device cuda \
      --out "docs/robustness_table_probe_${NAME}.md" \
      --selection "docs/selection_probe_${NAME}.json" \
      --out-dir "outputs/rungs_probe_${NAME}"
  log "[$NAME] DONE"
}

# batch 16 and 12 workers, both raised from 8 when the crop arm was cut. The
# old numbers were sized for TWO towers sharing 20 GB and 24 cores; one tower
# has the card to itself. The workers matter more than the batch: two arms used
# to hide one's decode behind the other's forward, and with one arm that
# interleaving is gone, so the single arm is MORE dataloader-bound than either
# arm was. Raising them is what stops the cut from making the survivor slower.
log "launching the band arm on $BACKBONE"
arm band band > logs/probe_band.log 2>&1 &
PID_BAND=$!
log "band pid $PID_BAND"

FAIL=0
wait $PID_BAND || { log "BAND ARM FAILED (see logs/probe_band.log)"; FAIL=1; }

wait $PID_CONTROL || log "content-blind control FAILED (see logs/content_blind_probe.log)"

# ---- 5. the report ------------------------------------------------------
# Not a verdict any more -- the policy was decided in docs/crop_vs_band_ablation.md
# and this run does not re-open it. Two things to read.
log "=============== UNION / dinov2l / band ==============="
python - <<'PY'
import json, os

p = "docs/selection_probe_band.json"
if not os.path.exists(p):
    print("band: no selection.json -- the arm did not finish")
else:
    print("band:", json.dumps(json.load(open(p)))[:600])
print()
print("Read `heldout_robust_tpr_at_1pct`, which is the SS6.4 selection metric and")
print("was fixed before any result existed. Two comparisons are legitimate:")
print("  * across RUNGS here, which is what the ladder is for; and")
print("  * against the FROZEN dinov3l ladder in docs/robustness_table.md, which")
print("    is the same five rungs, the same metric and the same seed. That is")
print("    the comparison this run exists for. The bar, per rung:")
print("      a0 0.8611 | a1 0.9037 | a2 0.9037 | a3 0.9012 | a7_norecon 0.0296")
print("Two caveats on that comparison, both real:")
print("  * SCALE. 20,000 rows here against the frozen bank's 138,116, and a")
print("    4,000-image eval against 25,332. Corpus is not quite the only")
print("    variable; run_ablation.py has no row-subset flag, so a scale-matched")
print("    frozen baseline is not a one-liner.")
print("  * POPULATION. The metric is val_internal reals vs heldout_generator")
print("    fakes, and those rows differ between corpora by construction. The")
print("    held-out PAIR is pinned (SDwithAdaptor_controlnet, VQGAN), so the")
print("    generator families match even though the rows do not.")
print("NOT against docs/crop_vs_band_ablation.md's 0.1903. Different corpus,")
print("different backbone, 10,000 rows; only its POLICY verdict transfers.")

# The crop arm's one surviving piece. `metadata_control` was never run in the
# ablation (its SS9.6: only the banks came back from Kaggle, not the images), so
# this is the first measurement of what crop standardisation does to CONTENT on
# a corpus we hold. It qualifies a policy already chosen rather than choosing
# one: a large positive difference means crop's cleaner confound proxies
# (laplacian_var 0.6345 vs band's 0.7508) were bought with a field-of-view
# shortcut, which is the number to publish beside that claim -- and evidence
# for tiling (SS10.1) over a wider crop_side.
p = "docs/content_blind_probe_union.json"
if os.path.exists(p):
    c = json.load(open(p))
    print()
    print("--- content-blind control (16x16 AFTER standardisation, CPU) ---")
    for m in ("crop", "band"):
        if m in c:
            pooled = c[m]["pooled"]
            auc = pooled.get("auc", pooled.get("auc_unverified_branch_provenance"))
            print(f"  {m:5s} pooled AUC {auc:.4f}   {pooled['verdict']}")
    if "crop" in c and "band" in c:
        d = c["crop"]["pooled"]["auc"] - c["band"]["pooled"]["auc"]
        print(f"  crop - band = {d:+.4f}"
              + ("   <-- crop's proxy advantage is bought with content"
                 if d > 0.03 else "   <-- no content shortcut at this size"))
else:
    print("\n(content-blind control produced no output -- see logs/content_blind_probe.log)")
PY
exit $FAIL
