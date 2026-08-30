#!/bin/bash
# Pull the full union corpus onto a pod and assemble the two roots it needs.
#
# RUN THIS IN THE BACKGROUND WHILE THE PROBE IS RUNNING. The pull is
# network-bound and the probe is GPU- and CPU-bound, so ~16 minutes of download
# at 1155 Mbps costs nothing if it overlaps the ~80 minutes of probe arms. By
# the time the probe says which backbone and which policy, the corpus is
# already here and the real extraction starts with no wait.
#
# LARGEST FIRST, and that is not cosmetic. `kaggle datasets download --unzip`
# writes the zip, extracts it, then deletes the zip -- so the disk peak is
# (everything so far) + 2x (the one in flight). Pulling ntire's 61 GiB when the
# disk is empty caps the peak at ~136 GiB; pulling it last would spike to
# ~188 GiB and overrun a 200 GB allocation. On 500 GB there is room either way,
# but the ordering is free.
#
# SEVEN DATASETS, NOT FIVE. Five carry images, one carries the two manifests
# (deliberately NOT named `techjam-aigc-union-*`, so it cannot be swept up as a
# sixth image mount), and one carries the organisers' benchmark, which is the
# `demo/` half of every eval rel_path.
#
#   ROOT=/data bash scripts/pull_union.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-/data/union}"
TRAIN="$ROOT/train"
BENCH="$ROOT/bench"
EVAL_ROOT="$ROOT/eval_root"

# Largest first. ntire is ~61 GiB; open_images ~9.
IMAGE_SLUGS=(
  justinbersamin/techjam-aigc-union-ntire
  justinbersamin/techjam-aigc-union-sid-set
  justinbersamin/techjam-aigc-union-coco-train2017
  justinbersamin/techjam-aigc-union-wildfake
  justinbersamin/techjam-aigc-union-open-images
)
MANIFEST_SLUG=justinbersamin/techjam-aigc-manifests-union
BENCH_SLUG=justinbersamin/techjam-aigc-benchmark

# Frozen 2026-08-30. A bank whose config reports a different fingerprint was
# built against a different manifest and is not comparable with anyone else's.
EXPECT_TRAIN_SHA=3cca88d94fbb573bb229f3ffe9a9370e2c5def42c78758c05275f421be23c406
EXPECT_EVAL_SHA=b950e04aea1e09fcdd72d5918536b8796062f90de64f041df1b7c933c0d53cd7

RETRIES="${RETRIES:-30}"          # 30 x 120s = up to an hour of waiting
RETRY_WAIT="${RETRY_WAIT:-120}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

[ -f "$HOME/.kaggle/kaggle.json" ] || die "no ~/.kaggle/kaggle.json"
chmod 600 "$HOME/.kaggle/kaggle.json"
command -v kaggle >/dev/null || python -m pip install -q kaggle
mkdir -p "$TRAIN" "$BENCH" "$EVAL_ROOT"

avail_gib() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
log "free at $ROOT: $(avail_gib "$ROOT") GiB (need ~140 peak, ~127 steady)"

pull() {   # slug, dest
  local slug=$1 dest=$2 name=${1##*/}
  # Idempotent: a re-run after a dropped connection must not re-pull 61 GiB.
  if [ -f "$dest/.pulled_$name" ]; then
    log "$name: already pulled"; return 0
  fi
  log "$name: pulling into $dest  (free $(avail_gib "$dest") GiB)"
  # Retry rather than die. Kaggle unpacks a large upload ASYNCHRONOUSLY -- a
  # 62 GB Dataset lists at 0 bytes and refuses to download for a while after
  # its upload returns 0. This script is meant to be fired and forgotten
  # alongside the probe, so it waits the processing out instead of making you
  # come back and restart it.
  local try
  for try in $(seq 1 "$RETRIES"); do
    if kaggle datasets download -d "$slug" -p "$dest" --unzip; then break; fi
    [ "$try" -eq "$RETRIES" ] && return 1
    log "  $name: attempt $try failed (still processing on Kaggle?) -- "\
        "retrying in ${RETRY_WAIT}s"
    sleep "$RETRY_WAIT"
  done
  # Kaggle may hand back the per-directory tars rather than an extracted tree,
  # depending on how it processed the upload. Handle both -- the difference
  # only shows up as "no rows resolve", hours later.
  shopt -s nullglob
  for t in "$dest"/*.tar; do log "  extracting $(basename "$t")"; tar -xf "$t" -C "$dest" && rm -f "$t"; done
  shopt -u nullglob
  touch "$dest/.pulled_$name"
}

for slug in "${IMAGE_SLUGS[@]}"; do
  pull "$slug" "$TRAIN" || die "pull failed: $slug
     If this is techjam-aigc-union-ntire, check it has finished PROCESSING on
     Kaggle -- a 62 GB upload is unpacked asynchronously and lists at 0 bytes
     until it is done."
done
pull "$MANIFEST_SLUG" "$ROOT" || die "manifest pull failed"
pull "$BENCH_SLUG" "$BENCH"   || die "benchmark pull failed"

# The eval manifest is rooted ONE LEVEL ABOVE the training root: its rel_paths
# start with `normalized_union/` or `demo/`. Two symlinks, not a copy.
ln -sfn "$TRAIN" "$EVAL_ROOT/normalized_union"
ln -sfn "$BENCH" "$EVAL_ROOT/demo"

log "train root:  $TRAIN"; ls "$TRAIN" | grep -v '^\.pulled' | sed 's/^/    /'
log "eval root:   $EVAL_ROOT"; ls "$EVAL_ROOT" | sed 's/^/    /'
log "size: $(du -sh --apparent-size "$TRAIN" 2>/dev/null | cut -f1)"

# ---- prove it before any GPU is spent
log "verifying"
python - "$ROOT" "$TRAIN" "$EVAL_ROOT" "$EXPECT_TRAIN_SHA" "$EXPECT_EVAL_SHA" <<'PY' || exit 1
import os, sys
import pandas as pd
sys.path.insert(0, "src")
from aigcdet.features.bank import manifest_fingerprint

root, train, eval_root, want_tr, want_ev = sys.argv[1:6]
for name, base, want in (("manifest_union.parquet", train, want_tr),
                         ("eval_manifest_union.parquet", eval_root, want_ev)):
    path = os.path.join(root, name)
    assert os.path.exists(path), f"missing {path}"
    df = pd.read_parquet(path)
    got = manifest_fingerprint(df)
    assert got == want, (
        f"{name} fingerprint is {got}, expected {want}. This is a DIFFERENT "
        f"manifest -- nothing extracted against it is comparable.")
    # A tree can list correctly and resolve to nothing: a directory level
    # shifted by an extract, a Dataset that arrived under another name.
    miss = [x for x in df["rel_path"].sample(500, random_state=20260827)
            if not os.path.exists(os.path.join(base, x))]
    assert not miss, f"{name}: {len(miss)}/500 do not resolve, e.g. {miss[:3]}"
    print(f"    {name}: {len(df)} rows, fingerprint {got[:16]}..., 500 resolve")
PY

log "READY."
log "  stage A:   AIGCDET_DATA_ROOT=$TRAIN  --manifest $ROOT/manifest_union.parquet"
log "  eval bank: --root $EVAL_ROOT         --manifest $ROOT/eval_manifest_union.parquet"
