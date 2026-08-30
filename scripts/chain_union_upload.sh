#!/bin/bash
# Publish the union corpus to Kaggle ONE SOURCE AT A TIME, as each finishes.
#
# The union normalises to ~111 GiB and `build_dataset` writes it source by
# source in alphabetical order. Waiting for the whole tree before starting the
# upload serialises two multi-hour jobs that have no reason to be serial: the
# moment `coco_train2017/` stops growing, its 18 GiB can be on its way to
# Kaggle while `ntire/` is still being written.
#
# Splitting across several Datasets costs nothing on the far side. The
# notebook already unifies MULTIPLE image mounts -- that is the mechanism five
# teammates' shards use -- and `kb.unify_mounts` locates each corpus root by
# the top-level names the manifest reports, so N mounts behave as one tree.
#
# Completion signal is the SUCCESSOR, not a file count. Normalisation legally
# skips unreadable images, so "count == expected" may never become true and
# the chain would hang on a source that finished fine. A source is done when
# the next one alphabetically has appeared; the last one is done when the
# build process exits.
#
# Uploads run one at a time on purpose. Each `--dir-mode zip` materialises a
# full-size archive in TMPDIR before sending, so two at once is two archives
# and two streams competing for the same uplink.
set -u
BUILD_PID="${1:?usage: chain_union_upload.sh <build_pid>}"
NORM="${2:-/mnt/berstorage/techjam/experiments/data/normalized_union}"
STAGE_ROOT="${3:-/mnt/berstorage/techjam/experiments/data}"
export TMPDIR="${TMPDIR:-/home/administrator/.cache/kaggle_tmp}"
mkdir -p "$TMPDIR"

# Alphabetical, matching the order build_dataset walks them.
SOURCES=(coco_train2017 ntire open_images sid_set wildfake)

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_source_done() {
  local idx=$1 name=${SOURCES[$1]} next=""
  [ $((idx + 1)) -lt ${#SOURCES[@]} ] && next="${SOURCES[$((idx + 1))]}"
  while true; do
    if [ -n "$next" ] && [ -d "$NORM/$next" ]; then
      # The successor exists, so `name` is finished. Give the writer a moment
      # to flush the last few files rather than racing its final batch.
      sleep 20; log "$name done (successor $next started)"; return 0
    fi
    if ! kill -0 "$BUILD_PID" 2>/dev/null; then
      log "$name done (build process exited)"; return 0
    fi
    sleep 60
  done
}

upload_source() {
  local name=$1
  local stage="$STAGE_ROOT/union_upload_$name"
  local slug="techjam-aigc-union-$(echo "$name" | tr '_' '-')"
  if [ -d "$stage" ]; then log "$name: staged already, skipping stage"; else
    mkdir -p "$stage"
    # Hardlinks: the bytes are already on this filesystem and the tree must
    # keep `<source>/<bucket>/...` intact, because that prefix IS the
    # manifest's rel_path and the notebook resolves <mount>/<rel_path>.
    cp -al "$NORM/$name" "$stage/$name" || { log "$name: cp -al FAILED"; return 1; }
    cat > "$stage/dataset-metadata.json" <<JSON
{
  "title": "TechJam Track5 AIGC Union - $name",
  "id": "justinbersamin/$slug",
  "licenses": [{"name": "other"}]
}
JSON
  fi
  log "$name: uploading as $slug ($(du -sh --apparent-size "$stage" | cut -f1))"
  set -a; . ~/.kaggle/env; set +a
  if kaggle datasets create -p "$stage" --dir-mode zip >> "logs/upload_union_$name.log" 2>&1; then
    log "$name: upload command returned 0"
  else
    log "$name: upload FAILED -- see logs/upload_union_$name.log"
  fi
}

for i in "${!SOURCES[@]}"; do
  name="${SOURCES[$i]}"
  if [ -f "logs/upload_union_$name.done" ]; then log "$name: already published"; continue; fi
  wait_for_source_done "$i"
  upload_source "$name" && touch "logs/upload_union_$name.done"
done
log "all sources published (or attempted); check logs/upload_union_*.log"
