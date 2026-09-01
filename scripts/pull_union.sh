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

# "<slug> <source>", largest first (ntire ~61 GiB, open_images ~9).
#
# THE SOURCE NAME IS NOT DECORATION AND NOT DERIVABLE FROM THE SLUG.
# `chain_union_upload.sh` stages `<source>/<bucket>/...` and uploads it with
# --dir-mode tar, which archives each TOP-LEVEL directory -- so each Dataset
# went up as one `<source>.tar`. Kaggle then unpacks it and DROPS THE OUTER
# LEVEL: the dataset root is the contents of `<source>/`, i.e. the buckets.
# Verified 2026-08-31 against the published Datasets:
#
#   techjam-aigc-union-sid-set  ->  real/0244438.png      (no `sid_set/`)
#   techjam-aigc-union-ntire    ->  ntire/0040000.png     (the BUCKET `ntire`)
#
# The manifest's rel_path is `<source>/<bucket>/<file>`, so extracting every
# Dataset into one flat directory yields `train/real/...` where
# `train/sid_set/real/...` is wanted, and NOTHING resolves. Each therefore
# extracts into its own `$TRAIN/<source>/`, which puts the stripped level back.
#
# The slug is also not a safe source: `techjam-aigc-union-coco-train2017`
# would give `coco-train2017`, and the manifest says `coco_train2017`.
IMAGE_SETS=(
  "justinbersamin/techjam-aigc-union-ntire ntire"
  "justinbersamin/techjam-aigc-union-sid-set sid_set"
  "justinbersamin/techjam-aigc-union-coco-train2017 coco_train2017"
  "justinbersamin/techjam-aigc-union-wildfake wildfake"
  "justinbersamin/techjam-aigc-union-open-images open_images"
)
MANIFEST_SLUG=justinbersamin/techjam-aigc-manifests-union
BENCH_SLUG=justinbersamin/techjam-aigc-benchmark

# Frozen 2026-08-30. A bank whose config reports a different fingerprint was
# built against a different manifest and is not comparable with anyone else's.
EXPECT_TRAIN_SHA=3cca88d94fbb573bb229f3ffe9a9370e2c5def42c78758c05275f421be23c406
EXPECT_EVAL_SHA=b950e04aea1e09fcdd72d5918536b8796062f90de64f041df1b7c933c0d53cd7

# Debian/Ubuntu images since 3.11 often ship `python3` and no `python` at all,
# and a bare `python` there is "command not found" three lines into a script on
# a box billed by the hour. Resolve it once.
PY_BIN="${PY_BIN:-$(command -v python || command -v python3)}"
[ -n "$PY_BIN" ] || { echo 'FATAL: no python or python3 on PATH' >&2; exit 1; }

# `pip install kaggle` drops its console script next to $PY_BIN, not onto PATH.
# With PY_BIN overridden to a venv interpreter (a Vast image ships no torch for
# the system python, so it must be), the bare `kaggle` further down is "command
# not found" AFTER the deps are installed and the checkpoints are cached -- the
# latest, most expensive point in the script at which to lose the corpus.
export PATH="$(dirname "$PY_BIN"):$PATH"

RETRIES="${RETRIES:-30}"          # 30 x 120s = up to an hour of waiting
RETRY_WAIT="${RETRY_WAIT:-120}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

kaggle_creds_present() {
  # THREE FORMS, because all three are in use. KAGGLE_API_TOKEN comes first
  # because the CLI prefers it over everything else: a `KGAT_`-prefixed Kaggle
  # ACCESS TOKEN is not a legacy API key, and putting one in kaggle.json's
  # `key` field authenticates as nobody -- public endpoints still answer (they
  # need no auth at all), so it looks like it worked right up until a PRIVATE
  # dataset returns 403. `kaggle.json` is what "Create New
  # Token" downloads; KAGGLE_USERNAME/KAGGLE_KEY is what this project's own
  # ~/.kaggle/env uses and what a rented pod is easiest to configure with.
  # Requiring only the file turned the env-var form into a confusing refusal.
  [ -n "${KAGGLE_API_TOKEN:-}" ] && return 0
  [ -f "$HOME/.kaggle/kaggle.json" ] && { chmod 600 "$HOME/.kaggle/kaggle.json"; return 0; }
  [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ] && return 0
  return 1
}
KAGGLE_HELP='no Kaggle credentials. Either export KAGGLE_API_TOKEN (a KGAT_ access
     token), write ~/.kaggle/kaggle.json
     ({"username":"...","key":"..."}, chmod 600) or export KAGGLE_USERNAME and
     KAGGLE_KEY. The probe and union Datasets are PRIVATE -- they carry NTIRE
     rows, which may not be published. Use a token created for THIS pod and
     revoke it when you destroy the pod: a rented host has root on the machine.'

kaggle_creds_present || die "$KAGGLE_HELP"
command -v kaggle >/dev/null || "$PY_BIN" -m pip install -q kaggle
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

# TWO PASSES, and the reason is availability rather than size.
#
# The first version pulled largest-first so the biggest archive landed while the
# disk was emptiest -- `--unzip` writes the zip, extracts, then deletes it, so
# the peak is everything-so-far plus 2x the one in flight. That mattered on a
# 200 GB allocation. It does not on 500 GB, and it had a cost: the largest
# source is also the one most likely to still be processing on Kaggle, so
# putting it first blocked 65 GiB of ready data behind an hour of retries on
# one that was not.
#
# So: try every source once, DEFER whatever is not ready, then retry only those.
# Kaggle finalises a large dataset's downloadable version well after its files
# are individually readable -- a whole-dataset download 404s while
# `datasets files` lists fine and single files fetch. Observed on
# techjam-aigc-union-ntire, 61 GiB, ~50 min after its upload returned.
deferred=()
for entry in "${IMAGE_SETS[@]}"; do
  slug=${entry%% *}; source=${entry##* }
  mkdir -p "$TRAIN/$source"
  if RETRIES=1 pull "$slug" "$TRAIN/$source"; then
    :
  else
    log "  $source: not ready yet -- deferred to the end"
    deferred+=("$entry")
  fi
done

for entry in "${deferred[@]}"; do
  slug=${entry%% *}; source=${entry##* }
  log "$source: retrying (up to $((RETRIES * RETRY_WAIT / 60)) min)"
  pull "$slug" "$TRAIN/$source" || die "pull failed: $slug
     Kaggle had not finalised this Dataset's downloadable version. Check
     https://www.kaggle.com/datasets/${slug#*/} -- if `kaggle datasets files`
     lists it but a whole-dataset download 404s, it is still processing and
     re-running this script later will pick it up (already-pulled sources are
     skipped)."
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
"$PY_BIN" - "$ROOT" "$TRAIN" "$EVAL_ROOT" "$EXPECT_TRAIN_SHA" "$EXPECT_EVAL_SHA" <<'PY' || exit 1
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
# pandas/pyarrow can abort in a static destructor at interpreter shutdown --
# 'terminate called without an active exception' -- AFTER the work is done and
# printed. That non-zero exit then killed a bootstrap whose checks had all
# passed. os._exit skips the teardown entirely.
import os as _os; _os._exit(0)
PY

log "READY."
log "  stage A:   AIGCDET_DATA_ROOT=$TRAIN  --manifest $ROOT/manifest_union.parquet"
log "  eval bank: --root $EVAL_ROOT         --manifest $ROOT/eval_manifest_union.parquet"
