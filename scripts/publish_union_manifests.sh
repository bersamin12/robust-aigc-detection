#!/bin/bash
# Publish the union's two manifests as one small Kaggle Dataset.
#
# WHY THIS IS NOT PART OF THE IMAGE UPLOAD
# The corpus goes up as five per-source Datasets (`chain_union_upload.sh`),
# and none of them carries a manifest: the chain stages `<source>/` and
# nothing else, because that prefix IS the manifest's rel_path and anything
# extra in the tree becomes an extra file the verify gate has to explain. So
# the manifests need a Dataset of their own -- a few MB against ~128 GB, and
# it can go up the moment the build freezes rather than hours later.
#
# WHY THE SLUG DOES NOT START WITH `techjam-aigc-union`
# The notebook finds its image mounts with a PREFIX glob,
# `/kaggle/input/techjam-aigc-union*`. A Dataset called
# `techjam-aigc-union-manifests` would match it, arrive as a sixth image
# mount, and `kb.unify_mounts` would symlink two parquet files into the
# corpus root -- which then fails, if it fails at all, as an unexplained
# extra-file count an hour into a session. `techjam-aigc-manifests-union`
# cannot match that glob. The notebook's `manifest_glob` /
# `eval_manifest_glob` for the union streams name this slug exactly.
#
# BOTH PARQUETS IN ONE DATASET, because they are attached together every
# time: the training manifest drives Stage A and the eval manifest drives the
# eval bank, and a session that has one and not the other cannot finish.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/mnt/berstorage/techjam/experiments/data
MANIFEST=$DATA/manifest_union.parquet
EVAL_MANIFEST=$DATA/eval_manifest_union.parquet
STAGE=$DATA/union_upload_manifests
SLUG=techjam-aigc-manifests-union
BUILD_PID="${1:-}"
export TMPDIR="${TMPDIR:-/home/administrator/.cache/kaggle_tmp}"
mkdir -p "$TMPDIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -n "$BUILD_PID" ]; then
  log "waiting for build pid $BUILD_PID"
  while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 60; done
fi

# The eval manifest is built by the probe runner, not by build_dataset, so it
# can lag the training manifest by a minute or two. Wait for it rather than
# publishing half of what a session needs.
for _ in $(seq 60); do
  [ -f "$MANIFEST" ] && [ -f "$EVAL_MANIFEST" ] && break
  sleep 30
done
for f in "$MANIFEST" "$EVAL_MANIFEST"; do
  [ -f "$f" ] || { log "FATAL: $f does not exist"; exit 1; }
done

rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$MANIFEST" "$EVAL_MANIFEST" "$STAGE/"
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "TechJam Track5 AIGC Union - manifests",
  "id": "justinbersamin/$SLUG",
  "licenses": [{"name": "other"}]
}
JSON

# `--dir-mode skip`: there are no directories here, only two parquet files,
# and they must arrive as FILES. Archiving them would make the notebook's
# glob -- which names `manifest_union.parquet` exactly -- match nothing.
log "publishing $SLUG ($(du -sh --apparent-size "$STAGE" | cut -f1))"
set -a; . ~/.kaggle/env; set +a
if kaggle datasets create -p "$STAGE" --dir-mode skip \
      >> logs/upload_union_manifests.log 2>&1; then
  log "$SLUG published"
  python - <<'PY'
import pandas as pd
# The REAL fingerprint function, imported rather than reimplemented. It is
# sha256 over `identity_paths` in row order with a trailing newline per entry
# -- a hand-rolled `"\n".join(df["rel_path"])` differs from it in both
# respects and would print a digest that matches nothing on disk, which is
# worse than printing none.
from aigcdet.features.bank import manifest_fingerprint
for name in ("manifest_union", "eval_manifest_union"):
    p = f"/mnt/berstorage/techjam/experiments/data/{name}.parquet"
    d = pd.read_parquet(p)
    print(f"{name}: {len(d):,} rows  manifest_sha256={manifest_fingerprint(d)}")
PY
else
  log "$SLUG FAILED -- see logs/upload_union_manifests.log"
  exit 1
fi
