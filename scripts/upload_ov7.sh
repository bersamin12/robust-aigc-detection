#!/bin/bash
# Publish AI-OV7 to Kaggle as ONE Dataset with two trees.
#
# Two trees, because they answer different questions and only one of them is
# reproducible from the other:
#
#   open_images_v7/   the RAW JPEG pairs, exactly as generated. This is the
#                     dataset. Each fake carries its real's 64 quantisation
#                     integers and its subsampling, so the encoder history is
#                     matched *as JPEG* -- the property the corpus exists for,
#                     and the one you cannot recover once it is re-encoded.
#   normalized_ov7/   what `build_dataset` produced: PNG, per-bucket running
#                     index, the tree `manifest_ov7.parquet`'s `rel_path`
#                     resolves against. Derived, but a Kaggle notebook cannot
#                     rebuild it inside a session, so it ships too.
#
# The manifest ships as well. Without it the normalized tree is anonymous --
# the filenames are running indices and carry no label, family or split.
#
# `--dir-mode tar`, matching `chain_union_upload.sh`: the payload is JPEG and
# PNG, both already entropy-coded, so zip would spend CPU re-compressing
# incompressible bytes for nothing.
#
# Hardlinks (`cp -al`) for the staging copy: same filesystem, so the tree costs
# no bytes, and -- more to the point -- it cannot re-encode an image the way a
# copy through any imaging library would.
set -euo pipefail

RAW="${RAW:-data/raw_ov7_src}"
NORM="${NORM:-data/normalized_ov7}"
MANIFEST="${MANIFEST:-data/manifest_ov7.parquet}"
STAGE="${STAGE:-data/ov7_upload}"
SLUG="${SLUG:-justinbersamin/techjam-aigc-ov7}"
export TMPDIR="${TMPDIR:-/home/administrator/.cache/kaggle_tmp}"
mkdir -p "$TMPDIR" logs

log() { echo "[$(date +%H:%M:%S)] $*"; }

for p in "$RAW/open_images_v7" "$NORM" "$MANIFEST" "$RAW/attribution.csv" \
         "$RAW/LICENCES.json" docs/ov7_dataset_README.md; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

if [ -d "$STAGE" ]; then
  log "staging dir exists, reusing: $STAGE"
else
  log "staging $STAGE"
  mkdir -p "$STAGE"
  cp -al "$RAW/open_images_v7" "$STAGE/open_images_v7"
  cp -al "$NORM" "$STAGE/normalized_ov7"
  cp "$MANIFEST" "$STAGE/manifest_ov7.parquet"
  cp "$RAW/attribution.csv" "$STAGE/attribution.csv"
  cp "$RAW/LICENCES.json" "$STAGE/LICENCES.json"
  cp docs/ov7_dataset_README.md "$STAGE/README.md"
  # One row per pair, every generation parameter that produced it: seed,
  # steps, guidance, strength, the prompt and where the prompt came from, the
  # crop box, and the real's measured JPEG quality and subsampling. Shipping
  # it is what makes the corpus auditable rather than merely downloadable.
  python - "$RAW" "$STAGE" <<'PY'
import glob, json, sys
import pandas as pd
raw, stage = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for f in sorted(glob.glob(f"{raw}/_rows/rows_*.jsonl"))
        for l in open(f) if l.strip()]
df = pd.DataFrame(rows).sort_values(["family", "image_id"]).reset_index(drop=True)
df.to_parquet(f"{stage}/pairs.parquet", index=False)
df.to_csv(f"{stage}/pairs.csv", index=False)
print(f"pairs.parquet: {len(df)} rows, {df.family.nunique()} families")
PY
  cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "AI-OV7: encoder-matched real/AI image pairs",
  "id": "$SLUG",
  "licenses": [{"name": "CC-BY-4.0"}],
  "keywords": ["computer science", "image", "deep learning", "art"]
}
JSON
fi

log "uploading $SLUG ($(du -sh --apparent-size "$STAGE" | cut -f1))"
set -a; . ~/.kaggle/env; set +a
if kaggle datasets status "$SLUG" >/dev/null 2>&1; then
  # `version`, not `create`: create 409s on a slug that already exists and the
  # error text does not say so plainly.
  cmd=(kaggle datasets version -p "$STAGE" --dir-mode tar -m "${MSG:-rebuild}")
else
  cmd=(kaggle datasets create -p "$STAGE" --dir-mode tar)
fi
if ionice -c3 nice -n 19 "${cmd[@]}" >> logs/upload_ov7.log 2>&1; then
  log "upload returned 0 -- see logs/upload_ov7.log"
  tail -3 logs/upload_ov7.log
else
  log "upload FAILED -- see logs/upload_ov7.log"; tail -20 logs/upload_ov7.log; exit 1
fi
