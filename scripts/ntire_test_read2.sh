#!/bin/bash
# The SECOND read of `ntire_test` -- declared before it happened, and the last.
#
# History, which the write-up reports in full:
#   read 1, 2026-10-04 12:00 UTC  crop-policy fusion (A2 + B2swa), via
#           scripts/ntire_test_read.sh -> docs/ntire/fusion_ntire2_final.json
#   read 2  squish-policy fusion, via THIS script, declared 2026-10-08 in
#           docs/ntire/paper_prereg.md before any squish model was trained
#
# Read 1 was the protocol's one read. Read 2 exists because the squish policy
# was found after read 1 was spent, from a probe on the fit splits. It is
# permitted under three conditions, all enforced here:
#   * it runs once: refuses if its output already exists;
#   * it runs last: refuses unless the pipeline's G stage (every gate and
#     table on the non-test splits) has completed, so nothing it shows can
#     feed back into a choice;
#   * it reads only what selection froze: both prefixes, checkpoints and
#     weights tags come from docs/ntire/paper/fusion_squish.json, whose
#     fusion was fitted on ntire_val alone.
# There is no read 3.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

DOCS="${DOCS:-docs/ntire/paper}"
FUSION="$DOCS/fusion_squish.json"
OUT="$DOCS/fusion_squish_final.json"
STAGE_G="${STAGE_G:-/scratch/justin_r13/aigcdet/outputs/paper/stages/stage_G.done}"

[ -f "$OUT" ] && { log "REFUSING: $OUT exists -- read 2 has happened. There is no read 3."; exit 1; }
[ -f "$STAGE_G" ] || { log "REFUSING: $STAGE_G missing -- read 2 runs only after every non-test result is written"; exit 1; }
[ -f "$FUSION" ] || { log "REFUSING: no frozen squish fusion at $FUSION"; exit 1; }
grep -q '"test_split": null' "$FUSION" || { log "REFUSING: $FUSION already names a test split"; exit 1; }

FUSION="$FUSION" OUT="$OUT" FIT_SPLITS=ntire_val bash scripts/ntire_test_read.sh

.venv/bin/python - "$OUT" <<'PYEOF' >> docs/ntire/results.md
import json, sys, datetime
r = json.load(open(sys.argv[1]))
t = r["splits"]["ntire_test"]
f = lambda v: "n/a" if v is None else f"{v:.4f}"
print(f"\n## Read 2 (declared 2026-10-08, run {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC)\n")
print("The squish-policy fusion, frozen on ntire_val before this read; see "
      "`scripts/ntire_test_read2.sh` for why a second read exists and the "
      "conditions it ran under. Read 1 (crop policy) is above, unchanged.\n")
print("| model | robust | clean | AUC | TPR@1%FPR |\n|---|---|---|---|---|")
for k in ("a", "b", "fused_equal", "fused"):
    m = t[k]
    print(f"| {k} | {f(m['robust_auc'])} | {f(m['clean_auc'])} | {f(m['auc'])} | {f(m['tpr_at_1pct_fpr'])} |")
PYEOF
log "TEST_READ2_DONE -> $OUT, appended to docs/ntire/results.md"
