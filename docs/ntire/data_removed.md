# Data removed from the r13 box, 2026-09-05

Deleted to make room for the remaining NTIRE train shards and the NTIRE
evaluation sets (user decision, 2026-09-05):

| tree | size | why it existed |
|---|---|---|
| `experiments/data/normalized_union/sid_set` | 25 GB | SID_Set authentic + pooled generated |
| `experiments/data/normalized_union/wildfake` | 16 GB | WildFake's 18 generator families + `real/` |

Consequences:

- `manifest_union.parquet`, `eval_manifest_union.parquet`, `manifest_plan.parquet`
  and the `probe/*.parquet` files still carry rows for both sources. Their
  `path`s no longer resolve on this box. The NTIRE-stream manifest
  (`scripts/build_ntire_manifest.py`) drops both **by name**
  (`DROPPED_SOURCES`), so a stray survivor cannot re-enter by accident.
- No feature bank on this box depended on either tree (`data/banks` did not
  exist here). Banks on other boxes that index the union manifest are
  unaffected; they never resolve paths on this machine.
- The `heldout_generator` split of the union manifest was WildFake-only
  (VQGAN, SDwithAdaptor_controlnet), so it is gone with it; the NTIRE stream
  has no held-out-generator population by design (NTIRE's own sets are the
  test).

Both trees are re-acquirable with `scripts/acquire_data.py --dataset sid_set`
/ `--dataset wildfake` and `scripts/build_dataset.py`, at the cost of a
rebuild of `normalized_union` numbering (see `build_dataset.py:406-407`).
