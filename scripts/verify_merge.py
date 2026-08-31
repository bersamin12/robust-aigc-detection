"""Does the merged bank actually contain its shards' bytes?

`finite_check` scans every row for NaN/Inf, so it would catch a torn write that
landed as garbage floats. It cannot catch a torn write that landed as VALID
floats from the wrong offset -- which is the specific risk when two merge
processes write one output path concurrently, as happened to
full_crop_siglipso400m at 09:29 (pid 7469 took SIGBUS while arm2 rewrote the
file underneath it). Row counts and finiteness both survive that; content does
not. So compare content, positionally, across every shard boundary.
"""
import sys, os, json
import numpy as np, pandas as pd

bank = sys.argv[1]
shards = sys.argv[2:]
f = np.load(os.path.join(bank, "feats.npy"), mmap_mode="r")
meta = pd.read_parquet(os.path.join(bank, "meta.parquet"))
print(f"merged: {f.shape} rows={len(meta)}")

rng = np.random.default_rng(0)
cursor = 0
bad_rows = 0
checked = 0
for sd in shards:
    sf = np.load(os.path.join(sd, "feats.npy"), mmap_mode="r")
    sm = pd.read_parquet(os.path.join(sd, "meta.parquet"))
    n = sf.shape[0]
    # every boundary, plus a random interior sample
    idx = sorted(set([0, 1, 2, n - 3, n - 2, n - 1]
                     + rng.integers(0, n, 60).tolist()))
    mism = 0
    for i in idx:
        if not np.array_equal(np.asarray(f[cursor + i]), np.asarray(sf[i])):
            mism += 1
        checked += 1
    # row_id alignment too: a shifted merge keeps content valid but misaligns meta
    rid_ok = (meta["row_id"].iloc[cursor:cursor + n].to_numpy()
              == sm["row_id"].to_numpy()).all() if "row_id" in meta else None
    print(f"  {os.path.basename(sd):40s} rows={n:6d} at [{cursor},{cursor+n})  "
          f"feat mismatches={mism}/{len(idx)}  row_id aligned={rid_ok}")
    bad_rows += mism
    cursor += n

print(f"\ncursor={cursor} vs merged {f.shape[0]}  "
      f"{'OK' if cursor == f.shape[0] else 'ROW COUNT MISMATCH'}")
print(f"checked {checked} rows, {bad_rows} content mismatches -> "
      f"{'VERIFIED' if bad_rows == 0 and cursor == f.shape[0] else 'CORRUPT'}")
