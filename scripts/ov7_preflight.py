import hashlib, os, sys
import pandas as pd
man = sys.argv[1]; root = sys.argv[2]
print("manifest sha256:", hashlib.sha256(open(man,"rb").read()).hexdigest())
df = pd.read_parquet(man)
print("rows:", len(df), "cols:", [c for c in df.columns][:12])
print(df.groupby(["split","label"]).size().to_string())
col = "rel_path" if "rel_path" in df.columns else "path"
missing = [p for p in df[col] if not os.path.exists(os.path.join(root, p))]
print(f"MISSING {len(missing)} of {len(df)} under {root}")
for p in missing[:5]: print("   ", p)
