"""Single-feature confound audit from cached proxies, at zero GPU cost."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from aigcdet.features.proxies import PROXY_NAMES

bank = "data/banks/siglip2l"
meta = pd.read_parquet(f"{bank}/meta.parquet")
prox = np.load(f"{bank}/proxies.npy", mmap_mode="r")[:, 0, :]   # view 0 = undegraded
man = pd.read_parquet("data/manifest.parquet")
man = man[man["split"].isin(["train", "val_internal"])].reset_index(drop=True)
assert (man["rel_path"].values == meta["rel_path"].values).all(), "bank/manifest misaligned"
y = meta["label"].to_numpy()

def auc(x, y):
    a = roc_auc_score(y, x)
    return max(a, 1 - a)

print(f"pool: {len(y)} rows  ({int((y==0).sum())} authentic / {int((y==1).sum())} generated)")
print(f"{'signal':<22}{'AUC':>8}")
for i, n in enumerate(PROXY_NAMES):
    print(f"{n:<22}{auc(np.asarray(prox[:, i], float), y):>8.4f}")
short = np.minimum(man["width"].to_numpy(), man["height"].to_numpy())
print(f"{'short side (manifest)':<22}{auc(short.astype(float), y):>8.4f}")

print("\nshort-side buckets (top 12 by count):")
df = pd.DataFrame({"short": short, "label": y})
g = df.groupby("short")["label"].agg(n="size", gen="sum")
g["real"] = g["n"] - g["gen"]
g["gen_share"] = g["gen"] / g["n"]
g = g.sort_values("n", ascending=False).head(12)
print(g.to_string())
single = g[(g["gen_share"] == 1.0) | (g["gen_share"] == 0.0)]["n"].sum()
tot_single = int(df.groupby("short")["label"].agg(lambda s: len(s) if s.nunique() == 1 else 0).sum())
print(f"\nrows in a single-class short-side bucket: {tot_single} ({tot_single/len(y):.1%})")
print(f"authentic images by source: {meta.groupby('source')['label'].apply(lambda s:(s==0).sum()).to_dict()}")
