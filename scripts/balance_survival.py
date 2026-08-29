import numpy as np, pandas as pd
from aigcdet.features.proxies import PROXY_NAMES
from aigcdet.data.splits import MIN_HELDOUT_IMAGES

bank = "data/banks/siglip2l"
meta = pd.read_parquet(f"{bank}/meta.parquet")
prox = np.asarray(np.load(f"{bank}/proxies.npy", mmap_mode="r")[:, 0, :], float)
man = pd.read_parquet("data/manifest.parquet")
man = man[man["split"].isin(["train","val_internal"])].reset_index(drop=True)
short = np.minimum(man["width"].to_numpy(), man["height"].to_numpy())
lap = prox[:, PROXY_NAMES.index("laplacian_var")]
y = meta["label"].to_numpy()
rng = np.random.default_rng(20260827)

def balance(strata):
    keep = []
    for k in pd.unique(strata):
        idx = np.flatnonzero(strata == k)
        a, b = idx[y[idx] == 0], idx[y[idx] == 1]
        n = min(len(a), len(b))
        if n == 0: continue
        keep += [rng.choice(a, n, replace=False), rng.choice(b, n, replace=False)]
    return np.concatenate(keep) if keep else np.array([], int)

qbins = pd.qcut(lap, 10, labels=False, duplicates="drop")
variants = {
    "short side (exact)": short.astype(str),
    "laplacian_var (10 quantile bins)": qbins.astype(str),
    "short side x laplacian_var": np.char.add(short.astype(str), qbins.astype(str)),
}
print(f"pool {len(y)} ({int((y==0).sum())} authentic / {int((y==1).sum())} generated)\n")
print(f"{'stratified on':<34}{'kept':>8}{'of pool':>9}{'authentic':>11}")
results = {}
for name, s in variants.items():
    k = balance(s); results[name] = k
    print(f"{name:<34}{len(k):>8}{len(k)/len(y):>8.1%}{int((y[k]==0).sum()):>11}")

k = results["laplacian_var (10 quantile bins)"]
fam = meta.iloc[k]
gen = fam[fam["label"] == 1]["generator"].value_counts()
allgen = meta[meta["label"] == 1]["generator"].value_counts()
print(f"\nafter laplacian_var balancing: {len(gen)} of {len(allgen)} generated families survive")
below = gen[gen < MIN_HELDOUT_IMAGES]
print(f"families below MIN_HELDOUT_IMAGES ({MIN_HELDOUT_IMAGES}): {below.to_dict() if len(below) else 'none'}")
ret = (gen / allgen.reindex(gen.index)).sort_values()
print(f"per-family retention: {ret.min():.1%} to {ret.max():.1%}")
print(f"authentic sources kept: {fam[fam['label']==0]['source'].value_counts().to_dict()}")
