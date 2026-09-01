#!/usr/bin/env python3
"""Can k-means on cached embeddings stand in for the generator labels NTIRE
does not ship?

NTIRE contributes 53% of this corpus's training fakes under ONE generator
string, so `PairedSampler` -- which draws a family uniformly, then an image
within it -- gives all 42 of its generators a single family's share of the
gradient. Whether that matters depends on how those 42 are internally
distributed, and there is no metadata to ask.

Clustering is the obvious substitute and it is only worth anything if it
actually tracks generator identity, so this measures that FIRST, on the
families where the answer is known. Sixteen of this corpus's families are a
single generator each; k-means with k=16 over their clean-view embeddings
either recovers them or it does not, and the adjusted Rand index says which.

  ARI near 1   -> clusters are generators; the NTIRE histogram below means
                  something.
  ARI near 0   -> the embedding does not separate generators at this
                  granularity, the histogram is noise, and the honest report
                  is that this question cannot be answered from features.

The control is the whole point. Without it a tidy 42-cluster histogram would
look like evidence whatever the truth was.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from aigcdet.features.bank import FeatureBank

SEED = 20260827
#: Families that are ONE generator. `ntire` (42) and `sid_set` (a collection)
#: are excluded by construction -- they are the thing being asked about.
COLLECTIONS = {"ntire", "sid_set", ""}


def _clean(bank, idx):
    """View 0 is the clean view by the bank's own invariant."""
    x = np.asarray(bank.feats[idx, 0, :]).astype(np.float32)
    # Standardised before k-means: the embedding's dimensions differ in scale
    # by more than an order of magnitude, and Lloyd's algorithm optimises plain
    # euclidean distance, so without this a handful of wide dimensions choose
    # the clusters.
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/banks/probe_crop_dinov2regl_local")
    ap.add_argument("--k-ntire", type=int, default=42,
                    help="generators NTIRE reports (arXiv 2604.11487)")
    ap.add_argument("--out", default="docs/ntire_cluster_probe.json")
    a = ap.parse_args()

    bank = FeatureBank.open(a.bank)
    gen = bank.meta["generator"].to_numpy().astype(str)
    lab = bank.meta["label"].to_numpy()
    src = bank.meta["source"].to_numpy().astype(str)

    # ---- control: do clusters recover KNOWN generators? --------------------
    known = np.array([i for i in range(len(gen))
                      if lab[i] == 1 and gen[i] not in COLLECTIONS])
    fams = sorted(set(gen[known]))
    print(f"control: {len(known)} fakes from {len(fams)} single-generator "
          f"families\n  {fams}")
    xk = _clean(bank, known)
    km = KMeans(n_clusters=len(fams), random_state=SEED, n_init=10).fit(xk)
    truth = np.array([fams.index(g) for g in gen[known]])
    ari = adjusted_rand_score(truth, km.labels_)

    # Purity: the share of images whose cluster's majority family is their own.
    purity = float(np.mean([
        (truth[km.labels_ == c] == np.bincount(truth[km.labels_ == c]).argmax()).mean()
        * (km.labels_ == c).sum() for c in range(len(fams))]) * len(fams) / len(truth))
    print(f"\n  adjusted Rand index = {ari:.3f}   purity = {purity:.3f}")

    verdict = ("clusters track generator identity" if ari > 0.5 else
               "clusters are WEAKLY related to generator identity" if ari > 0.2
               else "clusters do NOT track generator identity")
    print(f"  -> {verdict}")

    # ---- the question, reported either way, with its own caveat ------------
    nt = np.array([i for i in range(len(gen)) if lab[i] == 1 and src[i] == "ntire"])
    print(f"\nntire: {len(nt)} fakes, clustering into k={a.k_ntire}")
    xn = _clean(bank, nt)
    kn = KMeans(n_clusters=a.k_ntire, random_state=SEED, n_init=10).fit(xn)
    sizes = np.bincount(kn.labels_, minlength=a.k_ntire)
    sizes_sorted = np.sort(sizes)[::-1]
    ideal = len(nt) / a.k_ntire
    # Gini over cluster sizes: 0 = perfectly even, 1 = one cluster holds all.
    s = np.sort(sizes).astype(np.float64)
    gini = float((2 * np.arange(1, len(s) + 1) - len(s) - 1).dot(s)
                 / (len(s) * s.sum()))
    print(f"  even split would be {ideal:.0f} per cluster")
    print(f"  actual: min {sizes.min()}  median {int(np.median(sizes))}  "
          f"max {sizes.max()}   gini {gini:.3f}")
    print(f"  largest 5: {sizes_sorted[:5].tolist()}")
    print(f"  smallest 5: {sizes_sorted[-5:].tolist()}")

    if ari <= 0.2:
        print("\n  READ NOTHING INTO THE ABOVE. The control says this embedding "
              "does not separate generators it KNOWS are different, so the "
              "spread of these clusters is not evidence about how NTIRE's 42 "
              "are distributed.")

    rec = {"control_ari": ari, "control_purity": purity,
           "control_families": fams, "verdict": verdict,
           "ntire_n": int(len(nt)), "k": a.k_ntire,
           "cluster_sizes": sizes.tolist(), "gini": gini,
           "even_split": ideal, "bank": a.bank}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
