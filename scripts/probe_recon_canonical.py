"""Does canonicalisation destroy the reconstruction branch's signal?

A4's premise (spec 3.3) is that latent-diffusion outputs round-trip through
their own VAE with anomalously low error, and that the evidence "lives in the
high-frequency residual". `augment.canonical` band-limits every image to a
200px ceiling before any feature is computed, and `recon.attach_recon_to_bank`
canonicalises too, because it must replay the exact cached pixels.

So the branch is fed images whose detail above 200px Nyquist has been
destroyed -- and a diffusion image that has been downscaled and re-upscaled is
no longer a VAE output at all, which is the property being detected.

This probe computes `recon_features` twice on the same images, once on the
canonicalised clean view (what A4 would actually see) and once on the native
decode (what the branch was designed for), and reports how well each separates
the classes. Costs about a minute of GPU; A4 costs eight hours.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from PIL import Image

from aigcdet.augment.canonical import canonicalise
from aigcdet.data.manifest import read_manifest
from aigcdet.features.recon import RECON_FEATURE_NAMES, load_recon_models, recon_features


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="val_internal")
    ap.add_argument("--n", type=int, default=1200, help="images per class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root", default=None,
                    help="dataset root the manifest is rebased onto; "
                         "defaults to $AIGCDET_DATA_ROOT")
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args(argv)

    # read_manifest, never pd.read_parquet: the frozen manifest stores absolute
    # paths under the root it was built on, which no longer exists. This rebases
    # them from rel_path onto --root / $AIGCDET_DATA_ROOT.
    man = read_manifest(a.manifest, root=a.root)
    man = man[man["split"] == a.split]
    rng = np.random.default_rng(a.seed)
    take = np.concatenate([rng.choice(np.flatnonzero(man["label"].to_numpy() == c),
                                      min(a.n, int((man["label"] == c).sum())),
                                      replace=False) for c in (0, 1)])
    sub = man.iloc[np.sort(take)].reset_index(drop=True)
    print(f"{len(sub)} images from {a.split} "
          f"({int((sub['label']==0).sum())} authentic / {int((sub['label']==1).sum())} generated)")

    vae, lp = load_recon_models(a.device)
    canon, native, y = [], [], sub["label"].to_numpy()
    for p in sub["path"]:
        with Image.open(p) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        native.append(recon_features(base, vae, lp, a.device))
        canon.append(recon_features(canonicalise(base), vae, lp, a.device))
    canon, native = np.stack(canon), np.stack(native)

    def auc(x):
        v = roc_auc_score(y, x)
        return max(v, 1 - v)

    def probe_on(X, yy, seed):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        s = cross_val_predict(clf, Xs, yy, cv=5, method="decision_function")
        v = roc_auc_score(yy, s)
        return max(v, 1 - v)

    def probe(X):
        clf = LogisticRegression(max_iter=2000, random_state=a.seed)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        s = cross_val_predict(clf, Xs, y, cv=5, method="decision_function")
        v = roc_auc_score(y, s)
        return max(v, 1 - v)

    print(f"\n{'feature':<16}{'canonicalised':>15}{'native':>10}{'delta':>9}")
    for i, n in enumerate(RECON_FEATURE_NAMES):
        c, v = auc(canon[:, i]), auc(native[:, i])
        print(f"{n:<16}{c:>15.4f}{v:>10.4f}{v-c:>+9.4f}")
    pc, pn = probe(canon), probe(native)
    print(f"{'ALL 12 (probe)':<16}{pc:>15.4f}{pn:>10.4f}{pn-pc:>+9.4f}")

    # Within a resolution stratum short side is constant, so it cannot be what
    # the branch is reading. A canonicalised score that survives here is real
    # reconstruction evidence; one that collapses was resolution all along.
    short = np.minimum(sub["width"].to_numpy(), sub["height"].to_numpy())
    print(f"\n{'short side':<16}{'n':>6}{'canonicalised':>15}{'native':>10}")
    for v in np.unique(short):
        m = short == v
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            continue
        yy = y[m]
        pcs = probe_on(canon[m], yy, a.seed)
        pns = probe_on(native[m], yy, a.seed)
        print(f"{int(v):<16}{int(m.sum()):>6}{pcs:>15.4f}{pns:>10.4f}")
    np.savez("/tmp/claude-1000/recon_probe.npz", canon=canon, native=native,
             y=y, short=short)
    print("\nThe canonicalised column is what rung A4 would actually be trained on.\n"
          "If it sits near 0.5 while native does not, the 8-hour recon extraction\n"
          "would be measuring a signal canonicalisation has already removed.")


if __name__ == "__main__":
    main()
