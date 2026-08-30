"""Gate 1: measure the low-level confound of a corpus BEFORE extracting it.

`scripts/audit_confounds.py` reads a feature bank's cached proxies. That is
the right instrument once a bank exists and the wrong one here, because the
question this answers is whether to spend the GPU night that would produce it.

So this decodes images directly and runs them through the pipeline's real
front half -- `augment.canonical.canonicalise` under the policy being
proposed, then `augment.geometric.dihedral` if it is being proposed too, then
`features.proxies.proxy_vector` -- and reports how well each single proxy
predicts the label. Sampled, CPU-only, minutes rather than hours.

WHY IT IS A GATE AND NOT A REPORT
---------------------------------
`docs/low_level_confounds.md` measures the frozen corpus at var-Laplacian AUC
0.6721: a one-dimensional sharpness statistic separates the classes that well
after canonicalisation, without looking at content at all. Any corpus that
scores WORSE has handed the head an easier shortcut than the one we already
have, and a headline AUC from it means less, not more.

`--max-auc` makes that a refusal. Run it with the frozen figure and a small
tolerance, and a corpus that regresses exits non-zero before anything is
extracted.

CALIBRATION (2026-08-30)
------------------------
Run against the frozen manifest in band mode at `--n 6000`, this reproduces
`docs/low_level_confounds.md`'s table from the images alone -- no bank, no GPU:

                    published (bank)   this script
    jpeg_quality         0.5532          0.5583
    laplacian_var        0.6721          0.6815
    noise_floor          0.6374          0.6474
    short side           0.5992          0.5939

and the per-source split too (wildfake laplacian_var 0.7022 against the bank's
0.6944; sid_set noise_floor 0.7500 against 0.7314). Differences are sampling
noise at n=6000. That agreement is what makes a number from this script usable
as a threshold against those published figures.

WHAT IT CANNOT SEE
------------------
Content. A 200px crop of a 429px photograph is a detail while a 200px crop of
a 200px image is the whole frame, so crop standardisation trades a spectral
confound for a semantic one, and three pixel statistics are blind to that.
`eval/controls.py:content_blind_auc` is the instrument for it, and it needs a
bank. Passing this gate is necessary, not sufficient.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.canonical import CANON_CROP_SIDE, MODES, MODE_BAND, CanonPolicy
from aigcdet.augment.geometric import dihedral, geometric_rng, sample_dihedral
from aigcdet.data.manifest import read_manifest
from aigcdet.features.proxies import PROXY_NAMES, proxy_vector

#: Total rows sampled. Enough that an AUC's standard error is well under the
#: differences being read (~0.01 at n=3000 per class), small enough that the
#: whole pass is minutes.
SAMPLE_N = 6000

#: A source thinner than this in the sample gets reported but not scored: an
#: AUC over a few dozen rows moves by more than the effects being read.
MIN_GROUP = 200


def _one(task):
    """(index, path, policy, geometric, seed, row_id) -> proxy vector.

    Module-level and taking a plain tuple so it can go to a process pool; the
    work is decode-bound and embarrassingly parallel.
    """
    i, path, policy, geometric, seed, row_id = task
    try:
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # View 0's transform, exactly as `extract._prepare_image` builds it:
        # standardise, then orient. No recipe -- view 0 is the clean view, and
        # the degraded views' proxies measure the degradation, not the corpus.
        out = canonicalise_for(img, policy, seed, row_id)
        if geometric:
            out = dihedral(out, sample_dihedral(geometric_rng(seed, row_id, 0)))
        return i, proxy_vector(out)
    except Exception:
        # One unreadable file must not end a 20,000-image pass; the row is
        # dropped and counted.
        return i, None


def canonicalise_for(img, policy: CanonPolicy, seed: int, row_id: int):
    """Standardise as view 0 would be.

    Under crop mode the window is drawn from the same per-view key the
    extraction uses, so this measures the pixels that would actually be
    cached rather than a centre crop that only resembles them.
    """
    from aigcdet.augment.canonical import MODE_CROP, canonical_rng, canonicalise

    rng = canonical_rng(seed, row_id, 0) if policy.mode == MODE_CROP else None
    return canonicalise(img, policy=policy, rng=rng)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Orientation-corrected AUC: `max(a, 1 - a)`.

    A confound that predicts the label BACKWARDS is exactly as usable to a
    head as one that predicts it forwards, so the direction is not the
    quantity of interest -- the separability is.
    """
    x, y = np.asarray(x, float), np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    s = pd.Series(x).rank(method="average").to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    a = (s[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A PROPORTIONAL random sample of `n` rows, seeded.

    Proportional and not stratified, and the difference is the whole point of
    the gate. `--max-auc` is read against a figure measured on the corpus as
    it actually is -- 0.6721 on a pool that is 85% WildFake. A sample with
    equal rows per source measures a corpus nobody is building: run against
    the frozen manifest, stratified sampling put pooled `laplacian_var` at
    0.6118 and `noise_floor` at 0.6683, against the corpus's true 0.6721 and
    0.6374. Both per-source figures were right; the pooled ones were answers
    to a different question, and the threshold would have been compared to
    them.

    Per-source rows come out of this same sample, so a source that is thin in
    the corpus is thin here too -- which is honest, and `MIN_GROUP` reports it
    rather than scoring it. Raise `--n` if a small source needs reading.
    """
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    return df.iloc[np.sort(rng.choice(len(df), n, replace=False))]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="train,val_internal")
    ap.add_argument("--canon-mode", choices=MODES, default=MODE_BAND)
    ap.add_argument("--crop-side", type=int, default=CANON_CROP_SIDE)
    ap.add_argument("--geometric", action="store_true")
    ap.add_argument("--n", type=int, default=SAMPLE_N,
                    help="total rows to sample, proportionally. NOT per "
                         "source: the threshold is read against a figure "
                         "measured on the corpus as it is, so the sample has "
                         "to have the corpus's own source mix.")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-auc", type=float, default=None,
                    help="refuse (exit 1) if the worst single-proxy AUC "
                         "exceeds this. The frozen corpus scores 0.6721; pass "
                         "that plus a small tolerance to stop a corpus that "
                         "regresses before any GPU time is spent on it.")
    a = ap.parse_args(argv)

    policy = CanonPolicy(mode=a.canon_mode, crop_side=a.crop_side)
    df = read_manifest(a.manifest)
    splits = [s for s in a.split.split(",") if s]
    df = df[df["split"].isin(splits)]
    if df.empty:
        raise SystemExit(f"no rows in splits {splits}")
    full_mix = df["source"].value_counts(normalize=True).to_dict()
    df = sample_rows(df, a.n, a.seed)

    print(f"policy {policy.as_record()}  geometric={a.geometric}")
    print(f"sampling {len(df)} rows, proportionally")
    print("  corpus source mix: "
          + "  ".join(f"{k} {v:.1%}" for k, v in sorted(full_mix.items())) + "\n")

    tasks = [(i, r["path"], policy, a.geometric, a.seed, int(idx))
             for i, (idx, r) in enumerate(df.iterrows())]
    P = np.full((len(tasks), len(PROXY_NAMES)), np.nan, dtype=np.float64)
    # `workers <= 1` runs inline, the same convention as
    # `features.extract._iter_prepared`. It is not only for small runs: a
    # process pool needs `_one` to be importable by name in the child, which
    # it is not when this module has been loaded from a path rather than
    # imported -- so the inline path is what makes the script callable as a
    # library, and what the tests exercise.
    results = (map(_one, tasks) if a.workers <= 1 else
               ProcessPoolExecutor(max_workers=a.workers).map(
                   _one, tasks, chunksize=16))
    for i, vec in tqdm(results, total=len(tasks), desc="proxies"):
        if vec is not None:
            P[i] = vec

    ok = np.isfinite(P).all(axis=1)
    if not ok.all():
        print(f"\nskipped {int((~ok).sum())} unreadable rows")
    df = df.iloc[ok].copy()
    P = P[ok]
    y = df["label"].to_numpy()
    short = np.minimum(df["width"].to_numpy(), df["height"].to_numpy())

    def row(name, mask):
        sub_y = y[mask]
        vals = [auc(P[mask, i], sub_y) for i in range(len(PROXY_NAMES))]
        vals.append(auc(short[mask].astype(float), sub_y))
        return {"group": name, "n": int(mask.sum()),
                "real": int((sub_y == 0).sum()), "fake": int((sub_y == 1).sum()),
                **dict(zip([*PROXY_NAMES, "short_side"], vals))}

    rows = [row("POOLED", np.ones(len(y), bool))]
    # Per source, because the pooled figure is an average of leaks that do not
    # share a channel: WildFake leaks through sharpness and SID_Set through its
    # noise floor, and each dilutes the other (docs/low_level_confounds.md).
    for src in sorted(set(df["source"])):
        m = (df["source"] == src).to_numpy()
        if len(np.unique(y[m])) < 2:
            print(f"(source {src}: {int(m.sum())} rows, one class only -- no "
                  "within-source AUC exists. That is the normal case for a "
                  "source that contributes only reals.)")
        elif m.sum() < MIN_GROUP:
            print(f"(source {src}: {int(m.sum())} rows, under MIN_GROUP="
                  f"{MIN_GROUP} -- not scored. Raise --n to read it.)")
        else:
            rows.append(row(src, m))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pooled = out[out["group"] == "POOLED"].iloc[0]
    worst_name = max(PROXY_NAMES, key=lambda n: pooled[n])
    worst = float(pooled[worst_name])
    print(f"\nworst pooled proxy: {worst_name} = {worst:.4f}")
    print("  (frozen corpus, band mode: laplacian_var = 0.6721)")

    if a.max_auc is not None and worst > a.max_auc:
        raise SystemExit(
            f"\nREFUSED: worst pooled proxy {worst_name}={worst:.4f} exceeds "
            f"--max-auc {a.max_auc}. This corpus hands the head a stronger "
            "one-dimensional shortcut than the one we already have, so a "
            "headline AUC from it would mean less rather than more. Do not "
            "start the extraction.")
    return {"worst": worst, "worst_name": worst_name,
            "table": out.to_dict("records")}


if __name__ == "__main__":
    main()
