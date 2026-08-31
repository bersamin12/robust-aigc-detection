#!/usr/bin/env python3
"""Refuse a canonicalisation policy whose standardised view leaks the label.

`crop_clamp` makes how much an image is resampled a function of its native
resolution, and native resolution is not independent of the label. On the plan
manifest at crop_side=224 the upscale factor ALONE separates the classes at AUC
0.5430; the question this script answers is whether that survives into the
16x16 view the model actually receives.

The thresholds are this corpus's own measurements, not round numbers:

    crop (200/512), content-blind pooled   0.5081   the clean reference
    band, content-blind pooled             0.6105   the policy we rejected

So a proposed policy is PASSED below `--max-auc` (default 0.5500, roughly
halfway to band) and REFUSED at or above it. A refusal is a result to report,
not an obstacle to route around: it means the extra resolution was bought with
a shortcut, and the arm should run at the unclamped policy instead.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

CROP_REFERENCE = 0.5081
BAND_REJECTED = 0.6105


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--crop-side", type=int, required=True)
    ap.add_argument("--nominal-side", type=int, default=None)
    ap.add_argument("--crop-clamp", action="store_true")
    ap.add_argument("--limit", type=int, default=20000,
                    help="rows sampled; the probe is cross-validated so this "
                         "bounds the CI rather than biasing the estimate")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-auc", type=float, default=0.55)
    ap.add_argument("--out", default="docs/gate_crop_policy.json")
    a = ap.parse_args()

    cmd = [sys.executable, "scripts/content_blind_probe.py",
           "--manifest", a.manifest, "--mode", "crop",
           "--crop-side", str(a.crop_side), "--limit", str(a.limit),
           "--workers", str(a.workers), "--out", a.out]
    if a.root:
        cmd += ["--root", a.root]
    if a.nominal_side is not None:
        cmd += ["--nominal-side", str(a.nominal_side)]
    if a.crop_clamp:
        cmd += ["--crop-clamp"]
    print("gate: " + " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        print("GATE ERROR: the probe itself failed; nothing is proven either "
              "way, so this is a refusal too.")
        return 2

    with open(a.out) as f:
        res = json.load(f)
    pooled = res.get("crop", {}).get("pooled", {})
    auc = pooled.get("auc", pooled.get("auc_unverified_branch_provenance"))
    if auc is None:
        print("GATE ERROR: probe wrote no pooled AUC")
        return 2
    lo, hi = pooled.get("auc_ci", (float("nan"), float("nan")))

    print(f"\n=========== CROP POLICY GATE ===========")
    print(f"  policy      crop_side={a.crop_side} nominal_side={a.nominal_side} "
          f"clamp={a.crop_clamp}")
    print(f"  pooled AUC  {auc:.4f}  CI [{lo:.4f}, {hi:.4f}]")
    print(f"  references  crop {CROP_REFERENCE:.4f} (clean) | "
          f"band {BAND_REJECTED:.4f} (rejected)")
    print(f"  threshold   {a.max_auc:.4f}")
    if auc >= a.max_auc:
        print(f"  VERDICT     REFUSED -- {auc:.4f} >= {a.max_auc:.4f}. The "
              f"extra resolution was bought with a low-level shortcut. Run "
              f"the arm at the unclamped policy and report this number.")
        return 1
    print(f"  VERDICT     PASS -- {auc:.4f} < {a.max_auc:.4f}, and "
          f"{auc - CROP_REFERENCE:+.4f} against the clean crop reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
