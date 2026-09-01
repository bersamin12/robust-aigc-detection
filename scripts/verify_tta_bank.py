#!/usr/bin/env python3
"""Prove a TTA eval bank is the plain eval bank, eight ways.

The claim rung A6 rests on is that its bank holds the SAME images under the
SAME conditions as every other rung's, each looked at eight times. Nothing
about a violation of that is visible downstream: a bank whose composition order
is flipped, whose condition RNG was re-keyed per view, or whose flattening is
transposed has the right shape, the right dtype, the right row count and the
right condition names, and produces a robustness table that reads perfectly.

So the post-condition is checked against the plain bank rather than against
itself. Column `j * n_views` is condition `j` with the identity view applied, and the
identity view is the identity -- so it must equal column `j` of the plain bank.
That single equality fails under all four of the bugs above.

**Why the tolerance is one float16 ULP and not zero.** The PIXELS are identical
and the check on them is exact -- `tests/eval/test_tta_bank.py` runs a
deterministic stub embedder and asserts `array_equal`. What is compared here is
what the two banks STORE, which is a float16 cast of a GPU embedding, and the
plain bank embeds each image's 20 views in one batch while this one embeds 160.
Different batch shapes give different GEMM reduction orders, so the float32
embeddings differ in the last bits and the float16 cast lands one step apart on
some rows. Measured on a 24-image probe: 13 of 20 conditions differed, every one
of them by exactly 1 ULP (2.4e-4 at magnitude 0.25).

The tolerance cannot hide the bugs it is here to catch. Composing the TTA view
before the condition, re-keying the condition RNG per view, or transposing the
flattening all change which PICTURE was embedded, and two different pictures do
not agree to 3 decimal places -- they disagree by O(0.01) to O(1), three to four
orders of magnitude above this bar. The check reports the worst ULP distance it
saw, so a real divergence arrives as a number in the thousands, not as a
borderline pass.

Run after every merge. `scripts/run_tta_eval.sh` does.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from aigcdet.eval.grid import tta_axis
from aigcdet.features.bank import FeatureBank


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tta", required=True, help="the condition x tta_view bank")
    ap.add_argument("--plain", required=True, help="the condition-axis bank")
    ap.add_argument("--identity-view", default="identity",
                    help="the view that must reproduce the plain bank")
    ap.add_argument("--max-ulps", type=float, default=2.0,
                    help="how far apart the two banks' float16 values may land, "
                         "in units of the last place. The default of 2 covers "
                         "the GEMM reduction-order difference between a 20-view "
                         "and a 160-view batch; a genuine pixel difference is "
                         "thousands of ULPs, so raising this does not buy "
                         "safety, it only blinds the check. 0 demands "
                         "bit-exactness, which holds only if both banks were "
                         "embedded with the same batch composition.")
    a = ap.parse_args()

    tta, plain = FeatureBank.open(a.tta), FeatureBank.open(a.plain)
    views = tta_axis(tta)
    names = plain.config["conditions"]
    print(f"tta   {a.tta}\n  {len(tta.meta)} rows x {tta.config['n_views']} views "
          f"= {len(names)} conditions x {len(views)} tta views {views}")
    print(f"plain {a.plain}\n  {len(plain.meta)} rows x "
          f"{plain.config['n_views']} views")

    ok = True

    # Everything that must agree before the pixel check means anything. A
    # bit-exact identity column over the wrong manifest is still the wrong bank.
    from aigcdet.eval.grid import assert_tta_bank_matches
    try:
        assert_tta_bank_matches(plain, tta)
        print("axis/manifest/policy/order: OK")
    except ValueError as exc:
        print(f"axis/manifest/policy/order: FAILED\n  {exc}")
        return 1

    if a.identity_view not in views:
        print(f"NOTE: {a.identity_view!r} is not among this bank's views, so the "
              "bit-exactness check is skipped -- there is nothing in this bank "
              "the plain bank also holds. Everything above still applies.")
        return 0
    k0 = views.index(a.identity_view)

    n = len(views)
    worst, worst_cond, exact = 0.0, None, 0
    for j, cond in enumerate(names):
        got = np.asarray(tta.feats[:, j * n + k0, :])
        want = np.asarray(plain.feats[:, j, :])
        if np.array_equal(got, want):
            exact += 1
            continue
        # Distance in units of the LAST PLACE of the stored dtype, not an
        # absolute epsilon. An absolute bar would be far too loose on the large
        # activations and far too tight on the small ones, and this bank's
        # values span both.
        ulp = np.spacing(np.abs(want).astype(np.float16)).astype(np.float64)
        delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
        d = float(np.max(delta / np.maximum(ulp, np.finfo(np.float16).tiny)))
        if d > worst:
            worst, worst_cond = d, cond
    if worst <= a.max_ulps:
        print(f"identity view reproduces the plain bank on all {len(names)} "
              f"conditions: OK ({exact} bit-exact, worst {worst:.2f} ULP of "
              f"float16, bar {a.max_ulps:g})")
    else:
        ok = False
        print(f"IDENTITY CHECK FAILED: worst {worst:.1f} ULP on {worst_cond!r}, "
              f"bar {a.max_ulps:g}. At this magnitude the two banks were built "
              "from different PIXELS -- a reduction-order difference is 1-2 "
              "ULP, not this. So A6's row is not comparable with the rungs it "
              "would be tabulated beside. Likely causes: the TTA view was "
              "composed BEFORE the condition; the condition RNG was re-keyed "
              "on the flattened index (each view would get its own noise "
              "draw); the flattening is transposed (k*n_cond+j); or the two "
              "banks used different seeds or canonicalisation.")

    # Cheap, and it is the failure the 2026-08-29 banks shipped: a bank whose
    # only post-condition was its row count turned out to be 100% NaN.
    block = np.asarray(tta.feats[:, :, :])
    finite = bool(np.isfinite(block).all())
    # Summed in float64. |x| over 160 views x 1024 dims overflows float16's
    # 65504 on ordinary data, and the resulting `inf` is never 0, so the
    # all-zero check would pass on a bank that was entirely zeros -- the one
    # thing it exists to catch.
    zero_rows = int((np.abs(block, dtype=np.float64).sum(axis=(1, 2)) == 0).sum())
    print(f"finite {finite}  all-zero rows {zero_rows}/{len(tta.meta)}")
    if not finite or zero_rows:
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
