# A bank streamed between boxes does not reproduce on the box that received it

**2026-08-31, box #2 (`211.72.13.201`), probe corpus, `dinov2regl:crop`.**

## What happened

Rung A6's TTA eval bank was extracted on box #2 and checked against the plain
eval bank already there, using `scripts/verify_tta_bank.py`. The check compares
the bank's `identity` TTA view against the plain bank's matching condition:
those are the same pixels by construction, so they must agree.

They did not.

```
IDENTITY CHECK FAILED: worst 385.0 ULP on 'jpeg_q50', bar 2
```

## What it was not

Three candidates were ruled out before anything was rebuilt.

**Not the TTA composition.** The stored recipes agree condition for condition
between the two banks.

**Not batching.** Two extractions of the same 24 images at `--batch-size 32`
and `64` are bit-identical: 0 of 24 rows differ, worst `|delta|` exactly
0.000000. Different GEMM reduction orders were the obvious suspect and are not
the cause.

**Not the corpus.** The image bytes are identical on both boxes
(`md5 067fef3fffa045d2db3d4a905d56737f` for the same file on each).

The decisive observation was that `clean` — empty recipe, no RNG, deterministic
centre crop — differed on **3517 of 4000 rows**. Nothing about a TTA view can
affect that column, so the fault had to be upstream of the whole feature.

## What it was

A fresh extraction from box #2's own images does not reproduce the eval bank
that was streamed onto box #2:

```
fresh-from-this-box vs imported bank, same 24 rows:
  rows differing   21/24
  worst |delta|    0.004883   (feature |x| median 0.2017)
  clean view only  20/24 differ
```

Same bytes, same manifest fingerprint, different compute stack. Box #2 runs
`cv2=5.0.0 / PIL=12.3.0 / numpy=2.5.2 / torch=2.10.0+cu128`; the box that built
the imported bank ran something else. The only transform touching a clean view
is the resampling inside `canonicalise`, which is OpenCV's.

Streaming the banks was the right call at the time — it saved a full Stage A
run, and the fingerprints genuinely matched. The cost was invisible until
something computed locally had to sit in the same table.

## What it does and does not invalidate

**Valid.** `a3`, `a4`, `a4vq`, `a4both`, `aF`. Every one was trained on the
imported train bank and scored on the imported eval bank. That ladder compares
rungs, and its numbers stand.

**Invalid.** Placing anything computed on box #2 into that table. Rung A6's
bank was built here; the others' were not. That is "comparing an evaluation
rather than a rung", which is precisely the confound `assert_banks_comparable`
exists to prevent.

## Why the existing guards did not catch it

`assert_tta_bank_matches` compares `manifest_sha256`, the condition axis, the
canonicalisation policy and row order. All four legitimately agreed — the two
banks *are* the same evaluation, described identically, over the same images.
Only the pixels disagreed, and a fingerprint of the manifest cannot see that.

The check that did catch it compares stored FEATURES against a bank built from
the same pixels by another route, in units of the storage dtype's last place.
That is the only kind of check that could have.

## The fix

`scripts/rebuild_banks_local.sh` re-extracts both probe banks from this box's
images, as `*_local`. The imported banks are kept, not overwritten: they remain
the correct basis for the results already written up, and replacing them would
silently change the provenance of numbers in a report.

After the rebuild the same check passes against the local bank:

```
identity view reproduces the plain bank on all 20 conditions: OK
  (0 bit-exact, worst 1.00 ULP of float16, bar 2)
```

385 ULP against the imported bank, 1 ULP against the local one, from the same
TTA bank and the same code. The TTA path was correct throughout.

## The rule this leaves behind

**A bank may be copied between machines and read, but nothing computed on the
receiving machine may be tabulated beside it.** Mixing the two compares compute
stacks. If a box needs to compute anything new — an aux block, a TTA bank, a
live-pixel training rung — it needs its own banks, and the cheapest honest move
is to re-extract rather than to assume.

This applies with particular force to the unfreeze ladder (D0–D4), which trains
from live pixels and whose D0 rung is defined as reproducing cached `a3`. On
imported banks D0 would have failed to reproduce its own control, and the
failure would have looked like a bug in the trainer.
