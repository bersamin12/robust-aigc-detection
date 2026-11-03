# The resolution shortcut

Measured 2026-08-29 against the frozen manifest (`data/manifest.parquet`,
138,116 rows) and the organisers' demo benchmark (`data/demo`, 13,841 images).

**Summary: image resolution leaks the label, in our training pool and far more
severely in the scored benchmark. Any headline detection number is
uninterpretable until it is read against a dimensions-only baseline.**

## In the training pool

Short side against label, WildFake rows only:

| short side | authentic | generated | % generated |
| --- | --- | --- | --- |
| 128 | 0 | 1,262 | 100% |
| 200 | 40,000 | 11,516 | 22% |
| 224 | 0 | 7,002 | 100% |
| 256 | 0 | 25,728 | 100% |
| 450 | 0 | 6,158 | 100% |
| 512 | 15,000 | 10,316 | 41% |

**40,150 images — 29% of the whole dataset — sit at a resolution that is 100%
generated.** A single threshold on short side alone classifies at 72.6%
accuracy against a 52.9% majority-class baseline.

The cause is mundane and not a bug in our pipeline: generative models emit at
their native resolution (128, 224, 256, 450 are all standard generator output
sizes), while WildFake's authentic images were stored at 200 or 512.
Normalisation stores at short side 512 but never upscales, so the native sizes
survive.

## In the organisers' benchmark

Sampling 300 images from each half:

| half | label | short side |
| --- | --- | --- |
| COCO val2017 | authentic | **200 for every image sampled** |
| DALL·E 3 Advanced | generated | 618 – 1024, median 1024 |

The two halves do not overlap at all. `short_side >= 512 -> generated` is a
~100% accurate classifier on the scored benchmark **without reading a single
pixel**.

## Why this matters more than it looks

The backbones squish every input to a fixed square (384, or 224 for CLIP), so
absolute dimensions never reach the model. The leak survives as a *resampling
signature*: a 128px image upscaled to 384 is visibly soft; a 1024px image
downscaled to 384 is sharp. That softness is near-perfectly correlated with the
label across a third of our training data and across the entire benchmark.

A model trained and scored on this can post an excellent number having learned
"soft equals authentic" and nothing whatsoever about generation artefacts. The
failure is silent: it looks like success.

It is also directionally *consistent* between our training pool and the
benchmark (small equals authentic in both), which is the worst case. An
inconsistent shortcut would have shown up as a bad benchmark score and been
caught. A consistent one is rewarded.

## What we do about it

No data-level fix removes this, and we did not pretend otherwise. The source
datasets genuinely differ in resolution; re-normalising to a common size moves
the cue from dimensions into interpolation history rather than deleting it.
Three mitigations, none of which is a cure:

1. **A dimensions-only baseline** (`aigcdet.baselines.resolution`) is scored
   alongside every model and reported next to it. A model that does not
   substantially beat a classifier which never sees a pixel has not been shown
   to detect anything.
2. **Resolution canonicalisation** before condition transforms, applied
   identically to both classes and to both the training and evaluation paths,
   attenuating the resampling signature.
3. **Resolution-matched evaluation subsets**
   (`aigcdet.eval.resolution_control`), where within each stratum the classes
   are balanced so resolution carries no label information — reported with the
   count of rows discarded to achieve it, because a matched subset that drops
   most of the data is a weaker claim than one that drops little.

The robustness conditions the brief specifies (resize 0.5x/0.25x, crop 80%)
independently disturb this cue, which is a further reason the transformed
conditions are a more honest measurement than the clean one.

## Limitation we are stating rather than hiding

This is a property of the available public datasets, not of our method. Any
team using WildFake against the supplied benchmark faces it. We report the
dimensions-only baseline so our numbers can be read for what they are, and we
would treat a production version of this detector as unvalidated until it was
evaluated on a corpus where resolution and provenance are independent.
