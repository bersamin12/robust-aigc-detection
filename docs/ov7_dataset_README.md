# AI-OV7 — encoder-matched real / AI-generated image pairs

Every AI image in this dataset was generated **from one specific photograph**,
at that photograph's own pixel dimensions, and saved through that
photograph's own JPEG quantisation tables.

That is the whole point, and it is unusual. In a typical AIGC-detection corpus
the real images are photographs from one place and the fakes are generations
from another, so a detector can reach the right answer by reading resolution,
sharpness, or compression history instead of reading generation. On our own
frozen corpus a **single sharpness statistic separates real from fake at AUC
0.6721** — no content, no semantics, just variance of the Laplacian. Detectors
trained on corpora like that report numbers they cannot keep in the wild.

Here content, geometry and compression history are held fixed across each
pair, so what is left between the two images is as close to *the generator
alone* as we could build.

---

## 1. What is in here

```
open_images_v7/           the dataset: raw JPEG pairs, exactly as generated
  real/<ImageID>.jpg        the photograph, losslessly cropped
  sdxl_t2i/<ImageID>.jpg    its generated counterpart, one per family
  sd15_t2i/<ImageID>.jpg
  sdxl_self_cond/...
  sdxl_img2img/...
  sd15_img2img/...
  klein4b_t2i/...
  klein4b_ref_image/...

pairs.parquet / pairs.csv one row per pair: seed, steps, guidance, strength,
                          prompt, prompt source, crop box, the real's measured
                          JPEG quality and subsampling, model, licence, decoder
                          lineage, generation time
attribution.csv           per-image Author / Title / OriginalURL for the reals
LICENCES.json             machine-readable provenance for every bucket
manifest_ov7.parquet      labels, families and the frozen train/val/held-out
                          split, keyed to normalized_ov7/
normalized_ov7/           the same corpus after our normalisation pass: PNG,
                          per-bucket running index. What manifest_ov7's
                          `rel_path` resolves against. Derived from the raw
                          tree, shipped because a notebook cannot rebuild it
                          inside a session.
```

**The filename stem is the pairing key.** `open_images_v7/real/abc123.jpg` and
`open_images_v7/sdxl_t2i/abc123.jpg` are a pair, and that stem is the Open
Images V7 ImageID, so `attribution.csv` joins straight onto it. A real is used
by exactly one family — the families are disjoint, not stacked on the same
photographs.

## 2. How the pairing is enforced

Three things are held fixed, and each of them was a bug first.

**Dimensions.** Both images are the same size, to the pixel. Diffusion decoders
emit multiples of 8, so an earlier attempt at this corpus had fakes at
multiples of 8 against reals at their native size — and `width % 8 == 0`
separated the classes at ~100% without touching a single pixel. Each pair is
now generated at a centre crop whose **offset and size are both multiples of
16**.

**Compression history.** The real is cropped with `jpegtran -crop` on that
16-pixel grid, which is a lossless operation: at MCU boundaries it rewrites no
DCT coefficient, so the photograph keeps its original camera-and-Flickr
compression history exactly. The fake is then saved through the real's own 64
quantisation integers and its chroma subsampling. Alignment matters as much as
the tables do — the same earlier attempt copied the tables correctly but put
the fake's fresh 8×8 grid at a different phase from the real's preserved one,
and JPEG-quality AUC came back at **0.0000**, perfect inverse separation.

**Content.** The prompt for each generation is a caption of that exact
photograph, produced by Florence-2-large (`<DETAILED_CAPTION>`, MIT licence).

Parity is asserted on **every** pair at write time — size, all 64 quantisation
integers, subsampling — not on a sample.

## 3. Shape of the images

All portrait. Aspect 0.625–0.714, median 416×640, maximum 448×640.
**Zero square and zero landscape images**, by construction: the reals were
harvested at aspect ≤ 0.7, and generation happens at the real's own crop.
Public AIGC corpora are overwhelmingly square, which is itself a confound when
real-world inputs are not.

## 4. The generators

<!--FAMILIES-->

Conditioning is 76% fully synthetic (text-to-image and zero-mask
regeneration) and 24% image-conditioned.

Decoder **lineage** is recorded per row (`sdxl_vae`, `sd_vae`, `flux2_vae`)
and matters more than the model name: families sharing a VAE share most of
their decoder fingerprint. The frozen split holds out the **entire `flux2_vae`
lineage** — a different decoder *and* a different architecture (flow-matching
DiT against the UNets) — so the held-out rung measures a jump between decoder
families rather than generalising to a cousin that shares a VAE.

## 5. Licensing

**Reals: CC BY 2.0.** Sourced from Open Images V7, filtered at harvest to
`License == CC BY 2.0` — commercial use *and* redistribution permitted, with
attribution. `attribution.csv` carries Author, Title and OriginalURL for every
photograph and **must travel with the images**; our normalisation pass strips
image metadata, so the CSV is the only place that information survives.

**Generated images: ours to release.** Produced from weights under
`apache-2.0` (FLUX.2-klein-4B), `openrail++` (SDXL) and
`creativeml-openrail-m` (SD 1.5). All three grant ownership of the outputs and
permit commercial use; their use-based restrictions bind how you may use the
*model*, not how these images may be redistributed. The per-row `licence_tag`
in `pairs.parquet` lets you cut a pure-Apache subset without regenerating
anything.

This is the reason the corpus exists. The public datasets that would close the
modern-generator gap — OpenFake, Defactify — are licence-barred *structurally*:
their reals are web scrapes whose images keep individual copyrights, so no
compiler can grant commercial rights downstream. Generating over CC BY 2.0
reals is the way around that.

## 6. Confound gate

Measured with `scripts/gate_confounds.py`, which decodes the images directly
and asks how well each single low-level statistic predicts the label — no
model, no training. 0.5 is chance. Read against our frozen corpus, whose reals
and fakes come from different sources in the ordinary way:

<!--GATE-->

## 7. Three things to know before you train on this

**`self_cond` pairs are perceptually near-identical to their reals.** Zero-mask
regeneration reproduces the scene; the pairs sit at pHash Hamming distance 0–2.
Any within-corpus perceptual dedupe will silently delete that entire arm and
you will not be told.

**SD 1.5's VAE is a common reconstruction probe.** If your detector includes a
reconstruction-error feature that uses the SD 1.5 autoencoder, the `sd15_*`
families reconstruct at near-zero error and separate for free. That is not a
property of generated images in general — report reconstruction numbers per
lineage or they mean nothing.

**Around 0.8% of SD 1.5 generations were dropped.** 22 of 2,800 came back as a
completely black frame (pixel std exactly 0.00) — SD 1.5's known half-precision
VAE instability, where a NaN in the decoder surfaces as a flat image rather
than an error. Both files of such a pair were deleted, so there are no
orphans; the two SD 1.5 families land at 99.2% and 99.4% of target. It was
deliberately *not* patched mid-run: decoding that VAE in fp32 partway through
would leave some `sd15_*` images decoded at bf16 and the rest at fp32, a
within-family forensic difference, which is a worse problem than 0.8%
attrition in a corpus whose entire purpose is holding everything but the
generator fixed.

## 8. Reproducing it

Seeds are content-addressed — `blake2b(SEED, ImageID)` — so any pair
regenerates independently of the order or size of the run, and `pairs.parquet`
carries every other argument. Code, including the geometry and encoder-parity
functions and their tests, is in the `aigcdet` repository under
`src/aigcdet/generate/`.

## 9. Citation and attribution

If you use this dataset, please retain `attribution.csv` and credit the
photographers as CC BY 2.0 requires. Generated images may be used freely,
including commercially.
