"""The generation loop: reals in, matched pairs out.

Everything GPU-touching lives here; `geometry`, `encode`, `pool` and `registry`
are pure and tested. The loop's contract for one real is:

    crop_box(real)  ->  generate at exactly that size  ->  check the output
                    ->  save the fake through the real's encoder
                    ->  emit the real losslessly at the same box
                    ->  assert_parity(real, fake)

`assert_parity` is a post-condition on **every** pair, not a sample. Both
failures `docs/03` §1 records -- fakes always a multiple of 8 while reals kept
their native size, and a DCT phase mismatch that read as `jpeg_quality` AUC
0.0000 -- would have been caught by it on the first image instead of at the
gate, after the GPU time was spent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.generate.encode import (assert_parity, emit_real_cropped,
                                     save_matched, source_encoder)
from aigcdet.generate.geometry import seed_for
from aigcdet.generate.registry import MODELS, SUITE, FamilySpec, validate_suite

#: An output whose pixel std is below this is a flat frame -- the failure mode
#: of an OOM-degraded or mis-configured pipeline, which returns something
#: plausibly shaped and entirely black.
MIN_STD = 2.0

#: An `img2img`/`self_cond`/`ref_image` output within this mean absolute
#: distance of its real is a copy, not a generation. A corpus of copies would
#: train a detector to call photographs fake.
MIN_DELTA = 1.5


def check_licence(model_key: str) -> None:
    """Assert the model card still publishes the licence the registry claims.

    Licences change and `docs/02` §2 says to verify at the model page every
    time. This is that check, executed rather than remembered, and it runs
    before the weights are used rather than during a later audit.
    """
    from huggingface_hub import model_info
    spec = MODELS[model_key]
    # Every repo the pipeline pulls, not only the one it is addressed by: a
    # combined pipeline (Kandinsky 2.2) loads its prior from a second repo,
    # and checking one of the two licence-clears half the weights that made
    # the image.
    for hf_id in (spec.hf_id, *spec.companion_ids):
        published = (model_info(hf_id).cardData or {}).get("license")
        if published != spec.licence_tag:
            raise RuntimeError(
                f"{hf_id} publishes license={published!r}, registry claims "
                f"{spec.licence_tag!r}. Re-audit before generating anything "
                f"with it -- the whole licence position of this corpus rests "
                f"on the registry being true.")


def load(model_key: str, methods: set[str], device: str = "cuda"):
    """Load one model once, and return a pipeline per method it must serve.

    SDXL serves both `t2i` and `img2img`; loading it twice would cost 7 GB for
    an identical set of weights. `from_pipe` shares the components instead.
    """
    import torch
    from diffusers import (AutoPipelineForImage2Image, AutoPipelineForInpainting,
                           AutoPipelineForText2Image)

    spec = MODELS[model_key]
    check_licence(model_key)
    dtype = getattr(torch, spec.dtype)
    kw = dict(dtype=dtype, use_safetensors=True)

    if spec.arch == "flow_dit":
        # Klein's `image` argument is reference conditioning on the SAME
        # pipeline class, so t2i and ref_image are one object.
        from diffusers import Flux2KleinPipeline
        base = Flux2KleinPipeline.from_pretrained(spec.hf_id, **kw)
    elif spec.arch == "zimage_dit":
        # Its own pipeline class, and new enough that the auto class would
        # raise on it -- the same reason wuerstchen is dispatched explicitly.
        # `low_cpu_mem_usage=False` is the card's own example; it matters
        # because the offload path below re-places the components anyway.
        from diffusers import ZImagePipeline
        base = ZImagePipeline.from_pretrained(spec.hf_id, **kw)
    elif spec.arch == "wuerstchen":
        # Not in AutoPipelineForText2Image's mapping, so the auto class would
        # raise on it. The combined pipeline pulls the prior repo itself.
        from diffusers import WuerstchenCombinedPipeline
        base = WuerstchenCombinedPipeline.from_pretrained(spec.hf_id, **kw)
    elif "self_cond" in methods:
        base = AutoPipelineForInpainting.from_pretrained(spec.hf_id, **kw)
    else:
        try:
            base = AutoPipelineForText2Image.from_pretrained(
                spec.hf_id, variant="fp16", **kw)
        except Exception:
            base = AutoPipelineForText2Image.from_pretrained(spec.hf_id, **kw)

    if "self_cond" in methods:
        in_ch = base.unet.config.in_channels
        if in_ch != 9:
            raise RuntimeError(
                f"{spec.hf_id} has a {in_ch}-channel UNet; self_cond needs the "
                f"9-channel inpainting one (4 noisy latent + 1 mask + 4 masked "
                f"image). A 4-channel checkpoint here would silently run as "
                f"plain img2img and the family name would be a lie.")

    free = _free_vram_gb(device)
    if spec.vram_gb > free:
        print(f"  {model_key}: {spec.vram_gb} GB weights vs {free:.1f} GB free "
              f"-> {spec.offload_mode} CPU offload (slower, but not quantised)",
              flush=True)
        if spec.offload_mode == "model":
            base.enable_model_cpu_offload()
        elif spec.offload_mode == "sequential":
            base.enable_sequential_cpu_offload()
        else:
            raise ValueError(
                f"{model_key}: unknown offload_mode {spec.offload_mode!r}; "
                f"expected 'model' or 'sequential'")
    else:
        base = base.to(device)
    base.set_progress_bar_config(disable=True)

    pipes = {}
    for m in methods:
        if m in ("t2i", "ref_image", "self_cond"):
            pipes[m] = base
        elif m == "img2img":
            p = AutoPipelineForImage2Image.from_pipe(base)
            p.set_progress_bar_config(disable=True)
            pipes[m] = p
        else:
            raise KeyError(f"no pipeline wiring for method {m!r}")
    return base, pipes


def gpu_name(device: str = "cuda") -> str:
    """The card this process generates on, for the manifest.

    A corpus generated across mixed hardware cannot be stratified after the
    fact unless each row says what made it, and `docs/02` U8 plans exactly
    that: an A4500's rows and four 4090s' rows landing in one
    `pairs.parquet`. Per row rather than per run, because `--run-families`
    splits one shard's families across boxes -- "the run" is not one machine.
    """
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return str(device)
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return str(device)


def _free_vram_gb(device: str) -> float:
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 2 ** 30


def _round_up(n: int, mult: int) -> int:
    """Smallest multiple of `mult` that is >= n. Never rounds DOWN: a smaller
    output cannot be cropped up to the real's box, and `generate` raises."""
    if mult < 1:
        raise ValueError(f"size_multiple must be >= 1, got {mult}")
    return -(-n // mult) * mult


def generate(pipe, fam: FamilySpec, real: Image.Image, prompt: str,
             seed: int, size: tuple[int, int], device: str = "cuda") -> Image.Image:
    """Produce one fake at exactly `size` (w, h)."""
    import torch

    w, h = size
    # Ask at the model's own latent granularity and crop back to the real's
    # box below. Kandinsky 2.2 silently rounds a request up to a multiple of
    # 64 -- measured: 432x640 and 416x640 both came back 448x640, the same
    # image twice -- and Sana's 32x deep-compression autoencoder will not
    # accept a size 32 does not divide. Requesting the real's exact box and
    # hoping is how a family ends up with a dimension the pair does not
    # share, which is the leak `geometry.crop_box` exists to prevent.
    spec = MODELS[fam.model]
    mult = spec.size_multiple
    rw, rh = _round_up(w, mult), _round_up(h, mult)
    g = torch.Generator(device if str(device).startswith("cuda") else "cpu")
    g.manual_seed(seed % (2 ** 63))

    # `guidance_kw` because Wuerstchen has no `guidance_scale` -- it splits
    # into prior/decoder -- and diffusers swallows unknown kwargs rather than
    # raising, so the usual name would silently generate at the pipeline
    # default while the manifest recorded ours.
    kw = {"prompt": prompt, "num_inference_steps": fam.steps,
          spec.guidance_kw: fam.guidance, "generator": g}
    if fam.negative:
        kw["negative_prompt"] = fam.negative
    # Per-model call arguments. Sana's `use_resolution_binning=False` lives
    # here, and it is load-bearing: see `ModelSpec.call_kwargs`.
    kw.update(dict(spec.call_kwargs))

    if fam.method == "t2i":
        out = pipe(height=rh, width=rw, **kw)
    elif fam.method == "ref_image":
        out = pipe(image=[real], height=rh, width=rw, **kw)
    elif fam.method == "img2img":
        out = pipe(image=real, strength=fam.strength, **kw)
    elif fam.method == "self_cond":
        # An ALL-ZERO mask with strength 1.0: the latents start from pure
        # noise, and the 9-channel UNet is conditioned on the whole real
        # through its masked-image channels. That is "regenerate this image",
        # not "inpaint this region" -- the pipeline composites nothing back,
        # so every output pixel is generated.
        mask = Image.new("L", (w, h), 0)
        out = pipe(image=real, mask_image=mask, height=rh, width=rw,
                   strength=1.0, **kw)
    else:
        raise KeyError(f"unknown method {fam.method!r}")

    img = out.images[0].convert("RGB")
    if img.size != (w, h):
        # Includes the rounding above, and anything else the pipeline decided
        # on its own; either way the pair must end at the real's exact box.
        # Crop, never resize: a resample leaves the spectral signature
        # `docs/resolution_shortcut.md` measured, on the generated class only.
        if img.size[0] < w or img.size[1] < h:
            raise RuntimeError(f"pipeline returned {img.size}, smaller than the "
                               f"requested {(w, h)}; cannot crop up to it")
        l, t = (img.size[0] - w) // 2, (img.size[1] - h) // 2
        img = img.crop((l, t, l + w, t + h))
    return img


def check(fake: Image.Image, real: Image.Image, method: str) -> None:
    """Reject degenerate output before it costs a manifest row."""
    a = np.asarray(fake, dtype=np.float32)
    if a.std() < MIN_STD:
        raise ValueError(f"near-constant output (std {a.std():.2f} < {MIN_STD})")
    if method != "t2i":
        b = np.asarray(real.convert("RGB"), dtype=np.float32)
        delta = float(np.abs(a - b).mean())
        if delta < MIN_DELTA:
            raise ValueError(
                f"output is a copy of its real (mean |delta| {delta:.2f} < "
                f"{MIN_DELTA}); a fake that is its own real teaches the "
                f"detector to call photographs generated")


def _done_ids(rows_path: Path, out_root: Path, family: str) -> set[str]:
    """Image ids already emitted, confirmed against the files on disk."""
    if not rows_path.exists():
        return set()
    done = set()
    with rows_path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # truncated last line of a kill
            if ((out_root / r["fake_rel"]).exists()
                    and (out_root / r["real_rel"]).exists()):
                done.add(r["image_id"])
    return done


def run_family(family: str, sel: pd.DataFrame, captions: dict[str, str],
               pipes: dict, out_root: Path, rows_dir: Path, seed: int,
               *, suite: dict[str, FamilySpec] | None = None,
               device: str = "cuda", log_every: int = 25,
               max_fail_rate: float = 0.05) -> dict:
    """Generate every row of `sel` for one family. Resumable, and it stops if
    failures run away rather than burning hours producing nothing."""
    suite = SUITE if suite is None else suite
    fam = suite[family]
    spec = MODELS[fam.model]
    pipe = pipes[fam.method]
    rows_path = rows_dir / f"rows_{family}.jsonl"
    rows_dir.mkdir(parents=True, exist_ok=True)

    gpu = gpu_name(device)
    done = _done_ids(rows_path, out_root, family)
    todo = sel.loc[~sel["image_id"].isin(done)]
    stats = {"family": family, "done_before": len(done), "ok": 0, "failed": 0,
             "seconds": 0.0, "reasons": {}}
    if not len(todo):
        return stats

    t_start = time.time()
    with rows_path.open("a") as fh:
        for n, row in enumerate(todo.itertuples(index=False), 1):
            image_id = row.image_id
            box = (row.crop_l, row.crop_t, row.crop_r, row.crop_b)
            w, h = box[2] - box[0], box[3] - box[1]
            real_rel = f"open_images_v7/real/{image_id}.jpg"
            fake_rel = f"open_images_v7/{family}/{image_id}.jpg"
            try:
                enc = source_encoder(row.path)
                with Image.open(row.path) as im:
                    real_crop = im.convert("RGB").crop(box)
                prompt = captions.get(image_id, "")
                if not prompt:
                    raise ValueError("no caption; refusing to generate on an "
                                     "empty prompt (docs/03 §8)")
                s = seed_for(image_id, seed)
                t0 = time.time()
                fake = generate(pipe, fam, real_crop, prompt, s, (w, h), device)
                gen_s = time.time() - t0
                check(fake, real_crop, fam.method)

                save_matched(fake, out_root / fake_rel, enc)
                emit_real_cropped(row.path, out_root / real_rel, box)
                assert_parity(out_root / real_rel, out_root / fake_rel)
            except Exception as exc:
                stats["failed"] += 1
                reason = f"{type(exc).__name__}: {exc}"
                stats["reasons"][reason[:120]] = \
                    stats["reasons"].get(reason[:120], 0) + 1
                for rel in (fake_rel, real_rel):        # never leave a half pair
                    (out_root / rel).unlink(missing_ok=True)
                seen = stats["ok"] + stats["failed"]
                if seen >= 40 and stats["failed"] / seen > max_fail_rate:
                    raise RuntimeError(
                        f"{family}: {stats['failed']}/{seen} failed, above "
                        f"{max_fail_rate:.0%}. Stopping rather than spending "
                        f"hours on it. Reasons: {stats['reasons']}") from exc
                continue

            fh.write(json.dumps({
                "image_id": image_id, "family": family, "method": fam.method,
                "model": fam.model, "hf_id": spec.hf_id,
                "licence_tag": spec.licence_tag, "lineage": spec.lineage,
                "arch": spec.arch, "seed": s, "steps": fam.steps,
                "guidance": fam.guidance, "strength": fam.strength,
                "prompt": prompt, "prompt_source": "florence2",
                "width": w, "height": h, "crop_box": list(box),
                "src_width": int(row.width), "src_height": int(row.height),
                "jpeg_quality": float(enc["jpeg_quality"]),
                "subsampling": int(enc["subsampling"]),
                # What made this row. Neither was written before, so the
                # frozen 11,978-pair corpus cannot be stratified by hardware
                # after the fact; from here on it can be. docs/02 U7.6.
                "dtype": spec.dtype, "gpu": gpu,
                "real_rel": real_rel, "fake_rel": fake_rel,
                "gen_seconds": round(gen_s, 3),
            }) + "\n")
            fh.flush()
            stats["ok"] += 1
            if n % log_every == 0 or n == len(todo):
                el = time.time() - t_start
                print(f"  {family} {n}/{len(todo)} ok={stats['ok']} "
                      f"fail={stats['failed']} {el / n:.2f}s/img "
                      f"eta {(len(todo) - n) * el / n / 60:.1f}m", flush=True)
    stats["seconds"] = time.time() - t_start
    return stats


def run(sel: pd.DataFrame, captions: dict[str, str], out_root: str | Path,
        rows_dir: str | Path, seed: int, *,
        suite: dict[str, FamilySpec] | None = None,
        corpus: dict[str, FamilySpec] | None = None,
        device: str = "cuda") -> list[dict]:
    """Generate the whole selection, one MODEL at a time.

    Model-major rather than family-major so a 15 GB set of weights is loaded
    once and serves every family that needs it, and so only one model is
    resident at a time -- klein-4B alone is 15 GB of a 20 GB card.

    `corpus` is every family the manifest will hold once this run lands; the
    held-out invariants are properties of that and not of one run's slice
    (`registry.validate_suite`). Revalidated here and not merely at the CLI
    because `run` is the entry point that spends the GPU, and a caller that
    built its own suite dict never went through the CLI at all.
    """
    import gc
    import torch

    suite = SUITE if suite is None else suite
    validate_suite(suite, corpus=corpus)
    out_root, rows_dir = Path(out_root), Path(rows_dir)
    families = [f for f in sel["family"].unique() if f in suite]

    by_model: dict[str, list[str]] = {}
    for f in families:
        by_model.setdefault(suite[f].model, []).append(f)

    all_stats = []
    for model_key, fams in by_model.items():
        methods = {suite[f].method for f in fams}
        print(f"[load] {model_key} for {fams}", flush=True)
        base, pipes = load(model_key, methods, device)
        try:
            for f in sorted(fams):
                st = run_family(f, sel.loc[sel["family"] == f], captions,
                                pipes, out_root, rows_dir, seed,
                                suite=suite, device=device)
                print(f"[done] {st}", flush=True)
                all_stats.append(st)
        finally:
            del pipes, base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return all_stats
