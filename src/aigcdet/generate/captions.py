"""Captions for the text-to-image arm, from Florence-2-large (MIT).

`docs/02` §3.2 says "do not invent captions" and points at Open Images'
Localized Narratives -- human-written descriptions keyed by `ImageID`. That is
the right instinct and the wrong instrument here: the narratives cover 5.60% of
the Open Images train split and **6.48% of the CC BY 2.0 thumbnail rows this
corpus draws from**, about 3,900 of 60,000. Building a t2i arm on them would
either shrink the arm to a fifth of its planned size or fill the rest with
`prompt_source="MISSING"` -- an empty prompt, which is how `docs/03` §8's
`inpaint_box` arm ended up generating on no conditioning at all.

Captioning locally makes every real eligible and keeps the pairing genuine: the
caption describes *that photograph*, so its t2i counterpart is a real pairing
rather than an unrelated image. Florence-2-large is MIT, so the caption step
does not weaken the licence position of the corpus.

The caption is cached to parquet and keyed by `image_id`. It is an input to the
seed-deterministic generation, so it must not change between a smoke run and
the full run -- regenerate captions and you regenerate the corpus.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: `<CAPTION>` is a sentence fragment; `<MORE_DETAILED_CAPTION>` runs to a
#: paragraph and drifts off the image. `<DETAILED_CAPTION>` is one or two
#: sentences -- the length a diffusion text encoder actually attends to (SD's
#: CLIP truncates at 77 tokens).
TASK = "<DETAILED_CAPTION>"

# The `microsoft/Florence-2-*` uploads predate native Florence-2 support and
# load into transformers 5.x with the vision tower's conv norms mismatched
# (checkpoint 1024 vs model 512). `ignore_mismatched_sizes` would "fix" that by
# randomly reinitialising those norms -- silently, producing captions from a
# partly-untrained vision tower. `florence-community` is the official
# transformers-native re-upload of the same weights, still MIT.
MODEL_ID = "florence-community/Florence-2-large"


def load_captioner(device: str = "cuda", dtype: str = "bfloat16"):
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=getattr(torch, dtype)).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def clean(text: str) -> str:
    """Strip Florence-2's task tag and boilerplate opener.

    Every caption starting "The image shows..." spends ~4 of a 77-token budget
    saying nothing about the image, and does it identically across the whole
    corpus -- so the prompt distribution carries a constant the reals cannot.
    """
    t = text.replace(TASK, "").strip()
    for opener in ("The image shows ", "The image is ", "The image depicts ",
                   "This image shows ", "In this image "):
        if t.startswith(opener):
            t = t[len(opener):]
            break
    t = " ".join(t.split())
    return t[:1].upper() + t[1:] if t else t


def caption_batch(paths, model, processor, *, max_new_tokens: int = 96,
                  num_beams: int = 1) -> list[str]:
    """Greedy by default. Beam search at width 3 measured 4x slower -- 0.46 vs
    0.117 s/image -- for captions that are a prompt, not ground truth. That is
    ~50 minutes of GPU across the corpus, spent on wording."""
    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in paths]
    try:
        inputs = processor(text=[TASK] * len(images), images=images,
                           return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device, model.dtype)
                      if v.dtype.is_floating_point else v.to(model.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 num_beams=num_beams, do_sample=False)
        return [clean(t) for t in
                processor.batch_decode(ids, skip_special_tokens=True)]
    finally:
        for im in images:
            im.close()


def caption_pool(paths_by_id: dict[str, str], out_parquet: str | Path,
                 *, batch_size: int = 32, device: str = "cuda",
                 log_every: int = 10) -> pd.DataFrame:
    """Caption every real, resuming from `out_parquet` if it exists.

    Resumable because this is ~30 minutes of GPU on 60,000 images and the run
    that follows it is six hours: losing the captions to an OOM at minute 25
    should not cost the captions already earned.
    """
    out_parquet = Path(out_parquet)
    done: dict[str, str] = {}
    if out_parquet.exists():
        prev = pd.read_parquet(out_parquet)
        done = dict(zip(prev["image_id"], prev["caption"]))
    todo = [i for i in paths_by_id if i not in done]
    if not todo:
        return pd.read_parquet(out_parquet)

    model, processor = load_captioner(device)
    try:
        for b, at in enumerate(range(0, len(todo), batch_size)):
            chunk = todo[at:at + batch_size]
            for image_id, cap in zip(chunk, caption_batch(
                    [paths_by_id[i] for i in chunk], model, processor)):
                done[image_id] = cap
            if b % log_every == 0 or at + batch_size >= len(todo):
                print(f"captioned {min(at + batch_size, len(todo))}/{len(todo)}",
                      flush=True)
                _write(done, out_parquet)
    finally:
        _write(done, out_parquet)
    return pd.read_parquet(out_parquet)


def _write(done: dict[str, str], out_parquet: Path) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": list(done), "caption": list(done.values()),
                  "prompt_source": "florence2"}).to_parquet(out_parquet,
                                                            index=False)
