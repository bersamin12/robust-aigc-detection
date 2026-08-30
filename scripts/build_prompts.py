"""Build the generation prompts, once, and write them down.

`docs/03` §2 pairs every generated image to a specific real by `ImageID`. This
produces the text that pairing runs on, and it is a separate step from buying
because prompts are reusable and images are not: a prompt file is regenerated
for pennies, a purchased image is bought again.

WHY NOT JUST USE THE NARRATIVES
-------------------------------
Open Images ships **Localized Narratives** — descriptions spoken aloud by a
person looking at the photograph, then transcribed. `docs/02` §3.2 is right that
these beat inventing captions: a human genuinely saw that image.

They are also mediocre *prompts*, and the reason matters. They are spoken
transcripts, so they open with "In this image I can see ..." every time and stay
sparse: *"the human hand and the background is in white color"* is a faithful
description and almost nothing for a generator to work with. The result is a
clean, empty frame against a photograph full of skin texture, shadows, clutter
and sensor noise.

That gap is not cosmetic. It lands on `laplacian_var`, which at AUC 0.6721 is
already this project's worst confound (`docs/low_level_confounds.md`) — worse
than the compression leak encoder parity exists to close. A detector can learn
"busy means real" and never look at a generation artefact.

WHAT NTIRE DID, AND WHERE THIS GOES FURTHER
-------------------------------------------
The NTIRE 2026 challenge (arXiv 2604.11487 §2.1) captions each real with a
vision model and rewrites it into a concise prompt with an LLM. Their stated
reason is ours: *"By 'pairing' generated images with their real counterparts, we
ensure that both subsets reflect similar semantics and content distribution,
which should help detectors learn content-agnostic features."*

Their vision model is guessing what is in the picture. **We have a human
description of that exact picture**, which CC12M and RedCaps do not ship. So
`--source both` hands the captioner the image *and* the narrative: the narrative
anchors it to what is actually there, the model supplies the density the
narrative lacks. Strictly more information than either alone.

`--source narrative` (no model, free) and `--source vlm` (NTIRE's method
exactly) are kept so the three can be compared on the gate rather than argued
about.

FOUR THINGS THIS GETS RIGHT ON PURPOSE
--------------------------------------
1. **One captioner for every provider.** Captioning OpenAI's images with
   OpenAI's model and Google's with Google's would make the prompt distribution
   a per-provider variable, confounding §5.2's whole point. It is held constant.
   The default is a Qwen model for the same reason: it is not any of the four
   generators' labs.
2. **The output is written down, with the model id and the date.** Narratives
   are a fixed public dataset and reproduce by construction; model output does
   not. Without this file the eval set is not reproducible.
3. **§3.5 gets harder here, not easier.** A narrative says "a person". A good
   captioner says "a bald man in his fifties with wire-framed glasses" — more
   identifying, about photographs of real people. The system prompt forbids it
   AND `sanitise` still runs over the result, because an instruction is not an
   enforcement.
4. **Narratives are never overwritten.** They are kept in their own column, so
   a disappointing VLM run is reverted without re-fetching anything.

COST
----
Pennies, and dry run is still the default. A 512px image is ~1.5k input tokens;
at the default captioner's $0.065/M that is ~$0.0001 an image, so the whole
pilot's prompts cost less than a cent against ~$11 for its images.

USAGE
-----
    # narratives only, free, no key, no network beyond the public JSONL
    python scripts/build_prompts.py --reals <dir> --out prompts.csv --source narrative

    # the richer version -- dry run first
    export OPENROUTER_API_KEY=...
    python scripts/build_prompts.py --reals <dir> --out prompts.csv --source both
    python scripts/build_prompts.py --reals <dir> --out prompts.csv --source both --execute
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

from pilot_commercial_apis import sanitise

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
UA = "Mozilla/5.0 (compatible; aigcdet-research/1.0)"

#: Localized Narratives for the Open Images validation split. The train shards
#: are at .../open_images_train_v6_localized_narratives-0000{0..9}-of-00010.jsonl
#: if the pilot ever outgrows validation's ~41k images.
NARR_URL = ("https://storage.googleapis.com/localized-narratives/annotations/"
            "open_images_validation_localized_narratives.jsonl")

#: Deliberately not any of the four generators' labs (§1 above): a dedicated
#: vision-language model from a fifth lab, so the prompt distribution is a
#: constant across §5.2's per-provider comparison rather than a variable.
#:
#: Verified live 2026-08-30 at $0.000093/image with a 512px input. Slugs move:
#: `qwen/qwen3.5-flash` was rejected outright ("not a valid model ID") and
#: `qwen/qwen3.7-flash`, though real and cheaper, returns an empty
#: `choices[0].message` for this request shape. List what is actually available
#: with `GET https://openrouter.ai/api/v1/models` before changing this.
DEFAULT_CAPTIONER = "qwen/qwen3-vl-8b-instruct"

#: Sent at 512px. Captioning does not need more, and input tokens scale with
#: pixels, so this is most of the cost control.
SEND_SHORT_SIDE = 512

SOURCES = ("narrative", "vlm", "both")

#: The §3.5 instruction. `sanitise` still runs over whatever comes back: an
#: instruction to a model is a request, not a guarantee, and the subject here is
#: photographs of real people.
SYSTEM = (
    "You write short prompts for a text-to-image model, from a photograph.\n"
    "Rules, all mandatory:\n"
    "1. NEVER describe a person in a way that could identify them. No names, no "
    "faces, no distinguishing marks, no logos, no readable text, no landmarks. "
    "Say 'a person', 'a child', 'two people' and describe only clothing, pose "
    "and setting.\n"
    "2. Describe what is actually visible. Do not invent objects.\n"
    "3. Be concrete and dense: subject, what it is doing, setting, lighting, "
    "materials, textures, colours. The photograph is cluttered and detailed; the "
    "prompt should be too.\n"
    "4. One paragraph, 30-60 words, no preamble, no quotes, no 'this image'.\n"
    "5. Write it as a photograph, not an illustration."
)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_narratives(wanted: set[str], cache: str) -> dict[str, str]:
    """`ImageID -> caption`, streamed and stopped as soon as `wanted` is covered.

    The file is ~1.1 GB, almost all of it per-word audio timings we do not use,
    so this parses line by line and abandons the stream early rather than
    downloading it. The cache means a second run is instant.
    """
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            got = json.load(fh)
        if wanted <= set(got):
            print(f"  narratives: {len(got)} from cache")
            return got
    print(f"  narratives: streaming until {len(wanted)} ids are covered ...")
    found: dict[str, str] = {}
    req = urllib.request.Request(NARR_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            iid = rec.get("image_id")
            if iid in wanted and iid not in found:
                cap = (rec.get("caption") or "").strip()
                if cap:
                    found[iid] = cap
                if len(found) >= len(wanted):
                    break
    tmp = cache + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(found, fh)
    os.replace(tmp, cache)
    print(f"  narratives: {len(found)}/{len(wanted)} matched")
    return found


def as_data_uri(path: str) -> str:
    """Downscale to `SEND_SHORT_SIDE` and inline as base64. Never upscales."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if min(im.size) > SEND_SHORT_SIDE:
            k = SEND_SHORT_SIDE / min(im.size)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def caption(path: str, narrative: str, model: str, key: str) -> tuple[str, float, str]:
    """(prompt, cost, error). Sends the image, and the narrative when there is one."""
    ask = "Write the prompt for this photograph."
    if narrative:
        ask += ("\n\nA person who looked at this photograph described it as:\n"
                f"\"{narrative}\"\n"
                "Use that as ground truth for what is present, and add the visual "
                "detail it leaves out.")
    payload = {
        "model": model,
        "temperature": 0,          # reproducibility: see §2 of the docstring
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": ask},
                {"type": "image_url", "image_url": {"url": as_data_uri(path)}},
            ]},
        ],
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        CHAT_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return "", 0.0, f"HTTP {e.code}: {e.read()[:180].decode('utf-8', 'replace')}"
    except Exception as e:
        return "", 0.0, f"{type(e).__name__}: {e}"
    try:
        text = out["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        return "", 0.0, "no content in choices[0].message"
    return text, float((out.get("usage") or {}).get("cost") or 0.0), ""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reals", required=True, help="directory of <ImageID>.jpg reals")
    ap.add_argument("--out", required=True, help="prompts CSV to write")
    ap.add_argument("--source", choices=SOURCES, default="both")
    ap.add_argument("--captioner", default=DEFAULT_CAPTIONER)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--execute", action="store_true",
                    help="actually call the captioner. WITHOUT THIS NOTHING IS SPENT.")
    a = ap.parse_args(argv)

    ids = sorted(f[:-4] for f in os.listdir(a.reals) if f.lower().endswith(".jpg"))[:a.n]
    if not ids:
        raise SystemExit(f"no <ImageID>.jpg files in {a.reals}")

    narratives = {}
    if a.source in ("narrative", "both"):
        narratives = fetch_narratives(set(ids), os.path.join(
            os.path.dirname(os.path.abspath(a.out)) or ".", "narratives_cache.json"))

    done = {}
    if os.path.exists(a.out):                      # resume: never re-caption
        with open(a.out, newline="", encoding="utf-8") as fh:
            done = {r["image_id"]: r for r in csv.DictReader(fh)}
        print(f"  resuming: {len(done)} prompts already written")

    todo = [i for i in ids if i not in done]
    needs_model = a.source in ("vlm", "both")
    print(f"\n  {len(ids)} reals, {len(todo)} to build, source={a.source}"
          + (f", captioner={a.captioner}" if needs_model else " (no model, free)"))

    if needs_model and not a.execute:
        print(f"\n  DRY RUN — nothing sent, nothing charged. ~${len(todo) * 0.0001:.4f} "
              f"estimated for {len(todo)} captions.")
        if narratives:
            k = todo[0] if todo and todo[0] in narratives else next(iter(narratives), None)
            if k:
                print(f"\n  narrative for {k}:\n    {narratives[k][:200]}")
        print("\n  Re-run with --execute to build them.")
        return 0

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if needs_model and not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    rows, spent, refused, failed = list(done.values()), 0.0, 0, 0
    for n, iid in enumerate(todo, 1):
        narrative = narratives.get(iid, "")
        if a.source == "narrative":
            prompt, why = sanitise(narrative)
            model_used, err = "", ""
        else:
            # §3.5 first: a narrative naming someone must not be forwarded to
            # the captioner, and an image whose narrative names someone is
            # exactly the row that rule exists for.
            if narrative and sanitise(narrative)[1]:
                refused += 1
                continue
            text, cost, err = caption(os.path.join(a.reals, f"{iid}.jpg"),
                                      narrative if a.source == "both" else "",
                                      a.captioner, key)
            spent += cost
            model_used = a.captioner
            if err:
                failed += 1
                continue
            # The instruction above is a request; this is the enforcement.
            prompt, why = sanitise(text)
        if why:
            refused += 1
            continue
        rows.append({"image_id": iid, "narrative": narrative, "prompt": prompt,
                     "source": a.source, "captioner": model_used,
                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if n % 10 == 0 or n == len(todo):
            print(f"    {n}/{len(todo)}  kept={len(rows)} refused={refused} "
                  f"failed={failed}  ${spent:.4f}", flush=True)

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "narrative", "prompt",
                                           "source", "captioner", "ts"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n  wrote {len(rows)} prompts to {a.out}")
    print(f"  refused by §3.5 or too short: {refused}   captioner errors: {failed}")
    if spent:
        print(f"  captioning cost: ${spent:.4f}")
    if rows:
        print(f"\n  example:\n    {rows[-1]['prompt'][:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
