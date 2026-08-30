"""Buy the task 03 pilot: ~50 images per provider, and answer three questions at once.

`docs/03-commercial-apis-on-open-images.md` §3.2 asks for a small paid pilot
before the bulk buy. This is that pilot, and it is written so one purchase
answers everything the brief needs before the $114 commitment:

1. **The refusal rate**, per provider, measured rather than the guessed 20% in
   `docs/03a` §3. Task 02's reals are portrait-filtered, so the Localized
   Narratives that pair with them describe people, and §3.5 bars prompting for
   identifiable individuals. Refusals are likely and some providers bill them.
2. **The geometry question** in §3.1, which is task 03's genuine open problem.
   An API renders into fixed buckets, so something must close the gap to a
   ~427x640 real. Every purchased image is written out under BOTH
   `encoder_parity` geometries -- `resample` and `crop` -- from the same bytes,
   so the choice is decided on a measurement and costs one buy, not two.
3. **The real cost**, from the provider's own accounting rather than a price
   list. OpenRouter returns `usage.cost` per request; it is recorded per image.

WHY THE RAW BYTES ARE SAVED FIRST AND NEVER OVERWRITTEN
-------------------------------------------------------
`raw/` holds exactly what the provider returned. It is the only artifact here
that costs money, and every derived form is reproducible from it for free. So
the script writes raw first, then derives; a geometry bug is re-run offline
rather than re-bought. Nothing downstream may write into `raw/`.

MONEY
-----
**Dry run is the default.** Without `--execute` this contacts no API, spends
nothing, and prints the plan and the estimate. `--execute` is the only thing
that can charge a card, and it re-prints the estimate and requires
`--yes-spend` on top for a non-interactive run.

Resumes on the raw file's existence, so an interrupted run is restarted with
the same command and re-buys nothing.

PROVIDERS
---------
Most of the roster is reachable through **one** OpenRouter key: its unified
image API serves `openai/gpt-image-2`, `google/gemini-3.1-flash-image`,
`bytedance-seed/seedream-4.5` and others. **Ideogram is not on OpenRouter** and
keeps its own adapter and key.

One caveat to record rather than resolve here: `docs/03` §2 argues that for a
benchmark *"the served output is the thing being measured"*, which is an
argument for first-party endpoints over a router. OpenRouter proxies to the
provider for these models, but if a number looks anomalous for one family,
re-buy a handful direct before believing it.

USAGE
-----
    # what it would cost and what it would send -- no key needed, no spend
    python scripts/pilot_commercial_apis.py --reals <dir> --out <dir> --n 50

    # actually buy
    export OPENROUTER_API_KEY=... IDEOGRAM_API_KEY=...
    python scripts/pilot_commercial_apis.py --reals <dir> --out <dir> --n 50 \\
        --execute --yes-spend
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from PIL import Image

from aigcdet.data.encoder_parity import (
    GEOMETRIES, ParityError, read_profile, save_matched,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/images"
IDEOGRAM_URL = "https://api.ideogram.ai/v1/ideogram-v4/generate"

#: Portrait, natively. Asking for a square and cropping to the reals' ~0.67
#: aspect leaves ~614px of width against reals up to ~700 -- measured at 19% of
#: pairs refused by parity for being too small to use without upscaling. Every
#: adapter below asks for a tall bucket for this reason, not for aesthetics.
ASPECT_RATIO = "2:3"

#: Mid tier, deliberately. `docs/03` §3.4: the cheap tier does not merely cost
#: less, it produces different artefacts, so buying it would benchmark the
#: detector against cheap-tier output rather than against what the graded
#: benchmark contains.
QUALITY = "medium"


@dataclass(frozen=True)
class Provider:
    """One purchasable family. `family` becomes the generator bucket name.

    Per `docs/03` §4, each provider is its own held-out family and they are
    never merged into one `commercial_api` bucket -- the report has to be able
    to say which provider we fail on.
    """

    family: str
    adapter: str            # "openrouter" | "ideogram"
    model: str
    est_price: float        # per image, for the dry-run estimate only
    note: str = ""


#: Four providers, matching NTIRE 2026's held-out composition (`docs/03` §2.1).
#: Prices are the dry-run estimate only; the receipt records what was actually
#: charged, from `usage.cost` where the provider reports it.
PROVIDERS: tuple[Provider, ...] = (
    Provider("openai_gpt_image_2", "openrouter", "openai/gpt-image-2", 0.080,
             "portrait costs more than square: billed per output token"),
    Provider("google_gemini_31_flash_image", "openrouter",
             "google/gemini-3.1-flash-image", 0.067,
             "SynthID on every image, no opt-out -- flag its row (§5.7)"),
    Provider("bytedance_seedream_45", "openrouter",
             "bytedance-seed/seedream-4.5", 0.040,
             "terms not published for the image models -- see dataset_licences.md"),
    Provider("ideogram_40_turbo", "ideogram", "V_4_TURBO", 0.030,
             "edit endpoints share the per-image rate"),
)

#: Spoken-transcript boilerplate. Every Localized Narrative opens with some
#: variant of it, and it is dead weight in a generation prompt -- it describes
#: the act of looking at a picture rather than anything in the picture.
_BOILERPLATE = re.compile(
    r"^\s*(in this (image|picture)[, ]*)?(we|i) can see\s*|"
    r"^\s*in this (image|picture)[, ]*|"
    r"^\s*this (image|picture) (shows|contains)\s*",
    re.IGNORECASE,
)

#: A capitalised word that is not sentence-initial and is not a common
#: sentence-initial artefact. Used only to REFUSE a prompt, never to rewrite
#: one: §3.5 bars prompting for identifiable individuals, and a narrative that
#: names someone is the case that rule exists for. Dropping the row is cheap;
#: silently stripping a name and generating anyway is not.
_PROPER_NOUN = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")


def sanitise(caption: str) -> tuple[str, str]:
    """(prompt, reason_if_refused). Empty reason means the prompt is usable.

    Two jobs, and only the second is a safety rule:

    - Strip the narrative's spoken boilerplate. These captions were transcribed
      from speech, so they open with "In this image I can see ..." every time.
    - Refuse anything carrying a proper noun. `docs/03` §3.5 bars prompting for
      identifiable individuals, and the images behind these narratives are
      photographs of real people.

    **The name check runs on the ORIGINAL, before the boilerplate is stripped,
    and the order is the whole point.** A narrative's subject is usually its
    first content word, so after stripping, "Barack standing at a podium" puts
    the name at position 0 — where it is indistinguishable by capitalisation
    from an ordinary opener like "Shopping trolleys", and where a
    sentence-initial exclusion silently ignores it. Checked against the
    original, the boilerplate guarantees any name sits mid-sentence and is
    caught. This was a live bug, not a hypothetical: the first version checked
    the stripped text and let a named individual through.

    Residual gap, stated rather than papered over: a caption carrying no
    boilerplate at all that opens with a name. Every narrative in this corpus
    carries the boilerplate; a source that does not would need this tightened.
    """
    raw = (caption or "").strip()
    if not raw:
        return "", "empty narrative"
    names = _PROPER_NOUN.findall(raw)
    if names:
        return "", f"proper noun in narrative ({', '.join(sorted(set(names))[:3])}) — §3.5"
    text = _BOILERPLATE.sub("", raw).strip()
    text = text[:1].upper() + text[1:] if text else text
    if len(text) < 15:
        return "", "prompt too short after boilerplate strip"
    return text, ""


@dataclass
class Result:
    ok: bool
    data: bytes = b""
    media_type: str = ""
    cost: float | None = None
    reason: str = ""
    billed: bool = True     # assume a refusal was billed unless we know better


def _post(url: str, payload: dict, headers: dict, timeout: int = 180) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def call_openrouter(model: str, prompt: str, key: str) -> Result:
    """POST /api/v1/images. Returns base64 bytes at `data[0].b64_json`.

    `usage.cost` is the provider's own accounting for the request and is
    recorded rather than the price list -- §3.4 wants the number that was
    actually charged.
    """
    status, raw = _post(OPENROUTER_URL, {
        "model": model, "prompt": prompt, "n": 1,
        "resolution": "1K", "aspect_ratio": ASPECT_RATIO,
        "quality": QUALITY, "output_format": "png",
    }, {"Authorization": f"Bearer {key}"})
    try:
        body = json.loads(raw)
    except ValueError:
        return Result(False, reason=f"HTTP {status}: unparseable body")
    if status != 200:
        msg = (body.get("error") or {}).get("message") or raw[:200].decode("utf-8", "replace")
        # A 4xx from a content filter is a refusal; a 5xx is the provider
        # failing and is usually not billed. The distinction changes the retry
        # budget §3.4 needs, so it is recorded rather than flattened.
        return Result(False, reason=f"HTTP {status}: {msg}", billed=400 <= status < 500)
    items = body.get("data") or []
    if not items or not items[0].get("b64_json"):
        return Result(False, reason="200 with no image in data[0].b64_json")
    return Result(True, base64.b64decode(items[0]["b64_json"]),
                  items[0].get("media_type", "image/png"),
                  (body.get("usage") or {}).get("cost"))


def call_ideogram(model: str, prompt: str, key: str) -> Result:
    """Ideogram is not on OpenRouter, so it keeps its own adapter and key."""
    status, raw = _post(IDEOGRAM_URL, {
        "prompt": prompt, "rendering_speed": "TURBO",
        "aspect_ratio": ASPECT_RATIO.replace(":", "x"), "num_images": 1,
    }, {"Api-Key": key})
    try:
        body = json.loads(raw)
    except ValueError:
        return Result(False, reason=f"HTTP {status}: unparseable body")
    if status != 200:
        return Result(False, reason=f"HTTP {status}: {str(body)[:200]}",
                      billed=400 <= status < 500)
    items = body.get("data") or []
    if not items:
        return Result(False, reason="200 with empty data[]")
    url = items[0].get("url")
    if not url:
        return Result(False, reason="200 with no url in data[0]")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return Result(True, r.read(), "image/png", None)
    except Exception as e:
        return Result(False, reason=f"image fetch failed: {type(e).__name__}: {e}")


ADAPTERS = {"openrouter": call_openrouter, "ideogram": call_ideogram}
KEY_ENV = {"openrouter": "OPENROUTER_API_KEY", "ideogram": "IDEOGRAM_API_KEY"}


def load_pairs(reals_dir: str, attribution: str, n: int) -> list[dict]:
    """(ImageID, real_path, prompt) for rows whose narrative survives §3.5.

    Refused rows are dropped here, before any spend, and counted -- a prompt we
    would not send is not a refusal to charge to the provider's rate.
    """
    with open(attribution, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out, skipped = [], 0
    for r in rows:
        iid = r.get("ImageID") or r.get("image_id")
        real = os.path.join(reals_dir, f"{iid}.jpg")
        if not iid or not os.path.exists(real):
            continue
        prompt, why = sanitise(r.get("caption", ""))
        if why:
            skipped += 1
            continue
        out.append({"image_id": iid, "real": real, "prompt": prompt})
        if len(out) >= n:
            break
    if skipped:
        print(f"  {skipped} narratives refused locally by §3.5 before any spend")
    return out


def derive_geometries(raw_path: str, real_path: str, out_root: str,
                      family: str, image_id: str) -> dict:
    """Write the purchased image under every geometry. Free, and repeatable.

    Both are produced from the same bytes so §3.1's choice is a measurement
    over one purchase rather than two.
    """
    status = {}
    for geometry in GEOMETRIES:
        dst = os.path.join(out_root, f"parity_{geometry}", family, f"{image_id}.jpg")
        try:
            with Image.open(raw_path) as im:
                im.load()
                save_matched(im, dst, read_profile(real_path), geometry)
            status[geometry] = "ok"
        except ParityError as e:
            status[geometry] = str(e)
        except Exception as e:
            status[geometry] = f"{type(e).__name__}: {e}"
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reals", required=True, help="directory of <ImageID>.jpg reals")
    ap.add_argument("--attribution", default=None,
                    help="attribution.csv with ImageID and caption (default: <reals>/../attribution.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50, help="images per provider (§3.2 says ~50)")
    ap.add_argument("--only", default="", help="comma-separated family names to run")
    ap.add_argument("--execute", action="store_true",
                    help="actually call the APIs. WITHOUT THIS NOTHING IS SPENT.")
    ap.add_argument("--yes-spend", action="store_true",
                    help="skip the interactive confirmation (for non-interactive runs)")
    a = ap.parse_args(argv)

    attribution = a.attribution or os.path.join(os.path.dirname(a.reals.rstrip("/")),
                                                "attribution.csv")
    providers = [p for p in PROVIDERS
                 if not a.only or p.family in {s.strip() for s in a.only.split(",")}]
    if not providers:
        raise SystemExit(f"--only matched nothing. Known: {[p.family for p in PROVIDERS]}")

    pairs = load_pairs(a.reals, attribution, a.n)
    if not pairs:
        raise SystemExit(f"no usable pairs from {attribution}")

    print(f"\n  {len(pairs)} pairs x {len(providers)} providers "
          f"= {len(pairs) * len(providers)} images\n")
    print(f"  {'family':<32} {'model':<34} {'est $/img':>9} {'est total':>10}")
    print(f"  {'-'*32} {'-'*34} {'-'*9} {'-'*10}")
    estimate = 0.0
    for p in providers:
        sub = p.est_price * len(pairs)
        estimate += sub
        print(f"  {p.family:<32} {p.model:<34} {p.est_price:>9.3f} {sub:>10.2f}")
    print(f"  {'':<32} {'':<34} {'ESTIMATE':>9} {estimate:>10.2f}")
    for p in providers:
        if p.note:
            print(f"    {p.family}: {p.note}")

    if not a.execute:
        print("\n  DRY RUN — nothing was sent and nothing was charged.")
        print("  Re-run with --execute --yes-spend to buy.\n")
        print(f"  example prompt: {pairs[0]['prompt'][:150]}")
        return 0

    missing = sorted({KEY_ENV[p.adapter] for p in providers
                      if not os.environ.get(KEY_ENV[p.adapter])})
    if missing:
        raise SystemExit(f"missing key(s) in the environment: {', '.join(missing)}")

    if not a.yes_spend:
        if input(f"\n  This will charge about ${estimate:.2f}. Type 'spend' to continue: "
                 ).strip() != "spend":
            print("  aborted; nothing spent.")
            return 1

    os.makedirs(a.out, exist_ok=True)
    receipt_path = os.path.join(a.out, "pilot_receipt.jsonl")
    spent, counts = 0.0, {}

    for p in providers:
        key = os.environ[KEY_ENV[p.adapter]]
        call = ADAPTERS[p.adapter]
        raw_dir = os.path.join(a.out, "raw", p.family)
        os.makedirs(raw_dir, exist_ok=True)
        ok = refused = failed = 0
        t0 = time.time()
        print(f"\n  {p.family} ...", flush=True)

        for i, pair in enumerate(pairs):
            raw_path = os.path.join(raw_dir, f"{pair['image_id']}.png")
            if os.path.exists(raw_path):        # resume: never re-buy
                ok += 1
                continue
            res = call(p.model, pair["prompt"], key)
            rec = {"family": p.family, "model": p.model, "image_id": pair["image_id"],
                   "ok": res.ok, "cost": res.cost, "reason": res.reason,
                   "billed": res.billed if not res.ok else True,
                   "prompt": pair["prompt"], "ts": time.time()}
            if res.ok:
                tmp = raw_path + ".part"
                with open(tmp, "wb") as fh:
                    fh.write(res.data)
                os.replace(tmp, raw_path)
                rec["geometry"] = derive_geometries(raw_path, pair["real"], a.out,
                                                    p.family, pair["image_id"])
                ok += 1
            elif res.billed:
                refused += 1
            else:
                failed += 1
            if res.cost:
                spent += res.cost
            with open(receipt_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(pairs)}  ok={ok} refused={refused} "
                      f"failed={failed}  ${spent:.2f}", flush=True)

        counts[p.family] = (ok, refused, failed, time.time() - t0)

    print(f"\n  {'family':<32} {'ok':>5} {'refused':>8} {'failed':>7} {'refusal %':>10}")
    print(f"  {'-'*32} {'-'*5} {'-'*8} {'-'*7} {'-'*10}")
    for fam, (ok, refused, failed, _) in counts.items():
        tried = ok + refused
        print(f"  {fam:<32} {ok:>5} {refused:>8} {failed:>7} "
              f"{(refused / tried * 100 if tried else 0):>9.1f}%")

    print(f"\n  actually charged (where reported): ${spent:.2f} "
          f"against an estimate of ${estimate:.2f}")
    print(f"  receipt: {receipt_path}")
    print(f"\n  Next: run the gate over BOTH geometries and compare —\n"
          f"    for g in {' '.join(GEOMETRIES)}; do\n"
          f"      python scripts/prove_encoder_parity.py --reals {a.reals} \\\n"
          f"        --generated {a.out}/raw/<family> --out {a.out}/gate_$g --n {a.n}\n"
          f"    done\n"
          f"  then put the measured refusal rates above into docs/03a §3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
