#!/usr/bin/env python
"""Generate the AI-OV7 pairs: open-weight generators over Open Images V7 reals.

    python scripts/generate_ov7.py --total 10000
    python scripts/generate_ov7.py --total 14 --out data/ov7_smoke --smoke

Every fake is generated from one real, at that real's own MCU-aligned crop
dimensions, and saved through that real's own JPEG quantisation tables. See
`src/aigcdet/generate/` for why each of those is load-bearing, and `docs/02`
§5 for the gate this output has to clear before anything trains on it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from aigcdet.generate import registry
from aigcdet.generate.captions import caption_pool
from aigcdet.generate.pool import build_pool, select
from aigcdet.generate.run import run

DEFAULT_SEED = 20260830
PORTRAIT = "/mnt/berstorage/techjam/open_images/portrait"
ATTRIBUTION = "/mnt/berstorage/techjam/open_images/attribution.csv"


def used_elsewhere(rows_dir: Path, sel: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Reals this run would generate for that ALREADY have a fake elsewhere.

    `pool.select` promises one real produces exactly one fake for exactly one
    family, and it keeps that promise within a single deal. Across two deals
    it cannot: the family pattern is built from the suite's shares, so running
    a different suite re-deals every real in the block. A real that is
    `sdxl_t2i` under one suite can come out `sana1600m_t2i` under the next,
    and nothing downstream would object -- `_done_ids` is per family, so the
    second fake is simply generated. The corpus then carries one scene twice
    on the generated side against one real, and any content the head
    memorises from it arrives with a label prior attached.

    Returns (image_id, family that already owns it, family newly dealt it).
    Re-selecting the same real for the SAME family is a resume, not a clash.
    """
    owner: dict[str, str] = {}
    for path in sorted(Path(rows_dir).glob("rows_*.jsonl")):
        fam = path.name[len("rows_"):-len(".jsonl")]
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                owner[json.loads(line)["image_id"]] = fam
            except json.JSONDecodeError:
                continue  # truncated final line; `_done_ids` tolerates it too
    return [(i, owner[i], f) for i, f in zip(sel["image_id"], sel["family"])
            if owner.get(i, f) != f]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, required=True,
                    help="generated images across the whole suite")
    ap.add_argument("--out", default="data/raw_ov7_src",
                    help="corpus root; families land at "
                         "<out>/open_images_v7/<family>/")
    ap.add_argument("--rows-dir", default=None,
                    help="resumable per-family jsonl (default <out>/_rows)")
    ap.add_argument("--pool", default="data/ov7_pool.parquet")
    ap.add_argument("--captions", default="data/ov7_captions.parquet")
    ap.add_argument("--portrait-dir", default=PORTRAIT)
    ap.add_argument("--attribution", default=ATTRIBUTION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suite", default="ov7", choices=sorted(registry.SUITES),
                    help="which suite to run. 'ov7_lineage' is the additive "
                         "supplement: two new decoder lineages over reals the "
                         "first run never touched. Give it a --shard clear of "
                         "the reals already used, or the family deal is "
                         "recomputed and a real that already has a fake gets "
                         "a second one from another family.")
    ap.add_argument("--families", default=None,
                    help="comma-separated subset, for a smoke run")
    ap.add_argument("--smoke", action="store_true",
                    help="equal count per family instead of by share, and "
                         "refuse to write into the real corpus root")
    args = ap.parse_args(argv)

    # Resolved exactly once. The branch this was ported from reassigned its
    # output root to None further down and wrote the corpus to the CWD.
    out_root = Path(args.out).resolve()
    rows_dir = Path(args.rows_dir).resolve() if args.rows_dir else out_root / "_rows"
    if args.smoke and out_root == Path("data/raw_ov7_src").resolve():
        ap.error("--smoke writes throwaway images; give it its own --out")
    if shutil.which("jpegtran") is None:
        ap.error("jpegtran is not on PATH. It is required, not optional: it is "
                 "the only way to crop a real without re-encoding it. "
                 "Debian/Ubuntu: libjpeg-turbo-progs.")

    suite = dict(registry.SUITES[args.suite])
    corpus = registry.corpus_of(args.suite)
    if args.families:
        want = [f.strip() for f in args.families.split(",")]
        unknown = set(want) - set(suite)
        if unknown:
            ap.error(f"unknown families {sorted(unknown)}; "
                     f"suite has {sorted(suite)}")
        suite = {k: suite[k] for k in want}

    if args.smoke:
        per = max(1, args.total // len(suite))
        counts = {k: per for k in suite}
    else:
        registry.validate_suite(suite, corpus=corpus)
        counts = registry.resolve_suite(args.total, suite, corpus=corpus)
    print(f"[suite] {args.suite}: {counts}  (total {sum(counts.values())})",
          flush=True)

    pool_path = Path(args.pool)
    if pool_path.exists():
        pool = pd.read_parquet(pool_path)
    else:
        print(f"[pool] probing {args.portrait_dir} ...", flush=True)
        pool = build_pool(args.portrait_dir, args.attribution)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool.to_parquet(pool_path, index=False)
    print(f"[pool] {int(pool.eligible.sum())} eligible of {len(pool)}", flush=True)

    # Shares come from the suite, not from `counts`, so that raising --total
    # later resumes the same assignment instead of reshuffling it.
    shares = {k: v.share for k, v in suite.items()}
    if args.smoke or args.families:
        tot = sum(shares.values())
        shares = {k: v / tot for k, v in shares.items()}
    sel = select(pool, counts, seed=args.seed, shard=args.shard,
                 n_shards=args.n_shards, shares=shares)
    print(f"[select] {len(sel)} reals, disjoint across "
          f"{sel.family.nunique()} families", flush=True)

    clashes = used_elsewhere(rows_dir, sel)
    if clashes:
        head = ", ".join(f"{i} ({was} -> {now})" for i, was, now in clashes[:5])
        ap.error(
            f"{len(clashes)} of {len(sel)} reals already have a fake under a "
            f"different family: {head}"
            + (" ..." if len(clashes) > 5 else "")
            + f". One real produces one fake for one family; generating these "
              f"would put a scene in the corpus twice on the generated side "
              f"against one real. This is what happens when a second suite "
              f"re-deals reals the first one used -- run the supplement on a "
              f"--shard clear of them (--shard 1 --n-shards 5 starts at "
              f"position 10925, past the 0-9999 the ov7 suite consumed).")

    caps = caption_pool(dict(zip(sel["image_id"], sel["path"])), args.captions,
                        device=args.device)
    caps = dict(zip(caps["image_id"], caps["caption"]))
    missing = [i for i in sel["image_id"] if not caps.get(i)]
    if missing:
        print(f"[captions] {len(missing)} reals have no caption and will be "
              f"skipped rather than generated on an empty prompt", flush=True)

    stats = run(sel, caps, out_root, rows_dir, args.seed,
                suite=suite, corpus=corpus, device=args.device)

    ok = sum(s["ok"] for s in stats)
    failed = sum(s["failed"] for s in stats)
    print(f"\n[total] generated {ok}, failed {failed}", flush=True)
    for s in stats:
        if s["reasons"]:
            print(f"  {s['family']} failures: {s['reasons']}", flush=True)
    (rows_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
