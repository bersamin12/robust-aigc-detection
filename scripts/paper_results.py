#!/usr/bin/env python3
"""The squish paper's gates and results tables, computed from parquets only.

    python scripts/paper_results.py g1    # after the proxy arms; exit 3 = gate failed
    python scripts/paper_results.py full  # after full training + scoring

Every number written here comes out of a per-row parquet through
`paired_bootstrap.py`; nothing is typed by hand. The gates are the ones
pre-registered in docs/ntire/paper_prereg.md, which was committed before any
arm they judge had a number. They are evaluated here exactly as written
there -- if this file and that one ever disagree, that one wins and this one
is the bug.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_bootstrap as pb  # noqa: E402

SEEDS = (1, 2, 3)
#: Native-short-side buckets of the crop probe (docs/ntire/crop_probe.md).
BUCKETS = ((80, 336), (352, 512), (528, 736), (752, 4016))
EXT_BASES = ("ext_aigi_socialrf", "ext_aigi_community", "ext_synthbuster")
EXT_VARIANTS = ("", "_jpeg95", "_deg")


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


def boot(models: dict[str, str], baseline: str, n_boot: int, title: str) -> dict:
    """{name: parquet} -> paired bootstrap result (with its markdown)."""
    frames = [pb.load(n, p, None) for n, p in models.items()]
    df = pb.join(frames)
    res = pb.paired_bootstrap(df, list(models), baseline, n_boot=n_boot)
    res["inputs"] = models
    res["markdown"] = pb.to_markdown(res, baseline, title)
    return res


# ---------------------------------------------------------------------------
# G1: the proxy gate
# ---------------------------------------------------------------------------

def g1(scores: str, n_boot: int) -> tuple[dict, bool]:
    per_seed = {}
    for i in SEEDS:
        per_seed[f"s{i}"] = {}
        for split in ("ntire_val_hard", "ntire_val"):
            per_seed[f"s{i}"][split] = boot(
                {"crop": f"{scores}/abl_crop_s{i}_{split}.parquet",
                 "squish": f"{scores}/abl_squish_s{i}_{split}.parquet"},
                "crop", n_boot, f"seed {i}, {split}")
    hard = [per_seed[f"s{i}"]["ntire_val_hard"]["vs_baseline"]["squish"]
            for i in SEEDS]
    rob = [h["robust_auc"] for h in hard]
    cln = [h["clean_auc"] for h in hard]
    mean_rob = float(np.mean([r["diff"] for r in rob]))
    mean_cln = float(np.mean([c["diff"] for c in cln]))
    n_pos_ci = int(sum(r["ci"][0] > 0 for r in rob))
    checks = {
        "mean_robust_gain_gt_0.01": mean_rob > 0.01,
        "ci_above_zero_in_at_least_2_of_3": n_pos_ci >= 2,
        "mean_clean_loss_at_most_0.005": mean_cln >= -0.005,
    }
    passed = all(checks.values())
    return {"gate": "G1", "commit": commit(), "passed": passed,
            "checks": checks, "mean_robust_diff": mean_rob,
            "mean_clean_diff": mean_cln, "n_seed_ci_above_zero": n_pos_ci,
            "per_seed": per_seed}, passed


def g1_markdown(r: dict) -> str:
    lines = ["# G1 -- squish vs crop, proxy protocol, three seeds", "",
             "Pre-registered in `docs/ntire/paper_prereg.md`. Proxy: one "
             "dinov2regl224 tower, depth 24, 1 epoch over 25% of train, bank "
             "`ntire_meta`, mean pool, cosine + SWA, final weights. Each seed "
             "pair shares its training subset, head init and batch order; only "
             "`--canon-mode` differs.", "",
             "| seed | crop robust | squish robust | diff [95% CI] | crop clean |"
             " squish clean | diff [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for s, d in r["per_seed"].items():
        h = d["ntire_val_hard"]
        m, v = h["models"], h["vs_baseline"]["squish"]
        lines.append(
            f"| {s} | {m['crop']['robust_auc']['value']:.4f} | "
            f"{m['squish']['robust_auc']['value']:.4f} | "
            f"{v['robust_auc']['diff']:+.4f} [{v['robust_auc']['ci'][0]:+.4f}, "
            f"{v['robust_auc']['ci'][1]:+.4f}] | "
            f"{m['crop']['clean_auc']['value']:.4f} | "
            f"{m['squish']['clean_auc']['value']:.4f} | "
            f"{v['clean_auc']['diff']:+.4f} [{v['clean_auc']['ci'][0]:+.4f}, "
            f"{v['clean_auc']['ci'][1]:+.4f}] |")
    lines += ["", f"Mean robust diff {r['mean_robust_diff']:+.4f} (gate > +0.0100); "
              f"seeds with CI above zero {r['n_seed_ci_above_zero']}/3 (gate >= 2); "
              f"mean clean diff {r['mean_clean_diff']:+.4f} (gate >= -0.0050).", "",
              f"**G1 {'PASSED' if r['passed'] else 'FAILED'}** "
              + ("-- full training proceeds." if r["passed"] else
                 "-- the pipeline stops before full training, as registered."),
              "", f"Commit `{r['commit']}`.", "", "## ntire_val (secondary)", ""]
    for s, d in r["per_seed"].items():
        lines.append(d["ntire_val"]["markdown"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# full: G2, G3 and the secondary tables
# ---------------------------------------------------------------------------

def _fused(fusion_dir: str, tag: str, split: str) -> str:
    return f"{fusion_dir}/fusion_{tag}_{split}_fused.parquet"


def fov_table(parquets: dict[str, str], manifest: str) -> str:
    m = pd.read_parquet(manifest, columns=["path", "width", "height"])
    rows = ["| model | " + " | ".join(f"{lo}-{hi}px" for lo, hi in BUCKETS)
            + " |", "|---|" + "---|" * len(BUCKETS)]
    for name, p in parquets.items():
        d = pb.load(name, p, None).merge(m, on="path", how="left")
        if d["width"].isna().any():
            raise SystemExit(f"{p}: {int(d['width'].isna().sum())} rows not in {manifest}")
        short = d[["width", "height"]].min(axis=1).to_numpy()
        y = d["label"].to_numpy()
        dist = d["is_distorted"].to_numpy()
        cells = []
        for lo, hi in BUCKETS:
            k = (short >= lo) & (short <= hi) & (dist == 1)
            cells.append(f"{pb.auc(y[k], d[name].to_numpy()[k]):.4f} (n={int(k.sum())})")
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def full(scores: str, fusion_dir: str, picks: dict, n_boot: int,
         manifest: str, ext_prov: str) -> dict:
    out = {"commit": commit(), "picks": picks, "gates": {}, "tables": {}}

    # G2: fused, hard-val, z and w fitted on ntire_val alone for BOTH policies
    g2 = boot({"crop": _fused(fusion_dir, "crop", "ntire_val_hard"),
               "squish": _fused(fusion_dir, "squish", "ntire_val_hard")},
              "crop", n_boot, "G2 -- fused, ntire_val_hard")
    v = g2["vs_baseline"]["squish"]["robust_auc"]
    out["gates"]["G2"] = {"passed": bool(v["diff"] > 0 and v["ci"][0] > 0),
                          "robust_diff": v["diff"], "ci": v["ci"]}
    out["tables"]["G2"] = g2

    # G3: each external set, pooled AUC, the files as distributed
    out["gates"]["G3"] = {}
    for base in EXT_BASES:
        for var in EXT_VARIANTS:
            sp = base + var
            pc, ps = _fused(fusion_dir, "crop", sp), _fused(fusion_dir, "squish", sp)
            if not (os.path.exists(pc) and os.path.exists(ps)):
                out["tables"][sp] = {"missing": [p for p in (pc, ps)
                                                 if not os.path.exists(p)]}
                continue
            r = boot({"crop": pc, "squish": ps}, "crop", n_boot, f"fused, {sp}")
            out["tables"][sp] = r
            if var == "":
                d = r["vs_baseline"]["squish"]["auc"]
                out["gates"]["G3"][base] = {
                    "passed": bool(d["diff"] > 0 and d["ci"][0] > 0),
                    "auc_diff": d["diff"], "ci": d["ci"]}

    # Secondary: per branch on hard-val -- crop as trained, crop checkpoint
    # presented whole at test time, and trained under squish.
    for br, crop_prefix, probe in (("A", "outputs/scores/A2", "outputs/scores/probe_A2"),
                                   ("B", "outputs/scores/B2swa", "outputs/scores/probe_B2swa")):
        sq = picks[br]
        out["tables"][f"branch_{br}"] = boot(
            {"crop": f"{crop_prefix}_ntire_val_hard.parquet",
             "crop_testtime_squish": f"{probe}_squish_ntire_val_hard.parquet",
             "trained_squish": f"{sq}_ntire_val_hard.parquet"},
            "crop", n_boot, f"branch {br}, ntire_val_hard")

    out["tables"]["fov_buckets_md"] = fov_table(
        {"crop": _fused(fusion_dir, "crop", "ntire_val_hard"),
         "squish": _fused(fusion_dir, "squish", "ntire_val_hard")}, manifest)
    if os.path.exists(ext_prov):
        out["external_sets"] = json.load(open(ext_prov))
    return out


def full_markdown(r: dict) -> str:
    L = ["# Squish paper -- results", "",
         f"Commit `{r['commit']}`. Gates as registered in "
         "`docs/ntire/paper_prereg.md`. Fusion z-scores and weight are fitted on "
         "`ntire_val` alone for both policies; every other split is a readout.",
         "", "## Gates", ""]
    g2 = r["gates"]["G2"]
    L.append(f"- **G2 {'PASSED' if g2['passed'] else 'FAILED'}**: fused robust "
             f"AUC on ntire_val_hard, squish - crop = {g2['robust_diff']:+.4f} "
             f"[{g2['ci'][0]:+.4f}, {g2['ci'][1]:+.4f}]")
    for base, g in r["gates"]["G3"].items():
        L.append(f"- **G3 {base}: {'PASSED' if g['passed'] else 'FAILED'}**: "
                 f"pooled AUC squish - crop = {g['auc_diff']:+.4f} "
                 f"[{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]")
    missing = [b for b in EXT_BASES if b not in r["gates"]["G3"]]
    if missing:
        L.append(f"- G3 not evaluated (set absent): {missing}")
    L += ["", "## G2 -- ntire_val_hard, fused", "", r["tables"]["G2"]["markdown"],
          "", "## Robust AUC by native short side (fused, distorted rows)", "",
          r["tables"]["fov_buckets_md"], "",
          "## External benchmarks, fused", "",
          "`ext_X` = files as distributed; `_jpeg95` = every file re-encoded at "
          "JPEG q95 (format equalised); `_deg` = one seeded ntire-pool chain per "
          "file on the native image (our synthesis, not the organisers').", ""]
    ext = r.get("external_sets", {}).get("sets", {})
    if ext:
        L += ["| set | n | real/fake | short p50 | >512px | short-side-vs-label AUC "
              "| format-vs-label AUC |", "|---|---|---|---|---|---|---|"]
        for sp, p in ext.items():
            L.append(f"| `{sp}` | {p['n']} | {p['n_real']}/{p['n_fake']} | "
                     f"{p['short_side']['p50']:.0f} | {p['frac_above_512']:.0%} | "
                     f"{p['shortside_vs_label_auc']:.3f} | "
                     f"{p['format_vs_label_auc']:.3f} |")
        L.append("")
    for base in EXT_BASES:
        for var in EXT_VARIANTS:
            t = r["tables"].get(base + var)
            if t is None:
                continue
            L.append(t["markdown"] if "markdown" in t
                     else f"`{base + var}`: missing {t['missing']}")
            L.append("")
    L += ["## Per branch, ntire_val_hard: crop, crop presented whole at test time, "
          "trained under squish", ""]
    for br in ("A", "B"):
        L += [r["tables"][f"branch_{br}"]["markdown"], ""]
    L += ["Picks (final vs SWA, chosen on ntire_val alone): "
          + ", ".join(f"{k} -> `{v}`" for k, v in r["picks"].items())]
    return "\n".join(L) + "\n"


def _strip_md(r):
    """json without the duplicated markdown blobs."""
    if isinstance(r, dict):
        return {k: _strip_md(v) for k, v in r.items() if k != "markdown"}
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("g1", "full"))
    ap.add_argument("--scores", default="outputs/scores/paper")
    ap.add_argument("--fusion-dir", default="docs/ntire/paper")
    ap.add_argument("--picks", default="docs/ntire/paper/picks.json")
    ap.add_argument("--manifest", default="data/manifest_ntire.parquet")
    ap.add_argument("--ext-provenance", default="docs/ntire/external_sets.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="stem; .json and .md are written")
    a = ap.parse_args(argv)

    if a.what == "g1":
        r, ok = g1(a.scores, a.n_boot)
        stem = a.out or "docs/ntire/paper/squish_proxy_gate"
        md = g1_markdown(r)
    else:
        picks = json.load(open(a.picks))
        r = full(a.scores, a.fusion_dir, picks, a.n_boot, a.manifest,
                 a.ext_provenance)
        ok = True
        stem = a.out or "docs/ntire/paper/paper_results"
        md = full_markdown(r)
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    with open(stem + ".json", "w") as f:
        json.dump(_strip_md(r), f, indent=2)
    with open(stem + ".md", "w") as f:
        f.write(md)
    print(md)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
