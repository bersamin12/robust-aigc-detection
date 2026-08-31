#!/usr/bin/env python3
"""a3 / a4 / aF / a4both on the d8 tower, read on OV7 and on WildFake.

The control is `a3` retrained HERE, on cached d8 features -- not the 0.7054
from the live-pixel d8 run. Those two differ in the training path (live tower
forward in float32 against cached float16 features), and reading a4 against the
live number would fold that difference into the recon branch's effect. The
ablation is only interpretable inside the cached family.
"""
from __future__ import annotations
import argparse, json, os
from aigcdet.eval.errors import (
    SELECTION_METRIC, SELECTION_TARGET_FPR, heldout_robust_tpr)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector


def load_head(ckpt, device):
    """Rebuild a trained head through the ONE canonical loader.

    This used to hand-roll the reconstruction and pass `use_recon_vq=` and
    `use_freq=` straight to `Detector`, which accepts neither -- so every aux
    rung died on a TypeError before a single score was computed. The real
    error was having a second copy at all: `Detector` takes a single
    `recon_dim`, and turning three flags into that one width is exactly what
    `bank.aux_width` exists to do (its docstring says so: "`train_rung` and
    `load_detector` compute it identically"). `load_detector` is the only
    function that should consume it, and eight other scripts already do.
    """
    model, ck = load_detector(ckpt, device=device)
    return model, ck["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", required=True,
                    metavar="NAME=CONFIG:CHECKPOINT")
    ap.add_argument("--eval-bank", action="append", required=True,
                    metavar="LABEL=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rungs = {}
    for s in a.rung:
        n, rest = s.split("=", 1); cfg_path, ckpt = rest.rsplit(":", 1)
        rungs[n] = (cfg_path, ckpt)
    evals = {}
    for s in a.eval_bank:
        lbl, path = s.split("=", 1); evals[lbl] = path

    report = {"metric": SELECTION_METRIC, "target_fpr": SELECTION_TARGET_FPR,
              "tower": "d8", "note": (
                  "a3 here is retrained on CACHED d8 features and is the only "
                  "valid control for a4/aF; the live-pixel d8 run reads 0.7054 "
                  "on OV7 by a different training path."),
              "results": {}}

    for lbl, path in evals.items():
        bank = FeatureBank.open(path)
        splits = bank.meta["split"].to_numpy()
        print(f"\n=== {lbl}  ({path}, tower "
              f"{str(bank.config.get('tower_sha256'))[:12]})")
        report["results"][lbl] = {"eval_bank": path,
                                  "tower_sha256": bank.config.get("tower_sha256"),
                                  "rungs": {}}
        base = None
        for n, (cfg_path, ckpt) in rungs.items():
            if not os.path.exists(ckpt):
                print(f"  {n}: checkpoint absent, skipped"); continue
            head, cfg = load_head(ckpt, a.device)
            df = score_grid(head, bank,
                            use_recon=cfg.get("use_recon", False),
                            use_recon_vq=cfg.get("use_recon_vq", False),
                            use_freq=cfg.get("use_freq", False),
                            device=a.device)
            v = heldout_robust_tpr(df, splits, target_fpr=SELECTION_TARGET_FPR)
            flags = "+".join([k.replace("use_", "") for k in
                              ("use_recon", "use_recon_vq", "use_freq")
                              if cfg.get(k)]) or "backbone only"
            if n == "a3":
                base = v
            delta = "" if base is None or n == "a3" else f"  {v - base:+.4f} vs a3"
            print(f"  {n:8s} {flags:22s} {SELECTION_METRIC}={v:.4f}{delta}")
            report["results"][lbl]["rungs"][n] = {
                SELECTION_METRIC: v, "config": cfg_path, "checkpoint": ckpt,
                "use_recon": bool(cfg.get("use_recon")),
                "use_recon_vq": bool(cfg.get("use_recon_vq")),
                "use_freq": bool(cfg.get("use_freq"))}
        if base is not None:
            report["results"][lbl]["deltas_vs_a3"] = {
                n: r[SELECTION_METRIC] - base
                for n, r in report["results"][lbl]["rungs"].items() if n != "a3"}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
