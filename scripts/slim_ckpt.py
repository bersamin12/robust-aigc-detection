#!/usr/bin/env python3
"""Drop the optimiser and RNG state from a finished checkpoint.

A full-depth checkpoint carries the fp32 tower, the AdamW moments (twice the
tower again) and, once SWA has started, a second copy of every weight. For a
run that is over, only the weights are ever read again -- by
`scripts/score_plan_splits.py`, `scripts/predict.py` and the fusion step --
so the per-epoch snapshots can shed two thirds of their size.

Writes `<name>_slim.pt` BESIDE the original and never replaces it: a slim
checkpoint cannot be resumed from, and the resume logic looks for
`checkpoint.pt` by name. Delete the fat one yourself, once the run is done.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

DROP = ("optimizer_state_dict", "sampler_rng_state", "torch_rng_state")


def slim(path: str, out: str | None = None, keep_swa: bool = True) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in DROP:
        ck.pop(k, None)
    if not keep_swa:
        for k in ("swa_state_dict", "swa_tower_state_dict", "swa_tower_state_dicts"):
            ck[k] = None
        ck["swa_n"] = 0
    ck["slimmed"] = {"dropped": list(DROP), "from": os.path.abspath(path),
                     "keep_swa": keep_swa}
    out = out or os.path.splitext(path)[0] + "_slim.pt"
    tmp = out + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--no-swa", action="store_true",
                    help="also drop the SWA copy (halves the file again)")
    ap.add_argument("--delete-original", action="store_true",
                    help="remove the fat checkpoint after the slim one is "
                         "written; refused for a file named checkpoint.pt")
    a = ap.parse_args(argv)
    for p in a.ckpt:
        before = os.path.getsize(p)
        out = slim(p, keep_swa=not a.no_swa)
        after = os.path.getsize(out)
        print(f"{p}: {before / 1e9:.2f} GB -> {out}: {after / 1e9:.2f} GB")
        if a.delete_original:
            if os.path.basename(p) == "checkpoint.pt":
                print("  keeping checkpoint.pt (resume reads it by name)")
            else:
                os.remove(p)
                print("  removed original")
    return 0


if __name__ == "__main__":
    sys.exit(main())
