"""Stage B CLI.

    python scripts/train_rung.py --config configs/rungs/a3.yaml \
        --bank banks/dinov3l --device cuda
"""
from __future__ import annotations

import argparse

import yaml

from aigcdet.train.train_head import RungConfig, train_rung


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", default="outputs/rungs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--manifest", default=None,
                     help="frozen manifest.parquet; when given, verifies the bank is "
                          "still positionally aligned with it before training")
    a = ap.parse_args()

    with open(a.config) as f:
        raw = yaml.safe_load(f)
    cfg = RungConfig(bank_dir=a.bank, out_dir=a.out, device=a.device, seed=a.seed,
                      manifest_path=a.manifest, **raw)
    res = train_rung(cfg)
    print(f"{cfg.name}: val_auc={res['val_auc']:.4f} "
          f"mean_views={res['val_auc_mean_views']:.4f} -> {res['checkpoint']}")


if __name__ == "__main__":
    main()
