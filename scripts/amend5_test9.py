#!/usr/bin/env python3
"""Amendment 5, Test 9, step 2: a learned combiner over the nine `grid9` windows.

    python scripts/amend5_test9.py

Question. The paper says "nine native windows barely help": averaging the
logits of nine 200 px windows (`grid9`) is far below squish. Is that a
failure of the windows, or only of plain averaging? A learned combiner that
can weight the informative windows is the strongest version of the
multi-window alternative.

Registered run (paper_prereg.md, amendment 5, Test 9):
  - inputs: the pooled features (the head's input) of each grid9 window of
    frozen Branch A (A2) and Branch B (B2swa), from amend5_test9_features.py;
  - model: gated-attention MIL head (Ilse et al. 2018), hidden 128,
    dropout 0.1, one per branch;
  - fit on ntire_val, 90% fit / 10% early stopping, seed 20261019;
  - evaluate on ntire_val_hard (a fit split for squish) against squish.

    Pass  robust AUC (squish - combiner) CI above zero on both branches
    Tie   the CI includes zero
    Fail  the combiner's CI is above squish's

Choices the registration leaves open, fixed here before the first fit:
  - the gated attention reads the window features directly (Ilse et al.'s
    "embedding" is the frozen tower); the bag embedding is the
    attention-weighted mean of window features, then one linear layer gives
    the logit;
  - features are standardised with the mean / std of the fit windows;
  - AdamW, lr 1e-3, weight decay 1e-4, batch 64 bags, at most 50 epochs;
    early stopping on the 10% split's AUC, patience 5 epochs, best epoch
    kept;
  - the 90/10 split is stratified by label x is_distorted.

Writes docs/ntire/amend5/test9_mil-combiner.json and _fig.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc, paired_bootstrap  # noqa: E402
from paper_analyses import SEED, commit  # noqa: E402

OUT_DIR = "docs/ntire/amend5"
STEM = f"{OUT_DIR}/test9_mil-combiner"
FEATURE_DIR = "/scratch/dion_r13/dion_extensions/outputs/test9"
BRANCHES = {"A": "A2", "B": "B2swa"}
FIT_SPLIT, EVAL_SPLIT = "ntire_val", "ntire_val_hard"
SPLIT_SEED = 20261019
HIDDEN, DROPOUT = 128, 0.1
LR, WEIGHT_DECAY, BATCH_BAGS, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 64, 50, 5
N_BOOT = 2000


class GatedAttentionMIL(nn.Module):
    """Gated-attention MIL pooling (Ilse et al. 2018, eq. 9) and a linear classifier.

    For a bag of window features h_1..h_K:
        a_k = w^T (tanh(V h_k) * sigmoid(U h_k)),   softmax over the K windows
        z   = sum_k a_k h_k
        logit = c^T z + b
    Padding windows (bags have 1-9 windows) are masked out of the softmax.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.attention_tanh = nn.Linear(dim, HIDDEN)
        self.attention_gate = nn.Linear(dim, HIDDEN)
        self.attention_score = nn.Linear(HIDDEN, 1)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, windows: torch.Tensor, valid: torch.Tensor):
        """windows [B, K, D], valid [B, K] bool -> (logit [B], attention [B, K])."""
        h = self.dropout(windows)
        gated = torch.tanh(self.attention_tanh(h)) * torch.sigmoid(self.attention_gate(h))
        scores = self.attention_score(gated).squeeze(-1)
        scores = scores.masked_fill(~valid, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        bag = (attention.unsqueeze(-1) * h).sum(dim=1)
        return self.classifier(bag).squeeze(-1), attention


def load_bags(tag: str, split: str) -> dict:
    """Window features as a padded [n_images, 9, D] array plus per-image metadata."""
    assert split != "ntire_test", "Test 9 never reads ntire_test"
    # Read each array out of the .npz ONCE: indexing z["features"] inside the
    # loop would re-read the whole array from disk on every access.
    z = np.load(f"{FEATURE_DIR}/grid9_features_{tag}_{split}.npz", allow_pickle=True)
    features, window_logit, offsets = z["features"], z["window_logit"], z["offsets"]
    counts = np.diff(offsets)
    n_images, dim = len(counts), features.shape[1]
    padded = np.zeros((n_images, counts.max(), dim), dtype=np.float32)
    valid = np.zeros((n_images, counts.max()), dtype=bool)
    grid9_mean_logit = np.empty(n_images)
    for i in range(n_images):
        rows = slice(offsets[i], offsets[i + 1])
        padded[i, :counts[i]] = features[rows]
        valid[i, :counts[i]] = True
        grid9_mean_logit[i] = window_logit[rows].mean()
    meta = pd.DataFrame({"path": z["path"], "label": z["label"].astype(int),
                         "is_distorted": z["is_distorted"].astype(int),
                         "width": z["width"], "height": z["height"], "n_windows": counts,
                         "grid9": grid9_mean_logit})
    return {"windows": padded, "valid": valid, "meta": meta}


def stratified_split(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """90% fit / 10% early-stop indices, stratified by label x is_distorted."""
    rng = np.random.default_rng(SPLIT_SEED)
    fit_idx, stop_idx = [], []
    for _, group in meta.groupby(["label", "is_distorted"]):
        idx = rng.permutation(group.index.to_numpy())
        n_stop = int(round(0.10 * len(idx)))
        stop_idx.append(idx[:n_stop])
        fit_idx.append(idx[n_stop:])
    return np.sort(np.concatenate(fit_idx)), np.sort(np.concatenate(stop_idx))


def predict(model, windows, valid, device) -> np.ndarray:
    """Combiner logits for every bag, in eval mode, in chunks."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), 512):
            w = torch.from_numpy(windows[i:i + 512]).to(device)
            v = torch.from_numpy(valid[i:i + 512]).to(device)
            out.append(model(w, v)[0].double().cpu().numpy())
    return np.concatenate(out)


def fit_combiner(fit_bags: dict, device: str) -> tuple[GatedAttentionMIL, dict, np.ndarray, np.ndarray]:
    """Train on 90% of ntire_val, keep the epoch with the best 10% AUC."""
    torch.manual_seed(SPLIT_SEED)
    meta = fit_bags["meta"]
    fit_idx, stop_idx = stratified_split(meta)

    # Standardise with statistics of the real (non-padding) fit windows only.
    fit_windows = fit_bags["windows"][fit_idx][fit_bags["valid"][fit_idx]]
    mean, std = fit_windows.mean(axis=0), fit_windows.std(axis=0) + 1e-6
    windows = ((fit_bags["windows"] - mean) / std) * fit_bags["valid"][..., None]
    windows = windows.astype(np.float32)
    labels = meta["label"].to_numpy()

    model = GatedAttentionMIL(windows.shape[-1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SPLIT_SEED)
    best = {"auc": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(fit_idx)
        epoch_loss = 0.0
        for i in range(0, len(order), BATCH_BAGS):
            batch = order[i:i + BATCH_BAGS]
            w = torch.from_numpy(windows[batch]).to(device)
            v = torch.from_numpy(fit_bags["valid"][batch]).to(device)
            y = torch.from_numpy(labels[batch].astype(np.float32)).to(device)
            logit, _ = model(w, v)
            loss = loss_fn(logit, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(batch)
        stop_logit = predict(model, windows[stop_idx], fit_bags["valid"][stop_idx], device)
        stop_auc = auc(labels[stop_idx], stop_logit)
        history.append({"epoch": epoch, "fit_loss": epoch_loss / len(fit_idx), "stop_auc": stop_auc})
        print(f"  epoch {epoch}: fit loss {epoch_loss / len(fit_idx):.4f}, early-stop AUC {stop_auc:.4f}", flush=True)
        if stop_auc > best["auc"]:
            best = {"auc": stop_auc, "epoch": epoch,
                    "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}
        elif epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    record = {"n_fit": int(len(fit_idx)), "n_stop": int(len(stop_idx)), "best_epoch": best["epoch"],
              "best_stop_auc": best["auc"], "history": history}
    return model, record, mean, std


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"commit": commit(), "seed": SEED, "split_seed": SPLIT_SEED, "n_boot": N_BOOT,
           "hyperparameters": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
                               "batch_bags": BATCH_BAGS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
           "branches": {}}
    passes, fails = {}, {}
    for branch, tag in BRANCHES.items():
        fit_bags = load_bags(tag, FIT_SPLIT)
        eval_bags = load_bags(tag, EVAL_SPLIT)
        model, record, mean, std = fit_combiner(fit_bags, device)

        eval_windows = (((eval_bags["windows"] - mean) / std) * eval_bags["valid"][..., None]).astype(np.float32)
        eval_meta = eval_bags["meta"].copy()
        eval_meta["combiner"] = predict(model, eval_windows, eval_bags["valid"], device)

        squish = pd.read_parquet(f"outputs/scores/probe_{tag}_squish_{EVAL_SPLIT}.parquet")[["path", "logit"]]
        joined = eval_meta.merge(squish.rename(columns={"logit": "squish"}), on="path", validate="1:1")
        assert len(joined) == len(eval_meta)

        # Baseline combiner: vs_baseline["squish"] is (squish - combiner).
        boot = paired_bootstrap(joined, ["combiner", "squish", "grid9"], "combiner", n_boot=N_BOOT, seed=SEED)
        squish_minus_combiner = boot["vs_baseline"]["squish"]["robust_auc"]
        passes[branch] = squish_minus_combiner["ci"][0] > 0
        fails[branch] = squish_minus_combiner["ci"][1] < 0
        res["branches"][branch] = {"fit": record, "n_eval": int(len(joined)), "bootstrap": boot,
                                   "robust_squish_minus_combiner": squish_minus_combiner}
        joined.to_parquet(f"{FEATURE_DIR}/test9_rows_{branch}.parquet", index=False)

    if all(passes.values()):
        res["verdict"] = "PASS"
    elif all(fails.values()):
        res["verdict"] = "FAIL"
    elif any(fails.values()):
        res["verdict"] = "MIXED (" + ", ".join(f"{b}: {'pass' if passes[b] else 'fail' if fails[b] else 'tie'}"
                                                for b in BRANCHES) + ")"
    else:
        res["verdict"] = "TIE" if not any(passes.values()) else "PARTIAL PASS (" + ", ".join(
            f"{b}: {'pass' if passes[b] else 'tie'}" for b in BRANCHES) + ")"
    with open(f"{STEM}.json", "w") as f:
        json.dump(res, f, indent=2)
    figure(res, f"{STEM}_fig.pdf")
    for branch, e in res["branches"].items():
        m = e["bootstrap"]["models"]
        d = e["robust_squish_minus_combiner"]
        print(branch, "best epoch", e["fit"]["best_epoch"], "stop AUC", round(e["fit"]["best_stop_auc"], 4),
              "robust combiner/squish/grid9", *(round(m[a]["robust_auc"]["value"], 4) for a in ("combiner", "squish", "grid9")),
              f"squish-combiner {d['diff']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    print("verdict", res["verdict"])
    return 0


def figure(res: dict, pdf_path: str) -> None:
    """Robust AUC of grid9 mean, MIL combiner and squish per branch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    arms = (("grid9", "0.6"), ("combiner", "#b2182b"), ("squish", "#2166ac"))
    for i, branch in enumerate(BRANCHES):
        models = res["branches"][branch]["bootstrap"]["models"]
        for j, (arm, colour) in enumerate(arms):
            m = models[arm]["robust_auc"]
            x = i * 4 + j
            ax.bar(x, m["value"], color=colour, width=0.8, label=arm if i == 0 else None)
            ax.errorbar(x, m["value"], yerr=[[m["value"] - m["ci"][0]], [m["ci"][1] - m["value"]]],
                        color="0.2", lw=0.8, capsize=2)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(list(BRANCHES))
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("robust AUC, ntire_val_hard")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(pdf_path)


if __name__ == "__main__":
    sys.exit(main())
