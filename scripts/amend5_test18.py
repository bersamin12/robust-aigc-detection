#!/usr/bin/env python3
"""Amendment 10, Test 18, scoring: damage before vs after the shrink (`shrinkorder`).

    python scripts/amend5_test18.py rows        # CPU: the row rule and the window budget, no scoring
    python scripts/amend5_test18.py selfcheck   # CPU: view construction on a synthetic image
    python scripts/amend5_test18.py score       # GPU: both branches, resumable
    python scripts/amend5_test18.py score --per-label 2 --smoke --out-dir .../smoke     # smoke run
    python scripts/amend5_test18.py score --per-label 10 --out-dir .../timing          # timing run

Question. Test 13 found that about half of Branch B's damage rise survives
only if squish is allowed to shrink the damage away (attenuation at squish's
shrink, 5-18x on large images). Test 17 found no attenuation at k = 2. Does the
damage reduction from the shrink appear only when the shrink is strong?

Registered design (paper_prereg.md, amendment 10, Test 18):

  Rows: Test 2's / Test 13's clean NTIRE rows (`amend5_test2.load_sources`,
  ntire_val + ntire_val_hard, is_distorted == 0, never ntire_test), cut by
  native short side with the registered row rule (`row_rule`): k in
  {1, 2, 4, 8} at short side >= 1,600 if that leaves >= 1,000 rows with both
  labels; otherwise k in {1, 2, 4} at short side >= 800.

  Damage: Test 2's heavy level of each op (JPEG q30, blur sigma 2.0, noise
  sigma 0.10), applied by `amend5_test2.damage`, so the noise generator is
  seeded per image exactly as in Tests 2 and 13, and identically in both
  orders. The parameters are fixed in the pixel grid where the damage is
  applied: a native blur of sigma 2 is about sigma 2/k after the shrink.
  That reduction is part of the damage reduction from the shrink, the
  quantity under test, so it is deliberately not compensated.

  Views, all Test 14's tiles(k, 1) (whole frame, shrunk by k with INTER_AREA,
  200 px windows spread as in Test 14, each resized to n, mean logit):
    clean_k<k>         shrink(image, k)                -> windows
    before_<op>_k<k>   shrink(damage(image), k)        -> windows
    after_<op>_k<k>    damage(shrink(image, k))        -> windows
  At k = 1 the shrink is the identity, so before == after pixel for pixel.
  The full run scores it once per op (as `before_<op>_k1`); the smoke run
  scores both and asserts identical logits.

Output: one parquet per branch with, for every view, the image logit (mean
over windows), the window count, and the per-window logits (kept so that a
different pooling rule needs no rescore; plan section 7). The analysis is
amend5_test18_analysis.py.

`amend5_test2.py`, `amend5_test14.py` and `amend5_test14_analysis.py` are
copied unchanged from wt/test17 (identical to wt/test13's and wt/test14's).
"""
from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aigcdet.augment.canonical import _DOWNSCALE_INTERP  # noqa: E402
from aigcdet.train.finetune import _forward_tower, _windowed  # noqa: E402
from amend5_test2 import OurBranch, damage, load_sources  # noqa: E402
from amend5_test14 import views_tiles  # noqa: E402
from public_detector_probe import decode  # noqa: E402

OPS = ("jpeg", "blur", "noise")
#: Test 2's heavy condition of each op (amend5_test2.CONDITIONS).
HEAVY = {"jpeg": "jpeg_heavy", "blur": "blur_heavy", "noise": "noise_heavy"}
WINDOW_SIDE = 200

#: The row rule (registered). Strong set first, fallback second.
STRONG_KS, STRONG_MIN_SHORT, STRONG_MIN_ROWS = (1, 2, 4, 8), 1600, 1000
FALLBACK_KS, FALLBACK_MIN_SHORT = (1, 2, 4), 800
#: What the rule gave on the row counts before any scoring (amendment 10).
REGISTERED_OUTCOME = {"ks": FALLBACK_KS, "min_short": FALLBACK_MIN_SHORT,
                      "n": 2833, "n_real": 1520, "n_fake": 1313}

OUT_ROOT = "/scratch/dion_r13/dion_extensions/outputs/test18"
KEEP_COLUMNS = ["path", "label", "width", "height", "split", "content_sha256"]


# ---------------------------------------------------------------- rows

def row_rule(clean_rows: pd.DataFrame) -> tuple[tuple[int, ...], int, pd.DataFrame]:
    """The registered row rule, decided from row counts only.

    Returns the k set, the short-side cut and the rows that pass it.
    """
    short_side = clean_rows[["width", "height"]].min(axis=1)
    strong_rows = clean_rows[short_side >= STRONG_MIN_SHORT]
    strong_has_both_labels = strong_rows["label"].nunique() == 2
    if len(strong_rows) >= STRONG_MIN_ROWS and strong_has_both_labels:
        return STRONG_KS, STRONG_MIN_SHORT, strong_rows
    fallback_rows = clean_rows[short_side >= FALLBACK_MIN_SHORT]
    return FALLBACK_KS, FALLBACK_MIN_SHORT, fallback_rows


def load_rows() -> tuple[tuple[int, ...], pd.DataFrame]:
    """Test 18's rows in Test 2's row order, and the k set, checked against the registration."""
    clean_rows = load_sources()["ntire_clean"]
    assert not clean_rows["split"].eq("ntire_test").any(), "Test 18 never reads ntire_test"
    ks, min_short, rows = row_rule(clean_rows)
    outcome = {"ks": ks, "min_short": min_short, "n": len(rows),
               "n_real": int((rows["label"] == 0).sum()), "n_fake": int((rows["label"] == 1).sum())}
    assert outcome == REGISTERED_OUTCOME, (outcome, REGISTERED_OUTCOME)
    return ks, rows[KEEP_COLUMNS].reset_index(drop=True)


def expected_windows(height: int, width: int, k: int) -> int:
    """Window count of tiles(k, 1) from the image size alone (Test 14's ceil(L / 200) per axis)."""
    shrunk_h, shrunk_w = round(height / k), round(width / k)
    return math.ceil(shrunk_h / WINDOW_SIDE) * math.ceil(shrunk_w / WINDOW_SIDE)


def view_names(ks: tuple[int, ...], both_orders_at_k1: bool) -> list[str]:
    """Every view scored per image. after_<op>_k1 only when both orders are scored at k = 1."""
    names = [f"clean_k{k}" for k in ks]
    for op in OPS:
        for k in ks:
            names.append(f"before_{op}_k{k}")
            if k > 1 or both_orders_at_k1:
                names.append(f"after_{op}_k{k}")
    return names


# ---------------------------------------------------------------- views

def shrink(img: np.ndarray, k: int) -> np.ndarray:
    """Test 14's shrink of the whole frame (views_tiles step 2 at f = 1). k = 1 is the identity."""
    if k == 1:
        return img
    height, width = img.shape[:2]
    return cv2.resize(img, (round(width / k), round(height / k)), interpolation=_DOWNSCALE_INTERP)


def windows(shrunk: np.ndarray, policy) -> list[np.ndarray]:
    """Test 14's windows over an image that is already shrunk: tiles(1, 1) of it.

    views_tiles at k = 1, f = 1 takes the whole frame and does not resize it,
    so windows(shrink(img, k)) == views_tiles(img, k, 1) (checked in selfcheck).
    Its assert also refuses any shrunk side under 200 px; the row rule leaves
    none (shortest shrunk side 200 px at k_max).
    """
    return views_tiles(shrunk, policy, 1, 1.0)


def image_views(img: np.ndarray, sha: str, policy, ks: tuple[int, ...], both_orders_at_k1: bool):
    """Every view of one image, keyed by view name."""
    views = {}
    damaged_native = {op: damage(img, HEAVY[op], sha) for op in OPS}
    for k in ks:
        clean_shrunk = shrink(img, k)
        views[f"clean_k{k}"] = windows(clean_shrunk, policy)
        for op in OPS:
            damage_then_shrink = shrink(damaged_native[op], k)
            views[f"before_{op}_k{k}"] = windows(damage_then_shrink, policy)
            if k > 1 or both_orders_at_k1:
                shrink_then_damage = damage(clean_shrunk, HEAVY[op], sha)
                views[f"after_{op}_k{k}"] = windows(shrink_then_damage, policy)
    return views


# ---------------------------------------------------------------- scoring

def window_logits(branch: OurBranch, views_per_image: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Per-window logits of each image: OurBranch.image_logits's forward, without the mean.

    The image logit is the mean of these, the same number image_logits returns
    (checked on the smoke run).
    """
    flat = [view for views in views_per_image for view in views]
    pooled = torch.cat([_forward_tower(tower, spec, flat, branch.device, torch.bfloat16, branch.chunk, pooler=pooler)
                        for tower, spec, pooler in zip(branch.towers, branch.specs, branch.poolers)], dim=-1)
    all_logits = branch.head(pooled.float(), None)["logit"].double().cpu().numpy()
    bounds = np.cumsum([0] + [len(views) for views in views_per_image])
    assert bounds[-1] == len(all_logits)
    return [all_logits[start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def score_branch(branch: OurBranch, rows: pd.DataFrame, ks, both_orders_at_k1: bool, workers: int):
    """Per-window logits per image and view, one image per forward batch (as Tests 14 and 17)."""
    from tqdm import tqdm

    def prepare(chunk: pd.DataFrame):
        per_view = {}
        for path, sha in zip(chunk["path"], chunk["content_sha256"]):
            for name, views in image_views(decode(path), sha, branch.policy, ks, both_orders_at_k1).items():
                per_view.setdefault(name, []).append(views)
        return per_view

    chunks = [rows.iloc[i:i + 1] for i in range(len(rows))]
    per_window = {}
    first_chunk_checked = False
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, chunks, 2 * workers), total=len(chunks)):
            for name, views_per_image in fut.result().items():
                logits = window_logits(branch, views_per_image)
                if both_orders_at_k1 and not first_chunk_checked:
                    # Smoke only: the mean of window_logits is OurBranch.image_logits.
                    reference = branch.image_logits(views_per_image)
                    assert np.allclose([z.mean() for z in logits], reference, atol=1e-6), (name, reference)
                per_window.setdefault(name, []).extend(logits)
            first_chunk_checked = True
    return per_window


def score(a) -> int:
    if a.device.startswith("cuda"):
        # Shared GPU: past the cap, this process gets an out-of-memory error instead of
        # taking memory from other users. The CUDA context (~0.5 GB) is outside the cap.
        index = torch.device(a.device).index or 0
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gib * 2**30 / total), index)

    ks, rows = load_rows()
    if a.per_label:
        rows = rows.groupby("label", sort=False).head(a.per_label).reset_index(drop=True)
    names = view_names(ks, both_orders_at_k1=a.smoke)
    os.makedirs(a.out_dir, exist_ok=True)

    for model_name in a.models.split(","):
        out = f"{a.out_dir}/{model_name}_shrinkorder.parquet"
        if os.path.exists(out):
            print(f"skip {out}", flush=True)
            continue
        branch = OurBranch(model_name, a.device)
        started = time.time()
        per_window = score_branch(branch, rows, ks, a.smoke, a.workers)
        assert sorted(per_window) == sorted(names), (sorted(per_window), names)

        frame = rows.drop(columns=["content_sha256"]).copy()
        for name in names:
            frame[f"logit_{name}"] = [float(z.mean()) for z in per_window[name]]
            frame[f"n_windows_{name}"] = [len(z) for z in per_window[name]]
            frame[f"window_logits_{name}"] = [z.tolist() for z in per_window[name]]
        for k in ks:
            expected = [expected_windows(h, w, k) for h, w in zip(frame["height"], frame["width"])]
            assert frame[f"n_windows_clean_k{k}"].tolist() == expected, k

        if a.smoke:
            for op in OPS:
                before, after = frame[f"logit_before_{op}_k1"].to_numpy(), frame[f"logit_after_{op}_k1"].to_numpy()
                assert np.array_equal(before, after), (op, before, after)
            print(f"{model_name}: k = 1 before == after, identical logits for {', '.join(OPS)}", flush=True)

        frame.to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        seconds = time.time() - started
        n_windows = int(sum(frame[f"n_windows_{name}"].sum() for name in names))
        peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(f"{model_name}: {len(rows)} images, {n_windows} windows in {seconds:.0f} s "
              f"({n_windows / seconds:.1f} windows/s), peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"peak RSS {peak_rss_gib:.1f} GiB -> {out}", flush=True)
        del branch
        torch.cuda.empty_cache()
    print("TEST18_SCORING_DONE", flush=True)
    return 0


# ---------------------------------------------------------------- CPU steps

def rows_step() -> int:
    """Print the row rule's outcome and the window budget of the full run. Scores nothing."""
    ks, rows = load_rows()
    print(f"row rule: k in {ks}, short side >= {REGISTERED_OUTCOME['min_short']}, "
          f"{len(rows)} rows ({(rows['label'] == 0).sum()} real, {(rows['label'] == 1).sum()} fake)")
    names = view_names(ks, both_orders_at_k1=False)
    per_k = {k: sum(expected_windows(h, w, k) for h, w in zip(rows["height"], rows["width"])) for k in ks}
    views_per_k = {k: sum(1 for name in names if name.endswith(f"_k{k}")) for k in ks}
    total = sum(per_k[k] * views_per_k[k] for k in ks)
    print(f"windows per view at each k: {per_k}; views at each k: {views_per_k}; total per branch {total}")
    return 0


def selfcheck() -> int:
    """View construction on a synthetic image, no model: the identities the design relies on."""
    policy = SimpleNamespace(crop_side=200, nominal_side=512)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (1013, 1601, 3), dtype=np.uint8)
    sha = "0123456789abcdef" * 4
    views = image_views(img, sha, policy, FALLBACK_KS, both_orders_at_k1=True)
    for k in FALLBACK_KS:
        # windows(shrink(img, k)) is Test 14's tiles(k, 1).
        tiles = views_tiles(img, policy, k, 1.0)
        assert all(np.array_equal(x, y) for x, y in zip(views[f"clean_k{k}"], tiles))
        assert len(views[f"clean_k{k}"]) == len(tiles) == expected_windows(*img.shape[:2], k)
        for op in OPS:
            assert len(views[f"before_{op}_k{k}"]) == len(views[f"after_{op}_k{k}"]) == len(tiles), (op, k)
    for op in OPS:
        # k = 1: the two orders are the same pixels.
        assert all(np.array_equal(x, y) for x, y in zip(views[f"before_{op}_k1"], views[f"after_{op}_k1"])), op
        # k > 1: they are not (otherwise there is nothing to test).
        assert not all(np.array_equal(x, y) for x, y in zip(views[f"before_{op}_k2"], views[f"after_{op}_k2"])), op
    print("selfcheck passed: tiles(k, 1) reproduced, window counts equal across orders, k = 1 orders identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["rows", "selfcheck", "score"])
    ap.add_argument("--out-dir", default=f"{OUT_ROOT}/scores")
    ap.add_argument("--models", default="B2swa,A2")
    ap.add_argument("--per-label", type=int, default=0, help="first N rows of each label (smoke / timing)")
    ap.add_argument("--smoke", action="store_true", help="also score after_<op>_k1 and assert it equals before")
    ap.add_argument("--workers", type=int, default=4, help="CPU threads building views (RAM: ~0.4 GB per image in flight)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-gib", type=float, default=10.0, help="cap on this process's GPU memory (shared GPU)")
    a = ap.parse_args()
    if a.step == "rows":
        return rows_step()
    if a.step == "selfcheck":
        return selfcheck()
    return score(a)


if __name__ == "__main__":
    raise SystemExit(main())
