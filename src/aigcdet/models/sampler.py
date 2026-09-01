"""Paired batch sampler (spec §3.5).

Each batch pairs `n_src` source images with `m_deg` degraded views apiece, so
every clean embedding in the batch has degraded partners drawn from the SAME
image -- the consistency loss in `aigcdet.models.losses` is only meaningful
if the clean/degraded pair share a source image, never two different ones.

Batches are class-balanced (half positive, half negative) and
generator-balanced within each class: a generator family is picked uniformly
at random, then an image is drawn from that family. This equalises
representation across generators regardless of how skewed the underlying
pool is, so no single generator's over-representation in the source dataset
dominates a gradient step (spec §3.5: "no generator family dominates a
gradient step").

View 0 is the clean view by the bank's own invariant
(`aigcdet.features.bank`, `check_invariants`). Degraded views are always
drawn from the structurally disjoint range 1..n_views-1 -- view 0 is never
eligible as a degraded partner, by construction, not by chance. If it were,
the consistency loss would compare the clean view against itself: KL and MSE
both go to zero, the loss looks healthy, and the mechanism silently
contributes nothing.

No global randomness: every draw goes through the caller-supplied
`numpy.random.Generator`, so two samplers built from equivalently-seeded
generators reproduce the exact same batch sequence.
"""
from __future__ import annotations

import numpy as np
import torch


class PairedSampler:
    """Iterable of class- and generator-balanced clean/degraded batches.

    Parameters
    ----------
    bank:
        An open `aigcdet.features.bank.FeatureBank`.
    indices:
        Pool of row indices (e.g. the train split) to draw from. Must
        contain both classes.
    n_src:
        Source images per batch. Must be even so batches split exactly in
        half between the two classes.
    m_deg:
        Degraded views drawn per source image. Batch size is
        `B = n_src * m_deg`.
    rng:
        An explicit `numpy.random.Generator`. Required -- this project's
        determinism rule forbids implicit/global seeding, and the caller
        needs to thread one stream across the whole sampler.
    use_recon:
        If True, also return `r_clean`/`r_deg` reconstruction features
        (`bank.recon` must already be attached).
    augmented_only:
        If True, the `m_deg` degraded views drawn for one source image are
        distinct (sampled without replacement from the augmented views);
        if False (default), each of the `m_deg` draws is independent and
        may repeat a view index. Either way the degraded pool is restricted
        to views `1..n_views-1` -- view 0 is never eligible regardless of
        this flag.
    device:
        torch device string for the returned tensors.
    """

    def __init__(self, bank, indices: np.ndarray, n_src: int = 32, m_deg: int = 2,
                 *, rng: np.random.Generator, use_recon: bool = False,
                 use_recon_vq: bool = False, use_freq: bool = False,
                 augmented_only: bool = False, device: str = "cpu"):
        if n_src % 2 != 0:
            raise ValueError("n_src must be even so batches can be class-balanced")
        if m_deg < 1:
            raise ValueError("m_deg must be >= 1")
        self.bank = bank
        self.n_src, self.m_deg = n_src, m_deg
        self.rng = rng
        self.use_recon = use_recon
        self.use_recon_vq = use_recon_vq
        self.use_freq = use_freq
        self.augmented_only = augmented_only
        self.device = device
        self.n_views = bank.config["n_views"]
        if augmented_only and m_deg > self.n_views - 1:
            raise ValueError(
                "m_deg cannot exceed the number of augmented views (n_views - 1) "
                "when augmented_only=True")

        indices = np.asarray(indices)
        labels = bank.meta["label"].to_numpy()[indices]
        self.pos = indices[labels == 1]
        self.neg = indices[labels == 0]
        if len(self.pos) == 0 or len(self.neg) == 0:
            raise ValueError("index pool must contain both classes")
        self.generators = bank.meta["generator"].to_numpy()
        # Group each class pool by generator family ONCE. The previous
        # implementation recomputed self.generators[pool], np.unique(...) and
        # a full boolean mask over the whole pool on every single draw, i.e.
        # O(n_src x |pool|) per batch: measured at 45k/45k with 12 families
        # and n_src=64 that was ~61 ms/batch, ~43 min/rung, ~5 h across the
        # ladder -- against a ~1M-parameter head whose actual training step
        # is milliseconds. Grouping up front makes a draw O(1).
        self._pos_groups = self._build_groups(self.pos)
        self._neg_groups = self._build_groups(self.neg)

    def _build_groups(self, pool: np.ndarray) -> list[np.ndarray]:
        """Split `pool` into one index array per generator family present.

        Groups come out in the same order `np.unique` would give (ascending
        family name), and each group keeps `pool`'s own ordering, so drawing
        group-then-offset reproduces the previous implementation's choices
        draw for draw from the same generator stream.
        """
        gens = self.generators[pool]
        order = np.argsort(gens, kind="stable")
        sorted_gens = gens[order]
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_gens[1:] != sorted_gens[:-1])))
        ends = np.concatenate((starts[1:], [len(pool)]))
        return [pool[order[a:b]] for a, b in zip(starts, ends)]

    def _groups_for(self, pool: np.ndarray) -> list[np.ndarray]:
        if pool is self.pos:
            return self._pos_groups
        if pool is self.neg:
            return self._neg_groups
        return self._build_groups(pool)

    def __len__(self) -> int:
        half = self.n_src // 2
        return max(1, min(len(self.pos), len(self.neg)) // half)

    def _draw_stratified(self, pool: np.ndarray, k: int) -> np.ndarray:
        """Generator-balanced draw of k indices from pool: pick a generator
        family uniformly, then an image within it. A family with 2 images
        gets the same per-draw probability as a family with 2000.

        Two `rng.integers` draws per element, in the same order and with the
        same bounds as before the grouping was precomputed, so the batch
        sequence for a given seed is unchanged -- only the cost is.
        """
        groups = self._groups_for(pool)
        n_groups = len(groups)
        chosen = np.empty(k, dtype=np.int64)
        for i in range(k):
            family = groups[self.rng.integers(n_groups)]
            chosen[i] = family[self.rng.integers(len(family))]
        return chosen

    def _draw_degraded_views(self, n_rows: int) -> np.ndarray:
        """Degraded view index per (source, m_deg-slot) row, always in
        1..n_views-1. View 0 is structurally excluded here, not merely
        unlikely: `low=1` in both branches below."""
        if self.augmented_only:
            return np.stack([
                self.rng.choice(np.arange(1, self.n_views), size=self.m_deg,
                                 replace=False)
                for _ in range(n_rows)
            ])
        return self.rng.integers(1, self.n_views, size=(n_rows, self.m_deg))

    def draw_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One batch's ROW AND VIEW CHOICES, before any feature is read.

        `(src, si, vi)`: the source rows, those rows repeated `m_deg` times,
        and one degraded view index per repeat.

        Split out from `__iter__` so a subclass can reuse the draw itself
        rather than reimplement it. `train.finetune.LiveViewSampler` does
        exactly that, and the reason it must is the point of the split: the
        unfreeze ladder's D0 rung is the frozen tower, whose whole job is to
        reproduce the cached-feature rung it is the control for. If the two
        samplers drew even slightly different batch sequences from the same
        seed -- one extra `rng` call, a reordered pair of draws -- D0 would
        differ from its cached twin for a reason that has nothing to do with
        unfreezing, and the ladder's baseline would be wrong. Sharing the code
        makes that identity structural instead of a thing to keep in step by
        hand.
        """
        half = self.n_src // 2
        src = np.concatenate([
            self._draw_stratified(self.pos, half),
            self._draw_stratified(self.neg, half),
        ])
        deg_views = self._draw_degraded_views(len(src))
        return src, np.repeat(src, self.m_deg), deg_views.reshape(-1)

    def __iter__(self):
        for _ in range(len(self)):
            src, si, vi = self.draw_batch()

            f_clean = np.repeat(
                np.asarray(self.bank.feats[src, 0]).astype(np.float32),
                self.m_deg, axis=0)
            f_deg = np.asarray(self.bank.feats[si, vi]).astype(np.float32)
            y = self.bank.meta["label"].to_numpy()[si].astype(np.float32)

            batch = {
                "f_clean": torch.from_numpy(f_clean).to(self.device),
                "f_deg": torch.from_numpy(f_deg).to(self.device),
                "y_clean": torch.from_numpy(y).to(self.device),
                "y_deg": torch.from_numpy(y).to(self.device),
                "presence_deg": torch.from_numpy(
                    np.asarray(self.bank.presence[si, vi]).astype(np.float32)
                ).to(self.device),
                "severity_deg": torch.from_numpy(
                    np.asarray(self.bank.severity[si, vi]).astype(np.float32)
                ).to(self.device),
                "r_clean": None,
                "r_deg": None,
            }
            if self.use_recon or self.use_recon_vq or self.use_freq:
                # `recon_blocks` owns which blocks and in what order; a head
                # trained on [kl | vq] and scored against [vq | kl] would not
                # raise, it would just be wrong.
                blocks = self.bank.aux_blocks(self.use_recon,
                                              self.use_recon_vq, self.use_freq)
                r_clean = np.concatenate(
                    [np.asarray(b[src, 0]).astype(np.float32) for b in blocks],
                    axis=-1)
                batch["r_clean"] = torch.from_numpy(
                    np.repeat(r_clean, self.m_deg, axis=0)).to(self.device)
                batch["r_deg"] = torch.from_numpy(np.concatenate(
                    [np.asarray(b[si, vi]).astype(np.float32) for b in blocks],
                    axis=-1)).to(self.device)
            yield batch
