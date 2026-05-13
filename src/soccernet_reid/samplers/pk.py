"""PK and PK-per-action batch samplers for metric learning.

A PK batch contains P classes × K samples per class = P·K total samples. This
batch structure is what enables miners like batch-hard / semi-hard triplet
mining (Hermans et al., "In Defense of the Triplet Loss for Person ReID"): for
each anchor sample in the batch there are K-1 positives and (P-1)·K negatives
to mine from.

For SoccerNet ReID specifically (see §3 of PLAN_EKSPERYMENTOW.md):
- Classes have very few samples (54.8% are singletons, 33.5% have exactly 2).
- Defaults to K=2 (because K=4 would discard 90 % of the data).
- Cross-action PK: P=16, K=2 → batch 32.
- PK-per-action (PK-SA): P=8, K=2 → batch 16 (39 % of actions qualify).

Both samplers IMPLICITLY skip singletons by requiring ≥K samples per class —
no explicit catalog filter needed. This is the convention from BoT-ReID and
other ReID baselines.

Sampling is with replacement at the class level (each batch independently picks
P classes from the qualifying pool). Within a chosen class, K samples are drawn
without replacement if K ≤ class size, which is guaranteed by the qualification
filter. The default `num_batches` defines an "epoch" as 5000 iterations (§5).
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """Cross-action PK batch sampler.

    Each batch:
    1. Pick P unique classes from those with ≥K samples in the dataset.
    2. From each chosen class, sample K dataset-indices uniformly without
       replacement.
    3. Yield the concatenated list of P·K indices.

    Args:
        class_ids: Sequence of per-sample class labels, length = dataset size.
            Position `i` in this sequence corresponds to dataset index `i`.
        P: number of classes per batch (e.g., 16).
        K: number of samples per class per batch (e.g., 2).
        num_batches: how many batches one epoch yields. Default 5000.
        seed: RNG seed for reproducibility.

    Raises:
        ValueError: if fewer than P classes qualify (have ≥K samples).
    """

    def __init__(
        self,
        class_ids: Sequence[int],
        P: int,
        K: int,
        num_batches: int = 5000,
        seed: int = 0,
    ) -> None:
        if P < 2:
            raise ValueError(f"P must be ≥2 to form pos/neg pairs in a batch, got {P}")
        if K < 2:
            raise ValueError(f"K must be ≥2 for metric learning to form positives, got {K}")
        if num_batches < 1:
            raise ValueError(f"num_batches must be ≥1, got {num_batches}")

        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.seed = seed

        # Group dataset indices by class. Build a list keyed by qualifying class.
        cls_to_indices: dict[int, list[int]] = {}
        for idx, cid in enumerate(class_ids):
            cid_int = int(cid)
            if cid_int < 0:  # unannotated rows (e.g., challenge) — skip
                continue
            cls_to_indices.setdefault(cid_int, []).append(idx)

        self._qualifying_classes: list[int] = sorted(
            c for c, idx_list in cls_to_indices.items() if len(idx_list) >= K
        )
        self._cls_to_indices: dict[int, np.ndarray] = {
            c: np.asarray(cls_to_indices[c], dtype=np.int64) for c in self._qualifying_classes
        }

        if len(self._qualifying_classes) < P:
            raise ValueError(
                f"Only {len(self._qualifying_classes)} classes have ≥K={K} samples, "
                f"but P={P} requested. Lower K, lower P, or use a different sampler."
            )

    @property
    def qualifying_classes(self) -> list[int]:
        """Class IDs eligible for sampling (length ≥ P guaranteed)."""
        return list(self._qualifying_classes)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        qual = np.asarray(self._qualifying_classes, dtype=np.int64)
        for _ in range(self.num_batches):
            chosen_classes = rng.choice(qual, size=self.P, replace=False)
            batch: list[int] = []
            for cid in chosen_classes:
                idx_pool = self._cls_to_indices[int(cid)]
                picks = rng.choice(idx_pool, size=self.K, replace=False)
                batch.extend(int(i) for i in picks)
            yield batch


class PKPerActionBatchSampler(Sampler[list[int]]):
    """PK sampler restricted to a single action per batch.

    Each batch is sampled entirely from one action — matching the official
    evaluation protocol where retrieval is action-local. Two-stage filtering:

    1. A class qualifies if it has ≥K samples in its action (== its (action,
       uid) pair has ≥K samples). Since `class_id` IS the (action, uid) pair,
       this is the same as ≥K samples per class.
    2. An action qualifies if it has ≥P qualifying classes.

    Each batch:
    a. Pick a qualifying action.
    b. Pick P unique qualifying classes from that action.
    c. Pick K samples from each class.

    Args:
        class_ids: per-sample class label.
        action_ids: per-sample action_idx.
        P, K, num_batches, seed: as in :class:`PKBatchSampler`.

    Raises:
        ValueError: if no action qualifies (< P classes with ≥K samples).
    """

    def __init__(
        self,
        class_ids: Sequence[int],
        action_ids: Sequence[int],
        P: int,
        K: int,
        num_batches: int = 5000,
        seed: int = 0,
    ) -> None:
        if len(class_ids) != len(action_ids):
            raise ValueError(
                f"class_ids and action_ids length mismatch: "
                f"{len(class_ids)} vs {len(action_ids)}"
            )
        if P < 2:
            raise ValueError(f"P must be ≥2, got {P}")
        if K < 2:
            raise ValueError(f"K must be ≥2, got {K}")
        if num_batches < 1:
            raise ValueError(f"num_batches must be ≥1, got {num_batches}")

        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.seed = seed

        # Group dataset indices by (action, class).
        action_cls_to_indices: dict[tuple[int, int], list[int]] = {}
        for idx, (cid, aid) in enumerate(zip(class_ids, action_ids, strict=True)):
            cid_int = int(cid)
            if cid_int < 0:
                continue
            key = (int(aid), cid_int)
            action_cls_to_indices.setdefault(key, []).append(idx)

        # Build action → list[(cls_id, np.array of indices)] for qualifying classes.
        action_to_classes: dict[int, list[tuple[int, np.ndarray]]] = {}
        for (aid, cid), idx_list in action_cls_to_indices.items():
            if len(idx_list) < K:
                continue
            action_to_classes.setdefault(aid, []).append(
                (cid, np.asarray(idx_list, dtype=np.int64))
            )

        self._qualifying_actions: list[int] = sorted(
            a for a, cls_list in action_to_classes.items() if len(cls_list) >= P
        )
        # Store as parallel lists per action: ([class_id, ...], [np.array, ...])
        self._action_to_class_ids: dict[int, np.ndarray] = {}
        self._action_class_indices: dict[tuple[int, int], np.ndarray] = {}
        for aid in self._qualifying_actions:
            classes = action_to_classes[aid]
            self._action_to_class_ids[aid] = np.asarray(
                [cid for cid, _ in classes], dtype=np.int64
            )
            for cid, arr in classes:
                self._action_class_indices[(aid, cid)] = arr

        if len(self._qualifying_actions) == 0:
            raise ValueError(
                f"No action has ≥P={P} classes with ≥K={K} samples. "
                f"Try lower P (e.g. 4) or K=2."
            )

    @property
    def qualifying_actions(self) -> list[int]:
        return list(self._qualifying_actions)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        actions_arr = np.asarray(self._qualifying_actions, dtype=np.int64)
        for _ in range(self.num_batches):
            aid = int(rng.choice(actions_arr))
            class_pool = self._action_to_class_ids[aid]
            chosen_classes = rng.choice(class_pool, size=self.P, replace=False)
            batch: list[int] = []
            for cid in chosen_classes:
                idx_pool = self._action_class_indices[(aid, int(cid))]
                picks = rng.choice(idx_pool, size=self.K, replace=False)
                batch.extend(int(i) for i in picks)
            yield batch
