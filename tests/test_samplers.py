"""Unit tests for PK and PK-per-action batch samplers (synthetic data, no torch GPU)."""
from __future__ import annotations

import numpy as np
import pytest

from soccernet_reid.samplers import PKBatchSampler, PKPerActionBatchSampler


def _make_class_ids(per_class_counts: dict[int, int]) -> list[int]:
    """Build a class_ids sequence with the given counts per class id."""
    out: list[int] = []
    for cid, count in sorted(per_class_counts.items()):
        out.extend([cid] * count)
    return out


class TestPKBatchSampler:
    def test_batch_shape_and_class_diversity(self) -> None:
        # 5 classes × 4 samples each → all qualifying for K=2
        class_ids = _make_class_ids({0: 4, 1: 4, 2: 4, 3: 4, 4: 4})
        sampler = PKBatchSampler(class_ids, P=4, K=2, num_batches=10, seed=0)

        assert len(sampler) == 10
        for batch in sampler:
            assert len(batch) == 8  # P*K
            # All indices in range
            assert all(0 <= i < len(class_ids) for i in batch)
            # P unique classes, K samples each
            picked_classes = [class_ids[i] for i in batch]
            unique_classes = set(picked_classes)
            assert len(unique_classes) == 4
            for c in unique_classes:
                assert picked_classes.count(c) == 2

    def test_skips_singletons_with_K_equals_2(self) -> None:
        # Mix: 3 classes with 2 samples (qualifying), 5 singletons (not qualifying)
        class_ids = _make_class_ids({0: 2, 1: 2, 2: 2, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1})
        sampler = PKBatchSampler(class_ids, P=2, K=2, num_batches=20, seed=0)

        for batch in sampler:
            for i in batch:
                # Index must point to one of the qualifying classes
                assert class_ids[i] in {0, 1, 2}, (
                    f"Sampler picked singleton class {class_ids[i]}"
                )

    def test_raises_when_too_few_qualifying_classes(self) -> None:
        class_ids = _make_class_ids({0: 2, 1: 2})  # only 2 qualifying
        with pytest.raises(ValueError, match="have ≥K=2 samples"):
            PKBatchSampler(class_ids, P=4, K=2, num_batches=10, seed=0)

    def test_raises_on_K_too_high(self) -> None:
        class_ids = _make_class_ids({0: 2, 1: 2, 2: 2})  # max class size = 2
        with pytest.raises(ValueError, match="have ≥K=4"):
            PKBatchSampler(class_ids, P=2, K=4, num_batches=10, seed=0)

    def test_seed_reproducibility(self) -> None:
        class_ids = _make_class_ids({i: 3 for i in range(20)})
        s1 = PKBatchSampler(class_ids, P=4, K=2, num_batches=5, seed=42)
        s2 = PKBatchSampler(class_ids, P=4, K=2, num_batches=5, seed=42)
        b1 = [list(b) for b in s1]
        b2 = [list(b) for b in s2]
        assert b1 == b2

    def test_different_seeds_yield_different_batches(self) -> None:
        class_ids = _make_class_ids({i: 3 for i in range(20)})
        s1 = PKBatchSampler(class_ids, P=4, K=2, num_batches=5, seed=0)
        s2 = PKBatchSampler(class_ids, P=4, K=2, num_batches=5, seed=1)
        b1 = [list(b) for b in s1]
        b2 = [list(b) for b in s2]
        assert b1 != b2

    def test_qualifying_classes_property(self) -> None:
        class_ids = _make_class_ids({0: 4, 1: 4, 2: 1, 3: 1})  # 2 singletons
        sampler = PKBatchSampler(class_ids, P=2, K=2, num_batches=1, seed=0)
        assert sorted(sampler.qualifying_classes) == [0, 1]

    def test_negative_class_id_excluded(self) -> None:
        # class_id = -1 represents "unannotated" (e.g., challenge split)
        class_ids = [0, 0, 0, 1, 1, 1, -1, -1, -1]
        sampler = PKBatchSampler(class_ids, P=2, K=2, num_batches=5, seed=0)
        for batch in sampler:
            for i in batch:
                assert class_ids[i] != -1

    def test_default_num_batches_is_5000(self) -> None:
        class_ids = _make_class_ids({i: 2 for i in range(20)})
        sampler = PKBatchSampler(class_ids, P=4, K=2)  # defaults
        assert len(sampler) == 5000

    def test_validates_K_at_least_2(self) -> None:
        class_ids = _make_class_ids({0: 5, 1: 5})
        with pytest.raises(ValueError, match="K must be ≥2"):
            PKBatchSampler(class_ids, P=2, K=1, num_batches=1)

    def test_validates_P_at_least_2(self) -> None:
        class_ids = _make_class_ids({0: 5, 1: 5})
        with pytest.raises(ValueError, match="P must be ≥2"):
            PKBatchSampler(class_ids, P=1, K=2, num_batches=1)


class TestPKPerActionBatchSampler:
    def test_all_batch_indices_share_action(self) -> None:
        # 3 actions, each with 4 classes of 3 samples
        class_ids, action_ids = [], []
        for action in range(3):
            for cls in range(4):
                class_id = action * 100 + cls  # globally unique
                for _ in range(3):
                    class_ids.append(class_id)
                    action_ids.append(action)
        sampler = PKPerActionBatchSampler(
            class_ids, action_ids, P=2, K=2, num_batches=20, seed=0
        )

        for batch in sampler:
            actions_in_batch = {action_ids[i] for i in batch}
            assert len(actions_in_batch) == 1, (
                f"PK-SA batch spans multiple actions: {actions_in_batch}"
            )

    def test_batch_shape_and_class_diversity(self) -> None:
        class_ids, action_ids = [], []
        for action in range(3):
            for cls in range(4):
                class_id = action * 100 + cls
                for _ in range(3):
                    class_ids.append(class_id)
                    action_ids.append(action)
        sampler = PKPerActionBatchSampler(
            class_ids, action_ids, P=3, K=2, num_batches=20, seed=0
        )
        for batch in sampler:
            assert len(batch) == 6  # P*K
            picked_classes = [class_ids[i] for i in batch]
            assert len(set(picked_classes)) == 3
            for c in set(picked_classes):
                assert picked_classes.count(c) == 2

    def test_raises_when_no_action_qualifies(self) -> None:
        # 2 actions, but each has only 1 class with ≥2 samples
        class_ids = [0, 0, 1, 100, 100, 101]
        action_ids = [0, 0, 0, 1, 1, 1]
        with pytest.raises(ValueError, match="No action has ≥P=2"):
            PKPerActionBatchSampler(class_ids, action_ids, P=2, K=2, num_batches=1, seed=0)

    def test_partial_qualification_skips_bad_action(self) -> None:
        # Action 0 has 3 classes with ≥2 samples (qualifies for P=2,K=2).
        # Action 1 has only 1 such class (doesn't qualify).
        # All batches should come from action 0.
        class_ids = (
            # Action 0: 3 qualifying classes
            [0, 0, 1, 1, 2, 2]
            # Action 1: 1 qualifying class + 2 singletons
            + [10, 10, 11, 12]
        )
        action_ids = [0] * 6 + [1] * 4
        sampler = PKPerActionBatchSampler(
            class_ids, action_ids, P=2, K=2, num_batches=30, seed=0
        )
        assert sampler.qualifying_actions == [0]
        for batch in sampler:
            for i in batch:
                assert action_ids[i] == 0

    def test_qualifying_actions_property(self) -> None:
        # Action 0: 2 classes × 2 samples → qualifies for P=2/K=2
        # Action 1: only 1 class with ≥2 samples → no
        # Action 2: 3 classes × 2 samples → qualifies
        class_ids = [0, 0, 1, 1,   100, 100, 101,   200, 200, 201, 201, 202, 202]
        action_ids = [0, 0, 0, 0,  1, 1, 1,         2, 2, 2, 2, 2, 2]
        sampler = PKPerActionBatchSampler(
            class_ids, action_ids, P=2, K=2, num_batches=1, seed=0
        )
        assert sorted(sampler.qualifying_actions) == [0, 2]

    def test_seed_reproducibility(self) -> None:
        class_ids, action_ids = [], []
        for action in range(5):
            for cls in range(5):
                class_id = action * 100 + cls
                for _ in range(2):
                    class_ids.append(class_id)
                    action_ids.append(action)
        s1 = PKPerActionBatchSampler(class_ids, action_ids, P=3, K=2, num_batches=5, seed=99)
        s2 = PKPerActionBatchSampler(class_ids, action_ids, P=3, K=2, num_batches=5, seed=99)
        assert [list(b) for b in s1] == [list(b) for b in s2]

    def test_length_input_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            PKPerActionBatchSampler([0, 0, 1, 1], [0, 0, 0], P=2, K=2, num_batches=1)
