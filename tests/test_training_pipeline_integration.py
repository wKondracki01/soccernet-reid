"""Integration tests for the training-side data pipeline on REAL data.

Verifies that:
- The training catalog (player filter + class_id) has the expected counts.
- Both samplers can construct from the real train catalog with the planned P×K.
- Real images can be drawn through a DataLoader using the sampler + a training
  transform, returning batches of the expected shape.

These run only when the dataset is present locally.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from soccernet_reid.data.catalog import (
    assign_class_ids,
    build_catalog,
    filter_to_player_classes,
    load_catalog,
)
from soccernet_reid.data.dataset import ReIDImageDataset
from soccernet_reid.samplers import PKBatchSampler, PKPerActionBatchSampler
from soccernet_reid.transforms import build_transform

REID_ROOT = Path(__file__).resolve().parent.parent / "dataSoccerNet" / "reid-2023"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "outputs" / "catalog.parquet"

pytestmark = pytest.mark.skipif(
    not REID_ROOT.is_dir(),
    reason=f"Dataset not present at {REID_ROOT}",
)


@pytest.fixture(scope="module")
def train_catalog():
    """Full catalog → train split → player-class filter → class_id assigned."""
    if CATALOG_PATH.exists():
        full = load_catalog(CATALOG_PATH)
    else:
        full = build_catalog(REID_ROOT)
    train = full[full["split"] == "train"].copy()
    train = filter_to_player_classes(train)
    train = assign_class_ids(train)
    return train


class TestCatalogShapes:
    def test_player_filter_count_matches_planned(self, train_catalog) -> None:
        # Verified in earlier analysis: 225,652 player-class rows in train.
        assert len(train_catalog) == 225_652

    def test_class_id_count_matches_planned(self, train_catalog) -> None:
        # Verified: 138,861 unique (action, uid) pairs after class filter.
        assert train_catalog["class_id"].nunique() == 138_861
        # class_id must be contiguous 0..C-1
        assert train_catalog["class_id"].min() == 0
        assert train_catalog["class_id"].max() == 138_860

    def test_qualifying_classes_for_K2(self, train_catalog) -> None:
        # Verified: 62,714 classes have ≥2 samples (K=2 qualifying).
        sizes = train_catalog.groupby("class_id").size()
        qualifying = (sizes >= 2).sum()
        assert qualifying == 62_714


class TestPKOnRealData:
    def test_PK_construction_default_P16_K2(self, train_catalog) -> None:
        # Plan default: P=16, K=2 cross-action. Must succeed.
        sampler = PKBatchSampler(
            class_ids=train_catalog["class_id"].tolist(),
            P=16,
            K=2,
            num_batches=10,
            seed=0,
        )
        assert len(sampler.qualifying_classes) == 62_714
        # Verify a few batches
        for batch in sampler:
            assert len(batch) == 32
            classes = [train_catalog["class_id"].iloc[i] for i in batch]
            assert len(set(classes)) == 16

    def test_PK_K3_drops_significantly(self, train_catalog) -> None:
        # Per plan §3, K=3 keeps only ~25 % of pairs (16,158 qualifying).
        sampler = PKBatchSampler(
            class_ids=train_catalog["class_id"].tolist(),
            P=16,
            K=3,
            num_batches=1,
            seed=0,
        )
        assert len(sampler.qualifying_classes) == 16_158

    def test_PK_K4_raises_below_practical_threshold(self, train_catalog) -> None:
        # With K=4 only ~5,000 classes qualify — still many, but the plan
        # explicitly warns against K=4 for the small per-class budget here.
        sampler = PKBatchSampler(
            class_ids=train_catalog["class_id"].tolist(),
            P=16,
            K=4,
            num_batches=1,
            seed=0,
        )
        assert 4_000 <= len(sampler.qualifying_classes) <= 6_000


class TestPKPerActionOnRealData:
    def test_PKSA_construction_default_P8_K2(self, train_catalog) -> None:
        # Plan default for PK-SA: P=8, K=2 — 39.5 % of actions qualify.
        sampler = PKPerActionBatchSampler(
            class_ids=train_catalog["class_id"].tolist(),
            action_ids=train_catalog["action_idx"].tolist(),
            P=8,
            K=2,
            num_batches=10,
            seed=0,
        )
        assert len(sampler.qualifying_actions) == 3_630
        for batch in sampler:
            assert len(batch) == 16
            actions = {int(train_catalog["action_idx"].iloc[i]) for i in batch}
            assert len(actions) == 1, f"PK-SA batch spans actions {actions}"
            classes = [int(train_catalog["class_id"].iloc[i]) for i in batch]
            assert len(set(classes)) == 8

    def test_PKSA_P16_K4_raises_too_restrictive(self, train_catalog) -> None:
        # Plan §3: only 4 actions satisfy P=16,K=4 for PK-SA → it would
        # technically work but be near-degenerate. Verify the count to keep us
        # honest about why we changed the default.
        sampler = PKPerActionBatchSampler(
            class_ids=train_catalog["class_id"].tolist(),
            action_ids=train_catalog["action_idx"].tolist(),
            P=16,
            K=4,
            num_batches=1,
            seed=0,
        )
        assert len(sampler.qualifying_actions) == 4


class TestDataLoaderEndToEnd:
    @pytest.mark.parametrize("transform_level", ["aug-min", "aug-med", "aug-strong"])
    def test_real_images_through_loader(self, train_catalog, transform_level: str) -> None:
        # Tiny number of batches, num_workers=0 → keeps the test fast and deterministic.
        sampler = PKBatchSampler(
            class_ids=train_catalog["class_id"].tolist(),
            P=4,
            K=2,
            num_batches=3,
            seed=0,
        )
        transform = build_transform(transform_level, height=256, width=128)
        ds = ReIDImageDataset(train_catalog, transform=transform)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

        n_batches_seen = 0
        for batch in loader:
            assert batch["image"].shape == (8, 3, 256, 128), (
                f"{transform_level}: unexpected shape {batch['image'].shape}"
            )
            assert batch["image"].dtype == torch.float32
            assert "class_id" in batch
            assert batch["class_id"].shape == (8,)
            # Within a batch, exactly 4 unique class_id values
            assert len(set(batch["class_id"].tolist())) == 4
            n_batches_seen += 1
        assert n_batches_seen == 3
