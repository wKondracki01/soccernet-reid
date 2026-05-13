"""Sanity tests for the training loop on tiny synthetic data.

Avoids the real dataset to stay fast and runs everywhere (CPU is enough).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from soccernet_reid.data.dataset import ReIDImageDataset
from soccernet_reid.losses import build_loss
from soccernet_reid.models import build_model
from soccernet_reid.samplers import PKBatchSampler
from soccernet_reid.training import (
    enable_determinism,
    pick_device,
    seed_everything,
    train_one_epoch,
)
from soccernet_reid.training.loop import cosine_lr_with_warmup
from soccernet_reid.transforms import build_transform


@pytest.fixture
def tiny_dataset(tmp_path) -> object:
    """8 random PNG images in 4 classes (PK-friendly with K=2)."""
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for cls_id in range(4):
        for sample in range(2):
            arr = rng.integers(0, 256, size=(80, 40, 3), dtype=np.uint8)
            path = tmp_path / f"cls{cls_id}_s{sample}.png"
            Image.fromarray(arr, mode="RGB").save(path)
            rows.append({
                "path": str(path),
                "bbox_idx": 1000 + cls_id * 10 + sample,
                "action_idx": 0,
                "person_uid": cls_id,
                "class_id": cls_id,
            })
    df = pd.DataFrame(rows)
    return df


def _make_loader(df, P=4, K=2, num_batches=10, seed=0):
    transform = build_transform("aug-min", height=256, width=128)
    ds = ReIDImageDataset(df, transform=transform)
    sampler = PKBatchSampler(
        class_ids=df["class_id"].tolist(), P=P, K=K, num_batches=num_batches, seed=seed,
    )
    return DataLoader(ds, batch_sampler=sampler, num_workers=0)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(42)
    a1 = torch.randn(5)
    np_a1 = np.random.rand(3)
    seed_everything(42)
    a2 = torch.randn(5)
    np_a2 = np.random.rand(3)
    torch.testing.assert_close(a1, a2)
    np.testing.assert_array_equal(np_a1, np_a2)


def test_pick_device_returns_valid_device() -> None:
    dev = pick_device("auto")
    assert dev.type in {"cuda", "mps", "cpu"}


def test_train_one_epoch_runs_on_cpu(tiny_dataset) -> None:
    seed_everything(0)
    enable_determinism(deterministic=False)  # speed > strict determinism for tests
    device = torch.device("cpu")
    model = build_model("R18", "projection", embedding_dim=64, pretrained=False).to(device)
    loss_module = build_loss("tri", embedding_dim=64, margin=0.3)
    loader = _make_loader(tiny_dataset, P=4, K=2, num_batches=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = train_one_epoch(
        model=model,
        loss_module=loss_module,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        log_every=100,
        epoch=0,
    )
    assert "train_loss_mean" in metrics
    assert np.isfinite(metrics["train_loss_mean"])


def test_loss_decreases_over_two_epochs(tiny_dataset) -> None:
    """Run two epochs of triplet on the same 8 images: loss should drop."""
    seed_everything(0)
    enable_determinism(deterministic=False)
    device = torch.device("cpu")
    model = build_model("R18", "projection", embedding_dim=64, pretrained=False).to(device)
    loss_module = build_loss("tri", embedding_dim=64, margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = cosine_lr_with_warmup(optimizer, warmup_iters=2, total_iters=20)

    loader = _make_loader(tiny_dataset, P=4, K=2, num_batches=10, seed=0)
    epoch1 = train_one_epoch(
        model=model, loss_module=loss_module, loader=loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        log_every=100, epoch=0,
    )
    loader = _make_loader(tiny_dataset, P=4, K=2, num_batches=10, seed=1)
    epoch2 = train_one_epoch(
        model=model, loss_module=loss_module, loader=loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        log_every=100, epoch=1,
    )
    assert epoch2["train_loss_mean"] < epoch1["train_loss_mean"], (
        f"Loss did not drop: {epoch1['train_loss_mean']:.4f} → {epoch2['train_loss_mean']:.4f}"
    )


def test_train_one_epoch_with_classifier_loss(tiny_dataset) -> None:
    """CE loss path: requires_classes=True, embeds via classifier-cut head."""
    seed_everything(0)
    enable_determinism(deterministic=False)
    device = torch.device("cpu")
    model = build_model("R18", "classifier_cut", embedding_dim=64, pretrained=False).to(device)
    loss_module = build_loss(
        "ce", embedding_dim=model.embedding_dim, num_classes=4, label_smoothing=0.1
    )
    # Move classifier weights to device too
    loss_module.call.to(device)
    loader = _make_loader(tiny_dataset, P=4, K=2, num_batches=5)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.call.parameters()), lr=1e-3
    )

    metrics = train_one_epoch(
        model=model, loss_module=loss_module, loader=loader,
        optimizer=optimizer, scheduler=None, device=device,
        log_every=100, epoch=0,
    )
    assert np.isfinite(metrics["train_loss_mean"])


def test_cosine_lr_warmup_then_decay() -> None:
    optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = cosine_lr_with_warmup(optim, warmup_iters=5, total_iters=20, min_lr_ratio=0.0)

    lrs = []
    for _ in range(20):
        lrs.append(optim.param_groups[0]["lr"])
        scheduler.step()

    # Warmup: linear ramp from 1/5=0.2 to 1.0 in steps 0..4
    assert lrs[0] == pytest.approx(0.2)
    assert lrs[4] == pytest.approx(1.0)
    # After warmup: monotonic decrease, ends near 0
    assert lrs[-1] < 0.05
    for i in range(5, len(lrs) - 1):
        assert lrs[i + 1] <= lrs[i] + 1e-9
