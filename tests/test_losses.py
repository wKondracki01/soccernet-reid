"""Unit tests for the loss factory. Synthetic embeddings/labels, no real data."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from soccernet_reid.losses import build_loss


# Use a PK-like batch: 4 classes × 2 samples = 8
P, K = 4, 2
B = P * K
D = 16
NUM_CLASSES = 10


def _random_pk_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random L2-normalized embeddings + PK-style labels."""
    torch.manual_seed(seed)
    embeddings = F.normalize(torch.randn(B, D), p=2, dim=1)
    labels = torch.tensor([cls for cls in range(P) for _ in range(K)])
    return embeddings, labels


@pytest.mark.parametrize("name", ["tri", "cont", "ms", "circle"])
def test_metric_losses_return_finite_scalar(name: str) -> None:
    module = build_loss(name, embedding_dim=D)
    assert module.requires_classes is False
    emb, lbl = _random_pk_batch()
    out = module.call(emb, lbl)
    assert out.ndim == 0
    assert torch.isfinite(out)
    # Loss should be > 0 for random embeddings
    assert out.item() > 0


@pytest.mark.parametrize("name", ["ce", "arc"])
def test_classifier_losses_return_finite_scalar(name: str) -> None:
    module = build_loss(name, embedding_dim=D, num_classes=NUM_CLASSES)
    assert module.requires_classes is True
    emb, lbl = _random_pk_batch()
    out = module.call(emb, lbl)
    assert out.ndim == 0
    assert torch.isfinite(out)
    assert out.item() > 0


@pytest.mark.parametrize("name", ["tri", "cont", "ms", "circle", "ce", "arc"])
def test_loss_produces_gradients(name: str) -> None:
    module = build_loss(
        name,
        embedding_dim=D,
        num_classes=NUM_CLASSES if name in ("ce", "arc") else None,
    )
    emb, lbl = _random_pk_batch()
    emb.requires_grad_(True)
    out = module.call(emb, lbl)
    out.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()
    # Gradient should be nonzero (loss > 0, so something updates)
    assert emb.grad.abs().sum().item() > 0


class TestErrors:
    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss("nope", embedding_dim=D)

    def test_classifier_loss_needs_num_classes(self) -> None:
        with pytest.raises(ValueError, match="requires num_classes"):
            build_loss("ce", embedding_dim=D)
        with pytest.raises(ValueError, match="requires num_classes"):
            build_loss("arc", embedding_dim=D)


class TestHyperparameters:
    def test_triplet_margin_override(self) -> None:
        # Just verify that hyperparam keyword is accepted (the actual value
        # is checked by pml itself)
        m = build_loss("tri", embedding_dim=D, margin=0.5)
        emb, lbl = _random_pk_batch()
        out = m.call(emb, lbl)
        assert torch.isfinite(out)

    def test_arc_margin_and_scale_override(self) -> None:
        m = build_loss("arc", embedding_dim=D, num_classes=NUM_CLASSES,
                       margin=0.4, scale=20.0)
        emb, lbl = _random_pk_batch()
        out = m.call(emb, lbl)
        assert torch.isfinite(out)


def test_optimization_step_decreases_loss_for_triplet() -> None:
    """Smoke check that backprop on a tiny model actually reduces a triplet loss."""
    torch.manual_seed(0)
    # Trivial 2-layer net producing D=16 embeddings from a B=8 input
    net = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, D),
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    module = build_loss("tri", embedding_dim=D, margin=0.3)

    # Build a fixed input + PK labels
    x = torch.randn(B, 32)
    labels = torch.tensor([cls for cls in range(P) for _ in range(K)])

    initial = None
    final = None
    for step in range(50):
        emb = F.normalize(net(x), p=2, dim=1)
        loss = module.call(emb, labels)
        if step == 0:
            initial = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final = loss.item()

    assert initial is not None and final is not None
    assert final < initial * 0.5, f"Loss did not drop: {initial:.4f} → {final:.4f}"
