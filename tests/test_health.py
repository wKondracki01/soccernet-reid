"""Tests for the training diagnostics in soccernet_reid.training.health."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from soccernet_reid.models import build_model
from soccernet_reid.training.health import (
    ALIVE_THRESHOLD,
    batch_embedding_stats,
    gradient_norm,
    weight_health,
)


class _Tiny(nn.Module):
    """conv(2x1x1x1) -> BN(2) -> flatten -> linear(2 -> 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 3, bias=True)


class TestWeightHealth:
    def test_fresh_model_is_fully_alive(self) -> None:
        torch.manual_seed(0)
        stats = weight_health(_Tiny())
        assert stats["alive_weight_frac"] == 1.0
        assert stats["alive_bn_channel_frac"] == 1.0
        assert "alive_embedding_dims" not in stats  # no head

    def test_counts_dead_weights_and_channels(self) -> None:
        m = _Tiny()
        with torch.no_grad():
            m.conv.weight.copy_(torch.tensor([1.0, 0.0]).view(2, 1, 1, 1))
            m.fc.weight.copy_(torch.tensor([[1.0, 1e-41], [0.0, 2.0], [3.0, 1e-20]]))
            m.fc.bias.zero_()                      # 1-D: not counted as a weight
            m.bn.weight.copy_(torch.tensor([0.5, 1e-41]))
        stats = weight_health(m)
        # 8 weights with ndim >= 2; alive: conv 1, fc 1.0 / 2.0 / 3.0 -> 4
        assert stats["alive_weight_frac"] == pytest.approx(4 / 8)
        assert stats["alive_bn_channel_frac"] == pytest.approx(1 / 2)
        assert stats["weight_norm"] == pytest.approx(math.sqrt(1 + 1 + 4 + 9))

    def test_threshold_separates_denormals_from_small_live_weights(self) -> None:
        m = _Tiny()
        with torch.no_grad():
            m.conv.weight.copy_(torch.tensor([1e-6, 1e-41]).view(2, 1, 1, 1))
            m.fc.weight.fill_(1.0)
        assert 1e-41 < ALIVE_THRESHOLD < 1e-6
        assert weight_health(m)["alive_weight_frac"] == pytest.approx(7 / 8)

    def test_projection_head_dims(self) -> None:
        model = build_model("R18", "projection", embedding_dim=64, pretrained=False)
        stats = weight_health(model)
        assert stats["alive_feature_dims"] == 512
        assert stats["alive_embedding_dims"] == 64
        with torch.no_grad():
            model.head.bn_out.weight[:10] = 0.0
            model.head.bn_in.weight[:100] = 0.0
        stats = weight_health(model)
        assert stats["alive_feature_dims"] == 412
        assert stats["alive_embedding_dims"] == 54

    def test_bnneck_and_headless_bn(self) -> None:
        assert weight_health(build_model("R18", "bnneck", 64, pretrained=False))["alive_embedding_dims"] == 64
        plain = weight_health(build_model("R18", "plain", 64, pretrained=False))
        assert "alive_embedding_dims" not in plain and "alive_feature_dims" not in plain

    def test_does_not_modify_the_model(self) -> None:
        torch.manual_seed(0)
        m = _Tiny()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        weight_health(m)
        for k, v in m.state_dict().items():
            assert torch.equal(v, before[k])
        assert all(p.grad is None for p in m.parameters())


class TestBatchEmbeddingStats:
    def test_known_geometry(self) -> None:
        # Two classes, two samples each, on the unit circle.
        emb = torch.tensor([[1.0, 0.0], [0.0, 1.0],      # class 0: 90 degrees apart
                            [-1.0, 0.0], [1.0, 0.0]])    # class 1: opposite; [3] equals [0]
        labels = torch.tensor([0, 0, 1, 1])
        s = batch_embedding_stats(emb, labels)
        r2 = math.sqrt(2.0)
        # pairs: (0,1) r2, (0,2) 2, (0,3) 0, (1,2) r2, (1,3) r2, (2,3) 2
        assert s["pairwise_dist_mean"] == pytest.approx((3 * r2 + 4) / 6, abs=1e-6)
        # farthest positive: r2, r2, 2, 2 ; nearest negative: 0, r2, r2, 0
        assert s["d_ap_mean"] == pytest.approx((2 * r2 + 4) / 4, abs=1e-6)
        assert s["d_an_hardest_mean"] == pytest.approx(2 * r2 / 4, abs=1e-6)
        assert s["hard_frac"] == pytest.approx(3 / 4)   # anchor 1 is a tie, not hard

    def test_scale_of_embeddings_does_not_matter(self) -> None:
        torch.manual_seed(1)
        emb = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        a = batch_embedding_stats(emb, labels)
        b = batch_embedding_stats(emb * 37.0, labels)
        assert a == pytest.approx(b, abs=1e-5)

    def test_collapsed_batch_has_zero_spread(self) -> None:
        emb = torch.ones(6, 8)
        s = batch_embedding_stats(emb, torch.tensor([0, 0, 1, 1, 2, 2]))
        assert s["pairwise_dist_mean"] == pytest.approx(0.0, abs=1e-3)
        assert s["hard_frac"] == 0.0

    def test_zero_vectors_count_as_collapsed(self) -> None:
        # a dead network outputs the zero vector for every image
        s = batch_embedding_stats(torch.zeros(4, 8), torch.tensor([0, 0, 1, 1]))
        assert all(math.isfinite(v) for v in s.values())
        assert s["pairwise_dist_mean"] == 0.0
        assert s["d_ap_mean"] == 0.0 and s["d_an_hardest_mean"] == 0.0

    def test_batch_without_positives_reports_only_spread(self) -> None:
        torch.manual_seed(2)
        s = batch_embedding_stats(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]))
        assert set(s) == {"pairwise_dist_mean"}

    def test_single_sample(self) -> None:
        assert batch_embedding_stats(torch.randn(1, 8), torch.tensor([0])) == {}

    def test_no_gradient_is_created(self) -> None:
        emb = torch.randn(4, 8, requires_grad=True)
        batch_embedding_stats(emb, torch.tensor([0, 0, 1, 1]))
        assert emb.grad is None


class TestGradientNorm:
    def test_matches_manual_norm(self) -> None:
        a = nn.Parameter(torch.tensor([3.0, 0.0]))
        b = nn.Parameter(torch.tensor([[4.0]]))
        c = nn.Parameter(torch.tensor([7.0]))       # never used: grad stays None
        (a.sum() * 3.0 + b.sum() * 4.0).backward()  # grads: [3, 3], [[4]]
        assert gradient_norm([a, b, c]) == pytest.approx(math.sqrt(9 + 9 + 16))
        assert gradient_norm([a, b, c], scale=2.0) == pytest.approx(math.sqrt(34) / 2)

    def test_no_gradients(self) -> None:
        assert gradient_norm([nn.Parameter(torch.ones(3))]) == 0.0
