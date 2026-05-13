"""Unit tests for heads and the full ReIDModel."""
from __future__ import annotations

import pytest
import torch

from soccernet_reid.models import (
    BNNeckHead,
    ClassifierCutHead,
    PlainHead,
    ProjectionHead,
    build_model,
    create_head,
)


# We must use BN-compatible batches (B≥2) since the heads use BatchNorm1d.
BATCH = 4
IN_DIM = 512   # R18 feature dim


class TestProjectionHead:
    def test_output_shape_and_l2_norm(self) -> None:
        head = ProjectionHead(in_dim=IN_DIM, embedding_dim=256).eval()
        x = torch.randn(BATCH, IN_DIM)
        out = head(x)
        assert out.shape == (BATCH, 256)
        norms = out.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_changes_dim(self) -> None:
        head = ProjectionHead(in_dim=IN_DIM, embedding_dim=128).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        assert out.shape == (BATCH, 128)


class TestPlainHead:
    def test_output_shape_no_norm(self) -> None:
        head = PlainHead(in_dim=IN_DIM, embedding_dim=512).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        assert out.shape == (BATCH, 512)
        norms = out.norm(dim=1)
        # Plain head does NOT normalize — norms should NOT all be 1
        assert not torch.allclose(norms, torch.ones(BATCH), atol=1e-3)


class TestBNNeckHead:
    def test_returns_dict_with_two_keys(self) -> None:
        head = BNNeckHead(in_dim=IN_DIM, embedding_dim=None).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        assert isinstance(out, dict)
        assert set(out) == {"embedding_metric", "embedding_retrieval"}

    def test_retrieval_branch_is_l2_normed(self) -> None:
        head = BNNeckHead(in_dim=IN_DIM, embedding_dim=None).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        norms = out["embedding_retrieval"].norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_metric_branch_not_normed(self) -> None:
        head = BNNeckHead(in_dim=IN_DIM, embedding_dim=None).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        norms = out["embedding_metric"].norm(dim=1)
        assert not torch.allclose(norms, torch.ones(BATCH), atol=1e-3)

    def test_with_dim_reduction(self) -> None:
        head = BNNeckHead(in_dim=IN_DIM, embedding_dim=256).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        assert out["embedding_metric"].shape == (BATCH, 256)
        assert out["embedding_retrieval"].shape == (BATCH, 256)


class TestClassifierCutHead:
    def test_identity_when_no_dim_reduction(self) -> None:
        head = ClassifierCutHead(in_dim=IN_DIM, embedding_dim=None).eval()
        x = torch.randn(BATCH, IN_DIM)
        out = head(x)
        assert out.shape == (BATCH, IN_DIM)
        torch.testing.assert_close(out, x)

    def test_with_bottleneck(self) -> None:
        head = ClassifierCutHead(in_dim=IN_DIM, embedding_dim=256).eval()
        out = head(torch.randn(BATCH, IN_DIM))
        assert out.shape == (BATCH, 256)


class TestFactory:
    @pytest.mark.parametrize(
        "name,embedding_dim,expected_dim",
        [
            ("projection", 512, 512),
            ("projection", 128, 128),
            ("plain", 256, 256),
            ("bnneck", None, IN_DIM),
            ("bnneck", 256, 256),
            ("classifier_cut", None, IN_DIM),
            ("classifier_cut", 128, 128),
        ],
    )
    def test_factory_produces_correct_dim(self, name, embedding_dim, expected_dim) -> None:
        head = create_head(name, in_dim=IN_DIM, embedding_dim=embedding_dim).eval()
        assert head.embedding_dim == expected_dim

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown head"):
            create_head("nonexistent", in_dim=IN_DIM, embedding_dim=512)


class TestReIDModelEndToEnd:
    """Test that the full backbone + head pipeline returns expected shapes."""

    def test_r18_projection(self) -> None:
        model = build_model("R18", "projection", embedding_dim=256, pretrained=False).eval()
        x = torch.randn(BATCH, 3, 256, 128)
        out = model(x)
        assert out.shape == (BATCH, 256)
        torch.testing.assert_close(out.norm(dim=1), torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_r18_bnneck(self) -> None:
        model = build_model("R18", "bnneck", embedding_dim=None, pretrained=False).eval()
        x = torch.randn(BATCH, 3, 256, 128)
        out = model(x)
        assert isinstance(out, dict)
        assert out["embedding_retrieval"].shape == (BATCH, 512)
        assert out["embedding_metric"].shape == (BATCH, 512)

    def test_extract_embedding_handles_both_head_types(self) -> None:
        # ProjectionHead path
        m1 = build_model("R18", "projection", embedding_dim=256, pretrained=False)
        x = torch.randn(BATCH, 3, 256, 128)
        emb1 = m1.extract_embedding(x)
        assert emb1.shape == (BATCH, 256)
        # BNNeck path
        m2 = build_model("R18", "bnneck", embedding_dim=None, pretrained=False)
        emb2 = m2.extract_embedding(x)
        assert emb2.shape == (BATCH, 512)
        torch.testing.assert_close(emb2.norm(dim=1), torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_gradient_flows_through_head(self) -> None:
        model = build_model("R18", "projection", embedding_dim=128, pretrained=False)
        model.train()
        x = torch.randn(BATCH, 3, 256, 128, requires_grad=False)
        out = model(x)
        loss = out.pow(2).sum()
        loss.backward()
        # Some parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
