"""ReIDModel = backbone + head, with a uniform forward interface."""
from __future__ import annotations

import torch
import torch.nn as nn

from soccernet_reid.models.backbones import create_backbone, feature_dim
from soccernet_reid.models.heads import create_head


class ReIDModel(nn.Module):
    """Combine a timm backbone with a swappable head.

    Forward returns either:
        Tensor [B, D]               — for projection / plain / classifier_cut heads
        dict[str, Tensor]           — for bnneck (returns embedding_metric +
                                       embedding_retrieval)
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, head_name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_name = head_name
        self.embedding_dim: int = head.embedding_dim  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Single embedding used at inference (cosine retrieval)."""
        self.eval()
        out = self.forward(x)
        if isinstance(out, dict):
            return out["embedding_retrieval"]
        return out


def build_model(
    backbone_name: str,
    head_name: str,
    embedding_dim: int | None = 512,
    pretrained: bool = True,
) -> ReIDModel:
    """End-to-end factory: create backbone, probe its feature dim, attach head."""
    backbone = create_backbone(backbone_name, pretrained=pretrained)
    in_dim = feature_dim(backbone)
    head = create_head(head_name, in_dim=in_dim, embedding_dim=embedding_dim)
    return ReIDModel(backbone=backbone, head=head, head_name=head_name)
