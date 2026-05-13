"""Embedding heads attached to the backbone output.

Plan §5 / §7.3 / §7.1 require swappable heads. Four variants:

- ``projection`` (default for metric learning):
      backbone → BN → FC(D) → BN → L2-norm
- ``plain`` (ablation §7.1):
      backbone → FC(D)              (no BN, no L2)
- ``bnneck`` (BoT-ReID, used by hybrid Wariant H in §7.3):
      backbone → [embed_metric] → BN → [embed_retrieval]
      Triplet/Contrastive consume embed_metric (pre-BN).
      Retrieval and CE classifier consume embed_retrieval (post-BN, L2-norm).
- ``classifier_cut`` (Wariant K in F0b / §7.3):
      Backbone → optional FC bottleneck → embedding; classifier owned by the loss.

All heads accept a backbone feature tensor [B, in_dim] and either return a
Tensor (single embedding) or a dict with explicit keys when multiple outputs
are exposed. The training loop checks the type and routes to the right loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    """Kaiming init for Linear, BatchNorm with weight=1, bias=0."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ProjectionHead(nn.Module):
    """Standard metric-learning head: BN → FC(D) → BN → L2-norm.

    The double BatchNorm (one before, one after the FC) is the most stable
    variant we found in early ReID literature and matches BoT-ReID's post-BNNeck
    formulation when D == in_dim.
    """

    def __init__(self, in_dim: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim
        _init_weights(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.bn_in(features)
        x = self.fc(x)
        x = self.bn_out(x)
        return F.normalize(x, p=2, dim=1)


class PlainHead(nn.Module):
    """FC-only embedding: no BN, no L2-norm. For ablation §7.1 (norm OFF)."""

    def __init__(self, in_dim: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        _init_weights(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class BNNeckHead(nn.Module):
    """BoT-ReID (Luo et al. 2019) hybrid head producing two embeddings.

    Output dict keys:
        "embedding_metric"    : pre-BN feature, used by triplet/contrastive.
        "embedding_retrieval" : post-BN feature, L2-normed; used at inference and
                                fed to the CE classifier in hybrid Wariant H.

    If `embedding_dim != in_dim`, an FC reduces dim BEFORE the BN.
    """

    def __init__(self, in_dim: int, embedding_dim: int | None = None) -> None:
        super().__init__()
        target_dim = embedding_dim if embedding_dim else in_dim
        if embedding_dim and embedding_dim != in_dim:
            self.fc = nn.Linear(in_dim, embedding_dim, bias=False)
        else:
            self.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(target_dim)
        # By convention BNNeck disables bias in the BN's affine to keep features symmetric.
        self.bn.bias.requires_grad_(False)
        self.embedding_dim = target_dim
        _init_weights(self)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_bn = self.fc(features)
        post_bn = self.bn(pre_bn)
        return {
            "embedding_metric": pre_bn,
            "embedding_retrieval": F.normalize(post_bn, p=2, dim=1),
        }


class ClassifierCutHead(nn.Module):
    """Wariant K head: optional bottleneck for embedding; classifier elsewhere.

    The classifier itself is owned by the LOSS (CE or ArcFace) — this matches
    pytorch-metric-learning's ArcFaceLoss which carries its own weight matrix.
    At training and inference this head returns the bottleneck output, which is
    what the loss layer's classifier consumes.

    If `embedding_dim` is None or equal to in_dim, the head is an Identity.
    """

    def __init__(self, in_dim: int, embedding_dim: int | None = None) -> None:
        super().__init__()
        if embedding_dim and embedding_dim != in_dim:
            self.bottleneck = nn.Sequential(
                nn.Linear(in_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
            )
            self.embedding_dim = embedding_dim
        else:
            self.bottleneck = nn.Identity()
            self.embedding_dim = in_dim
        _init_weights(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(features)


_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "projection": ProjectionHead,
    "plain": PlainHead,
    "bnneck": BNNeckHead,
    "classifier_cut": ClassifierCutHead,
}


def create_head(
    name: str,
    in_dim: int,
    embedding_dim: int | None = 512,
) -> nn.Module:
    """Factory: build a head identified by its plan code.

    Args:
        name: one of {"projection", "plain", "bnneck", "classifier_cut"}.
        in_dim: backbone feature dim (output of `feature_dim(backbone)`).
        embedding_dim: target D; for `bnneck`/`classifier_cut`, None means
            keep in_dim (no FC reduction).
    """
    if name not in _HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head {name!r}; supported: {sorted(_HEAD_REGISTRY)}"
        )
    cls = _HEAD_REGISTRY[name]
    if cls in (ProjectionHead, PlainHead):
        assert embedding_dim is not None, f"{name} requires embedding_dim"
        return cls(in_dim=in_dim, embedding_dim=embedding_dim)
    # bnneck / classifier_cut accept embedding_dim=None
    return cls(in_dim=in_dim, embedding_dim=embedding_dim)
