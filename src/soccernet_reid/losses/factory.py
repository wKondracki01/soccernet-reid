"""Loss factory unifying metric-learning and classifier losses from §2.B.

All built losses expose the same call signature:
    loss_module(embeddings: Tensor[B, D], labels: Tensor[B]) -> Tensor scalar

For losses that need an internal classifier (CE, ArcFace), the classifier is
built into the loss module — the embedding head (e.g. `projection` or
`classifier_cut`) does not need to know about it.

Six losses from §2.B:
    "tri"     — triplet loss with batch-hard mining
    "cont"    — contrastive loss, all-pairs (no miner)
    "ms"      — multi-similarity loss with MultiSimilarityMiner
    "circle"  — CircleLoss
    "ce"      — cross-entropy with label smoothing (classifier built-in)
    "arc"     — ArcFace via pytorch-metric-learning (classifier built-in)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners as pml_miners

_KNOWN: tuple[str, ...] = ("tri", "cont", "ms", "circle", "ce", "arc")


@dataclass
class LossModule:
    """Bundle of (loss callable, optional miner) + metadata.

    The training loop does:
        if module.requires_classes:
            # CE / Arc: need num_classes set at build time
            loss = module.call(embeddings, labels)
        else:
            # Metric: maybe run miner first
            pairs = module.miner(embeddings, labels) if module.miner else None
            loss = module.call(embeddings, labels, pairs)
    """

    name: str
    call: nn.Module               # operates on (embeddings, labels [, mined])
    miner: nn.Module | None       # only metric-style losses; None means all-pairs
    requires_classes: bool        # True for CE / ArcFace (need num_classes)
    embedding_dim: int            # the D the loss expects on its input


class _MetricLossWrapper(nn.Module):
    """Adapter: pml metric loss + optional miner.

    Calling pattern in the training loop:
        loss = wrapper(embeddings, labels)   # miner is applied internally.
    """

    def __init__(self, loss: nn.Module, miner: nn.Module | None = None) -> None:
        super().__init__()
        self.loss = loss
        self.miner = miner

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.miner is not None:
            mined = self.miner(embeddings, labels)
            return self.loss(embeddings, labels, mined)
        return self.loss(embeddings, labels)


class _CEWithClassifier(nn.Module):
    """nn.CrossEntropyLoss on top of a learned linear classifier (label smoothing)."""

    def __init__(self, embedding_dim: int, num_classes: int, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_out", nonlinearity="linear")
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(embeddings)
        return self.loss(logits, labels)


def build_loss(
    name: str,
    embedding_dim: int,
    num_classes: int | None = None,
    **kwargs: Any,
) -> LossModule:
    """Construct one of the six losses by name.

    Args:
        name: one of {"tri", "cont", "ms", "circle", "ce", "arc"}.
        embedding_dim: D, the embedding dimension the loss receives.
        num_classes: required for "ce" and "arc"; ignored otherwise.
        **kwargs: per-loss hyperparameters (e.g. `margin`, `scale`, `label_smoothing`).
            Defaults match §2.B of the plan.
    """
    if name not in _KNOWN:
        raise ValueError(f"Unknown loss {name!r}; supported: {sorted(_KNOWN)}")

    if name in ("ce", "arc") and num_classes is None:
        raise ValueError(f"Loss {name!r} requires num_classes")

    if name == "tri":
        margin = kwargs.get("margin", 0.3)
        # BatchHardMiner: hardest positive and hardest negative per anchor in the batch.
        miner = pml_miners.BatchHardMiner()
        loss = pml_losses.TripletMarginLoss(margin=margin)
        return LossModule(
            name=name,
            call=_MetricLossWrapper(loss=loss, miner=miner),
            miner=miner,
            requires_classes=False,
            embedding_dim=embedding_dim,
        )

    if name == "cont":
        # All-pairs contrastive (no miner). pml's ContrastiveLoss expects
        # pos_margin (closer than) and neg_margin (further than).
        pos_margin = kwargs.get("pos_margin", 0.0)
        neg_margin = kwargs.get("neg_margin", 0.5)
        loss = pml_losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)
        return LossModule(
            name=name,
            call=_MetricLossWrapper(loss=loss, miner=None),
            miner=None,
            requires_classes=False,
            embedding_dim=embedding_dim,
        )

    if name == "ms":
        # MultiSimilarityMiner + MultiSimilarityLoss are designed to be used together.
        epsilon = kwargs.get("epsilon", 0.1)
        miner = pml_miners.MultiSimilarityMiner(epsilon=epsilon)
        alpha = kwargs.get("alpha", 2.0)
        beta = kwargs.get("beta", 50.0)
        base = kwargs.get("base", 1.0)
        loss = pml_losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        return LossModule(
            name=name,
            call=_MetricLossWrapper(loss=loss, miner=miner),
            miner=miner,
            requires_classes=False,
            embedding_dim=embedding_dim,
        )

    if name == "circle":
        m = kwargs.get("m", 0.25)
        gamma = kwargs.get("gamma", 64)
        loss = pml_losses.CircleLoss(m=m, gamma=gamma)
        # CircleLoss is typically used with a hard-pair miner in BoT-ReID
        miner = pml_miners.BatchHardMiner()
        return LossModule(
            name=name,
            call=_MetricLossWrapper(loss=loss, miner=miner),
            miner=miner,
            requires_classes=False,
            embedding_dim=embedding_dim,
        )

    if name == "ce":
        label_smoothing = kwargs.get("label_smoothing", 0.1)
        ce = _CEWithClassifier(embedding_dim, num_classes, label_smoothing=label_smoothing)
        return LossModule(
            name=name,
            call=ce,
            miner=None,
            requires_classes=True,
            embedding_dim=embedding_dim,
        )

    # name == "arc"
    margin = kwargs.get("margin", 0.5)
    scale = kwargs.get("scale", 30.0)
    # pml's ArcFaceLoss owns a learnable weight matrix (num_classes, embedding_dim)
    arc = pml_losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_dim,
        margin=margin,
        scale=scale,
    )

    class _ArcAdapter(nn.Module):
        def __init__(self, arc_loss: nn.Module) -> None:
            super().__init__()
            self.arc = arc_loss

        def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return self.arc(embeddings, labels)

    return LossModule(
        name=name,
        call=_ArcAdapter(arc),
        miner=None,
        requires_classes=True,
        embedding_dim=embedding_dim,
    )
