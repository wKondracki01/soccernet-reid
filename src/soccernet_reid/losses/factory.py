"""Loss factory unifying metric-learning and classifier losses from §2.B and §2.C.

Public API
----------
    build_loss(
        name,                  # one of: tri, cont, ms, circle, ce, arc
        embedding_dim,         # D from the model head
        num_classes=None,      # required for ce/arc
        miner=None,            # override default miner: 'batch-hard' | 'semi-hard'
                               # | 'multi-similarity' | 'none' (all-pairs).
                               # None = sensible default per loss.
        xbm=False,             # wrap with Cross-Batch Memory (cross-batch negatives)
        xbm_memory_size=1024,  # XBM bank size; ignored if xbm=False
        miner_margin=0.3,      # used only by semi-hard TripletMarginMiner
        **kwargs               # loss-specific hyperparams (margin, alpha, ...)
    ) -> LossModule

Each LossModule has a callable that accepts (embeddings, labels) -> scalar loss
and applies any miner / XBM internally.

Loss × miner × XBM compatibility
--------------------------------
    Loss     | default miner    | accepts override | accepts XBM
    tri      | batch-hard       | yes              | yes
    cont     | none (all-pairs) | yes              | yes
    ms       | multi-similarity | yes              | yes
    circle   | batch-hard       | yes              | yes
    ce       | n/a              | rejected         | rejected
    arc      | n/a              | rejected         | rejected

This map mirrors the §2.C "pakiet" definitions:
    RAND        = sampler=random + miner=none
    PK-BH       = sampler=pk     + miner=batch-hard
    PK-SH       = sampler=pk     + miner=semi-hard
    PK-SA-BH    = sampler=pk_sa  + miner=batch-hard
    PK-BH-XBM   = sampler=pk     + miner=batch-hard + xbm=true
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners as pml_miners

_KNOWN_LOSSES: tuple[str, ...] = ("tri", "cont", "ms", "circle", "ce", "arc")
_KNOWN_MINERS: tuple[str, ...] = ("batch-hard", "semi-hard", "multi-similarity", "none")
_METRIC_LOSSES: frozenset[str] = frozenset({"tri", "cont", "ms", "circle"})
_CLASSIFIER_LOSSES: frozenset[str] = frozenset({"ce", "arc"})

# Default miner per metric loss — what we use if user doesn't override.
_DEFAULT_MINER: dict[str, str] = {
    "tri": "batch-hard",
    "cont": "none",
    "ms": "multi-similarity",
    "circle": "batch-hard",
}


@dataclass
class LossModule:
    """Bundle of loss callable + metadata.

    Training loop does:
        loss = module.call(embeddings, labels)   # mining + xbm done internally
    """

    name: str
    call: nn.Module
    requires_classes: bool
    embedding_dim: int
    miner_name: str | None       # for logging / debugging
    xbm_enabled: bool


# --- Miner factory ----------------------------------------------------------


def _build_miner(name: str, margin: float) -> nn.Module | None:
    """Build a miner by name; returns None for 'none' (all-pairs)."""
    if name == "none":
        return None
    if name == "batch-hard":
        return pml_miners.BatchHardMiner()
    if name == "semi-hard":
        # TripletMarginMiner with type_of_triplets="semihard" gives FaceNet-style
        # semi-hard mining: for each anchor, pick a negative that's farther than
        # the (hardest) positive but still within the margin.
        return pml_miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard")
    if name == "multi-similarity":
        return pml_miners.MultiSimilarityMiner()
    raise ValueError(f"Unknown miner {name!r}; supported: {_KNOWN_MINERS}")


# --- Loss adapters (call interface) -----------------------------------------


class _MetricLossWrapper(nn.Module):
    """Adapter: pml metric loss + optional miner. No XBM."""

    def __init__(self, loss: nn.Module, miner: nn.Module | None) -> None:
        super().__init__()
        self.loss = loss
        self.miner = miner

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.miner is not None:
            mined = self.miner(embeddings, labels)
            return self.loss(embeddings, labels, mined)
        return self.loss(embeddings, labels)


class _XBMWrapper(nn.Module):
    """Adapter: pml's CrossBatchMemory (memory bank of past embeddings).

    XBM (Wang et al. 2020) keeps a sliding window of recent embeddings and uses
    them as additional negatives in each step. Particularly effective for losses
    that benefit from many negatives per anchor (MS, CircleLoss, Contrastive).
    """

    def __init__(
        self,
        loss: nn.Module,
        embedding_dim: int,
        memory_size: int,
        miner: nn.Module | None,
    ) -> None:
        super().__init__()
        self.xbm = pml_losses.CrossBatchMemory(
            loss=loss,
            embedding_size=embedding_dim,
            memory_size=memory_size,
            miner=miner,  # XBM applies the miner over (current + memory) embeddings
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.xbm(embeddings, labels)


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


class _ArcAdapter(nn.Module):
    def __init__(self, arc_loss: nn.Module) -> None:
        super().__init__()
        self.arc = arc_loss

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.arc(embeddings, labels)


# --- Public factory ---------------------------------------------------------


def _build_metric_loss(name: str, **kwargs: Any) -> nn.Module:
    """Build the bare pml metric loss (without miner / XBM)."""
    if name == "tri":
        return pml_losses.TripletMarginLoss(margin=kwargs.get("margin", 0.3))
    if name == "cont":
        return pml_losses.ContrastiveLoss(
            pos_margin=kwargs.get("pos_margin", 0.0),
            neg_margin=kwargs.get("neg_margin", 0.5),
        )
    if name == "ms":
        return pml_losses.MultiSimilarityLoss(
            alpha=kwargs.get("alpha", 2.0),
            beta=kwargs.get("beta", 50.0),
            base=kwargs.get("base", 1.0),
        )
    if name == "circle":
        return pml_losses.CircleLoss(
            m=kwargs.get("m", 0.25),
            gamma=kwargs.get("gamma", 64),
        )
    raise ValueError(f"Not a metric loss: {name!r}")


def build_loss(
    name: str,
    embedding_dim: int,
    num_classes: int | None = None,
    miner: str | None = None,
    xbm: bool = False,
    xbm_memory_size: int = 1024,
    miner_margin: float = 0.3,
    **kwargs: Any,
) -> LossModule:
    """Construct one of the six losses with optional miner and XBM. See module docstring."""
    if name not in _KNOWN_LOSSES:
        raise ValueError(f"Unknown loss {name!r}; supported: {sorted(_KNOWN_LOSSES)}")

    # Classifier losses: no miner / xbm allowed
    if name in _CLASSIFIER_LOSSES:
        if miner is not None and miner != "none":
            raise ValueError(
                f"Loss {name!r} is classifier-based and does not accept miner={miner!r}"
            )
        if xbm:
            raise ValueError(f"Loss {name!r} is classifier-based and does not accept xbm=True")
        if num_classes is None:
            raise ValueError(f"Loss {name!r} requires num_classes")

        if name == "ce":
            ce = _CEWithClassifier(
                embedding_dim, num_classes,
                label_smoothing=kwargs.get("label_smoothing", 0.1),
            )
            return LossModule(
                name=name, call=ce, requires_classes=True,
                embedding_dim=embedding_dim, miner_name=None, xbm_enabled=False,
            )
        # arc
        arc = pml_losses.ArcFaceLoss(
            num_classes=num_classes, embedding_size=embedding_dim,
            margin=kwargs.get("margin", 0.5), scale=kwargs.get("scale", 30.0),
        )
        return LossModule(
            name=name, call=_ArcAdapter(arc), requires_classes=True,
            embedding_dim=embedding_dim, miner_name=None, xbm_enabled=False,
        )

    # Metric losses: build base loss, attach miner, optionally wrap with XBM
    base_loss = _build_metric_loss(name, **kwargs)

    miner_choice = miner if miner is not None else _DEFAULT_MINER[name]
    if miner_choice not in _KNOWN_MINERS:
        raise ValueError(f"Unknown miner {miner_choice!r}; supported: {_KNOWN_MINERS}")
    miner_module = _build_miner(miner_choice, margin=miner_margin)

    if xbm:
        callable_module = _XBMWrapper(
            loss=base_loss, embedding_dim=embedding_dim,
            memory_size=xbm_memory_size, miner=miner_module,
        )
    else:
        callable_module = _MetricLossWrapper(loss=base_loss, miner=miner_module)

    return LossModule(
        name=name, call=callable_module, requires_classes=False,
        embedding_dim=embedding_dim, miner_name=miner_choice, xbm_enabled=xbm,
    )
