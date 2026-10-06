"""Training diagnostics: how much of the network is still in use.

`torch.optim.Adam` applies `weight_decay` as an L2 term inside the adaptive
update, so a weight whose loss gradient is weak is pulled towards zero at close
to the full learning-rate speed, and a channel that reaches zero no longer
receives a gradient from the loss. In the May 2026 runs trained with the
triplet loss 97-99% of the convolution weights ended as numerical zeros (all of
them in the collapsed AUG-STRONG / AUG-BOT runs), which was only found in the
saved checkpoints months later. These helpers make it visible during training.

All functions only read the model / tensors they are given.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

# Dead weights sit around 1e-41 (float32 denormals); live ones are many orders above.
ALIVE_THRESHOLD = 1e-12

# Head attribute -> name of the count reported for its BatchNorm scale.
_HEAD_BN_KEYS: tuple[tuple[str, str], ...] = (
    ("bn_in", "alive_feature_dims"),      # backbone features the head still reads
    ("bn_out", "alive_embedding_dims"),   # embedding dimensions that still vary
    ("bn", "alive_embedding_dims"),       # bnneck
)


def weight_health(model: nn.Module, threshold: float = ALIVE_THRESHOLD) -> dict[str, float]:
    """Shares of weights and channels that are not numerically zero.

    Keys:
        alive_weight_frac      conv / linear weights (parameters with ndim >= 2)
        alive_bn_channel_frac  BatchNorm scales (gamma) in the whole model
        weight_norm            L2 norm of all conv / linear weights
        alive_feature_dims     see ``_HEAD_BN_KEYS`` (only for heads that have it)
        alive_embedding_dims   see ``_HEAD_BN_KEYS`` (only for heads that have it)

    A BatchNorm channel with gamma = 0 outputs the same value for every image,
    so it carries no information even when its bias is non-zero.
    """
    n_weights = 0
    alive_weights: torch.Tensor | int = 0
    sq_sum: torch.Tensor | float = 0.0
    for p in model.parameters():
        if p.ndim >= 2:
            w = p.detach()
            n_weights += w.numel()
            alive_weights = alive_weights + (w.abs() > threshold).sum()
            sq_sum = sq_sum + w.float().pow(2).sum()

    n_channels = 0
    alive_channels: torch.Tensor | int = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm) and m.weight is not None:
            g = m.weight.detach()
            n_channels += g.numel()
            alive_channels = alive_channels + (g.abs() > threshold).sum()

    stats = {
        "alive_weight_frac": float(alive_weights) / n_weights if n_weights else float("nan"),
        "alive_bn_channel_frac": float(alive_channels) / n_channels if n_channels else float("nan"),
        "weight_norm": float(sq_sum) ** 0.5,
    }
    head = getattr(model, "head", None)
    for attr, key in _HEAD_BN_KEYS:
        bn = getattr(head, attr, None)
        if isinstance(bn, _BatchNorm) and bn.weight is not None:
            stats[key] = float((bn.weight.detach().abs() > threshold).sum())
    return stats


def batch_embedding_stats(embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """How spread out the embeddings of one batch are, and how hard its triplets are.

    Distances are Euclidean between L2-normalised embeddings (range 0..2), as
    the triplet loss sees them.

    Keys:
        pairwise_dist_mean  mean distance over all pairs; 0 means every image
                            got the same embedding
        d_ap_mean           mean distance to the farthest positive       )  only if some
        d_an_hardest_mean   mean distance to the nearest negative        )  anchor has both
        hard_frac           share of anchors whose nearest negative is   )  a positive and
                            closer than their farthest positive          )  a negative
    """
    emb = F.normalize(embeddings.detach().float(), p=2, dim=1)
    n = emb.shape[0]
    if n < 2:
        return {}
    # |a - b|^2 = |a|^2 + |b|^2 - 2ab; the norms stay in so that all-zero
    # embeddings (a dead network) come out at distance 0, not sqrt(2).
    sq = (emb * emb).sum(dim=1)
    dist = (sq[:, None] + sq[None, :] - 2.0 * emb @ emb.T).clamp_min(0.0).sqrt()
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    stats = {"pairwise_dist_mean": float(dist[~eye].mean())}

    same = labels[:, None] == labels[None, :]
    pos, neg = same & ~eye, ~same
    valid = pos.any(dim=1) & neg.any(dim=1)
    if bool(valid.any()):
        d_ap = dist.masked_fill(~pos, float("-inf")).max(dim=1).values[valid]
        d_an = dist.masked_fill(~neg, float("inf")).min(dim=1).values[valid]
        stats["d_ap_mean"] = float(d_ap.mean())
        stats["d_an_hardest_mean"] = float(d_an.mean())
        stats["hard_frac"] = float((d_an < d_ap).float().mean())
    return stats


def gradient_norm(parameters: Iterable[nn.Parameter], scale: float = 1.0) -> float:
    """L2 norm of the gradients currently stored on ``parameters``.

    Call after ``backward()`` and before ``optimizer.step()``: this is the
    gradient of the loss alone, because Adam adds the weight-decay term inside
    its step. With AMP pass ``scale=scaler.get_scale()`` to undo the loss scaling.
    """
    total: torch.Tensor | None = None
    for p in parameters:
        if p.grad is not None:
            s = p.grad.detach().float().pow(2).sum()
            total = s if total is None else total + s
    return 0.0 if total is None else float(total.sqrt()) / scale
