"""Backbone factory wrapping `timm`.

For the smoke test (Krok 2) we use a feature-extractor-only backbone:
    timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")

This already gives us the post-GAP feature vector. The full projection head with
BN+FC+L2 (planned in §5) comes later in Krok 4. For evaluation purposes we just
need raw embeddings; we L2-normalize at inference time when using cosine.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


# Mapping plan codes → timm model names. Only includes backbones from §2.A.
_TIMM_NAMES: dict[str, str] = {
    "R18": "resnet18",
    "R34": "resnet34",
    "EB1": "efficientnet_b1",
    "EB2": "efficientnet_b2",
    "VGG16-BN": "vgg16_bn",
    "VGG11-BN": "vgg11_bn",
}


def create_backbone(
    name: str,
    pretrained: bool = True,
) -> nn.Module:
    """Build a feature-extractor backbone identified by our plan code.

    Args:
        name: one of the codes from §2.A (e.g., "R18", "EB1", "VGG16-BN").
        pretrained: load ImageNet weights.

    Returns:
        nn.Module that maps [B, 3, H, W] → [B, feat_dim]. No projection head.
    """
    if name not in _TIMM_NAMES:
        raise ValueError(
            f"Unknown backbone {name!r}; supported: {sorted(_TIMM_NAMES)}"
        )
    timm_name = _TIMM_NAMES[name]
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=0,         # remove classifier head
        global_pool="avg",     # GAP
    )
    return model


def feature_dim(backbone: nn.Module, input_hw: tuple[int, int] = (256, 128)) -> int:
    """Probe the backbone's output feature dimension.

    Uses the model's current device/dtype to avoid cross-device errors.
    """
    h, w = input_hw
    backbone.eval()
    param = next(backbone.parameters())
    with torch.no_grad():
        dummy = torch.zeros(1, 3, h, w, device=param.device, dtype=param.dtype)
        feat = backbone(dummy)
    return int(feat.shape[-1])
