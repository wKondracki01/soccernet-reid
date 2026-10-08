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
    "VGG11-BN": "vgg11_bn",
    "VGG16-BN": "vgg16_bn",
    "VGG11-BN-CONV": "vgg11_bn",
    "VGG16-BN-CONV": "vgg16_bn",
}

# VGG codes that stop after the convolutional layers (see _ConvOnlyVGG).
_CONV_ONLY: frozenset[str] = frozenset({"VGG11-BN-CONV", "VGG16-BN-CONV"})


class _ConvOnlyVGG(nn.Module):
    """Convolutional part of a timm VGG followed by global average pooling.

    timm's VGG with ``num_classes=0`` keeps its two fully connected layers (fc6
    as a 7x7 convolution, and fc7: about 120 M parameters, the same in VGG11 and
    VGG16) and returns a 4096-d vector. fc6 needs a map of at least 7x7, so for
    our 256x128 crops timm first stretches the 8x4 feature map to 8x7. The codes
    "VGG11-BN" / "VGG16-BN" are that model.

    This wrapper drops both layers, so VGG is used the way the other backbones
    are: last feature map -> average over positions -> 512-d vector.
    """

    def __init__(self, vgg: nn.Module) -> None:
        super().__init__()
        self.features = vgg.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(2, 3))


def create_backbone(
    name: str,
    pretrained: bool = True,
) -> nn.Module:
    """Build a feature-extractor backbone identified by our plan code.

    Args:
        name: one of the codes from §2.A (e.g., "R18", "EB1", "VGG16-BN"), or a
            "-CONV" VGG code for the convolutional layers alone.
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
    if name in _CONV_ONLY:
        return _ConvOnlyVGG(model)
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
