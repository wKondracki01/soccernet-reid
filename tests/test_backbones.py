"""Backbone factory: the convolution-only VGG codes next to the full timm VGG."""
from __future__ import annotations

import pytest
import timm
import torch

from soccernet_reid.models import build_model
from soccernet_reid.models.backbones import create_backbone, feature_dim


def _n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("code, timm_name", [("VGG11-BN-CONV", "vgg11_bn"), ("VGG16-BN-CONV", "vgg16_bn")])
def test_conv_only_vgg_is_the_conv_part_plus_average_pooling(code: str, timm_name: str) -> None:
    backbone = create_backbone(code, pretrained=False).eval()
    reference = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="avg").eval()
    reference.features.load_state_dict(backbone.features.state_dict())

    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        out = backbone(x)
        feature_map = reference.forward_features(x)

    assert feature_map.shape == (2, 512, 8, 4)          # no stretching to 8x7
    assert out.shape == (2, 512)
    torch.testing.assert_close(out, feature_map.mean(dim=(2, 3)))
    # only the convolutional layers are kept: no fc6 / fc7
    assert not any("pre_logits" in name for name, _ in backbone.named_parameters())
    assert _n_params(backbone) == _n_params(reference.features)
    assert _n_params(backbone) < 16_000_000


def test_full_vgg_code_is_unchanged() -> None:
    """ "VGG11-BN" stays the timm model with fc6 / fc7 (4096-d), as in the runs trained with it."""
    backbone = create_backbone("VGG11-BN", pretrained=False)
    assert feature_dim(backbone) == 4096
    assert any("pre_logits" in name for name, _ in backbone.named_parameters())
    assert _n_params(backbone) > 120_000_000


def test_conv_only_vgg_with_projection_head() -> None:
    model = build_model("VGG11-BN-CONV", "projection", embedding_dim=512, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 256, 128))
    assert out.shape == (4, 512)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)
    assert model.head.bn_in.num_features == 512


def test_unknown_backbone_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backbone"):
        create_backbone("VGG19-BN-CONV", pretrained=False)
