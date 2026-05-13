"""Unit tests for the augmentation pipelines.

We test:
- Output shape and dtype across all four presets
- Idempotence of `eval` (same input → same output)
- Stochasticity of augmentations (two calls produce different tensors)
- Normalization roughly matches expected ImageNet stats
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from soccernet_reid.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
)


def _random_pil(size: tuple[int, int] = (160, 80), seed: int = 0) -> Image.Image:
    """Make a deterministic random RGB PIL image of given (W, H)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestShapeAndDtype:
    @pytest.mark.parametrize("level", ["eval", "aug-min", "aug-med", "aug-strong"])
    def test_output_shape_is_3xHxW(self, level: str) -> None:
        tx = build_transform(level, height=256, width=128)
        img = _random_pil()
        torch.manual_seed(0)
        out = tx(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 256, 128), f"{level}: bad shape {out.shape}"
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("level", ["eval", "aug-min", "aug-med", "aug-strong"])
    def test_handles_small_input(self, level: str) -> None:
        # Some real bboxes are tiny (e.g. 30×15) — must still resize correctly
        tx = build_transform(level, height=256, width=128)
        img = _random_pil(size=(15, 30))
        torch.manual_seed(0)
        out = tx(img)
        assert out.shape == (3, 256, 128)


class TestDeterminism:
    def test_eval_is_deterministic(self) -> None:
        tx = build_transform("eval", height=256, width=128)
        img = _random_pil()
        torch.manual_seed(0)
        a = tx(img)
        torch.manual_seed(123)  # different torch seed, but eval has no randomness
        b = tx(img)
        torch.testing.assert_close(a, b)

    def test_aug_min_varies_across_calls(self) -> None:
        # With horizontal flip p=0.5, two calls with different torch seeds
        # should produce different outputs (for at least one of two attempts).
        tx = build_transform("aug-min", height=256, width=128)
        img = _random_pil()
        outs = []
        for seed in range(20):
            torch.manual_seed(seed)
            outs.append(tx(img))
        any_different = any(not torch.equal(outs[0], o) for o in outs[1:])
        assert any_different, "aug-min should produce different outputs across seeds"

    @pytest.mark.parametrize("level", ["aug-med", "aug-strong"])
    def test_strong_augmentations_vary(self, level: str) -> None:
        tx = build_transform(level, height=256, width=128)
        img = _random_pil()
        torch.manual_seed(0)
        a = tx(img)
        torch.manual_seed(1)
        b = tx(img)
        assert not torch.equal(a, b), f"{level}: two seeds gave identical outputs"


class TestNormalization:
    def test_eval_normalises_to_imagenet_stats(self) -> None:
        # With a random uniform image, per-channel mean post-normalize should be
        # approximately (0.5 - mu) / sigma for each channel.
        tx = build_transform("eval", height=256, width=128)
        img = _random_pil(size=(128, 256), seed=0)  # actually at target size already
        out = tx(img)
        # Expected: (0.5 - mean) / std per channel, with tolerance for randomness
        per_channel_mean = out.mean(dim=(1, 2))
        for c, (mu, sigma) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            expected = (0.5 - mu) / sigma
            assert abs(per_channel_mean[c].item() - expected) < 0.05, (
                f"Channel {c}: got {per_channel_mean[c].item():.3f}, "
                f"expected ~{expected:.3f}"
            )


class TestInvalidLevel:
    def test_unknown_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transform level"):
            build_transform("mega-strong", 256, 128)
