"""Image transforms — three augmentation presets matching §2.D of the plan.

Presets:
    "eval"       — no augmentation, just resize + normalize. Used for valid/test.
    "aug-min"    — eval + horizontal flip. The weakest training transform.
    "aug-med"    — aug-min + ColorJitter + RandomCrop with padding + Random Erasing.
    "aug-strong" — aug-med + RandAugment + GaussianBlur + RandomPerspective.
    "aug-bot"   — ReID-aware "strong": aug-med features + RandomGrayscale + stronger
                   ColorJitter. No RandAugment/Perspective/Blur. Recipe follows
                   BoT-ReID (Luo 2019) / MGN (Wang 2018).

Random Erasing is the same in all three (``RANDOM_ERASING``: p=0.5, 2-40% of the
area, aspect ratio 0.3-3.3 — the defaults of Zhong et al. and of BoT-ReID), so
the presets differ only in the operations listed above. Until October 2026
aug-med erased 2-33% while the other two erased 2-40%.

The collapse of the aug-strong / aug-bot runs of May 2026 was caused by the
optimizer's weight decay, not by these transforms (see configs/config.yaml).

MixUp / CutMix are deliberately omitted (see plan §2.D): they require label
mixing which is incompatible with pair-based metric learning losses.

All transforms accept a PIL.Image and return a [3, H, W] float32 tensor
normalised with ImageNet statistics.
"""
from __future__ import annotations

from typing import Literal

import torch
from torchvision.transforms import v2

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# One Random Erasing setting for every preset that uses it (see module docstring).
RANDOM_ERASING: dict = {"p": 0.5, "scale": (0.02, 0.4), "ratio": (0.3, 3.3)}

TransformLevel = Literal["eval", "aug-min", "aug-med", "aug-strong", "aug-bot"]
_LEVELS: tuple[str, ...] = ("eval", "aug-min", "aug-med", "aug-strong", "aug-bot")


def build_transform(
    level: TransformLevel = "eval",
    height: int = 256,
    width: int = 128,
    pad: int = 10,
) -> v2.Compose:
    """Construct one of the four standard transforms (see module docstring).

    Args:
        level: which preset to use.
        height, width: target image size (ReID standard 256×128).
        pad: padding used by AUG-MED/STRONG RandomCrop.

    Returns:
        A `v2.Compose` callable that maps PIL.Image → [3, H, W] float32 tensor.
    """
    if level not in _LEVELS:
        raise ValueError(f"Unknown transform level: {level!r}; choose from {_LEVELS}")

    # Common preamble: load as uint8 tensor at target size.
    preamble: list = [
        v2.PILToTensor(),
        v2.Resize((height, width), antialias=True),
    ]

    # Common postamble: float32 normalization to ImageNet stats.
    postamble: list = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if level == "eval":
        return v2.Compose([*preamble, *postamble])

    if level == "aug-min":
        return v2.Compose(
            [
                *preamble,
                v2.RandomHorizontalFlip(p=0.5),
                *postamble,
            ]
        )

    if level == "aug-med":
        return v2.Compose(
            [
                *preamble,
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                v2.Pad(pad),
                v2.RandomCrop((height, width)),
                *postamble,
                # RandomErasing on float-normalised tensor (per v2 convention).
                v2.RandomErasing(**RANDOM_ERASING),
            ]
        )

    if level == "aug-strong":
        return v2.Compose(
            [
                *preamble,
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                v2.Pad(pad),
                v2.RandomCrop((height, width)),
                # RandAugment / AutoAugment expect uint8 tensors -> still pre-postamble.
                v2.RandAugment(num_ops=2, magnitude=9),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                v2.RandomPerspective(distortion_scale=0.2, p=0.3),
                *postamble,
                v2.RandomErasing(**RANDOM_ERASING),
            ]
        )

    # level == "aug-bot"
    # ReID-aware "strong": no RandAugment/Perspective/Blur. Adds RandomGrayscale
    # (BoT-ReID/MGN canonical) and a stronger ColorJitter.
    return v2.Compose(
        [
            *preamble,
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomGrayscale(p=0.2),
            v2.Pad(pad),
            v2.RandomCrop((height, width)),
            *postamble,
            v2.RandomErasing(**RANDOM_ERASING),
        ]
    )
