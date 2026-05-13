"""Reproducibility, device selection, and small training utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU+CUDA+MPS)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism(deterministic: bool = True, warn_only: bool = True) -> None:
    """Best-effort deterministic CUDA / cuDNN. Off by default in training (perf cost)."""
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except Exception:
            # Some ops have no deterministic implementation; warn_only handles it.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def pick_device(preference: str = "auto") -> torch.device:
    """Pick a torch.device from a string preference.

    "auto" → cuda > mps > cpu (in that order).
    Otherwise a specific device string is passed through.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def amp_supported(device: torch.device) -> bool:
    """Return True if torch.amp is supported on this device (CUDA only as of 2.x)."""
    return device.type == "cuda"
