"""The defaults in configs/config.yaml that every run of a series relies on.

Both values below were different in May 2026 and both caused trouble (see the
comments in the config). A run started without overrides must get the safe ones.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def test_no_weight_decay_by_default() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["optimizer"]["name"] == "adam"
    assert float(cfg["optimizer"]["weight_decay"]) == 0.0


def test_full_precision_by_default() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["amp"] is False
