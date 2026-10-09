"""scripts/benchmark_speed.py: the measurements have the expected shape and units."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_speed.py"
    spec = importlib.util.spec_from_file_location("benchmark_speed_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result() -> dict:
    return _load_script().benchmark_backbone(
        "R18", torch.device("cpu"), batch_size=4, train_batch_size=4, warmup=1, iters=2, repeats=1,
    )


def test_reports_parameters_of_backbone_and_head(result: dict) -> None:
    from soccernet_reid.models import build_model

    model = build_model("R18", "projection", embedding_dim=512, pretrained=False)
    assert result["parameters"] == sum(p.numel() for p in model.parameters())
    assert result["backbone"] == "R18" and result["embedding_dim"] == 512 and result["input"] == [256, 128]


def test_inference_figures_are_consistent(result: dict) -> None:
    assert result["inference_images_per_s"] > 0
    # images per second and milliseconds per image describe the same measurement
    assert result["inference_images_per_s"] * result["inference_ms_per_image"] == pytest.approx(1000.0)
    assert result["latency_ms_single_image"] > 0


def test_training_figures_are_consistent(result: dict) -> None:
    assert result["train_batch_size"] == 4
    assert result["train_steps_per_s"] > 0
    assert result["train_images_per_s"] == pytest.approx(4 * result["train_steps_per_s"])


def test_training_measurement_can_be_skipped() -> None:
    out = _load_script().benchmark_backbone(
        "R18", torch.device("cpu"), batch_size=2, warmup=0, iters=1, repeats=1, train=False,
    )
    assert "train_steps_per_s" not in out and "inference_images_per_s" in out


def test_table_has_one_line_per_backbone(result: dict) -> None:
    table = _load_script().format_table([result, result])
    lines = table.splitlines()
    assert len(lines) == 3 and lines[0].startswith("backbone") and "train it/s" in lines[0]
    assert lines[1].startswith("R18")
