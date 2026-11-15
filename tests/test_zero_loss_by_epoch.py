"""scripts/zero_loss_by_epoch.py: the share of zero-loss steps and the mean loss in each epoch."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "zero_loss_by_epoch.py"
    spec = importlib.util.spec_from_file_location("zero_loss_by_epoch_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_loss_steps_are_counted_per_epoch() -> None:
    script = _load_script()
    losses = [0.0, 0.3, 0.0, 0.0, 0.2, 0.1, 0.0, 0.4]
    assert script.shares_by_epoch(losses, 4) == [0.75, 0.25]
    assert script.means_by_epoch(losses, 4) == pytest.approx([0.075, 0.175])


def test_steps_that_do_not_fill_whole_epochs_are_refused() -> None:
    script = _load_script()
    with pytest.raises(ValueError):
        script.shares_by_epoch([0.0] * 7, 4)
    with pytest.raises(ValueError):
        script.means_by_epoch([], 4)


def test_epoch_length_comes_from_the_printed_configuration(tmp_path) -> None:
    (tmp_path / "stdout.log").write_text("\n".join(
        ["=" * 70, "num_epochs: 40", "num_iters_per_epoch: 5000", "=" * 70, "Device: cpu"]))
    config = _load_script().run_config(tmp_path)
    assert (config["num_epochs"], config["num_iters_per_epoch"]) == (40, 5000)
