"""scripts/train.py: when the best checkpoint is uploaded to W&B (`wandb.upload_best`)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[object, list[str]]] = []

    def log_artifact(self, artifact, aliases):
        self.logged.append((artifact, aliases))


def _cfg(**wandb_keys):
    return OmegaConf.create({
        "experiment_name": "T_RUN",
        "backbone": {"name": "R18"}, "head": {"name": "projection"}, "loss": {"name": "tri"},
        "wandb": {"enabled": True, **wandb_keys},
    })


@pytest.mark.parametrize("mode", ["improvement", "end", "off"])
def test_known_modes_are_accepted(mode: str) -> None:
    assert _load_script()._upload_best_mode(_cfg(upload_best=mode)) == mode


def test_missing_key_means_upload_on_every_improvement() -> None:
    # configs written before the key existed must keep their behaviour
    assert _load_script()._upload_best_mode(_cfg()) == "improvement"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="upload_best"):
        _load_script()._upload_best_mode(_cfg(upload_best="always"))


def test_upload_logs_one_artifact_with_the_epoch_and_map(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb-cache"))
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"checkpoint")
    run = _FakeRun()
    metrics = {"mAP": 0.71234, "rank-1": 0.6, "rank-5": 0.9, "rank-10": 0.97}

    _load_script()._upload_best_checkpoint(run, _cfg(), ckpt, 12, metrics, 512)

    assert len(run.logged) == 1
    artifact, aliases = run.logged[0]
    assert artifact.name == "T_RUN-best"
    assert aliases == ["epoch-12", "map-0.7123", "best"]
    assert artifact.metadata["epoch"] == 12
    assert artifact.metadata["valid_mAP"] == pytest.approx(0.71234)
    assert artifact.metadata["valid_rank_10"] == pytest.approx(0.97)


def test_failed_upload_does_not_raise(tmp_path, capsys) -> None:
    class _Broken:
        def log_artifact(self, artifact, aliases):
            raise RuntimeError("network down")

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"checkpoint")
    _load_script()._upload_best_checkpoint(_Broken(), _cfg(), ckpt, 3, {"mAP": 0.5}, 512)
    assert "upload failed" in capsys.readouterr().out
