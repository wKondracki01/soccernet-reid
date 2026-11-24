"""Share of training steps with zero loss in every epoch, from the local W&B history of each run.

The review files keep this share only for the whole training (``zero_loss_frac``). The thesis also
shows how it changes from epoch to epoch in the augmentation stage and for AUG-MED against AUG-STRONG
in stage 5, so it is computed here from the
loss that training logged for every step (``train/loss_step``). A step of a margin loss has zero
loss when every triplet of its batch already meets the margin, and it then gives no gradient.

The length of an epoch and the number of epochs are read from the configuration that each run
printed at its start (stdout.log), and the number of logged steps must match them exactly. Only
the local files under outputs/runs/RUN are read; nothing is trained or evaluated.

Output: for every run the number of steps, the share over the whole training, the share in every
epoch and the mean loss of every epoch (the last one repeats ``train_loss_epoch`` of the review
files and serves as a check that the steps were read in the right order).

Usage
-----
    uv run python scripts/zero_loss_by_epoch.py     # writes outputs/_g/ZERO_LOSS_BY_EPOCH.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# the augmentation stage: AUG-MIN (the run of stage 1 with the same settings), AUG-MED, AUG-STRONG
# and AUG-COLOR, which is called aug-bot in the code
STAGE3_RUNS = ("G1_PK_SA_BH", "G3_AUG_MED", "G3_AUG_STRONG", "G3_AUG_BOT")
# stage 5 on ResNet-34: the three trainings of the base configuration (AUG-MED) and the three of the
# same setup with AUG-STRONG, which became the final configuration
STAGE5_RUNS = ("G5_REF", "G5_REF_S1", "G5_REF_S2", "G5_STRONG", "G5_STRONG_S1", "G5_STRONG_S2")


def run_config(run_dir: Path) -> dict:
    """The Hydra configuration printed at the start of a run (between the two rules in stdout.log)."""
    import yaml

    text = (run_dir / "stdout.log").read_text(errors="ignore")
    return yaml.safe_load(text.split("=" * 70)[1])


def step_losses(run_dir: Path) -> list[float]:
    """The loss logged at every training step, in order, from the last W&B run file of ``run_dir``."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    files = sorted(run_dir.glob("wandb/run-*/*.wandb"))
    if not files:
        raise FileNotFoundError(f"no W&B history under {run_dir}")
    store = datastore.DataStore()
    store.open_for_scan(str(files[-1]))
    losses: list[float] = []
    while True:
        try:
            data = store.scan_data()
        except Exception:      # as in the review scripts; a short read is caught by the step count
            break
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        for item in record.history.item:
            if (item.key or "/".join(item.nested_key)) == "train/loss_step":
                losses.append(float(json.loads(item.value_json)))
    return losses


def _epochs(losses: list[float], steps_per_epoch: int) -> list[list[float]]:
    if not losses or len(losses) % steps_per_epoch:
        raise ValueError(f"{len(losses)} steps do not fill whole epochs of {steps_per_epoch} steps")
    return [losses[i:i + steps_per_epoch] for i in range(0, len(losses), steps_per_epoch)]


def shares_by_epoch(losses: list[float], steps_per_epoch: int) -> list[float]:
    """Share of the steps of each epoch whose loss is exactly zero."""
    return [sum(v == 0 for v in epoch) / steps_per_epoch for epoch in _epochs(losses, steps_per_epoch)]


def means_by_epoch(losses: list[float], steps_per_epoch: int) -> list[float]:
    """Mean loss of the steps of each epoch."""
    return [sum(epoch) / steps_per_epoch for epoch in _epochs(losses, steps_per_epoch)]


def summarize(run_dir: Path) -> dict:
    config = run_config(run_dir)
    steps_per_epoch, epochs = int(config["num_iters_per_epoch"]), int(config["num_epochs"])
    losses = step_losses(run_dir)
    if len(losses) != steps_per_epoch * epochs:
        raise ValueError(f"{run_dir.name}: {len(losses)} logged steps, expected {epochs} x {steps_per_epoch}")
    return {
        "steps": len(losses),
        "steps_per_epoch": steps_per_epoch,
        "zero_loss_frac": sum(v == 0 for v in losses) / len(losses),
        "zero_loss_by_epoch": shares_by_epoch(losses, steps_per_epoch),
        "loss_by_epoch": means_by_epoch(losses, steps_per_epoch),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--names", nargs="+", default=[*STAGE3_RUNS, *STAGE5_RUNS])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "_g" / "ZERO_LOSS_BY_EPOCH.json")
    args = parser.parse_args()

    result = {}
    for name in args.names:
        result[name] = summary = summarize(args.runs / name)
        print(f"{name}: {summary['steps']} steps, zero loss in {100 * summary['zero_loss_frac']:.2f}% of them, "
              f"in the last epoch {100 * summary['zero_loss_by_epoch'][-1]:.2f}%")
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
