"""Sequentially run a chain of training configurations.

Used for unattended execution of an entire experimental phase (e.g., Faza 1
or Faza 2) without requiring per-run intervention. Wait-for is optional —
useful when an externally-launched run is still in progress (e.g., F2_CONT
launched separately, chain queued behind it).

Each config in `configs` is a string `"<name>:<hydra_args>"` where
`<hydra_args>` is a space-separated list of Hydra overrides. The training
is run by spawning scripts/train.py synchronously; the chain blocks until
each training exits before moving to the next.

Usage
-----
    # Stand-alone chain (start immediately)
    python scripts/run_chain.py \\
        "F2_MS:loss=ms sampler=pk_sa num_epochs=40" \\
        "F2_CIRCLE:loss=circle sampler=pk_sa num_epochs=40" \\
        "F2_ARC:loss=arc sampler=random num_epochs=40"

    # Wait for an already-running training to finish before starting the chain
    python scripts/run_chain.py --wait-for F2_CONT \\
        "F1_PK_BH_XBM_v2:+loss.xbm=true num_epochs=60" \\
        ...

Output
------
Chain progress lines (prefixed `[chain N/M] ...`) go to this script's stdout,
which when wrapped by `launch_bg.py` ends up in
`outputs/runs/<chain_name>/stdout.log`. Final marker line `ALL CHAIN DONE`
signals successful completion.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DONE_SENTINEL = "Done. Best valid mAP"
POLL_INTERVAL_S = 60


def _venv_python() -> Path:
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wait_for_completion_via_log(run_name: str) -> None:
    """Poll the log file for a previously-launched (sibling) run.

    Does NOT detect crashes; this is intentionally a passive observer. If the
    awaited run never writes the sentinel (e.g., it crashed before completion)
    this function loops forever and a human needs to intervene.
    """
    log_path = PROJECT_ROOT / "outputs" / "runs" / run_name / "stdout.log"
    print(f"[{_now()}] [chain] waiting for {run_name!r} (polling {log_path})", flush=True)
    while True:
        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                content = ""
            if DONE_SENTINEL in content:
                print(f"[{_now()}] [chain] {run_name} finished — sentinel found", flush=True)
                return
        time.sleep(POLL_INTERVAL_S)


def run_training(name: str, hydra_args: list[str], python_exe: Path, train_script: Path) -> int:
    """Spawn train.py synchronously with the given Hydra overrides."""
    cmd = [
        str(python_exe),
        str(train_script),
        f"experiment_name={name}",
        *hydra_args,
    ]
    print(f"[{_now()}] [chain] starting {name}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    print(f"[{_now()}] [chain] {name} exit code: {result.returncode}", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wait-for",
        default=None,
        help="If set, poll outputs/runs/<NAME>/stdout.log for the 'Done. Best valid mAP' "
        "sentinel before starting the chain. Useful when another run is in progress.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help='Configs in the form "<name>:<hydra_args>", e.g. '
        '"F2_MS:loss=ms sampler=pk_sa num_epochs=40"',
    )
    args = parser.parse_args()

    python_exe = _venv_python()
    if not python_exe.exists():
        print(f"ERROR: venv python not found at {python_exe}", file=sys.stderr)
        return 1
    train_script = PROJECT_ROOT / "scripts" / "train.py"

    print(f"[{_now()}] [chain] starting; {len(args.configs)} configs queued", flush=True)
    for i, cfg in enumerate(args.configs, 1):
        if ":" not in cfg:
            print(f"ERROR: config {cfg!r} missing ':' separator", file=sys.stderr)
            return 2
        name, hydra_str = cfg.split(":", 1)
        print(f"  [{i}/{len(args.configs)}] {name}  ({hydra_str})", flush=True)

    if args.wait_for:
        wait_for_completion_via_log(args.wait_for)

    for i, cfg in enumerate(args.configs, 1):
        name, hydra_str = cfg.split(":", 1)
        hydra_args = hydra_str.split()
        print(f"[{_now()}] [chain {i}/{len(args.configs)}] === {name} ===", flush=True)
        rc = run_training(name, hydra_args, python_exe, train_script)
        if rc != 0:
            print(
                f"[{_now()}] [chain] ABORTED at {name} (exit {rc}); "
                "remaining configs NOT run.",
                flush=True,
            )
            return 1
        print(f"[{_now()}] [chain {i}/{len(args.configs)}] OK {name}", flush=True)

    print(f"[{_now()}] ALL CHAIN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
