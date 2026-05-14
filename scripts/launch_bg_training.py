"""Launch a training run in the background, with logs to disk.

Why this exists
---------------
On Windows, nesting `uv run` inside PowerShell `Start-Process` failed silently
(no env / TTY for uv's trampoline). So we don't try; instead we run *this*
script via `uv run python ...` interactively, and *this* script spawns the
training process directly from the venv's `python.exe` with detach flags
(Windows: DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP; Unix: setsid).
The training keeps running after this script and the SSH session exit.

Usage
-----
    # From a regular terminal (or via SSH):
    uv run python scripts/launch_bg_training.py <experiment_name> [hydra_overrides...]

Examples
--------
    # Reference Faza 0a (uses default config: R18 + TRI + PK + AUG-MIN, 60 epochs)
    uv run python scripts/launch_bg_training.py F0a_reference

    # Smoke test (2 epochs x 30 iters, no W&B)
    uv run python scripts/launch_bg_training.py smoke +experiment=smoke

    # Faza 2 example: try MS loss with EB1 backbone
    uv run python scripts/launch_bg_training.py F2_EB1_MS backbone=eb1 loss=ms

Output
------
Logs go to:
    outputs/runs/<experiment_name>/stdout.log
    outputs/runs/<experiment_name>/stderr.log

Use the printed `Get-Content -Wait` / `tail -f` command to monitor in real time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _venv_python() -> Path:
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "experiment_name",
        help="Identifier for the run; becomes the output directory + W&B run name",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Hydra-style overrides passed straight through (e.g. backbone=eb1 loss=ms)",
    )
    args = parser.parse_args()

    exp = args.experiment_name
    log_dir = PROJECT_ROOT / "outputs" / "runs" / exp
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"

    python_exe = _venv_python()
    if not python_exe.exists():
        print(
            f"ERROR: venv python not found at {python_exe}\n"
            f"Run 'uv sync --all-extras' first to create the venv.",
            file=sys.stderr,
        )
        return 1

    train_script = PROJECT_ROOT / "scripts" / "train.py"
    cmd = [
        str(python_exe),
        str(train_script),
        f"experiment_name={exp}",
        *args.hydra_overrides,
    ]

    print(f"Launching training in background")
    print(f"  experiment:  {exp}")
    print(f"  hydra args:  {' '.join(args.hydra_overrides) if args.hydra_overrides else '(defaults from configs/config.yaml)'}")
    print(f"  working dir: {PROJECT_ROOT}")
    print(f"  stdout log:  {stdout_log}")
    print(f"  stderr log:  {stderr_log}")
    print()

    # Open in unbuffered binary mode so the OS doesn't buffer behind us.
    out_f = stdout_log.open("wb", buffering=0)
    err_f = stderr_log.open("wb", buffering=0)

    if sys.platform == "win32":
        # Detach completely from the launching console: child survives parent exit
        # (incl. SSH disconnect), runs without a console window of its own.
        flags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            stdin=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            creationflags=flags,
            close_fds=True,
        )
    else:
        # Unix: start_new_session calls setsid(2) so we leave the parent's
        # process group. Resists SIGHUP from terminal hangup.
        proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            stdin=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
            close_fds=True,
        )

    print(f"PID: {proc.pid}")
    print()
    print("Monitor (live tail):")
    if sys.platform == "win32":
        print(f"  Get-Content -Path '{stdout_log}' -Wait -Tail 50")
        print(f"  # from Mac via SSH:")
        print(f"  ssh rtx3080 \"Get-Content -Path '{stdout_log}' -Wait -Tail 50\"")
    else:
        print(f"  tail -f {stdout_log}")
    print()
    print("Stop the run:")
    if sys.platform == "win32":
        print(f"  Stop-Process -Id {proc.pid} -Force")
    else:
        print(f"  kill {proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
