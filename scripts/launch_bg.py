"""Launch ANY long-running script (training, dataset download, etc.) in the background.

Why this exists
---------------
On Windows, naive backgrounding patterns (`Start-Process -NoNewWindow`,
`uv run X &`, `start /B`) all fail in subtle ways: the child inherits the
SSH session's console and gets killed when the SSH command returns. This
launcher uses platform-specific *full detach* primitives:

  - Windows: `creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`
  - Unix:    `start_new_session=True` (calls setsid(2))

The child runs in its own process group, with no controlling terminal,
and survives SSH disconnect. Logs go to disk so we can monitor with
`Get-Content -Wait` or `tail -f`.

Two convenience modes:
  --kind train     (default) → spawns scripts/train.py with Hydra overrides
  --kind download  → spawns scripts/download_dataset.py with given args
  --kind raw       → spawns whatever `--cmd` you pass

Usage
-----
    # Reference training run (Faza 0a)
    uv run python scripts/launch_bg.py --kind train --name F0a_reference

    # Smoke test
    uv run python scripts/launch_bg.py --kind train --name smoke +experiment=smoke

    # Dataset download
    uv run python scripts/launch_bg.py --kind download --name dataset_dl -- \\
        --target dataSoccerNet --extract --verify

    # Anything else
    uv run python scripts/launch_bg.py --kind raw --name custom -- \\
        scripts/build_catalog.py --reid-root dataSoccerNet/reid-2023 ...

Output
------
    outputs/runs/<name>/stdout.log
    outputs/runs/<name>/stderr.log
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


def _build_cmd(kind: str, args: list[str], python_exe: Path) -> list[str]:
    """Assemble the full command line to launch."""
    if kind == "train":
        return [str(python_exe), str(PROJECT_ROOT / "scripts" / "train.py"), *args]
    if kind == "download":
        return [
            str(python_exe),
            str(PROJECT_ROOT / "scripts" / "download_dataset.py"),
            *args,
        ]
    if kind == "raw":
        if not args:
            raise ValueError("--kind raw requires a command after `--`")
        # Treat first positional as a script path (resolve relative to project root)
        # and everything else as args. Run via venv's python.
        first, *rest = args
        first_path = Path(first)
        if not first_path.is_absolute():
            first_path = PROJECT_ROOT / first_path
        return [str(python_exe), str(first_path), *rest]
    raise ValueError(f"Unknown --kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kind",
        choices=("train", "download", "raw"),
        default="train",
        help="What to launch (default: train)",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Run name; logs go to outputs/runs/<name>/",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the spawned script. For Hydra overrides "
        "(--kind train) prefix them with key=value. For other kinds use `--` "
        "before the args.",
    )
    args = parser.parse_args()

    # Strip leading `--` if present (REMAINDER keeps it)
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra

    # For --kind train, automatically inject experiment_name=<name> if not present
    if args.kind == "train" and not any(a.startswith("experiment_name=") for a in extra):
        extra = [f"experiment_name={args.name}", *extra]

    python_exe = _venv_python()
    if not python_exe.exists():
        print(
            f"ERROR: venv python not found at {python_exe}\n"
            "Run 'uv sync --all-extras' first to create the venv.",
            file=sys.stderr,
        )
        return 1

    cmd = _build_cmd(args.kind, extra, python_exe)

    log_dir = PROJECT_ROOT / "outputs" / "runs" / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"

    print(f"Launching ({args.kind}) in background")
    print(f"  name:        {args.name}")
    print(f"  command:     {' '.join(cmd)}")
    print(f"  cwd:         {PROJECT_ROOT}")
    print(f"  stdout log:  {stdout_log}")
    print(f"  stderr log:  {stderr_log}")
    print()

    out_f = stdout_log.open("wb", buffering=0)
    err_f = stderr_log.open("wb", buffering=0)

    if sys.platform == "win32":
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
        print(f"  Get-Content -Path '{stderr_log}' -Wait -Tail 50  # for tqdm progress bars")
        print(f"  # from Mac via SSH:")
        print(f"  ssh rtx3080 \"Get-Content -Path '{stderr_log}' -Wait -Tail 50\"")
    else:
        print(f"  tail -f {stdout_log}")
        print(f"  tail -f {stderr_log}")
    print()
    print("Stop the run:")
    if sys.platform == "win32":
        print(f"  Stop-Process -Id {proc.pid} -Force")
    else:
        print(f"  kill {proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
