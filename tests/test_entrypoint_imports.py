"""Entry-point scripts must import pyarrow.dataset before torch.

On Windows (the training laptop) loading pyarrow's dataset DLLs after torch kills
the process with an access violation and no traceback, so reading the catalog
parquet exits with code 1 (scripts/eval_checkpoint.py did exactly that). The
crash cannot be reproduced on macOS/Linux, so this checks the import order
statically.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _top_level_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _is_torch(name: str) -> bool:
    return name == "torch" or name.startswith("torch.")


TORCH_SCRIPTS = sorted(
    p for p in SCRIPTS_DIR.glob("*.py") if any(_is_torch(n) for n in _top_level_imports(p))
)


def test_known_torch_entrypoints_are_checked() -> None:
    names = {p.name for p in TORCH_SCRIPTS}
    assert {"train.py", "eval_checkpoint.py", "smoke_eval.py"} <= names


@pytest.mark.parametrize("script", TORCH_SCRIPTS, ids=lambda p: p.name)
def test_pyarrow_dataset_imported_before_torch(script: Path) -> None:
    names = _top_level_imports(script)
    first_torch = next(i for i, n in enumerate(names) if _is_torch(n))
    assert "pyarrow.dataset" in names[:first_torch], (
        f"{script.name}: add `import pyarrow.dataset` before the first torch import"
    )
