"""Evaluate a saved training checkpoint on valid or test split.

Use cases
---------
- Re-evaluate a checkpoint with a *different* distance metric (cosine vs. euclidean)
  without re-running training.
- Run final evaluation on test/ for a chosen best valid configuration.
- Sanity-check a checkpoint after copying between machines (Mac <-> 3080 <-> cloud).
- Compare several runs on identical conditions for the thesis tables.

The checkpoint format produced by `scripts/train.py` includes the full Hydra
config used at training time. This script reconstructs the model from that
config so callers don't need to remember which backbone/head/D was used.

Usage
-----
    # Evaluate a local checkpoint on valid (default)
    uv run python scripts/eval_checkpoint.py outputs/runs/F0a_reference/best.pt

    # On the test split with euclidean distance
    uv run python scripts/eval_checkpoint.py outputs/runs/F0a_reference/best.pt \\
        --split test --distance euclidean

    # Pull a checkpoint from W&B Artifacts and evaluate
    uv run python scripts/eval_checkpoint.py wandb://entity/project/F0a_reference-best:best \\
        --split valid

    # Use a non-default catalog or report extra ranks
    uv run python scripts/eval_checkpoint.py best.pt --catalog outputs/catalog.parquet \\
        --ranks 1 5 10 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow running without `uv run -m ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.data.catalog import build_catalog, load_catalog  # noqa: E402
from soccernet_reid.models import build_model  # noqa: E402
from soccernet_reid.training import evaluate_model, pick_device  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_checkpoint(ckpt_arg: str) -> Path:
    """Return a local file path for a checkpoint argument.

    Supports:
      * regular file paths (absolute or relative to project root or CWD)
      * `wandb://entity/project/artifact:version` URIs — downloaded via wandb.Api
    """
    if ckpt_arg.startswith("wandb://"):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb URIs require the 'training' extra: uv sync --all-extras"
            ) from e
        spec = ckpt_arg[len("wandb://"):]
        api = wandb.Api()
        artifact = api.artifact(spec, type="model")
        local_dir = Path(artifact.download(root=str(PROJECT_ROOT / "outputs" / "wandb_artifacts")))
        # The artifact contains best.pt — return the .pt file
        pt_files = list(local_dir.glob("*.pt"))
        if len(pt_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .pt file in artifact {spec!r}, got {pt_files}"
            )
        print(f"Downloaded W&B artifact to {pt_files[0]}")
        return pt_files[0]

    # Local path: try as-is, then relative to project root
    p = Path(ckpt_arg)
    if p.is_file():
        return p
    p2 = PROJECT_ROOT / ckpt_arg
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg!r} (also tried {p2})")


def _rebuild_model_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct the model from the config stored in the checkpoint."""
    cfg = ckpt.get("config")
    if cfg is None:
        raise KeyError(
            "Checkpoint missing 'config' key — cannot reconstruct model. "
            "Was this checkpoint saved by scripts/train.py?"
        )
    backbone_name = cfg["backbone"]["name"]
    head_name = cfg["head"]["name"]
    embedding_dim = cfg.get("embedding_dim", 512)

    print(f"Rebuilding model: {backbone_name} + {head_name} (D={embedding_dim})")
    model = build_model(
        backbone_name=backbone_name,
        head_name=head_name,
        embedding_dim=embedding_dim,
        pretrained=False,  # we'll load weights from the checkpoint
    ).to(device)

    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    if not missing and not unexpected:
        print("  state_dict loaded cleanly")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="Path to .pt file OR wandb://entity/project/artifact:version")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet",
                        help="Pre-built catalog parquet (default: outputs/catalog.parquet)")
    parser.add_argument("--reid-root", type=Path,
                        default=PROJECT_ROOT / "dataSoccerNet" / "reid-2023",
                        help="Used to rebuild catalog if --catalog doesn't exist")
    parser.add_argument("--split", default="valid", choices=("valid", "test"),
                        help="Which split to evaluate on (default: valid)")
    parser.add_argument("--distance", default="cosine", choices=("cosine", "euclidean"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 5, 10],
                        help="Rank-k values to compute")
    parser.add_argument("--device", default="auto",
                        help="auto / cuda / mps / cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional: dump metrics as JSON to this path")
    args = parser.parse_args()

    # 1. Catalog
    if args.catalog.exists():
        print(f"Loading catalog from {args.catalog}")
        catalog = load_catalog(args.catalog)
    else:
        print(f"Building catalog from {args.reid_root}")
        catalog = build_catalog(args.reid_root)

    # 2. Resolve and load checkpoint
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  epoch:    {ckpt.get('epoch', '?')}")
    print(f"  best_mAP: {ckpt.get('best_mAP', '?'):.4f}" if "best_mAP" in ckpt else "")

    # 3. Device + model
    device = pick_device(args.device)
    print(f"Device: {device}")
    model = _rebuild_model_from_checkpoint(ckpt, device)
    model.eval()

    # 4. Evaluate
    print(f"\nEvaluating on {args.split} (distance={args.distance}) ...")
    metrics = evaluate_model(
        model=model,
        catalog=catalog,
        split=args.split,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distance=args.distance,
        ranks=tuple(args.ranks),
    )

    # 5. Report
    print("\n=== Results ===")
    print(f"  split:    {args.split}")
    print(f"  distance: {args.distance}")
    print(f"  mAP:      {metrics['mAP']:.4f}")
    for r in args.ranks:
        print(f"  Rank-{r}:   {metrics[f'rank-{r}']:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "split": args.split,
                "distance": args.distance,
                "epoch": ckpt.get("epoch"),
                **metrics,
            },
            indent=2,
        ))
        print(f"\nWrote metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
