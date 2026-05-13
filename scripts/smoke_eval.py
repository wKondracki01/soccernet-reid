"""End-to-end smoke test: extract features → rank → mAP/Rank-k → compare with official.

This is the Krok 2 deliverable from §10:
    "Evaluator zgodny z oficjalnym + smoke test"

Default behavior: R18-ImageNet (no fine-tuning), full valid set, cosine similarity.
Use --limit-actions or --limit-images to cut down for quick local checks.

Usage:
    uv run python scripts/smoke_eval.py
    uv run python scripts/smoke_eval.py --limit-actions 50
    uv run python scripts/smoke_eval.py --backbone EB1 --device mps
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Allow running directly without `uv run -m ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.data.catalog import build_catalog, load_catalog  # noqa: E402
from soccernet_reid.data.dataset import ReIDImageDataset, default_eval_transform  # noqa: E402
from soccernet_reid.eval.metrics import compute_metrics  # noqa: E402
from soccernet_reid.eval.official import (  # noqa: E402
    catalog_to_groundtruth_dict,
    run_official_evaluator,
)
from soccernet_reid.eval.ranking import compute_rankings  # noqa: E402
from soccernet_reid.models.backbones import create_backbone, feature_dim  # noqa: E402


def _pick_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def _filter_subset(df: pd.DataFrame, limit_actions: int | None, limit_images: int | None) -> pd.DataFrame:
    """Keep both query and gallery rows for the SAME set of actions (limited)."""
    if limit_actions is not None:
        actions_sorted = sorted(df["action_idx"].unique())[:limit_actions]
        df = df[df["action_idx"].isin(actions_sorted)]
    if limit_images is not None and len(df) > limit_images:
        # Sample whole actions to stay valid for evaluation
        rng = np.random.default_rng(0)
        actions = list(df["action_idx"].unique())
        rng.shuffle(actions)
        keep_actions: list[int] = []
        running = 0
        for a in actions:
            sub = df[df["action_idx"] == a]
            running += len(sub)
            keep_actions.append(int(a))
            if running >= limit_images:
                break
        df = df[df["action_idx"].isin(keep_actions)]
    return df.reset_index(drop=True)


def _filter_queries_with_positives(query_df: pd.DataFrame, gallery_df: pd.DataFrame) -> pd.DataFrame:
    """Drop queries whose person_uid has no occurrence in gallery of the same action.

    Required because the official evaluator raises ValueError for such queries.
    """
    gallery_pairs = set(
        map(tuple, gallery_df[["action_idx", "person_uid"]].itertuples(index=False, name=None))
    )
    mask = query_df.apply(
        lambda r: (int(r["action_idx"]), int(r["person_uid"])) in gallery_pairs, axis=1
    )
    return query_df[mask].reset_index(drop=True)


@torch.no_grad()
def extract_features(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    transform = default_eval_transform(height=256, width=128)
    ds = ReIDImageDataset(df, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    feats: list[np.ndarray] = []
    t0 = time.perf_counter()
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        out = model(imgs)
        feats.append(out.float().cpu().numpy())
    arr = np.concatenate(feats, axis=0)
    dt = time.perf_counter() - t0
    print(f"  extracted {arr.shape[0]:,} features (dim={arr.shape[1]}) in {dt:.1f}s "
          f"= {arr.shape[0] / max(dt, 0.001):.1f} img/s")
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=Path("outputs/catalog.parquet"),
                        help="Pre-built catalog parquet (default: outputs/catalog.parquet)")
    parser.add_argument("--reid-root", type=Path, default=Path("dataSoccerNet/reid-2023"),
                        help="Used only if --catalog doesn't exist; rebuilds catalog from JSON")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--backbone", type=str, default="R18",
                        help="Backbone code (R18/R34/EB1/EB2/VGG16-BN/VGG11-BN)")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / mps / cpu / auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--limit-actions", type=int, default=None,
                        help="Limit to first N actions (for quick smoke test)")
    parser.add_argument("--limit-images", type=int, default=None,
                        help="Limit total images (rounds up to whole actions)")
    parser.add_argument("--compare-official", action="store_true",
                        help="Run the official evaluator side-by-side and assert parity")
    args = parser.parse_args()

    # 1. Load catalog
    if args.catalog.exists():
        print(f"Loading catalog from {args.catalog} ...")
        df = load_catalog(args.catalog)
    else:
        print(f"Catalog not cached, building from {args.reid_root} ...")
        df = build_catalog(args.reid_root)

    split_df = df[df["split"] == args.split].copy()
    if split_df.empty:
        print(f"ERROR: no rows for split={args.split!r}", file=sys.stderr)
        return 1

    # 2. Apply optional limits
    if args.limit_actions or args.limit_images:
        before = len(split_df)
        split_df = _filter_subset(split_df, args.limit_actions, args.limit_images)
        print(f"Subset: {len(split_df):,} rows from {before:,} (kept {split_df['action_idx'].nunique()} actions)")

    query_df = split_df[split_df["role"] == "query"].reset_index(drop=True)
    gallery_df = split_df[split_df["role"] == "gallery"].reset_index(drop=True)
    print(f"\n{args.split}: {len(query_df):,} queries, {len(gallery_df):,} gallery items, "
          f"{split_df['action_idx'].nunique()} actions")

    # Drop queries with no positive in gallery (official evaluator constraint)
    before_q = len(query_df)
    query_df = _filter_queries_with_positives(query_df, gallery_df)
    if len(query_df) < before_q:
        print(f"  dropped {before_q - len(query_df)} queries with no positives in gallery")

    # 3. Load model
    print(f"\nLoading backbone {args.backbone} (pretrained=True) ...")
    device = _pick_device(args.device)
    print(f"  device: {device}")
    model = create_backbone(args.backbone, pretrained=True).to(device).eval()
    d = feature_dim(model)
    print(f"  feature dim: {d}")

    # 4. Extract features
    print("\nExtracting query features ...")
    query_feats = extract_features(query_df, model, device, args.batch_size, args.num_workers)
    print("Extracting gallery features ...")
    gallery_feats = extract_features(gallery_df, model, device, args.batch_size, args.num_workers)

    # 5. Build rankings
    print("\nBuilding rankings ...")
    t0 = time.perf_counter()
    rankings = compute_rankings(
        query_feats=query_feats,
        gallery_feats=gallery_feats,
        query_bbox_idx=query_df["bbox_idx"].astype(int).tolist(),
        gallery_bbox_idx=gallery_df["bbox_idx"].astype(int).tolist(),
        query_actions=query_df["action_idx"].astype(int).tolist(),
        gallery_actions=gallery_df["action_idx"].astype(int).tolist(),
        distance=args.distance,
    )
    print(f"  built {len(rankings):,} rankings in {time.perf_counter() - t0:.1f}s")

    # 6. Build groundtruth dict (in-memory, same shape as official JSON)
    keep_actions = set(query_df["action_idx"].unique()) | set(gallery_df["action_idx"].unique())
    subset_df = df[(df["split"] == args.split) & (df["action_idx"].isin(keep_actions))]
    keep_bbox = set(query_df["bbox_idx"].astype(int)) | set(gallery_df["bbox_idx"].astype(int))
    subset_df = subset_df[subset_df["bbox_idx"].isin(keep_bbox)]
    gt = catalog_to_groundtruth_dict(subset_df, split=args.split)
    # Restrict to the queries we actually evaluated
    gt["query"] = {k: v for k, v in gt["query"].items() if int(k) in keep_bbox}

    # 7. Our metrics
    print("\nComputing our metrics ...")
    ours = compute_metrics(rankings, gt["query"], gt["gallery"], ranks=(1, 5, 10))
    print(f"  mAP:     {ours['mAP']:.4f}")
    print(f"  Rank-1:  {ours['rank-1']:.4f}")
    print(f"  Rank-5:  {ours['rank-5']:.4f}")
    print(f"  Rank-10: {ours['rank-10']:.4f}")

    # 8. Optional parity check
    if args.compare_official:
        print("\nRunning official evaluator ...")
        theirs = run_official_evaluator(rankings, gt)
        print(f"  Official mAP:    {theirs['mAP']:.6f}")
        print(f"  Official Rank-1: {theirs['rank-1']:.6f}")
        diff_map = abs(ours["mAP"] - theirs["mAP"])
        diff_r1 = abs(ours["rank-1"] - theirs["rank-1"])
        print(f"\n|ΔmAP|    = {diff_map:.2e}")
        print(f"|ΔRank-1| = {diff_r1:.2e}")
        if diff_map > 1e-5 or diff_r1 > 1e-5:
            print("  PARITY FAILED — abs diff > 1e-5", file=sys.stderr)
            return 2
        print("  PARITY OK (both metrics within 1e-5)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
