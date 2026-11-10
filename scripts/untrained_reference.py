"""Validation scores of the reference points that need no training on SoccerNet.

    random ranking       i.i.d. Gaussian vectors, one draw per seed; the thesis reports the mean
                         of the draws. The exact expectation over uniformly random orderings of
                         each gallery is saved next to it as a check.
    same vector          one vector for every crop: all similarities tie and the stable sort
                         leaves the gallery in the order of the data.
    ResNet-18 ImageNet   pooled backbone features with the ImageNet weights that training starts
                         from, without the projection head and without fine-tuning.

Queries and gallery are selected exactly as in the evaluations during training. Only the
validation split is used: the test split is reserved for the final configuration.

Usage
-----
    uv run python scripts/untrained_reference.py        # writes outputs/_g/UNTRAINED_valid.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
# Must precede torch: on Windows, loading pyarrow's dataset DLLs after torch crashes
# the process (see scripts/eval_checkpoint.py).
import pyarrow.dataset  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soccernet_reid.models.backbones import create_backbone  # noqa: E402
from soccernet_reid.training.loop import (  # noqa: E402
    _extract_split_features,
    _query_gallery_frames,
    metrics_from_embeddings,
)

SPLIT = "valid"
SEEDS = (0, 1, 2, 3, 4)
RANDOM_DIM = 512


def expected_random_scores(query_df: pd.DataFrame, gallery_df: pd.DataFrame,
                           ranks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Exact mean of mAP and Rank-k when every gallery is ordered uniformly at random.

    For a gallery of m crops with n of the query's person, the expected AP is
    (H_m + (n - 1)(m - H_m)/(m - 1)) / m, where H_m is the m-th harmonic number, and
    the chance of a hit in the first k places is 1 - C(m - n, k) / C(m, k).
    """
    sizes = gallery_df.groupby("action_idx").size().to_dict()
    positives = gallery_df.groupby(["action_idx", "person_uid"]).size().to_dict()
    aps: list[float] = []
    hits: dict[int, list[float]] = {k: [] for k in ranks}
    for action, person in zip(query_df["action_idx"], query_df["person_uid"], strict=True):
        m, n = sizes[int(action)], positives[(int(action), int(person))]
        harmonic = sum(1.0 / i for i in range(1, m + 1))
        aps.append(1.0 if m == 1 else (harmonic + (n - 1) * (m - harmonic) / (m - 1)) / m)
        for k in ranks:
            hits[k].append(1.0 if m - n < k else 1.0 - comb(m - n, k) / comb(m, k))
    out = {"mAP": float(np.mean(aps))}
    out.update({f"rank-{k}": float(np.mean(v)) for k, v in hits.items()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "_g" / "UNTRAINED_valid.json")
    parser.add_argument("--device", default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    catalog = pd.read_parquet(args.catalog)
    _, query_df, gallery_df = _query_gallery_frames(catalog, SPLIT)
    ids = {
        "query_bbox_idx": query_df["bbox_idx"].to_numpy(dtype=np.int64),
        "gallery_bbox_idx": gallery_df["bbox_idx"].to_numpy(dtype=np.int64),
        "query_action_idx": query_df["action_idx"].to_numpy(dtype=np.int64),
        "gallery_action_idx": gallery_df["action_idx"].to_numpy(dtype=np.int64),
    }
    n_query, n_gallery = len(query_df), len(gallery_df)
    print(f"{SPLIT}: {n_query} queries, {n_gallery} gallery crops, {gallery_df['action_idx'].nunique()} actions")

    per_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        per_seed[str(seed)] = metrics_from_embeddings(
            dict(ids, query_feats=rng.standard_normal((n_query, RANDOM_DIM)).astype(np.float32),
                 gallery_feats=rng.standard_normal((n_gallery, RANDOM_DIM)).astype(np.float32)),
            catalog, split=SPLIT)
        print(f"random ranking, seed {seed}: {per_seed[str(seed)]}")
    metrics = list(per_seed[str(SEEDS[0])])
    random_mean = {m: float(np.mean([scores[m] for scores in per_seed.values()])) for m in metrics}

    same_vector = metrics_from_embeddings(
        dict(ids, query_feats=np.ones((n_query, 8), np.float32), gallery_feats=np.ones((n_gallery, 8), np.float32)),
        catalog, split=SPLIT)
    print(f"same vector: {same_vector}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = create_backbone("R18", pretrained=True).to(device).eval()
    start = time.perf_counter()
    features = {"query_feats": _extract_split_features(query_df, model, device, 64, args.num_workers),
                "gallery_feats": _extract_split_features(gallery_df, model, device, 64, args.num_workers)}
    resnet = metrics_from_embeddings(dict(ids, **features), catalog, split=SPLIT)
    print(f"ResNet-18 ImageNet: {resnet}")

    result = {
        "date": date.today().isoformat(),
        "split": SPLIT,
        "distance": "cosine",
        "queries": n_query,
        "gallery": n_gallery,
        "actions": int(gallery_df["action_idx"].nunique()),
        "random_ranking": {
            "method": f"{RANDOM_DIM}-dimensional standard normal vectors from numpy.random.default_rng(seed)",
            "seeds": list(SEEDS),
            "per_seed": per_seed,
            "mean": random_mean,
            "expected": expected_random_scores(query_df, gallery_df),
        },
        "same_vector": same_vector,
        "resnet18_imagenet": resnet,
        "resnet18_imagenet_run": {"device": str(device), "feature_dim": int(features["query_feats"].shape[1]),
                                  "seconds": round(time.perf_counter() - start, 1)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
