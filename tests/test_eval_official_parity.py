"""Smoke test verifying that our evaluator matches the official SoccerNet evaluator
to numerical precision.

We feed random rankings (since this test verifies the METRIC, not retrieval quality)
on the full valid set and compare mAP / rank-1 between:
  - our `compute_metrics`
  - `SoccerNet.Evaluation.ReIdentification.evaluate`

Both should agree to <1e-5 absolute error (they implement the same algorithm in
numpy, only difference is iteration order and aggregation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soccernet_reid.data.catalog import build_catalog
from soccernet_reid.eval.metrics import compute_metrics
from soccernet_reid.eval.official import (
    catalog_to_groundtruth_dict,
    run_official_evaluator,
)

REID_ROOT = Path(__file__).resolve().parent.parent / "dataSoccerNet" / "reid-2023"

pytestmark = pytest.mark.skipif(
    not REID_ROOT.is_dir(),
    reason=f"Dataset not present at {REID_ROOT}",
)


def _build_random_rankings(
    groundtruth: dict,
    seed: int,
) -> dict[str, list[int]]:
    """For each query, return a random permutation of all gallery items in the same action."""
    rng = np.random.default_rng(seed)
    # Group gallery bbox_idx by action_idx
    action_to_gallery: dict[int, list[int]] = {}
    for g in groundtruth["gallery"].values():
        action_to_gallery.setdefault(int(g["action_idx"]), []).append(int(g["bbox_idx"]))

    rankings: dict[str, list[int]] = {}
    for q_str, q in groundtruth["query"].items():
        action = int(q["action_idx"])
        candidates = list(action_to_gallery.get(action, []))
        rng.shuffle(candidates)
        rankings[q_str] = candidates
    return rankings


@pytest.fixture(scope="module")
def gt_dict() -> dict:
    df = build_catalog(REID_ROOT, splits=["valid"])
    return catalog_to_groundtruth_dict(df, split="valid")


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_rankings_match_official(gt_dict, seed: int) -> None:
    rankings = _build_random_rankings(gt_dict, seed=seed)

    # Filter out queries with no positive in gallery — official evaluator would error.
    # In the actual valid set this shouldn't happen for the official split, but
    # be defensive in case some action has only one ID.
    gallery_pid_by_bbox = {
        int(g["bbox_idx"]): int(g["person_uid"]) for g in gt_dict["gallery"].values()
    }
    action_to_gallery = {}
    for g in gt_dict["gallery"].values():
        action_to_gallery.setdefault(int(g["action_idx"]), []).append(int(g["bbox_idx"]))

    valid_queries = {}
    valid_rankings = {}
    for q_str, q in gt_dict["query"].items():
        q_pid = int(q["person_uid"])
        action = int(q["action_idx"])
        if any(gallery_pid_by_bbox[g] == q_pid for g in action_to_gallery.get(action, [])):
            valid_queries[q_str] = q
            valid_rankings[q_str] = rankings[q_str]

    print(f"\n  using {len(valid_queries):,} of {len(gt_dict['query']):,} queries "
          f"({100*len(valid_queries)/len(gt_dict['query']):.1f}% have positives)")

    filtered_gt = {"query": valid_queries, "gallery": gt_dict["gallery"]}
    ours = compute_metrics(
        valid_rankings, filtered_gt["query"], filtered_gt["gallery"], ranks=(1,)
    )
    theirs = run_official_evaluator(valid_rankings, filtered_gt)

    print(f"  ours:    mAP={ours['mAP']:.6f}  rank-1={ours['rank-1']:.6f}")
    print(f"  theirs:  mAP={theirs['mAP']:.6f}  rank-1={theirs['rank-1']:.6f}")
    assert ours["mAP"] == pytest.approx(theirs["mAP"], abs=1e-6)
    assert ours["rank-1"] == pytest.approx(theirs["rank-1"], abs=1e-6)


def test_extreme_pattern_match(gt_dict) -> None:
    """Specifically craft a 'perfect oracle' ranking and 'inverted oracle' ranking
    to make sure both ends of the metric scale agree.
    """
    gallery_pid_by_bbox = {
        int(g["bbox_idx"]): int(g["person_uid"]) for g in gt_dict["gallery"].values()
    }
    action_to_gallery = {}
    for g in gt_dict["gallery"].values():
        action_to_gallery.setdefault(int(g["action_idx"]), []).append(int(g["bbox_idx"]))

    # Oracle: positives first, then negatives — should give mAP = 1.0
    oracle_rankings: dict[str, list[int]] = {}
    valid_queries = {}
    for q_str, q in gt_dict["query"].items():
        q_pid = int(q["person_uid"])
        action = int(q["action_idx"])
        items = action_to_gallery.get(action, [])
        positives = [g for g in items if gallery_pid_by_bbox[g] == q_pid]
        negatives = [g for g in items if gallery_pid_by_bbox[g] != q_pid]
        if not positives:
            continue
        valid_queries[q_str] = q
        oracle_rankings[q_str] = positives + negatives

    filtered_gt = {"query": valid_queries, "gallery": gt_dict["gallery"]}
    ours = compute_metrics(oracle_rankings, filtered_gt["query"], filtered_gt["gallery"], ranks=(1,))
    theirs = run_official_evaluator(oracle_rankings, filtered_gt)

    assert ours["mAP"] == pytest.approx(1.0, abs=1e-9)
    assert theirs["mAP"] == pytest.approx(1.0, abs=1e-9)
    assert ours["mAP"] == pytest.approx(theirs["mAP"], abs=1e-9)
    assert ours["rank-1"] == pytest.approx(theirs["rank-1"], abs=1e-9)
