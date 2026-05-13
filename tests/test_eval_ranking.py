"""Unit tests for embedding → ranking pipeline (no torch needed, numpy only)."""
from __future__ import annotations

import numpy as np
import pytest

from soccernet_reid.eval.ranking import compute_rankings, evaluate_embeddings


def test_cosine_ranking_correct_order() -> None:
    # Single query, gallery with 3 items in same action. Crafted embeddings:
    # query = [1, 0]
    # gallery [10] = [1, 0] (most similar)
    # gallery [11] = [0.5, 0.5] (less similar)
    # gallery [12] = [0, 1] (orthogonal, least similar)
    q_feats = np.array([[1.0, 0.0]], dtype=np.float32)
    g_feats = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    rankings = compute_rankings(
        query_feats=q_feats,
        gallery_feats=g_feats,
        query_bbox_idx=[0],
        gallery_bbox_idx=[10, 11, 12],
        query_actions=[0],
        gallery_actions=[0, 0, 0],
        distance="cosine",
    )
    assert rankings == {"0": [10, 11, 12]}


def test_cross_action_gallery_excluded() -> None:
    # Query in action 0, gallery items in actions 0 and 1.
    # Ranking should ONLY include action 0 gallery items.
    q_feats = np.array([[1.0, 0.0]], dtype=np.float32)
    g_feats = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    rankings = compute_rankings(
        query_feats=q_feats,
        gallery_feats=g_feats,
        query_bbox_idx=[0],
        gallery_bbox_idx=[10, 11, 12, 99],
        query_actions=[0],
        gallery_actions=[0, 0, 0, 1],   # last item in different action
        distance="cosine",
    )
    assert 99 not in rankings["0"], "Cross-action gallery item leaked into ranking"
    assert set(rankings["0"]) == {10, 11, 12}


def test_euclidean_ranking_correct_order() -> None:
    # With euclidean, smaller distance = better
    q_feats = np.array([[0.0, 0.0]], dtype=np.float32)
    g_feats = np.array([[0.1, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    rankings = compute_rankings(
        query_feats=q_feats, gallery_feats=g_feats,
        query_bbox_idx=[0], gallery_bbox_idx=[10, 11, 12],
        query_actions=[0], gallery_actions=[0, 0, 0],
        distance="euclidean",
    )
    assert rankings == {"0": [10, 11, 12]}


def test_evaluate_embeddings_end_to_end() -> None:
    # 2 queries, 2 actions, gallery sized to give a known mAP
    query_meta = {
        "0": {"bbox_idx": 0, "person_uid": 1, "action_idx": 0},
        "1": {"bbox_idx": 1, "person_uid": 1, "action_idx": 1},
    }
    # Gallery in action 0: bbox 10 (pid=1, match), bbox 11 (pid=2)
    # Gallery in action 1: bbox 20 (pid=2), bbox 21 (pid=1, match)
    gallery_meta = {
        "10": {"bbox_idx": 10, "person_uid": 1, "action_idx": 0},
        "11": {"bbox_idx": 11, "person_uid": 2, "action_idx": 0},
        "20": {"bbox_idx": 20, "person_uid": 2, "action_idx": 1},
        "21": {"bbox_idx": 21, "person_uid": 1, "action_idx": 1},
    }
    # Embeddings: queries and their positives are identical
    q_feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    g_feats = np.array([
        [1.0, 0.0],   # bbox 10 (match for query 0)
        [0.0, 1.0],   # bbox 11 (will rank below match in action 0)
        [1.0, 0.0],   # bbox 20 (low sim to query 1, ranks below match)
        [0.0, 1.0],   # bbox 21 (match for query 1)
    ], dtype=np.float32)
    m = evaluate_embeddings(
        query_feats=q_feats, gallery_feats=g_feats,
        query_meta=query_meta, gallery_meta=gallery_meta,
        distance="cosine", ranks=(1, 2),
    )
    # Both queries: top-1 is the match → mAP=1.0, Rank-1=1.0
    assert m["mAP"] == pytest.approx(1.0)
    assert m["rank-1"] == pytest.approx(1.0)


def test_feature_dim_mismatch_raises() -> None:
    q_feats = np.zeros((2, 3), dtype=np.float32)
    g_feats = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="feature dim mismatch"):
        compute_rankings(q_feats, g_feats, [0, 1], [10, 11], [0, 0], [0, 0])


def test_metadata_length_mismatch_raises() -> None:
    q_feats = np.zeros((2, 3), dtype=np.float32)
    g_feats = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatches"):
        compute_rankings(q_feats, g_feats, [0, 1, 2], [10, 11], [0, 0], [0, 0])
