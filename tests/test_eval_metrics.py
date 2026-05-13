"""Unit tests for mAP/Rank-k computation. No torch, no real data needed.

The expected values are computed by hand from the official algorithm.
"""
from __future__ import annotations

import numpy as np
import pytest

from soccernet_reid.eval.metrics import compute_metrics, validate_rankings_complete


def _make_meta(rows):
    """Build a {bbox_idx_str: {fields}} dict.

    Each row: (bbox_idx, person_uid, action_idx).
    """
    return {
        str(b): {"bbox_idx": b, "person_uid": p, "action_idx": a}
        for b, p, a in rows
    }


class TestSingleQueryAP:
    """AP for a single query, varying ranking quality."""

    def test_perfect_ranking_all_matches(self) -> None:
        # Single action with 1 query (p=1) and 4 gallery items all p=1
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 1, 0), (11, 1, 0), (12, 1, 0), (13, 1, 0)])
        rankings = {"0": [10, 11, 12, 13]}
        m = compute_metrics(rankings, query, gallery, ranks=(1, 5))
        assert m["mAP"] == pytest.approx(1.0)
        assert m["rank-1"] == pytest.approx(1.0)
        assert m["rank-5"] == pytest.approx(1.0)

    def test_known_ap_5_over_6(self) -> None:
        # Gallery match pattern [+, -, +, -] → AP = (1 + 2/3) / 2 = 5/6
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 1, 0), (11, 2, 0), (12, 1, 0), (13, 3, 0)])
        rankings = {"0": [10, 11, 12, 13]}
        m = compute_metrics(rankings, query, gallery, ranks=(1, 2, 5))
        assert m["mAP"] == pytest.approx(5 / 6, abs=1e-9)
        assert m["rank-1"] == pytest.approx(1.0)
        assert m["rank-2"] == pytest.approx(1.0)

    def test_known_ap_one_half(self) -> None:
        # Gallery match pattern [-, +, -, +] → AP = (1/2 + 2/4) / 2 = 0.5
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 2, 0), (11, 1, 0), (12, 3, 0), (13, 1, 0)])
        rankings = {"0": [10, 11, 12, 13]}
        m = compute_metrics(rankings, query, gallery, ranks=(1, 2, 4, 5))
        assert m["mAP"] == pytest.approx(0.5, abs=1e-9)
        assert m["rank-1"] == pytest.approx(0.0)
        assert m["rank-2"] == pytest.approx(1.0)
        assert m["rank-4"] == pytest.approx(1.0)
        # rank-5 padded from rank-4=1 → still 1
        assert m["rank-5"] == pytest.approx(1.0)


class TestMultiQueryAggregation:
    def test_mean_ap_two_queries(self) -> None:
        # query 0: [+, -, +, -] → AP=5/6, Rank-1=1
        # query 1: [-, +, -, +] → AP=1/2, Rank-1=0
        query = _make_meta([(0, 1, 0), (1, 1, 1)])
        gallery = _make_meta([
            (10, 1, 0), (11, 2, 0), (12, 1, 0), (13, 3, 0),
            (20, 2, 1), (21, 1, 1), (22, 3, 1), (23, 1, 1),
        ])
        rankings = {"0": [10, 11, 12, 13], "1": [20, 21, 22, 23]}
        m = compute_metrics(rankings, query, gallery, ranks=(1, 2))
        # Mean AP = (5/6 + 1/2) / 2 = 8/12 = 2/3
        assert m["mAP"] == pytest.approx(2 / 3, abs=1e-9)
        # Mean Rank-1 = (1 + 0)/2 = 0.5
        assert m["rank-1"] == pytest.approx(0.5)


class TestValidation:
    def test_missing_query_ranking(self) -> None:
        query = _make_meta([(0, 1, 0), (1, 1, 0)])
        gallery = _make_meta([(10, 1, 0)])
        rankings = {"0": [10]}  # missing "1"
        with pytest.raises(ValueError, match="No ranking provided for query '1'"):
            validate_rankings_complete(rankings, query, gallery)

    def test_duplicate_in_ranking(self) -> None:
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 1, 0), (11, 2, 0)])
        rankings = {"0": [10, 10]}
        with pytest.raises(ValueError, match="duplicate"):
            validate_rankings_complete(rankings, query, gallery)

    def test_missing_gallery_from_action(self) -> None:
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 1, 0), (11, 2, 0), (12, 3, 0)])
        rankings = {"0": [10, 11]}  # 12 missing
        with pytest.raises(ValueError, match="missing"):
            validate_rankings_complete(rankings, query, gallery)

    def test_cross_action_gallery_rejected(self) -> None:
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 1, 0), (20, 1, 1)])  # 20 is in different action
        rankings = {"0": [10, 20]}
        with pytest.raises(ValueError, match="NOT from action"):
            validate_rankings_complete(rankings, query, gallery)

    def test_no_positive_in_gallery(self) -> None:
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 2, 0), (11, 3, 0)])  # no person_uid=1 in gallery
        rankings = {"0": [10, 11]}
        with pytest.raises(ValueError, match="no positive"):
            validate_rankings_complete(rankings, query, gallery)


class TestRankPadding:
    def test_rank_k_padded_when_ranking_shorter_than_k(self) -> None:
        # Gallery has 3 items, but we ask for rank-10
        query = _make_meta([(0, 1, 0)])
        gallery = _make_meta([(10, 1, 0), (11, 2, 0), (12, 3, 0)])
        rankings = {"0": [10, 11, 12]}
        m = compute_metrics(rankings, query, gallery, ranks=(1, 10))
        # Rank-1: match at position 0 → 1.0
        # Rank-10: only 3 items in ranking, but cmc was clipped to 1 at position 0,
        #          stays 1 forever via padding → 1.0
        assert m["rank-1"] == pytest.approx(1.0)
        assert m["rank-10"] == pytest.approx(1.0)


class TestNumericalStability:
    def test_many_queries_same_pattern(self) -> None:
        # 100 queries all with identical pattern → mAP should be exactly the per-query AP
        rows_q = [(i, 1, i) for i in range(100)]
        rows_g = []
        rankings = {}
        for i in range(100):
            # action i has gallery [+, -, +, -]
            base = 1000 + i * 10
            rows_g.extend([(base + 0, 1, i), (base + 1, 2, i), (base + 2, 1, i), (base + 3, 3, i)])
            rankings[str(i)] = [base + 0, base + 1, base + 2, base + 3]
        query = _make_meta(rows_q)
        gallery = _make_meta(rows_g)
        m = compute_metrics(rankings, query, gallery, ranks=(1,))
        assert m["mAP"] == pytest.approx(5 / 6, abs=1e-9)
        assert m["rank-1"] == pytest.approx(1.0)
