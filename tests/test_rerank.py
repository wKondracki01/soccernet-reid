"""Tests for per-action re-ranking (k-reciprocal and dual-softmax)."""
from __future__ import annotations

import numpy as np
import pytest

from soccernet_reid.eval.ranking import compute_rankings
from soccernet_reid.eval.rerank import (
    combined_scores,
    compute_reranked_rankings,
    dual_softmax_log_shares,
    dual_softmax_scores,
    dual_softmax_shares,
    k_reciprocal_components,
    k_reciprocal_distances,
    rerank_action_scores,
)


def _unit(rng: np.random.Generator, n: int, d: int = 16) -> np.ndarray:
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _reference_re_ranking(prob_fea, gal_fea, k1, k2, lambda_value):
    """NumPy port of the reference ``re_ranking.py`` by Zhong et al. (CVPR 2017).

    Kept deliberately close to the original, including its variable names. The
    only changes: NumPy instead of torch for the distance matrix, and float64
    instead of float16 (the original used float16 to save memory on galleries
    with tens of thousands of images, which also rounds the result).
    """
    query_num = prob_fea.shape[0]
    all_num = query_num + gal_fea.shape[0]
    feat = np.concatenate([prob_fea, gal_fea])
    sq = np.power(feat, 2).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS flags
        original_dist = sq + sq.T - 2.0 * feat @ feat.T
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                candidate_k_reciprocal_index
            ):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num])
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:query_num, query_num:]


class TestKReciprocalAgainstReference:
    @pytest.mark.parametrize(
        ("n_q", "n_g", "k1", "k2", "lam"),
        [
            (5, 18, 4, 2, 0.3),    # typical SoccerNet action
            (5, 18, 2, 1, 0.5),    # k2=1 skips the local query expansion
            (7, 19, 6, 3, 0.1),
            (1, 18, 3, 2, 0.7),    # single query
            (12, 60, 20, 6, 0.3),  # the paper's defaults on a large action
            (3, 4, 20, 6, 0.3),    # k1 and k2 larger than the action: lists truncate
            (2, 1, 1, 1, 0.0),
        ],
    )
    def test_matches_reference_implementation(self, n_q, n_g, k1, k2, lam) -> None:
        rng = np.random.default_rng(n_q * 1000 + n_g * 10 + k1)
        q, g = _unit(rng, n_q), _unit(rng, n_g)
        ours = k_reciprocal_distances(q, g, k1=k1, k2=k2, lambda_value=lam)
        ref = _reference_re_ranking(q, g, k1, k2, lam)
        assert ours.shape == (n_q, n_g)
        np.testing.assert_allclose(ours, ref, atol=1e-9, rtol=0)

    def test_components_combine_into_final_distance(self) -> None:
        rng = np.random.default_rng(0)
        q, g = _unit(rng, 6), _unit(rng, 20)
        jaccard, original = k_reciprocal_components(q, g, k1=4, k2=2)
        for lam in (0.0, 0.3, 1.0):
            np.testing.assert_allclose(
                (1 - lam) * jaccard + lam * original,
                k_reciprocal_distances(q, g, k1=4, k2=2, lambda_value=lam),
            )

    def test_unnormalised_features_give_the_same_result(self) -> None:
        rng = np.random.default_rng(1)
        q, g = _unit(rng, 4), _unit(rng, 15)
        scaled_q = q * rng.uniform(0.5, 5.0, size=(4, 1))
        scaled_g = g * rng.uniform(0.5, 5.0, size=(15, 1))
        np.testing.assert_allclose(
            k_reciprocal_distances(q, g), k_reciprocal_distances(scaled_q, scaled_g), atol=1e-12
        )


def _toy_split(rng: np.random.Generator, n_actions: int = 6):
    """Features and metadata for a few actions of varying size."""
    qf, gf, qb, gb, qa, ga = [], [], [], [], [], []
    next_q, next_g = 0, 10_000
    for action in range(n_actions):
        n_q = int(rng.integers(1, 7))
        n_g = int(rng.integers(1, 22))
        qf.append(_unit(rng, n_q))
        gf.append(_unit(rng, n_g))
        qb += list(range(next_q, next_q + n_q))
        gb += list(range(next_g, next_g + n_g))
        qa += [action] * n_q
        ga += [action] * n_g
        next_q += n_q
        next_g += n_g
    return np.concatenate(qf), np.concatenate(gf), qb, gb, qa, ga


class TestInvariants:
    def test_lambda_one_reproduces_plain_cosine_ranking(self) -> None:
        # final = original distance, a monotone function of cosine within a row
        split = _toy_split(np.random.default_rng(2))
        plain = compute_rankings(*split, distance="cosine")
        reranked = compute_reranked_rankings(*split, method="k_reciprocal", lambda_value=1.0)
        assert reranked == plain

    def test_dual_softmax_with_huge_temperature_keeps_plain_order(self) -> None:
        split = _toy_split(np.random.default_rng(3))
        plain = compute_rankings(*split, distance="cosine")
        reranked = compute_reranked_rankings(*split, method="dual_softmax", temperature=1e9)
        assert reranked == plain

    def test_both_with_huge_temperature_equals_k_reciprocal(self) -> None:
        split = _toy_split(np.random.default_rng(4))
        kr = compute_reranked_rankings(*split, method="k_reciprocal", k1=3, k2=2, lambda_value=0.4)
        both = compute_reranked_rankings(
            *split, method="both", k1=3, k2=2, lambda_value=0.4, temperature=1e9
        )
        assert both == kr

    def test_single_query_is_untouched_by_dual_softmax(self) -> None:
        rng = np.random.default_rng(5)
        q, g = _unit(rng, 1), _unit(rng, 12)
        sim = q @ g.T
        scores = dual_softmax_scores(sim, temperature=0.05)
        assert np.array_equal(np.argsort(-scores[0], kind="stable"), np.argsort(-sim[0], kind="stable"))

    def test_each_ranking_is_a_permutation_of_its_action_gallery(self) -> None:
        qf, gf, qb, gb, qa, ga = _toy_split(np.random.default_rng(6))
        for method in ("k_reciprocal", "dual_softmax", "both"):
            rankings = compute_reranked_rankings(qf, gf, qb, gb, qa, ga, method=method)
            assert set(rankings) == {str(b) for b in qb}
            for q_bbox, action in zip(qb, qa, strict=True):
                expected = sorted(b for b, a in zip(gb, ga, strict=True) if a == action)
                assert sorted(rankings[str(q_bbox)]) == expected

    def test_actions_do_not_influence_each_other(self) -> None:
        rng = np.random.default_rng(7)
        qf, gf, qb, gb, qa, ga = _toy_split(rng)
        before = compute_reranked_rankings(qf, gf, qb, gb, qa, ga, method="both")
        qf2, gf2 = qf.copy(), gf.copy()
        last = max(qa)
        qf2[np.asarray(qa) == last] = _unit(rng, int(np.sum(np.asarray(qa) == last)))
        gf2[np.asarray(ga) == last] = _unit(rng, int(np.sum(np.asarray(ga) == last)))
        after = compute_reranked_rankings(qf2, gf2, qb, gb, qa, ga, method="both")
        for q_bbox, action in zip(qb, qa, strict=True):
            if action != last:
                assert after[str(q_bbox)] == before[str(q_bbox)]

    def test_deterministic(self) -> None:
        split = _toy_split(np.random.default_rng(8))
        a = compute_reranked_rankings(*split, method="both")
        b = compute_reranked_rankings(*split, method="both")
        assert a == b

    def test_k_reciprocal_can_change_the_order(self) -> None:
        split = _toy_split(np.random.default_rng(9), n_actions=30)
        plain = compute_rankings(*split, distance="cosine")
        reranked = compute_reranked_rankings(*split, method="k_reciprocal", k1=4, k2=2, lambda_value=0.3)
        assert reranked != plain


class TestDualSoftmax:
    def test_competing_query_takes_the_crop(self) -> None:
        # Crop X looks like A (0.80) but much more like B (0.95); crop Y fits only A.
        sim = np.array([[0.80, 0.78],   # query A vs crops X, Y
                        [0.95, 0.40]])  # query B
        assert np.argmax(sim[0]) == 0  # plain ranking puts X first for A
        scores = dual_softmax_scores(sim, temperature=0.1)
        assert np.argmax(scores[0]) == 1  # after normalisation A prefers Y
        assert np.argmax(scores[1]) == 0  # B keeps X

    def test_shares_sum_to_one_over_queries(self) -> None:
        rng = np.random.default_rng(10)
        shares = dual_softmax_shares(rng.uniform(-1, 1, size=(5, 9)), temperature=0.07)
        np.testing.assert_allclose(shares.sum(axis=0), np.ones(9))
        assert (shares >= 0).all()

    def test_distractor_crop_is_shared_evenly(self) -> None:
        # a crop equally (dis)similar to every query gives equal shares
        sim = np.array([[0.9, 0.1], [0.2, 0.1], [0.3, 0.1]])
        shares = dual_softmax_shares(sim, temperature=0.05)
        np.testing.assert_allclose(shares[:, 1], np.full(3, 1 / 3))

    def test_scores_are_the_log_of_similarity_times_share(self) -> None:
        rng = np.random.default_rng(12)
        sim = rng.uniform(-0.9, 1.0, size=(6, 11))
        product = (sim + 1.0) / 2.0 * dual_softmax_shares(sim, temperature=0.07)
        scores = dual_softmax_scores(sim, temperature=0.07)
        np.testing.assert_allclose(np.exp(scores), product, rtol=1e-12)
        for row in range(sim.shape[0]):  # hence the same order as the plain product
            assert np.array_equal(
                np.argsort(-scores[row], kind="stable"), np.argsort(-product[row], kind="stable")
            )

    def test_combined_scores_are_the_log_of_the_product(self) -> None:
        rng = np.random.default_rng(13)
        sim = rng.uniform(-0.9, 1.0, size=(4, 7))
        final = rng.uniform(0.0, 0.99, size=(4, 7))
        product = (1.0 - final) * dual_softmax_shares(sim, temperature=0.1)
        np.testing.assert_allclose(np.exp(combined_scores(final, sim, 0.1)), product, rtol=1e-12)

    def test_low_temperature_does_not_underflow_into_ties(self) -> None:
        # Query B wins every crop by a wide margin. As a plain product A's shares
        # are exp(-8000), exp(-6500), exp(-3000) = 0.0, i.e. three tied crops left
        # in gallery order; in the log domain A still ranks them by the gap to B.
        sim = np.array([[0.10, 0.30, 0.20],
                        [0.90, 0.95, 0.50]])
        assert (dual_softmax_shares(sim, temperature=1e-4)[0] == 0.0).all()
        scores = dual_softmax_scores(sim, temperature=1e-4)
        assert np.isfinite(scores).all()
        assert list(np.argsort(-scores[0], kind="stable")) == [2, 1, 0]   # gaps 0.30 < 0.65 < 0.80
        assert list(np.argsort(-scores[1], kind="stable")) == [1, 0, 2]   # B's own order by similarity
        both = combined_scores(np.full((2, 3), 0.5), sim, temperature=1e-4)
        assert np.isfinite(both).all()
        assert list(np.argsort(-both[0], kind="stable")) == [2, 1, 0]

    def test_log_shares_match_shares(self) -> None:
        rng = np.random.default_rng(14)
        sim = rng.uniform(-1, 1, size=(5, 9))
        np.testing.assert_allclose(
            np.exp(dual_softmax_log_shares(sim, 0.05)), dual_softmax_shares(sim, 0.05), rtol=1e-12
        )

    def test_zero_base_score_stays_finite(self) -> None:
        # cosine -1 and a k-reciprocal distance of exactly 1 both give a base score of 0
        sim = np.array([[-1.0, 0.5], [0.2, 0.1]])
        assert np.isfinite(dual_softmax_scores(sim, temperature=0.1)).all()
        scores = combined_scores(np.array([[1.0, 0.2], [0.3, 0.4]]), sim, temperature=0.1)
        assert np.isfinite(scores).all()
        assert np.argmin(scores[0]) == 0


class TestEdgeCases:
    @pytest.mark.parametrize(("n_q", "n_g"), [(1, 1), (1, 2), (2, 1), (3, 2)])
    @pytest.mark.parametrize("method", ["k_reciprocal", "dual_softmax", "both"])
    def test_tiny_actions(self, n_q: int, n_g: int, method: str) -> None:
        rng = np.random.default_rng(n_q * 10 + n_g)
        scores = rerank_action_scores(_unit(rng, n_q), _unit(rng, n_g), method, k1=20, k2=6)
        assert scores.shape == (n_q, n_g)
        assert np.isfinite(scores).all()

    def test_identical_features_do_not_produce_nan(self) -> None:
        q = np.ones((2, 8))
        g = np.ones((5, 8))
        for method in ("k_reciprocal", "dual_softmax", "both"):
            assert np.isfinite(rerank_action_scores(q, g, method)).all()

    def test_query_without_gallery_gets_empty_ranking(self) -> None:
        rng = np.random.default_rng(11)
        rankings = compute_reranked_rankings(
            _unit(rng, 2), _unit(rng, 3), [1, 2], [10, 11, 12], [0, 5], [0, 0, 0]
        )
        assert sorted(rankings["1"]) == [10, 11, 12]
        assert rankings["2"] == []


def _load_eval_rerank_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "eval_rerank.py"
    spec = importlib.util.spec_from_file_location("eval_rerank_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEvalRerankScript:
    """scripts/eval_rerank.py scores the grid from cached per-action pieces.

    Its rankings must equal the library function's, otherwise the tuned numbers
    would describe a different procedure than the one documented.
    """

    @pytest.mark.parametrize(
        ("method", "params"),
        [
            ("k_reciprocal", {"k1": 3, "k2": 2, "lambda_value": 0.4}),
            ("k_reciprocal", {"k1": 1, "k2": 1, "lambda_value": 0.9}),
            ("dual_softmax", {"temperature": 0.05}),
            ("both", {"k1": 4, "k2": 2, "lambda_value": 0.3, "temperature": 0.1}),
        ],
    )
    def test_cached_scores_match_library_rankings(self, method: str, params: dict) -> None:
        from soccernet_reid.eval.rerank import rankings_from_action_scores

        script = _load_eval_rerank_script()
        qf, gf, qb, gb, qa, ga = _toy_split(np.random.default_rng(20), n_actions=12)
        qa = qa + [99]          # one query whose action has no gallery
        qb = qb + [777_777]
        qf = np.concatenate([qf, _unit(np.random.default_rng(21), 1)])
        emb = {
            "query_feats": (qf * 3.0).astype(np.float32),   # unnormalised, float32 like the .npz
            "gallery_feats": (gf * 0.5).astype(np.float32),
            "query_bbox_idx": np.asarray(qb), "gallery_bbox_idx": np.asarray(gb),
            "query_action_idx": np.asarray(qa), "gallery_action_idx": np.asarray(ga),
        }
        actions = script.Actions(emb)
        from_script = rankings_from_action_scores(actions.scores(method, params), qb, gb, qa, ga)
        from_library = compute_reranked_rankings(
            emb["query_feats"], emb["gallery_feats"], qb, gb, qa, ga, method=method, **params
        )
        assert from_script == from_library
        assert from_script["777777"] == []


class TestValidation:
    def test_unknown_method(self) -> None:
        rng = np.random.default_rng(12)
        with pytest.raises(ValueError, match="Unknown re-ranking method"):
            rerank_action_scores(_unit(rng, 2), _unit(rng, 3), "nope")  # type: ignore[arg-type]

    @pytest.mark.parametrize(("k1", "k2"), [(0, 2), (2, 0), (1.5, 2)])
    def test_bad_k(self, k1, k2) -> None:
        rng = np.random.default_rng(13)
        with pytest.raises(ValueError, match="must be an integer >= 1"):
            k_reciprocal_distances(_unit(rng, 2), _unit(rng, 3), k1=k1, k2=k2)

    @pytest.mark.parametrize("lam", [-0.1, 1.1])
    def test_bad_lambda(self, lam: float) -> None:
        rng = np.random.default_rng(14)
        with pytest.raises(ValueError, match="lambda_value must be in"):
            k_reciprocal_distances(_unit(rng, 2), _unit(rng, 3), lambda_value=lam)

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_bad_temperature(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature must be > 0"):
            dual_softmax_shares(np.zeros((2, 2)), temperature)

    @pytest.mark.parametrize("method", ["k_reciprocal", "dual_softmax", "both"])
    def test_non_finite_embeddings_fail_loudly(self, method: str) -> None:
        # e.g. a model whose BatchNorm running statistics turned NaN (F3_EB4)
        rng = np.random.default_rng(16)
        q, g = _unit(rng, 2), _unit(rng, 4)
        g[1, 3] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            rerank_action_scores(q, g, method)

    def test_length_mismatch(self) -> None:
        rng = np.random.default_rng(15)
        with pytest.raises(ValueError, match="query_feats rows must match"):
            compute_reranked_rankings(_unit(rng, 2), _unit(rng, 3), [1], [10, 11, 12], [0, 0], [0, 0, 0])
