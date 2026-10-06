"""Per-action re-ranking: post-processing of the gallery order, no retraining.

In SoccerNet ReID a query is ranked only against the gallery of its own action,
so everything here works on ONE action at a time: all queries of the action
plus its gallery (median 24 crops in total on the valid split).

Two methods, usable separately or together:

``k_reciprocal``
    k-reciprocal re-ranking (Zhong et al., "Re-ranking Person Re-identification
    with k-reciprocal Encoding", CVPR 2017). The final distance mixes the
    original distance with a Jaccard distance between k-reciprocal neighbour
    sets::

        final = (1 - lambda) * jaccard + lambda * original

    The neighbour graph is built on the joint set [queries; gallery] of the
    action, as in the reference implementation. The defaults of the paper
    (k1=20, k2=6) target galleries with thousands of images and many shots per
    identity; here the whole action is ~24 crops and 73% of queries have a
    single positive, so k1 and k2 must be small and are tuned on valid.
    Neighbour lists are simply truncated when an action has fewer than k1+1
    crops.

``dual_softmax``
    Normalisation across the queries of an action. Within one action every
    query is a different person, and a gallery crop shows one person, so a crop
    that matches another query much better is unlikely to be this query's
    positive. For each gallery crop the similarities to all queries are turned
    into shares with a softmax (temperature ``T``), and the query's own
    similarity is multiplied by its share. With one query, or with a very large
    temperature, the order is unchanged. This relies on a property of the
    benchmark (distinct identities among the queries of an action) and must be
    reported as such.

``both``
    The k-reciprocal similarity ``1 - final`` multiplied by the dual-softmax
    shares.

The plain cosine ranking stays in :mod:`soccernet_reid.eval.ranking`, which is
kept bit-identical to the official evaluator; nothing here is used unless a
re-ranking method is requested explicitly.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

RerankMethod = Literal["k_reciprocal", "dual_softmax", "both"]
_METHODS: tuple[str, ...] = ("k_reciprocal", "dual_softmax", "both")


def _unit_rows(x) -> np.ndarray:
    """Rows of ``x`` as float64 unit vectors (zero rows stay zero)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2-D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b.T`` for unit rows, with a real finiteness check.

    Some BLAS builds (Accelerate on Apple silicon) raise spurious divide /
    overflow / invalid flags in matmul on perfectly finite inputs, so the flags
    are silenced here and the result is checked instead. Non-finite embeddings
    (e.g. a model whose BatchNorm statistics turned NaN) fail loudly rather than
    yielding an arbitrary order.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = a @ b.T
    if not np.isfinite(sim).all():
        raise ValueError("embeddings contain NaN or Inf; cannot re-rank")
    return sim


def _check_k_params(k1: int, k2: int) -> None:
    if int(k1) != k1 or k1 < 1:
        raise ValueError(f"k1 must be an integer >= 1, got {k1!r}")
    if int(k2) != k2 or k2 < 1:
        raise ValueError(f"k2 must be an integer >= 1, got {k2!r}")


def k_reciprocal_components(
    query_feats,
    gallery_feats,
    k1: int,
    k2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Jaccard and original distance blocks for one action.

    Returns ``(jaccard, original)``, both ``[n_q, n_g]``. The final k-reciprocal
    distance is ``(1 - lambda) * jaccard + lambda * original``; returning the
    two parts lets a parameter search sweep ``lambda`` without recomputing the
    neighbour sets.

    Follows the reference implementation step by step: squared Euclidean
    distance on unit vectors, each row scaled by its maximum, k-reciprocal sets
    with the 2/3-overlap expansion, Gaussian-kernel weights, local query
    expansion over the ``k2`` nearest neighbours, then the Jaccard distance.
    """
    _check_k_params(k1, k2)
    q = _unit_rows(query_feats)
    g = _unit_rows(gallery_feats)
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"feature dim mismatch: query D={q.shape[1]} gallery D={g.shape[1]}")
    n_q, n_g = q.shape[0], g.shape[0]
    n = n_q + n_g
    if n_q == 0 or n_g == 0:
        return np.zeros((n_q, n_g)), np.zeros((n_q, n_g))

    feats = np.concatenate([q, g], axis=0)
    dist = np.clip(2.0 - 2.0 * _similarity(feats, feats), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    row_max = dist.max(axis=1, keepdims=True)
    row_max[row_max == 0.0] = 1.0  # all crops identical: keep zeros, avoid 0/0
    original = dist / row_max

    initial_rank = np.argsort(original, axis=1, kind="stable")
    half = int(np.around(k1 / 2))

    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.where(backward == i)[0]]
        expanded = reciprocal
        for candidate in reciprocal:
            cand_forward = initial_rank[candidate, : half + 1]
            cand_backward = initial_rank[cand_forward, : half + 1]
            cand_reciprocal = cand_forward[np.where(cand_backward == candidate)[0]]
            overlap = len(np.intersect1d(cand_reciprocal, reciprocal))
            if overlap > 2.0 / 3.0 * len(cand_reciprocal):
                expanded = np.append(expanded, cand_reciprocal)
        expanded = np.unique(expanded)
        w = np.exp(-original[i, expanded])
        weights[i, expanded] = w / w.sum()

    if k2 != 1:
        expanded_weights = np.zeros_like(weights)
        for i in range(n):
            expanded_weights[i] = weights[initial_rank[i, :k2]].mean(axis=0)
        weights = expanded_weights

    # sum_k min(V[i, k], V[j, k]) for every query i and every crop j
    shared = np.minimum(weights[:n_q, None, :], weights[None, :, :]).sum(axis=2)
    jaccard = 1.0 - shared / (2.0 - shared)
    return jaccard[:, n_q:], original[:n_q, n_q:]


def k_reciprocal_distances(
    query_feats,
    gallery_feats,
    k1: int = 4,
    k2: int = 2,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Final k-reciprocal distance ``[n_q, n_g]`` for one action (lower = closer)."""
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(f"lambda_value must be in [0, 1], got {lambda_value!r}")
    jaccard, original = k_reciprocal_components(query_feats, gallery_feats, k1, k2)
    return (1.0 - lambda_value) * jaccard + lambda_value * original


def dual_softmax_shares(similarity: np.ndarray, temperature: float) -> np.ndarray:
    """Share of each query in each gallery crop: softmax over queries (axis 0)."""
    if not temperature > 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError(f"similarity must be [n_q, n_g], got shape {sim.shape}")
    if sim.shape[0] == 0:
        return np.zeros_like(sim)
    logits = sim / temperature
    logits = logits - logits.max(axis=0, keepdims=True)
    shares = np.exp(logits)
    return shares / shares.sum(axis=0, keepdims=True)


def dual_softmax_scores(similarity: np.ndarray, temperature: float) -> np.ndarray:
    """Cosine similarity mapped to [0, 1], times the query's share (higher = better)."""
    sim = np.asarray(similarity, dtype=np.float64)
    return (sim + 1.0) / 2.0 * dual_softmax_shares(sim, temperature)


def rerank_action_scores(
    query_feats,
    gallery_feats,
    method: RerankMethod,
    *,
    k1: int = 4,
    k2: int = 2,
    lambda_value: float = 0.3,
    temperature: float = 0.1,
) -> np.ndarray:
    """Scores ``[n_q, n_g]`` for one action; a higher score ranks earlier."""
    if method not in _METHODS:
        raise ValueError(f"Unknown re-ranking method {method!r}; choose from {_METHODS}")
    q = _unit_rows(query_feats)
    g = _unit_rows(gallery_feats)
    if method == "dual_softmax":
        return dual_softmax_scores(_similarity(q, g), temperature)
    final = k_reciprocal_distances(q, g, k1=k1, k2=k2, lambda_value=lambda_value)
    if method == "k_reciprocal":
        return -final
    return (1.0 - final) * dual_softmax_shares(_similarity(q, g), temperature)


def group_positions_by_action(actions: Sequence[int]) -> dict[int, np.ndarray]:
    """``{action_idx: positions}`` with positions in their original order."""
    arr = np.asarray(actions, dtype=np.int64)
    return {int(a): np.where(arr == a)[0] for a in np.unique(arr)}


def rankings_from_action_scores(
    scores_by_action: dict[int, np.ndarray],
    query_bbox_idx: Sequence[int],
    gallery_bbox_idx: Sequence[int],
    query_actions: Sequence[int],
    gallery_actions: Sequence[int],
) -> dict[str, list[int]]:
    """Turn per-action score matrices into ``{query_bbox_idx: [gallery_bbox_idx, ...]}``.

    ``scores_by_action[a]`` is ``[n_q_a, n_g_a]`` with rows / columns in the
    order the action's queries / gallery crops appear in the inputs. Ties keep
    the gallery order (stable sort), as the plain ranking does. A query whose
    action has no gallery gets an empty ranking.
    """
    qb = np.asarray(query_bbox_idx, dtype=np.int64)
    gb = np.asarray(gallery_bbox_idx, dtype=np.int64)
    q_groups = group_positions_by_action(query_actions)
    g_groups = group_positions_by_action(gallery_actions)

    rankings: dict[str, list[int]] = {}
    for action, q_pos in q_groups.items():
        g_pos = g_groups.get(action)
        if g_pos is None or len(g_pos) == 0:
            for qp in q_pos:
                rankings[str(int(qb[qp]))] = []
            continue
        scores = np.asarray(scores_by_action[action])
        if scores.shape != (len(q_pos), len(g_pos)):
            raise ValueError(
                f"action {action}: scores have shape {scores.shape}, "
                f"expected {(len(q_pos), len(g_pos))}"
            )
        gallery_ids = gb[g_pos]
        for row, qp in enumerate(q_pos):
            order = np.argsort(-scores[row], kind="stable")
            rankings[str(int(qb[qp]))] = [int(b) for b in gallery_ids[order]]
    return rankings


def compute_reranked_rankings(
    query_feats,
    gallery_feats,
    query_bbox_idx: Sequence[int],
    gallery_bbox_idx: Sequence[int],
    query_actions: Sequence[int],
    gallery_actions: Sequence[int],
    method: RerankMethod = "k_reciprocal",
    *,
    k1: int = 4,
    k2: int = 2,
    lambda_value: float = 0.3,
    temperature: float = 0.1,
) -> dict[str, list[int]]:
    """Re-ranked counterpart of :func:`soccernet_reid.eval.ranking.compute_rankings`.

    Same inputs and the same output format, so the result goes straight into
    ``compute_metrics``. Each action is processed on its own.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown re-ranking method {method!r}; choose from {_METHODS}")
    qf = _unit_rows(query_feats)
    gf = _unit_rows(gallery_feats)
    if qf.shape[0] != len(query_bbox_idx) or qf.shape[0] != len(query_actions):
        raise ValueError("query_feats rows must match query_bbox_idx / query_actions")
    if gf.shape[0] != len(gallery_bbox_idx) or gf.shape[0] != len(gallery_actions):
        raise ValueError("gallery_feats rows must match gallery_bbox_idx / gallery_actions")

    q_groups = group_positions_by_action(query_actions)
    g_groups = group_positions_by_action(gallery_actions)
    scores_by_action: dict[int, np.ndarray] = {}
    for action, q_pos in q_groups.items():
        g_pos = g_groups.get(action)
        if g_pos is None or len(g_pos) == 0:
            continue
        scores_by_action[action] = rerank_action_scores(
            qf[q_pos], gf[g_pos], method,
            k1=k1, k2=k2, lambda_value=lambda_value, temperature=temperature,
        )
    return rankings_from_action_scores(
        scores_by_action, query_bbox_idx, gallery_bbox_idx, query_actions, gallery_actions
    )
