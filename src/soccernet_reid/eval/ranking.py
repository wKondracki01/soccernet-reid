"""Build rankings from embedding tensors (numpy or torch, accepts both).

The retrieval protocol:
    For each query, sort gallery items from the SAME action by decreasing similarity.
    Cross-action gallery items are never included in the ranking (official rule).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from soccernet_reid.eval.metrics import compute_metrics


def _to_numpy(x) -> np.ndarray:
    """Accept torch tensors or numpy arrays; return float32 numpy on CPU."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of x in-place style (returns new array)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def compute_rankings(
    query_feats,
    gallery_feats,
    query_bbox_idx: Sequence[int],
    gallery_bbox_idx: Sequence[int],
    query_actions: Sequence[int],
    gallery_actions: Sequence[int],
    distance: Literal["cosine", "euclidean"] = "cosine",
) -> dict[str, list[int]]:
    """Build per-query rankings restricted to same-action gallery items.

    Args:
        query_feats: [N_q, D] embeddings (numpy or torch).
        gallery_feats: [N_g, D] embeddings.
        query_bbox_idx, gallery_bbox_idx: bbox_idx values, lengths N_q and N_g.
        query_actions, gallery_actions: action_idx values, same lengths.
        distance: similarity metric for sorting. "cosine" L2-normalizes internally.

    Returns:
        {query_bbox_idx_str: [gallery_bbox_idx_int, ...]} — ordered by descending
        similarity (or ascending distance for euclidean).
    """
    qf = _to_numpy(query_feats)
    gf = _to_numpy(gallery_feats)
    qb = np.asarray(query_bbox_idx, dtype=np.int64)
    gb = np.asarray(gallery_bbox_idx, dtype=np.int64)
    qa = np.asarray(query_actions, dtype=np.int64)
    ga = np.asarray(gallery_actions, dtype=np.int64)

    if qf.shape[0] != len(qb) or qf.shape[0] != len(qa):
        raise ValueError(f"query_feats N={qf.shape[0]} mismatches metadata lengths {len(qb)},{len(qa)}")
    if gf.shape[0] != len(gb) or gf.shape[0] != len(ga):
        raise ValueError(f"gallery_feats N={gf.shape[0]} mismatches metadata lengths {len(gb)},{len(ga)}")
    if qf.shape[1] != gf.shape[1]:
        raise ValueError(f"feature dim mismatch: query D={qf.shape[1]} gallery D={gf.shape[1]}")

    if distance == "cosine":
        qf = _l2_normalize(qf)
        gf = _l2_normalize(gf)
    elif distance != "euclidean":
        raise ValueError(f"Unknown distance: {distance!r}")

    # Bucket gallery indices by action for O(N_q + N_g) lookups
    action_to_gallery_pos: dict[int, np.ndarray] = {}
    for action_idx in np.unique(ga):
        action_to_gallery_pos[int(action_idx)] = np.where(ga == action_idx)[0]

    rankings: dict[str, list[int]] = {}
    for q_pos in range(len(qb)):
        q_action = int(qa[q_pos])
        if q_action not in action_to_gallery_pos:
            # Shouldn't happen with the official dataset, but be defensive
            rankings[str(int(qb[q_pos]))] = []
            continue
        g_positions = action_to_gallery_pos[q_action]
        g_feats_sub = gf[g_positions]   # [M, D]
        q_vec = qf[q_pos]                # [D]

        # Suppress benign matmul warnings on subnormal/edge-case feature pairs.
        # Verified that features themselves are finite (no NaN/Inf) on both CPU and MPS;
        # the warnings come from numpy's BLAS reporting subnormal underflow which does
        # not affect ranking order. Parity with the official evaluator is exact.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if distance == "cosine":
                sims = g_feats_sub @ q_vec   # higher is better
                order = np.argsort(-sims, kind="stable")
            else:  # euclidean
                diff = g_feats_sub - q_vec[None, :]
                dists = np.einsum("ij,ij->i", diff, diff)  # squared euclidean
                order = np.argsort(dists, kind="stable")    # lower is better

        ranked_bbox_idx = gb[g_positions][order]
        rankings[str(int(qb[q_pos]))] = [int(b) for b in ranked_bbox_idx]

    return rankings


def evaluate_embeddings(
    query_feats,
    gallery_feats,
    query_meta: Mapping[str, Mapping[str, object]],
    gallery_meta: Mapping[str, Mapping[str, object]],
    distance: Literal["cosine", "euclidean"] = "cosine",
    ranks: Sequence[int] = (1, 5, 10),
    validate: bool = True,
) -> dict[str, float]:
    """End-to-end: features → rankings → metrics.

    `query_meta` / `gallery_meta` are bbox_info-style dicts. We extract bbox_idx and
    action_idx from them to match against the feature rows by position (so the order
    of features must match dict iteration order, which in Python ≥3.7 is insertion order).
    """
    q_items = list(query_meta.values())
    g_items = list(gallery_meta.values())
    rankings = compute_rankings(
        query_feats=query_feats,
        gallery_feats=gallery_feats,
        query_bbox_idx=[int(q["bbox_idx"]) for q in q_items],
        gallery_bbox_idx=[int(g["bbox_idx"]) for g in g_items],
        query_actions=[int(q["action_idx"]) for q in q_items],
        gallery_actions=[int(g["action_idx"]) for g in g_items],
        distance=distance,
    )
    return compute_metrics(rankings, query_meta, gallery_meta, ranks=ranks, validate=validate)
