"""mAP and Rank-k computation, 1:1 with the official SoccerNet evaluator.

Algorithm taken directly from `SoccerNet.Evaluation.ReIdentification.evaluate`:

    matches = [g_pid == q_pid for g_pid in gallery_ranking_pid]
    raw_cmc = np.array(matches)
    if not np.any(raw_cmc): ERROR   # query has no positive in gallery

    # AP (no interpolation, Wikipedia-style):
    num_rel = raw_cmc.sum()
    tmp = raw_cmc.cumsum() / (np.arange(len(raw_cmc)) + 1)
    AP = (tmp * raw_cmc).sum() / num_rel

    # Rank-k CMC: 1 if a positive lies in top-k, else 0 (per query)
    cmc = raw_cmc.cumsum()
    cmc[cmc > 1] = 1   # clip — once we hit a match, all subsequent ranks are 1
    rank_k = cmc[k - 1]

The official evaluator hardcodes max_rank=1; we extend to arbitrary k.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Number of decimal places to which our metrics must match the official.
AP_TIES_TOLERANCE = 1e-6


def validate_rankings_complete(
    rankings: Mapping[str, Sequence[int]],
    query_meta: Mapping[str, Mapping[str, object]],
    gallery_meta: Mapping[str, Mapping[str, object]],
) -> None:
    """Verify that rankings satisfy the official-evaluator preconditions:

    1. Every query has a ranking.
    2. Each ranking lists exactly the gallery bbox_idx values from the same action,
       no duplicates, no cross-action items.
    3. Each query identity has at least one positive in its gallery (otherwise the
       official evaluator raises ValueError).

    Raises ValueError with a descriptive message on any violation. Returns None
    on success.
    """
    # Pre-build action_idx -> set of gallery bbox_idx
    action_to_gallery_idx: dict[int, set[int]] = {}
    gallery_pid: dict[int, int] = {}
    for g_str, g in gallery_meta.items():
        action_idx = int(g["action_idx"])
        bbox_idx = int(g["bbox_idx"])
        action_to_gallery_idx.setdefault(action_idx, set()).add(bbox_idx)
        gallery_pid[bbox_idx] = int(g["person_uid"])

    for q_str, q in query_meta.items():
        if q_str not in rankings:
            raise ValueError(f"No ranking provided for query {q_str!r}")
        q_action = int(q["action_idx"])
        q_pid = int(q["person_uid"])

        expected = action_to_gallery_idx.get(q_action, set())
        provided = list(rankings[q_str])
        if len(provided) != len(set(provided)):
            raise ValueError(f"Ranking for query {q_str!r} contains duplicate gallery indices")
        provided_set = set(provided)
        missing = expected - provided_set
        extra = provided_set - expected
        if missing:
            raise ValueError(
                f"Ranking for query {q_str!r} missing {len(missing)} gallery items "
                f"from same action {q_action}; first 5: {sorted(missing)[:5]}"
            )
        if extra:
            raise ValueError(
                f"Ranking for query {q_str!r} contains {len(extra)} gallery items "
                f"NOT from action {q_action}; first 5: {sorted(extra)[:5]}"
            )

        # Positive existence: at least one gallery in same action must be the same person
        positives = sum(1 for g_idx in expected if gallery_pid[g_idx] == q_pid)
        if positives == 0:
            raise ValueError(
                f"Query {q_str!r} (person_uid={q_pid}, action={q_action}) has no "
                f"positive in gallery (this means the official evaluator would fail)"
            )


def compute_metrics(
    rankings: Mapping[str, Sequence[int]],
    query_meta: Mapping[str, Mapping[str, object]],
    gallery_meta: Mapping[str, Mapping[str, object]],
    ranks: Sequence[int] = (1, 5, 10),
    validate: bool = True,
) -> dict[str, float]:
    """Compute mAP and Rank-k metrics matching the official SoccerNet evaluator.

    Args:
        rankings: {query_bbox_idx_str: [gallery_bbox_idx, ...]} — full ordered ranking
            of all gallery items from the same action as the query.
        query_meta, gallery_meta: bbox_info-style dicts {bbox_idx_str: {person_uid, action_idx, ...}}.
        ranks: which Rank-k values to compute. Default (1, 5, 10).
        validate: if True, run :func:`validate_rankings_complete` first.

    Returns:
        Dict with keys "mAP" and "rank-{k}" for each k in `ranks`. Values in [0, 1].
    """
    if validate:
        validate_rankings_complete(rankings, query_meta, gallery_meta)

    # Map gallery bbox_idx (int) -> person_uid (int) for O(1) lookup
    gallery_pid: dict[int, int] = {
        int(g["bbox_idx"]): int(g["person_uid"]) for g in gallery_meta.values()
    }

    max_rank = max(ranks)
    aps: list[float] = []
    cmc_per_query: list[np.ndarray] = []

    for q_str in query_meta.keys():
        q_pid = int(query_meta[q_str]["person_uid"])
        ranking = list(rankings[q_str])

        # Binary match vector along the ranking
        raw_cmc = np.fromiter(
            (gallery_pid[g_idx] == q_pid for g_idx in ranking),
            dtype=np.int64,
            count=len(ranking),
        )
        if not raw_cmc.any():
            # validate_rankings_complete already catches this, but guard anyway
            raise ValueError(f"Query {q_str!r} has no positive in its ranked gallery")

        # AP — exactly the official formula
        num_rel = int(raw_cmc.sum())
        cumsum = raw_cmc.cumsum()
        precision_at_k = cumsum / np.arange(1, len(raw_cmc) + 1)
        ap = float((precision_at_k * raw_cmc).sum() / num_rel)
        aps.append(ap)

        # Rank-k: cumulative match, clipped to [0,1]
        cmc = cumsum.copy()
        cmc[cmc > 1] = 1
        # Pad with last value if the ranking is shorter than max_rank (rare for full rankings,
        # but guard for edge cases). Once cmc reaches 1 it stays 1.
        if len(cmc) < max_rank:
            pad = np.full(max_rank - len(cmc), cmc[-1], dtype=cmc.dtype)
            cmc = np.concatenate([cmc, pad])
        cmc_per_query.append(cmc[:max_rank].astype(np.float32))

    mean_cmc = np.stack(cmc_per_query, axis=0).mean(axis=0)

    out: dict[str, float] = {"mAP": float(np.mean(aps))}
    for k in ranks:
        out[f"rank-{k}"] = float(mean_cmc[k - 1])
    return out
