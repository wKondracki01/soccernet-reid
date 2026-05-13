"""Bridges to the official SoccerNet evaluator.

The official `SoccerNet.Evaluation.ReIdentification.evaluate` reads two JSON files:
groundtruth (bbox_info style) and rankings ({query_idx_str: [gallery_idx_int, ...]}).
This module produces those structures from our DataFrame catalog and wraps the
official call for smoke tests.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def catalog_to_groundtruth_dict(
    df: pd.DataFrame,
    split: str = "valid",
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the official groundtruth dict from our catalog DataFrame.

    Returns a dict shaped like the official `bbox_info.json`:
        {"query": {bbox_idx_str: {bbox_idx, action_idx, person_uid, ...}, ...},
         "gallery": {...}}
    """
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"No rows for split={split!r} in catalog")

    out: dict[str, dict[str, dict[str, Any]]] = {"query": {}, "gallery": {}}
    for role in ("query", "gallery"):
        role_df = sub[sub["role"] == role]
        for _, row in role_df.iterrows():
            bbox_idx = int(row["bbox_idx"])
            out[role][str(bbox_idx)] = {
                "bbox_idx": bbox_idx,
                "action_idx": int(row["action_idx"]),
                "person_uid": int(row["person_uid"]),
                "frame_idx": int(row["frame_idx"]),
                "clazz": str(row["clazz"]),
                "id": str(row["id_"]),
                "UAI": str(row["uai"]),
                "relative_path": str(row["relative_path"]),
                "height": int(row["height"]),
                "width": int(row["width"]),
            }
    return out


def rankings_to_official_dict(
    rankings: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
    """Convert our rankings (already in the right shape) to a JSON-serializable dict."""
    return {q: [int(g) for g in gs] for q, gs in rankings.items()}


def run_official_evaluator(
    rankings: Mapping[str, Sequence[int]],
    groundtruth: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float]:
    """Invoke the official `SoccerNet.Evaluation.ReIdentification.evaluate` on our data.

    Writes both inputs to temporary JSON files, runs the evaluator, returns its result.
    """
    from SoccerNet.Evaluation.ReIdentification import evaluate  # imported lazily

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gt_path = tmp_path / "groundtruth.json"
        rk_path = tmp_path / "rankings.json"
        with gt_path.open("w") as f:
            json.dump(groundtruth, f)
        with rk_path.open("w") as f:
            json.dump(rankings_to_official_dict(rankings), f)
        return evaluate(str(gt_path), str(rk_path))
