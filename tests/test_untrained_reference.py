"""scripts/untrained_reference.py: the exact expectation of a random ranking."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "untrained_reference.py"
    spec = importlib.util.spec_from_file_location("untrained_reference_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _brute_force(gallery_people: list[int], query_person: int, k: int) -> tuple[float, float]:
    """Mean AP and Rank-k over every ordering of a small gallery."""
    aps, hits = [], []
    for order in itertools.permutations(gallery_people):
        match = np.array([p == query_person for p in order], dtype=float)
        precision = match.cumsum() / np.arange(1, len(match) + 1)
        aps.append((precision * match).sum() / match.sum())
        hits.append(float(match[:k].any()))
    return float(np.mean(aps)), float(np.mean(hits))


@pytest.mark.parametrize("gallery_people", [[1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5], [1, 1, 1, 2, 2, 3, 4]])
def test_expected_random_scores_match_every_ordering(gallery_people: list[int]) -> None:
    script = _load_script()
    gallery = pd.DataFrame({"action_idx": 0, "person_uid": gallery_people})
    query = pd.DataFrame({"action_idx": [0], "person_uid": [1]})
    expected = script.expected_random_scores(query, gallery, ranks=(1, 2, 5))
    for k in (1, 2, 5):
        ap, hit = _brute_force(gallery_people, 1, k)
        assert expected["mAP"] == pytest.approx(ap)
        assert expected[f"rank-{k}"] == pytest.approx(hit)
