"""scripts/make_thesis_figures.py: the numbers that go onto the figures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "make_thesis_figures.py"
    spec = importlib.util.spec_from_file_location("make_thesis_figures_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(best: float, curve: list[float] | None = None) -> dict:
    curve = curve or [best]
    evals = [[5 * (i + 1), v, 0.0, 0.0, 0.0] for i, v in enumerate(curve)]
    return {"evals": evals, "best": [evals[-1][0], best, 0.0, 0.0, 0.0]}


@pytest.fixture
def runs() -> dict[str, dict]:
    r = {name: _run(v) for name, v in {
        "G1_PK_BH": 0.68, "G1_PK_SA_BH": 0.75, "G3_AUG_MED": 0.79, "G4_R34": 0.81,
        "G5_STRONG": 0.812, "G5_STRONG_S1": 0.813, "G5_STRONG_S2": 0.810,
        "G6_D64": 0.807, "G6_D128": 0.808, "G6_D256": 0.812, "G6_D1024": 0.809,
    }.items()}
    r["G4_EB1"] = _run(0.78, [0.70, 0.78, 0.77])
    return r


def test_decimal_comma() -> None:
    script = _load_script()
    assert script.pl(0.8116) == "0,8116"
    assert script.pl(6.71, 1) == "6,7"


def test_reviews_are_merged_and_a_later_file_wins(tmp_path) -> None:
    script = _load_script()
    (tmp_path / "REVIEW_axis1.json").write_text(json.dumps({"A": _run(0.5), "B": _run(0.6)}))
    (tmp_path / "REVIEW_final.json").write_text(json.dumps({"B": _run(0.7)}))
    merged = script.load_reviews(tmp_path)
    assert script.best_map(merged["A"]) == 0.5 and script.best_map(merged["B"]) == 0.7


def test_curve_returns_epochs_and_map(runs) -> None:
    assert _load_script().curve(runs["G4_EB1"]) == ([5, 10, 15], [0.70, 0.78, 0.77])


def test_ladder_ends_with_the_mean_of_the_final_seeds(runs, tmp_path) -> None:
    script = _load_script()
    steps = script.ladder_steps(runs, tmp_path)          # no re-ranking files yet
    assert [v for _, v in steps[:4]] == [0.68, 0.75, 0.79, 0.81]
    assert steps[-1][1] == pytest.approx((0.812 + 0.813 + 0.810) / 3)
    assert len(steps) == 5


def test_ladder_adds_reranking_only_when_all_three_seeds_have_it(runs, tmp_path) -> None:
    script = _load_script()
    for name, value in zip(script.FINAL_SEEDS, (0.833, 0.832, 0.831), strict=True):
        steps = script.ladder_steps(runs, tmp_path)
        assert len(steps) == 5
        (tmp_path / name).mkdir()
        (tmp_path / name / "rerank_valid.json").write_text(
            json.dumps({"best": {"both": {"metrics": {"mAP": value}}}}))
    steps = script.ladder_steps(runs, tmp_path)
    assert len(steps) == 6 and steps[-1][1] == pytest.approx(0.832)


def test_learning_curves_are_saved_as_three_separate_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    script = _load_script()
    names = ("G1_PK_SA_BH", "G3_AUG_MED", "G3_AUG_STRONG", "G2_CE", "G2_CE_WD", *script.BASE_SEEDS, *script.FINAL_SEEDS)
    runs = {name: _run(0.8, [0.70, 0.75, 0.78, 0.80]) for name in names}
    script.draw_curves(runs, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{name}.{ext}" for name in script.CURVE_FIGURES for ext in ("pdf", "png"))
    assert len(script.CURVE_FIGURES) == 3


def test_dimension_points_are_sorted_and_512_carries_the_seed_range(runs) -> None:
    pts = _load_script().dimension_points(runs)
    assert [p[0] for p in pts] == [64, 128, 256, 512, 1024]
    d512 = pts[3]
    assert d512[1] == pytest.approx((0.812 + 0.813 + 0.810) / 3) and (d512[2], d512[3]) == (0.810, 0.813)
    assert pts[0][1:] == (0.807, 0.807, 0.807)


def test_cost_points_join_speed_with_the_backbone_axis(runs) -> None:
    script = _load_script()
    speed = {"results": [
        {"backbone": "R34", "inference_ms_per_image": 0.5, "parameters": 21_500_000},
        {"backbone": "EB1", "inference_ms_per_image": 0.9, "parameters": 7_200_000},
    ]}
    pts = script.cost_points(runs, speed)
    assert [(p["label"], p["ms"], p["mAP"]) for p in pts] == [("ResNet-34", 0.5, 0.81), ("EfficientNet-B1", 0.9, 0.78)]


def test_thesis_name_of_the_colour_preset() -> None:
    assert _load_script().AUGMENT_LABELS["aug-bot"] == "AUG-COLOR"


def test_cmc_counts_queries_answered_within_k_results() -> None:
    import numpy as np
    import pandas as pd

    script = _load_script()
    rows = [("valid", "query", 0, 0, 1), ("valid", "query", 1, 0, 2), ("valid", "query", 2, 1, 1)]
    # action 0: gallery crops of persons 1, 9 (distractor), 1, 2;  action 1: persons 5 and 1
    rows += [("valid", "gallery", 0, 0, 1), ("valid", "gallery", 1, 0, 9), ("valid", "gallery", 2, 0, 1),
             ("valid", "gallery", 3, 0, 2), ("valid", "gallery", 4, 1, 5), ("valid", "gallery", 5, 1, 1)]
    catalog = pd.DataFrame(rows, columns=["split", "role", "bbox_idx", "action_idx", "person_uid"])

    def unit(*v):
        a = np.asarray(v, dtype=np.float32)
        return a / np.linalg.norm(a)

    emb = {
        "query_bbox_idx": np.array([0, 1, 2]), "query_action_idx": np.array([0, 0, 1]),
        "gallery_bbox_idx": np.array([0, 1, 2, 3, 4, 5]), "gallery_action_idx": np.array([0, 0, 0, 0, 1, 1]),
        # query 0: the distractor is closest, a correct crop second;  query 1: correct first;
        # query 2: the wrong person first, the correct one second
        "query_feats": np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]),
        "gallery_feats": np.stack([unit(1, 0.5, 0), unit(1, 0.1, 0), unit(1, 1, 0), unit(0.2, 1, 0),
                                   unit(0, 0.2, 1), unit(0, 1, 1)]),
    }
    cmc = script.cmc_curve(emb, catalog, "valid", max_rank=3)
    assert cmc == pytest.approx([1 / 3, 1.0, 1.0])
