"""scripts/visualize_retrieval.py: the rows it draws must agree with the evaluation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soccernet_reid.eval.ranking import compute_rankings


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "visualize_retrieval.py"
    spec = importlib.util.spec_from_file_location("visualize_retrieval_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toy():
    """Two actions. Query and gallery bbox indices overlap on purpose, as in the real data."""
    rows = []
    # action 0: queries of persons 1 and 2; gallery: person 1 twice, person 2 once, a distractor (9)
    rows += [("valid", "query", 0, 0, 1), ("valid", "query", 1, 0, 2)]
    rows += [("valid", "gallery", 0, 0, 1), ("valid", "gallery", 1, 0, 9), ("valid", "gallery", 2, 0, 1), ("valid", "gallery", 3, 0, 2)]
    # action 1: one query (person 1 again: a different class, labels are per action)
    rows += [("valid", "query", 2, 1, 1)]
    rows += [("valid", "gallery", 4, 1, 5), ("valid", "gallery", 5, 1, 1)]
    cat = pd.DataFrame(rows, columns=["split", "role", "bbox_idx", "action_idx", "person_uid"])
    cat["path"] = [f"/img/{r}_{b}.png" for r, b in zip(cat["role"], cat["bbox_idx"], strict=True)]

    def unit(*v):
        a = np.asarray(v, dtype=np.float32)
        return a / np.linalg.norm(a)

    emb = {
        "split": np.array("valid"),
        "query_bbox_idx": np.array([0, 1, 2]), "query_action_idx": np.array([0, 0, 1]),
        "gallery_bbox_idx": np.array([0, 1, 2, 3, 4, 5]), "gallery_action_idx": np.array([0, 0, 0, 0, 1, 1]),
        "query_feats": np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]),
        # for query 0 (action 0): g1 (distractor) closest, then g0 (correct), g2 (correct), g3
        "gallery_feats": np.stack([unit(1, 0.5, 0), unit(1, 0.1, 0), unit(1, 1, 0), unit(0.2, 1, 0),
                                   unit(0, 0, 1), unit(0, 1, 1)]),
    }
    return emb, cat


def test_rows_follow_the_evaluation_ranking(toy) -> None:
    script = _load_script()
    emb, cat = toy
    ranked = script.rank_queries(emb, cat, "valid")
    official = compute_rankings(
        emb["query_feats"], emb["gallery_feats"], [0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 0, 1], [0, 0, 0, 0, 1, 1],
        distance="cosine",
    )
    for row in ranked:
        assert [g["bbox_idx"] for g in row["gallery"]] == official[str(row["bbox_idx"])]
        d = [g["distance"] for g in row["gallery"]]
        assert d == sorted(d)                              # nearest first
        assert all(-1e-6 <= x <= 2 + 1e-6 for x in d)


def test_correct_flags_ap_and_first_correct(toy) -> None:
    script = _load_script()
    emb, cat = toy
    q0, q1, q2 = script.rank_queries(emb, cat, "valid")
    # query 0: order g1 (wrong), g0, g2 (both correct), g3 (wrong)
    assert [(g["bbox_idx"], g["correct"]) for g in q0["gallery"]] == [(1, False), (0, True), (2, True), (3, False)]
    assert q0["first_correct"] == 2
    assert q0["ap"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert q0["path"] == "/img/query_0.png" and q0["gallery"][0]["path"] == "/img/gallery_1.png"
    # query 1 (person 2): its only match g3 is nearest
    assert q1["first_correct"] == 1 and q1["ap"] == pytest.approx(1.0)
    # query 2 is ranked only against the gallery of action 1; g4 is identical but another person
    assert [g["bbox_idx"] for g in q2["gallery"]] == [4, 5]
    assert q2["gallery"][0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert q2["first_correct"] == 2 and q2["ap"] == pytest.approx(0.5)


def test_selection_modes(toy) -> None:
    script = _load_script()
    emb, cat = toy
    ranked = script.rank_queries(emb, cat, "valid")
    assert [r["bbox_idx"] for r in script.select_queries(ranked, "successes", 5, seed=0)] == [1]
    assert sorted(r["bbox_idx"] for r in script.select_queries(ranked, "failures", 5, seed=0)) == [0, 2]
    mixed = script.select_queries(ranked, "mixed", 2, seed=0)
    assert [r["first_correct"] == 1 for r in mixed] == [True, False]
    a = [r["bbox_idx"] for r in script.select_queries(ranked, "random", 2, seed=7)]
    assert a == [r["bbox_idx"] for r in script.select_queries(ranked, "random", 2, seed=7)]
    with pytest.raises(ValueError, match="Unknown selection mode"):
        script.select_queries(ranked, "best", 2, seed=0)


def test_draw_figure_writes_an_image(toy, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image

    script = _load_script()
    emb, cat = toy
    for p in cat["path"]:
        f = tmp_path / Path(p).name
        Image.fromarray(np.random.default_rng(0).integers(0, 255, size=(60, 30, 3), dtype=np.uint8)).save(f)
    cat = cat.assign(path=[str(tmp_path / Path(p).name) for p in cat["path"]])
    ranked = script.rank_queries(emb, cat, "valid")
    out = tmp_path / "fig.png"
    script.draw_figure(ranked, top_k=3, out_path=out, title="toy")   # action 1 has only 2 gallery crops
    assert out.stat().st_size > 1000


def test_pick_queries_keeps_the_given_order(toy) -> None:
    script = _load_script()
    emb, catalog = toy
    ranked = script.rank_queries(emb, catalog, "valid")
    assert [r["bbox_idx"] for r in script.pick_queries(ranked, [2, 0])] == [2, 0]
    with pytest.raises(ValueError, match="No such query"):
        script.pick_queries(ranked, [0, 999])


def test_pair_rows_puts_the_second_model_under_each_query(toy, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image

    script = _load_script()
    emb, catalog = toy
    for p in catalog["path"]:
        f = tmp_path / Path(p).name
        Image.fromarray(np.random.default_rng(0).integers(0, 255, size=(60, 30, 3), dtype=np.uint8)).save(f)
    catalog = catalog.assign(path=[str(tmp_path / Path(p).name) for p in catalog["path"]])
    ranked = script.rank_queries(emb, catalog, "valid")
    other = dict(emb)
    other["gallery_feats"] = emb["gallery_feats"][::-1].copy()   # a different model: other rankings
    ranked_other = script.rank_queries(other, catalog, "valid")

    rows = script.pair_rows(ranked[:2], ranked_other, ("final", "baseline"))

    assert [r["label"] for r in rows] == ["final", "baseline", "final", "baseline"]
    assert [r["bbox_idx"] for r in rows] == [ranked[0]["bbox_idx"]] * 2 + [ranked[1]["bbox_idx"]] * 2
    assert rows[1]["gallery"] == {r["bbox_idx"]: r for r in ranked_other}[ranked[0]["bbox_idx"]]["gallery"]
    out = tmp_path / "pair.png"
    script.draw_figure(rows, top_k=3, out_path=out, title=None, lang="pl")
    assert out.stat().st_size > 0



def test_select_differing_takes_queries_where_one_model_is_right(toy) -> None:
    script = _load_script()
    emb, catalog = toy
    ranked = script.rank_queries(emb, catalog, "valid")
    # a second model that agrees on query 1 and flips the outcome of queries 0 and 2
    other = [dict(r) for r in ranked]
    for r in other:
        if r["bbox_idx"] in (0, 2):
            r["first_correct"] = 1 if r["first_correct"] != 1 else 2
    picked = script.select_differing(ranked, other, 5, seed=0)
    assert sorted(r["bbox_idx"] for r in picked) == [0, 2]
    assert script.select_differing(ranked, ranked, 5, seed=0) == []
