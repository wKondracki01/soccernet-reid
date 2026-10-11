"""scripts/make_thesis_schematics.py: which crops the explanatory figures show."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "make_thesis_schematics.py"
    spec = importlib.util.spec_from_file_location("make_thesis_schematics_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_action(tmp_path: Path) -> pd.DataFrame:
    """One validation action: two queries; the gallery holds a match for each and one distractor."""
    from PIL import Image

    rows = []
    for i, (role, person) in enumerate((("query", 7), ("query", 3), ("gallery", 3), ("gallery", 99), ("gallery", 7))):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (20, 40), (10 * i, 120, 60)).save(path)
        rows.append(("valid", role, i, 6, person, str(path)))
    return pd.DataFrame(rows, columns=["split", "role", "bbox_idx", "action_idx", "person_uid", "path"])


def test_gallery_crops_carry_the_number_of_their_query(tmp_path) -> None:
    script = _load_script()
    queries, gallery = script.action_example(_valid_action(tmp_path), action=6)
    assert [Path(p).name for p in queries] == ["0.png", "1.png"]
    assert [(Path(p).name, n) for p, n in gallery] == [("2.png", 2), ("3.png", None), ("4.png", 1)]


def test_action_and_model_figures_are_written(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    script = _load_script()
    monkeypatch.setattr(script, "ACTION", 6)
    catalog = _valid_action(tmp_path)
    out = tmp_path / "out"
    script.draw_action(catalog, out)
    script.draw_model(catalog, out)
    assert sorted(p.name for p in out.iterdir()) == ["przykladowa_akcja.pdf", "przykladowa_akcja.png",
                                                     "schemat_modelu.pdf", "schemat_modelu.png"]


def test_sampler_batches_follow_the_two_samplers() -> None:
    script = _load_script()
    rows = []
    for action in range(4):                           # four actions, eight players each, two crops per player
        for person in range(8):
            for crop in range(2):
                rows.append(("train", action, person, "Player_team_left", f"a{action}/p{person}/{crop}.png"))
    catalog = pd.DataFrame(rows, columns=["split", "action_idx", "person_uid", "clazz", "path"])
    pk, pk_sa = script.sampler_batches(catalog)
    assert len(pk) == 32 and len(pk_sa) == 16
    people = lambda paths: [p.rsplit("/", 1)[0] for p in paths]                      # noqa: E731
    assert all(a == b for a, b in zip(people(pk)[0::2], people(pk)[1::2], strict=True))   # crops come in pairs
    assert len(set(people(pk))) == 16 and len(set(people(pk_sa))) == 8
    assert len({p.split("/")[0] for p in pk_sa}) == 1                                # PK-SA: one action
    assert len({p.split("/")[0] for p in pk}) > 1                                    # PK: several actions
