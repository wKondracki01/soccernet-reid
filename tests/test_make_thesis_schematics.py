"""scripts/make_thesis_schematics.py: which crops the explanatory figures show, and at what size they are drawn."""
from __future__ import annotations

import importlib.util
import re
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


def _printed_width(pdf: Path) -> float:
    """Width of the page of a one-page PDF, in PostScript points."""
    box = re.search(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", pdf.read_bytes())
    return float(box.group(3)) - float(box.group(1))


def test_gallery_crops_carry_the_number_of_their_query(tmp_path) -> None:
    script = _load_script()
    queries, gallery = script.action_example(_valid_action(tmp_path), action=6)
    assert [Path(p).name for p in queries] == ["0.png", "1.png"]
    assert [(Path(p).name, n) for p, n in gallery] == [("2.png", 2), ("3.png", None), ("4.png", 1)]


def test_action_figure_is_drawn_at_its_printed_width(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    script = _load_script()
    monkeypatch.setattr(script, "ACTION", 6)
    out = tmp_path / "out"
    script.draw_action(_valid_action(tmp_path), out)
    assert sorted(p.name for p in out.iterdir()) == ["zapytania-galeria-akcja.pdf", "zapytania-galeria-akcja.png"]
    assert _printed_width(out / "zapytania-galeria-akcja.pdf") == pytest.approx(0.8 * script.TEXT_WIDTH_PT, abs=0.1)


@pytest.mark.parametrize(("draw", "name"), [
    ("draw_preprocessing", "przetwarzanie-wycinkow"),
    ("draw_model", "metodyka-model"),
    ("draw_samplers", "metodyka-partie"),
    ("draw_query_normalization", "normalizacja_po_zapytaniach"),
])
def test_diagrams_are_drawn_at_the_text_width(tmp_path, draw: str, name: str) -> None:
    """Drawn at the printed size, a 9-point label prints at 9 points (the normalization figure
    also checks its shares, scores and orders against rerank.py while it is drawn)."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    script = _load_script()
    getattr(script, draw)(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{name}.pdf", f"{name}.png"]
    assert _printed_width(tmp_path / f"{name}.pdf") == pytest.approx(script.TEXT_WIDTH_PT, abs=0.1)


def test_text_width_is_that_of_the_thesis_template() -> None:
    """154 mm: A4 (210 mm) less two 25 mm margins and the 6 mm binding offset of main.tex."""
    width = _load_script().TEXT_WIDTH_PT
    assert width == pytest.approx((210 - 2 * 25 - 6) / 25.4 * 72)
