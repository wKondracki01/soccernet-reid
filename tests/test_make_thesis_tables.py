"""scripts/make_thesis_tables.py: formatting, the consistency check of a table, reading of results."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "make_thesis_tables.py"
    spec = importlib.util.spec_from_file_location("make_thesis_tables_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


CONFIG = """backbone:
  name: R34
  pretrained: true
loss:
  name: tri
  margin: 0.3
sampler:
  name: pk_sa
  P: 8
  K: 2
transform:
  level: aug-bot
seed: 0
embedding_dim: 512
num_epochs: 60
optimizer:
  weight_decay: 0.0
eval:
  every_n_epochs: 5
"""


def _write_run(runs_dir: Path, name: str, config: str = CONFIG, evals: tuple[tuple[float, float], ...] = ()) -> None:
    (runs_dir / name).mkdir()
    lines = ["=" * 70, config, "=" * 70, "Device: cpu"]
    lines += [f"  Eval: mAP={m:.4f} R-1={r:.4f} time=1.0s" for m, r in evals]
    (runs_dir / name / "stdout.log").write_text("\n".join(lines))


def test_numbers_are_written_the_polish_way() -> None:
    script = _load_script()
    assert script.num(248234) == "248\\,234" and script.num(512) == "512"
    assert script.percent(0.548) == "54,8\\%"
    assert script.points(0.0085) == "+0,85" and script.points(-0.0514) == "$-$5,14"
    assert script.points(-0.00001) == "0,00"          # no sign on a difference that rounds to zero
    assert script.short(1.0) == "1" and script.short(0.25) == "0{,}25" and script.short(80) == "80"
    assert script.tt("G1_PK_BH") == "\\texttt{G1\\_PK\\_BH}"


def test_the_best_value_of_each_column_is_bold() -> None:
    cells = _load_script().metric_cells([(0.75, 0.60), (0.81, 0.55)])
    assert cells == [["0,7500", "\\textbf{0,6000}"], ["\\textbf{0,8100}", "0,5500"]]


def test_column_count_understands_paragraph_columns() -> None:
    script = _load_script()
    assert script.column_count("lrrr") == 4
    assert script.column_count("l" + script.par(0.3) + "r@{}") == 3
    with pytest.raises(ValueError):
        script.column_count("l|r")


def test_render_refuses_a_row_that_does_not_fill_the_columns() -> None:
    script = _load_script()
    good = script.Table("t", "Podpis", "lrr", [["a", "b", "c"]], [["x", "1", "2"], None, ["\\multicolumn{3}{l}{razem}"]])
    text = script.render(good)
    assert "\\caption{Podpis}" in text and "\\label{tab:t}" in text and text.count("\\hline") == 4
    assert text.index("\\caption") < text.index("\\begin{tabular}")       # the caption goes above the table
    with pytest.raises(ValueError):
        script.render(script.Table("t", "Podpis", "lrr", [["a", "b", "c"]], [["x", "1"]]))


def test_a_table_with_a_width_is_stretched_between_its_columns() -> None:
    script = _load_script()
    text = script.render(script.Table("t", "Podpis", "lr", [["a", "b"]], [["x", "1"]], width="0.97\\linewidth"))
    assert "\\begin{tabular*}{0.97\\linewidth}{@{\\hspace{\\tabcolsep}\\extracolsep{\\fill}}lr}" in text
    assert "\\end{tabular*}" in text and "\\begin{tabular}" not in text


def test_long_table_repeats_its_header() -> None:
    script = _load_script()
    text = script.render(script.Table("t", "Podpis", "lr", [["a", "b"]], [["x", "1"]], long=True))
    assert "\\begin{longtable}{lr}" in text and "\\endfirsthead" in text and text.count("a & b \\\\") == 2


def test_run_is_described_by_what_distinguishes_it(tmp_path) -> None:
    script = _load_script()
    _write_run(tmp_path, "A")
    config = script.run_config(tmp_path, "A")
    assert script.selection_code(config) == "PK-SA-BH"
    assert script.describe(config) == "ResNet-34, trójkowa, PK-SA-BH, AUG-COLOR"
    config["loss"] = {"name": "circle", "m": 0.4, "gamma": 80}
    config["backbone"]["pretrained"] = False
    config["seed"] = 2
    assert script.describe(config) == "ResNet-34, bez wag ImageNet, Circle ($m = 0{,}4$, $\\gamma = 80$), PK-SA, AUG-COLOR, ziarno 2"
    config["loss"] = {"name": "ce", "label_smoothing": 0.1}
    config["sampler"] = {"name": "random", "batch_size": 32}
    config["optimizer"]["weight_decay"] = 0.0005
    assert "entropia krzyżowa, sampler losowy, regularyzacja wag" in script.describe(config)


def test_scores_come_from_the_re_evaluation_and_fall_back_to_the_log(tmp_path) -> None:
    script = _load_script()
    _write_run(tmp_path, "A", evals=((0.41, 0.30), (0.45, 0.32), (0.23, 0.10)))
    assert script.best_scores(tmp_path, "A") == (10, (0.45, 0.32, None, None))    # stopped run: only the log
    assert script.logged_evals(tmp_path, "A")[-1][0] == 15
    (tmp_path / "A" / "eval_valid.json").write_text(
        json.dumps({"epoch": 59, "mAP": 0.8, "rank-1": 0.7, "rank-5": 0.9, "rank-10": 0.95}))
    assert script.best_scores(tmp_path, "A") == (60, (0.8, 0.7, 0.9, 0.95))        # epochs are counted from 1


def test_difference_is_found_whichever_way_round_it_was_saved() -> None:
    script = _load_script()
    saved = {"A - B": {"d_map": 0.02, "d_map_ci": [0.01, 0.03]}}
    assert script.difference(saved, "A", "B") == (0.02, 0.01, 0.03)
    assert script.difference(saved, "B", "A") == (-0.02, -0.03, -0.01)
    assert script.interval(-0.0042, 0.0043) == "[$-$0,42; +0,43]"


def test_dataset_tables_count_classes_and_sampler_pools() -> None:
    pd = pytest.importorskip("pandas")
    script = _load_script()
    rows = []
    for action in range(2):                       # action 0: eight players with two crops; action 1: one with one
        for person in range(8 if action == 0 else 1):
            for _ in range(2 if action == 0 else 1):
                rows.append(("train", None, action, person, "Player_team_left", "liga", "2015", "mecz"))
    rows.append(("train", None, 0, 99, "Main_referee", "liga", "2015", "mecz"))          # filtered out
    rows += [("valid", "query", 5, 1, "Player_team_left", "liga", "2015", "inny"),
             ("valid", "gallery", 5, 1, "Player_team_left", "liga", "2015", "inny")]
    catalog = pd.DataFrame(rows, columns=["split", "role", "action_idx", "person_uid", "clazz", "championship",
                                          "season", "game"])
    splits, per_class, pools = script.dataset_tables(catalog)
    assert splits.rows[0] == ["treningowy", "18", "--", "--", "2", "1"]
    assert splits.rows[1] == ["walidacyjny", "2", "1", "1", "1", "1"]
    assert per_class.rows[0] == ["1", "1", "11,1\\%"] and per_class.rows[1] == ["2", "8", "88,9\\%"]
    assert pools.rows[1][1:3] == ["8", "16"] and pools.rows[2][1:3] == ["8", "16"]
    for table in (splits, per_class, pools):
        script.render(table)


def test_method_tables_are_consistent() -> None:
    script = _load_script()
    speed = {"results": [{"backbone": code, "parameters": 1_000_000 * (i + 1)} for i, code in enumerate(script.BACKBONES)]}
    tables = script.method_tables(speed)
    assert [t.name for t in tables] == ["warianty-doboru", "funkcje-straty", "zestawy-augmentacji", "ekstraktory-cech",
                                        "plan-etapow", "ustawienia-treningu"]
    for table in tables:
        script.render(table)
    plan = tables[4]
    names = [name for _, group in script.STAGES for name in group]
    assert len(names) == len(set(names)) == 47
    assert len(plan.rows) == len(script.STAGES)


def test_backbone_results_hold_only_the_scores_sorted_by_map(tmp_path) -> None:
    script = _load_script()
    expected = []
    for i, (code, run) in enumerate(script.BACKBONE_RUNS.items(), start=1):
        _write_run(tmp_path, run)
        values = (0.5 + i / 100, 0.4 + i / 100, 0.8 + i / 100, 0.9 + i / 100)
        (tmp_path / run / "eval_valid.json").write_text(json.dumps({"epoch": i, **dict(zip(script.METRICS, values))}))
        expected.append((script.BACKBONES[code][0], values))
    table = script.backbone_results_table(tmp_path)
    for row, (name, values) in zip(table.rows, reversed(expected), strict=True):
        assert row[0] == name
        assert [cell.removeprefix("\\textbf{").removesuffix("}") for cell in row[1:]] == [
            script.pl(value) for value in values]
    assert all(cell.startswith("\\textbf{") for cell in table.rows[0][1:])
    assert not any("\\textbf{" in cell for row in table.rows[1:] for cell in row[1:])
    # first column named as in chapter 4; the parameters and the throughput are in the cost table, so this
    # one has the scores only; no note, as the text explains "+ fc"; laid out as table 5.1, with the scores
    # close together at the right, and the name column of a quarter of the line the author chose for it
    assert table.header == [["Ekstraktor", *script.METRIC_HEADER]]
    assert "przepustowo" not in table.caption and not table.notes
    assert table.spec == script.par(0.25) + "rrrr" and table.width == "0.97\\linewidth" and table.tabcolsep is None
    script.render(table)


def test_computational_cost_keeps_family_order_and_code_mapping() -> None:
    script = _load_script()
    speed_rows = [{"backbone": code, "parameters": (i + 1) * 1_000_000, "inference_images_per_s": 1000 * i + 0.4,
                   "latency_ms_single_image": i + 0.125, "train_steps_per_s": i + 10.25}
                  for i, code in enumerate(script.BACKBONES)]
    table = script.computational_cost_table({"results": list(reversed(speed_rows))})
    # no value is bold: the author chose so, as what is best differs between the columns
    assert table.rows == [[name, f"{i + 1},0", script.num(1000 * i), script.pl(i + 0.125, 2), script.pl(i + 10.25, 1)]
                          for i, (name, _, _) in enumerate(script.BACKBONES.values())]
    # every measure of cost in one table, the parameters and the throughput included
    assert table.header == [["Ekstraktor", "Parametry [mln]", "Wycinki/s", "Opóźnienie [ms]", "Kroki uczenia/s"]]
    assert "RTX" not in table.caption and "na karcie \\emph{NVIDIA" in table.notes   # the author's wording here
    # the text defines the throughput; the note keeps only how each speed was measured
    assert "czyli" not in table.notes
    assert "dla partii 64 wycinków" in table.notes and "dla partii 16 wycinków" in table.notes
    # four number columns of one width, stretched to the width of the note
    assert table.spec == script.COST_GRID and table.width == "0.97\\linewidth" and table.tabcolsep == 3
    script.render(table)


def test_cost_grid_has_four_number_columns_of_one_width() -> None:
    script = _load_script()
    column = ">{\\raggedleft\\arraybackslash}p{0.18\\linewidth}"
    assert script.column_count(script.COST_GRID) == 5
    # the second number column, the throughput, stands a little left of the right edge of its share
    assert script.COST_GRID == "l" + 2 * column + "<{\\hspace*{0.02\\linewidth}}" + 2 * column
    assert script.column_count("l<{\\hspace*{1em}}r@{}") == 2
