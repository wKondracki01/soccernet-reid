"""scripts/check_thesis_text.py: what the static check reports and how the prose is measured."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "check_thesis_text.py"
    spec = importlib.util.spec_from_file_location("check_thesis_text_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _thesis(tmp_path: Path, chapter: str) -> Path:
    (tmp_path / "tex" / "tabele").mkdir(parents=True)
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "rysunek.pdf").write_bytes(b"%PDF")
    (tmp_path / "main.tex").write_text("\\input{tex/1-rozdzial}\n", encoding="utf-8")
    (tmp_path / "tex" / "tabele" / "tabela.tex").write_text("\\begin{table}\\label{tab:tabela}\\end{table}\n", encoding="utf-8")
    (tmp_path / "tex" / "2-de-finibus.tex").write_text("\\cite{szablon} \\ref{nic}\n", encoding="utf-8")   # template sample
    (tmp_path / "bibliografia.bib").write_text("@article{znana2020,\n  title = {T},\n}\n", encoding="utf-8")
    (tmp_path / "tex" / "1-rozdzial.tex").write_text(chapter, encoding="utf-8")
    return tmp_path


GOOD = """\\section{Tytuł}\\label{sec:tytul}
Zdanie z~odwołaniem (tabela~\\ref{tab:tabela}) i~cytowaniem \\cite{znana2020}.
% \\cite{usunieta} \\ref{fig:brak} zakomentowany tekst nie jest sprawdzany
\\input{tex/tabele/tabela}
\\includegraphics[width=0.5\\linewidth]{rysunek.pdf}
"""


def test_clean_sources_report_nothing(tmp_path) -> None:
    script = _load_script()
    assert script.check_repository(_thesis(tmp_path, GOOD)) == []


def test_missing_targets_are_reported(tmp_path) -> None:
    script = _load_script()
    bad = GOOD + "\\ref{fig:brak} \\cite{znana2020,nieznana} \\input{tex/tabele/brak} \\includegraphics{brak.pdf} {\n"
    problems = "\n".join(script.check_repository(_thesis(tmp_path, bad)))
    for expected in ("fig:brak", "nieznana", "tex/tabele/brak", "brak.pdf", "nawiasów"):
        assert expected in problems
    assert "znana2020}" not in problems and "szablon" not in problems


def test_duplicate_label_is_reported(tmp_path) -> None:
    script = _load_script()
    problems = script.check_repository(_thesis(tmp_path, GOOD + "\\label{tab:tabela}\n"))
    assert len(problems) == 1 and "tab:tabela" in problems[0]


def test_prose_is_measured_without_commands_comments_and_floats() -> None:
    script = _load_script()
    text = """\\subsection{Nagłówek}
Pierwsze zdanie ma pięć słów. Drugie zdanie zawiera \\textbf{\\emph{pogrubiony}} wyraz oraz przypis\\footnote{Treść przypisu.} i~nic więcej.
% Zakomentowane zdanie nie liczy się wcale.
\\begin{figure}\\caption{Podpis rysunku.}\\end{figure}
Trzecie stanowi przykład; ma średnik.
"""
    stats = script.text_stats(text)
    assert stats["sentences"] == 3 and stats["words"] == 5 + 10 + 5
    assert stats["short"] == 2 and stats["long"] == 0
    assert stats["discouraged"]["średnik"] == 1 and stats["discouraged"]["stanowi"] == 1
