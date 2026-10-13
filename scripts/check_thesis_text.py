"""Check the LaTeX sources of the thesis without compiling them, and measure the prose of a chapter.

There is no LaTeX on the development machine (Overleaf compiles the thesis), so this
script catches what would otherwise only show up there: a missing \\input or image, a
\\ref without its \\label, a \\cite without its bibliography entry, unbalanced braces.
Commented-out text is ignored, the sample chapters of the template are skipped.

With ``--chapter`` it also reports the figures used to keep the prose even: number of
words and sentences, mean sentence length, shares of short and long sentences, and how
often a few discouraged constructions occur.

Usage
-----
    python scripts/check_thesis_text.py --thesis ~/Projects/LaTeX/6ac7f20401f35b6ac4f68ab2
    python scripts/check_thesis_text.py --thesis PATH --chapter tex/3-zbior-danych.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TEMPLATE_SAMPLES = ("1-wstep.tex", "2-de-finibus.tex", "3-code-listings.tex")
POLISH_LETTERS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
# Constructions the style rules of the thesis ask to avoid in running text.
DISCOURAGED = {
    "myślnik w zdaniu": r" [-–—] ",
    "średnik": r";",
    "stanowi": r"\bstanowi",
    "posiada": r"\bposiada",
    "kluczowy": r"\bkluczow",
    "warto (zauważyć, podkreślić)": r"\b[Ww]arto\b",
    "nie tylko ... ale": r"\bnie tylko\b",
    "związany z": r"\bzwiązan\w* z\b",
}


def strip_comments(text: str) -> str:
    """Remove everything from an unescaped % to the end of each line."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.split("\n"))


def thesis_files(root: Path) -> list[Path]:
    chapters = [p for p in sorted((root / "tex").glob("*.tex")) if p.name not in TEMPLATE_SAMPLES]
    return [root / "main.tex", *chapters, *sorted((root / "tex" / "tabele").glob("*.tex"))]


def check_repository(root: Path) -> list[str]:
    """Problems found in the sources; an empty list means nothing was found."""
    problems: list[str] = []
    bibliography = (root / "bibliografia.bib").read_text(encoding="utf-8") if (root / "bibliografia.bib").exists() else ""
    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bibliography))
    labels: dict[str, Path] = {}
    references: list[tuple[str, Path]] = []
    for path in thesis_files(root):
        text = strip_comments(path.read_text(encoding="utf-8"))
        name = path.relative_to(root)
        bare = text.replace("\\%", "").replace("\\{", "").replace("\\}", "").replace("\\$", "")
        if path.name != "main.tex":                       # main.tex carries template macros with their own syntax
            if bare.count("{") != bare.count("}"):
                problems.append(f"{name}: {bare.count('{')} nawiasów otwierających i {bare.count('}')} zamykających")
            if bare.count("$") % 2:
                problems.append(f"{name}: nieparzysta liczba znaków $")
            odd = sorted({c for c in text if ord(c) > 127 and c not in POLISH_LETTERS})
            if odd:
                problems.append(f"{name}: znaki spoza ASCII i polskich liter: {' '.join(odd)}")
        if sorted(re.findall(r"\\begin\{(\w+\*?)\}", text)) != sorted(re.findall(r"\\end\{(\w+\*?)\}", text)):
            problems.append(f"{name}: niesparowane \\begin i \\end")
        for target in re.findall(r"\\input\{([^}]+)\}", text):
            if not (root / f"{target}.tex").exists() and not (root / target).exists():
                problems.append(f"{name}: brak pliku dla \\input{{{target}}}")
        for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            if not (root / "img" / image).exists() and not (root / image).exists():
                problems.append(f"{name}: brak obrazu {image}")
        for label in re.findall(r"\\label\{([^}]+)\}", text):
            if label in labels:
                problems.append(f"etykieta {label} występuje dwa razy: {labels[label].relative_to(root)} i {name}")
            labels[label] = path
        references += [(ref, path) for ref in re.findall(r"\\ref\{([^}]+)\}", text)]
        for group in re.findall(r"\\cite\{([^}]+)\}", text):
            for key in (k.strip() for k in group.split(",")):
                if key not in keys:
                    problems.append(f"{name}: brak pozycji bibliografii dla \\cite{{{key}}}")
    for ref, path in references:
        if ref not in labels:
            problems.append(f"{path.relative_to(root)}: \\ref do nieistniejącej etykiety {ref}")
    return problems


def prose(text: str) -> str:
    """Running text of a chapter: no comments, floats, headings, footnotes or commands."""
    text = strip_comments(text)
    text = re.sub(r"\\begin\{(figure|table|longtable)\}.*?\\end\{\1\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(section|subsection|subsubsection|label|input|cite|ref|footnote)\*?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(textbf|textit|emph|texttt)\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\(textbf|textit|emph|texttt)\{([^{}]*)\}", r"\2", text)      # one level of nesting
    text = text.replace("~", " ").replace("\\,", "").replace("\\%", "%")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", prose(text)) if s.strip()]


def text_stats(text: str) -> dict:
    found = sentences(text)
    lengths = [len(s.split()) for s in found]
    body = prose(text)
    return {
        "words": sum(lengths),
        "sentences": len(found),
        "mean": sum(lengths) / len(found) if found else 0.0,
        "short": sum(n <= 8 for n in lengths),            # the author's own prose: about one sentence in ten
        "long": sum(n > 25 for n in lengths),             # the author's own prose: about one in four or five
        "discouraged": {name: len(re.findall(pattern, body)) for name, pattern in DISCOURAGED.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thesis", type=Path, required=True, help="root of the thesis repository (the Overleaf clone)")
    parser.add_argument("--chapter", type=Path, help="chapter file, relative to the thesis root, to measure")
    args = parser.parse_args()

    root = args.thesis.expanduser()
    problems = check_repository(root)
    print(f"sprawdzono {len(thesis_files(root))} plików")
    for problem in problems:
        print("PROBLEM:", problem)
    if not problems:
        print("nie znaleziono problemów")
    if args.chapter:
        stats = text_stats((root / args.chapter).read_text(encoding="utf-8"))
        print(f"{args.chapter}: {stats['words']} słów, {stats['sentences']} zdań, średnio {stats['mean']:.1f} słowa w zdaniu, "
              f"{stats['short']} zdań do 8 słów, {stats['long']} zdań ponad 25 słów")
        for name, count in stats["discouraged"].items():
            if count:
                print(f"  {name}: {count}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
