"""Write the tables of the thesis as LaTeX files from the saved results of series G.

Nothing is trained or evaluated here: every result is read from a file written by
an earlier step, and the dataset tables are counted from the catalog.

Inputs
    outputs/_g/REVIEW_*.json                validation curve and best epoch of each run
    outputs/_g/BOOTSTRAP_*.json             paired bootstrap of differences between runs
    outputs/_g/speed.json                   scripts/benchmark_speed.py
    outputs/runs/RUN/stdout.log             the configuration the run was started with
    outputs/runs/RUN/eval_valid.json        re-evaluation of the saved best checkpoint
    outputs/runs/RUN/eval_test.json         final models only
    outputs/runs/RUN/rerank_{valid,test}.json   final models only
    outputs/runs/RUN/embeddings_valid.npz   only for the table on loss settings
    outputs/catalog.parquet                 only for the three dataset tables

Every table goes to its own file NAME.tex and carries the label tab:NAME, so the
thesis can \\input it next to the text that discusses it. Text is Polish, numbers
use a decimal comma, the best value of a metric column is bold.

Usage
-----
    python scripts/make_thesis_tables.py --out outputs/tables
    python scripts/make_thesis_tables.py --only results appendix
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_thesis_figures import (  # noqa: E402
    AUGMENT_LABELS,
    BASE_SEEDS,
    FINAL_SEEDS,
    load_reviews,
    pl,
)

METRICS = ("mAP", "rank-1", "rank-5", "rank-10")
METRIC_HEADER = ["mAP", "Rank-1", "Rank-5", "Rank-10"]
DASH = "--"

# Validation scores that need no training. The first two follow from the protocol alone
# (random embeddings, mean of the seeds 0-4; one constant vector for every crop), the
# third is ResNet-18 with its ImageNet weights (scripts/smoke_eval.py). All three were
# recomputed on 2026-10-11 and agree with these values to the fourth decimal place.
UNTRAINED = (
    ("losowy ranking", (0.1942, 0.0673, 0.3135, 0.5755)),
    ("ten sam wektor dla każdego wycinka", (0.2808, 0.1235, 0.5143, 0.7762)),
    ("ResNet-18 z wagami ImageNet, bez douczania", (0.3295, 0.2253, 0.5058, 0.6884)),
)

# Leaderboard of the SoccerNet Re-Identification challenge 2023 (challenge split, per cent), as
# published in the README of the official repository github.com/SoccerNet/sn-reid; read 2026-10-11.
CHALLENGE_2023 = (
    ("UniBw Munich - VIS", 93.26, 91.26),
    ("Baseline (Inspur)", 91.68, 89.41),
    ("sjtu-lenovo", 91.51, 89.17),
    ("MTVACV", 90.11, 87.04),
    ("ErxianBridge", 85.76, 82.33),
    ("cm_test", 42.60, 28.73),
)

BACKBONES = {  # code -> (name in the thesis, pretrained weights in timm, dimension of the pooled feature)
    "R18": ("ResNet-18", "resnet18.a1_in1k", 512),
    "R34": ("ResNet-34", "resnet34.a1_in1k", 512),
    "EB1": ("EfficientNet-B1", "efficientnet_b1.ra4_e3600_r240_in1k", 1280),
    "EB2": ("EfficientNet-B2", "efficientnet_b2.ra_in1k", 1408),
    "VGG11-BN-CONV": ("VGG11-BN", "vgg11_bn.tv_in1k", 512),
    "VGG16-BN-CONV": ("VGG16-BN", "vgg16_bn.tv_in1k", 512),
    "VGG11-BN": ("VGG11-BN + fc", "vgg11_bn.tv_in1k", 4096),
    "VGG16-BN": ("VGG16-BN + fc", "vgg16_bn.tv_in1k", 4096),
}
BACKBONE_RUNS = {
    "R18": "G3_AUG_MED", "R34": "G4_R34", "EB1": "G4_EB1", "EB2": "G4_EB2",
    "VGG11-BN-CONV": "G4_VGG11_BN_CONV", "VGG16-BN-CONV": "G4_VGG16_BN_CONV",
    "VGG11-BN": "G4_VGG11_BN", "VGG16-BN": "G4_VGG16_BN",
}

# Every training of the thesis, in the order of the appendix: (heading, runs).
STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Punkty odniesienia", ("G0a_reference", "G0b_variant_K", "G0c_variant_K_WD")),
    ("Etap 1: dobór przykładów", ("G1_RAND", "G1_PK_SH", "G1_PK_BH", "G1_PK_BH_XBM", "G1_PK_SA_SH", "G1_PK_SA_BH",
                                  "G1_PK_SA_BH_XBM")),
    ("Etap 2: funkcje straty", ("G2_CONT", "G2_CONT_M1", "G2_MS", "G2_MS_B05", "G2_CIRCLE", "G2_CIRCLE_M04",
                                "G2_CE", "G2_CE_WD", "G2_ARC", "G2_ARC_WD")),
    ("Etap 3: augmentacja", ("G3_AUG_MED", "G3_AUG_STRONG", "G3_AUG_BOT")),
    ("Etap 4: ekstraktory cech", ("G4_R34", "G4_EB1", "G4_EB2", "G4_VGG11_BN_CONV", "G4_VGG16_BN_CONV",
                                  "G4_VGG11_BN", "G4_VGG16_BN")),
    ("Etap 5: weryfikacja wyborów", ("G5_REF", "G5_REF_S1", "G5_REF_S2", "G5_STRONG", "G5_XBM", "G5_CIRCLE",
                                     "G5_MS", "G5_PK", "G5_ARC_WD")),
    ("Konfiguracja końcowa: dodatkowe treningi", ("G5_STRONG_S1", "G5_STRONG_S2")),
    ("Badania uzupełniające", ("G6_D64", "G6_D128", "G6_D256", "G6_D1024", "G6_SCRATCH", "G6_SCRATCH_CIRCLE")),
)


# --------------------------------------------------------------------------- formatting

def num(value: int) -> str:
    """An integer with thin spaces between groups of three digits."""
    return f"{value:,}".replace(",", "\\,")


def percent(share: float, digits: int = 1) -> str:
    return pl(100 * share, digits) + "\\%"


def points(delta: float, digits: int = 2) -> str:
    """A difference between two scores in percentage points, with an explicit sign."""
    text = pl(abs(100 * delta), digits)
    if float(text.replace(",", ".")) == 0:
        return text                                   # a difference that rounds to zero has no sign
    return ("$-$" if delta < 0 else "+") + text


def short(value: float) -> str:
    """A setting such as 0.5 or 80 without trailing zeros, for use inside math mode."""
    return pl(value, 2).rstrip("0").rstrip(",").replace(",", "{,}")


def tt(text: str) -> str:
    return "\\texttt{" + text.replace("_", "\\_") + "}"


def bold(text: str) -> str:
    return "\\textbf{" + text + "}"


def metric_cells(rows: list[tuple[float, ...]], digits: int = 4, mark_best: bool = True) -> list[list[str]]:
    """Format columns of scores; the highest value of every column is bold."""
    best = [max(col) for col in zip(*rows, strict=True)]
    out = []
    for row in rows:
        cells = []
        for value, top in zip(row, best, strict=True):
            text = pl(value, digits)
            cells.append(bold(text) if mark_best and pl(top, digits) == text else text)
        out.append(cells)
    return out


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


# --------------------------------------------------------------------------- tables as data

@dataclass
class Table:
    name: str                       # file stem; the label is tab:NAME
    caption: str
    spec: str                       # column specification of the tabular
    header: list[list[str]]
    rows: list[list[str] | None]    # None draws a horizontal rule
    size: str = "\\small"
    long: bool = False              # longtable that may break across pages


_MULTICOLUMN = re.compile(r"^\\multicolumn\{(\d+)\}")


def par(width: float) -> str:
    """A paragraph column of the given share of the line width, set ragged right."""
    return f">{{\\raggedright\\arraybackslash}}p{{{width}\\linewidth}}"


def column_count(spec: str) -> int:
    """Number of columns in a specification made of l, c, r and p{...}, with optional @{...} and >{...}."""
    stripped = re.sub(r"[@>]\{[^{}]*\}", "", spec)
    stripped = re.sub(r"p\{[^{}]*\}", "p", stripped)
    if not re.fullmatch(r"[lcrp]+", stripped):
        raise ValueError(f"unsupported column specification: {spec!r}")
    return len(stripped)


def row_span(row: list[str]) -> int:
    total = 0
    for cell in row:
        match = _MULTICOLUMN.match(cell)
        total += int(match.group(1)) if match else 1
    return total


def render(table: Table) -> str:
    """LaTeX source of a table; raises if a row does not fill the declared columns."""
    n = column_count(table.spec)
    for row in [*table.header, *[r for r in table.rows if r is not None]]:
        if row_span(row) != n:
            raise ValueError(f"{table.name}: a row has {row_span(row)} columns, the table has {n}: {row}")
        for cell in row:
            if cell.count("{") != cell.count("}"):
                raise ValueError(f"{table.name}: unbalanced braces in {cell!r}")

    def line(row: list[str] | None) -> str:
        return "\\hline" if row is None else " & ".join(row) + " \\\\"

    head = [line(r) for r in table.header]
    body = [line(r) for r in table.rows]
    if table.long:
        out = [
            "{" + table.size, "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{longtable}}{{{table.spec}}}",
            f"\\caption{{{table.caption}}}\\label{{tab:{table.name}}} \\\\",
            "\\hline", *head, "\\hline", "\\endfirsthead",
            "\\hline", *head, "\\hline", "\\endhead",
            "\\hline", "\\endfoot",
            *body, "\\end{longtable}", "}",
        ]
    else:
        out = [
            "\\begin{table}[!htbp]", "\\centering", f"\\caption{{{table.caption}}}", f"\\label{{tab:{table.name}}}",
            table.size, f"\\begin{{tabular}}{{{table.spec}}}",
            "\\hline", *head, "\\hline", *body, "\\hline", "\\end{tabular}", "\\end{table}",
        ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- reading results

def run_config(runs_dir: Path, name: str) -> dict:
    """The Hydra configuration printed at the start of a run (between the two rules in stdout.log)."""
    import yaml

    text = (runs_dir / name / "stdout.log").read_text(errors="ignore")
    return yaml.safe_load(text.split("=" * 70)[1])


_EVAL_LINE = re.compile(r"Eval: mAP=([0-9.]+) R-1=([0-9.]+)")


def logged_evals(runs_dir: Path, name: str) -> list[tuple[int, float, float]]:
    """(epoch, mAP, Rank-1) of every evaluation printed in the training log."""
    every = run_config(runs_dir, name)["eval"]["every_n_epochs"]
    text = (runs_dir / name / "stdout.log").read_text(errors="ignore")
    return [(every * (i + 1), float(m.group(1)), float(m.group(2))) for i, m in enumerate(_EVAL_LINE.finditer(text))]


def best_scores(runs_dir: Path, name: str) -> tuple[int, tuple[float | None, ...]]:
    """(epoch, scores) of the saved best checkpoint; the epoch is counted from 1.

    A run that was stopped early has no re-evaluation file; its mAP and Rank-1 then come
    from the training log, which does not print Rank-5 and Rank-10 (returned as None).
    """
    path = runs_dir / name / "eval_valid.json"
    if not path.exists():
        epoch, m_ap, rank1 = max(logged_evals(runs_dir, name), key=lambda e: e[1])
        return epoch, (m_ap, rank1, None, None)
    data = json.loads(path.read_text())
    return int(data["epoch"]) + 1, tuple(float(data[m]) for m in METRICS)


def scores(runs_dir: Path, name: str) -> tuple[float, ...]:
    return best_scores(runs_dir, name)[1]


def load_bootstrap(review_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(review_dir.glob("BOOTSTRAP_*.json")):
        out.update(json.loads(path.read_text()))
    return out


def difference(bootstrap: dict[str, dict], a: str, b: str) -> tuple[float, float, float]:
    """mAP of ``a`` minus mAP of ``b`` with its 95% interval, whichever way round it was saved."""
    if f"{a} - {b}" in bootstrap:
        entry = bootstrap[f"{a} - {b}"]
        return entry["d_map"], entry["d_map_ci"][0], entry["d_map_ci"][1]
    entry = bootstrap[f"{b} - {a}"]
    return -entry["d_map"], -entry["d_map_ci"][1], -entry["d_map_ci"][0]


def interval(low: float, high: float) -> str:
    return f"[{points(low)}; {points(high)}]"


def embedding_geometry(runs_dir: Path, name: str) -> tuple[float, float]:
    """(mean cosine of all pairs of different crops, participation ratio) of the validation embeddings."""
    import numpy as np

    with np.load(runs_dir / name / "embeddings_valid.npz", allow_pickle=False) as data:
        feats = np.concatenate([data["query_feats"], data["gallery_feats"]]).astype(np.float64)
    unit = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    n = len(unit)
    mean_cosine = (float(np.sum(unit.sum(axis=0) ** 2)) - n) / (n * (n - 1))
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(unit, rowvar=False)), 0, None)
    return mean_cosine, float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


# --------------------------------------------------------------------------- describing a run

SAMPLER_CODES = {"random": "RAND", "pk": "PK", "pk_sa": "PK-SA"}
MINER_CODES = {"batch-hard": "BH", "semi-hard": "SH"}


def selection_code(config: dict) -> str:
    """Code of the example-selection variant, e.g. PK-SA-BH-XBM (only meaningful for the triplet loss)."""
    loss, sampler = config["loss"], SAMPLER_CODES[config["sampler"]["name"]]
    if sampler == "RAND":
        return sampler
    code = f"{sampler}-{MINER_CODES[loss.get('miner', 'batch-hard')]}"
    return code + "-XBM" if loss.get("xbm") else code


def loss_text(loss: dict) -> str:
    name = loss["name"]
    if name == "tri":
        return "trójkowa"
    if name == "cont":
        return f"kontrastywna (margines {pl(loss['neg_margin'], 1)})"
    if name == "ms":
        return f"multi-similarity ($\\lambda = {short(loss['base'])}$)"
    if name == "circle":
        return f"Circle ($m = {short(loss['m'])}$, $\\gamma = {short(loss['gamma'])}$)"
    if name == "ce":
        return "entropia krzyżowa"
    if name == "arc":
        return "ArcFace"
    raise ValueError(f"unknown loss: {name!r}")


def describe(config: dict) -> str:
    """One line with everything that distinguishes a run from the others."""
    loss = config["loss"]
    parts = [BACKBONES[config["backbone"]["name"]][0]]
    if not config["backbone"]["pretrained"]:
        parts.append("bez wag ImageNet")
    parts.append(loss_text(loss))
    if loss["name"] == "tri":
        parts.append(selection_code(config))
    else:
        parts.append("sampler losowy" if config["sampler"]["name"] == "random" else SAMPLER_CODES[config["sampler"]["name"]])
    if config["optimizer"]["weight_decay"] > 0:
        parts.append("weight decay")
    parts.append(AUGMENT_LABELS[config["transform"]["level"]])
    if config["embedding_dim"] != 512:
        parts.append(f"embedding {config['embedding_dim']}")
    if config["seed"] != 0:
        parts.append(f"ziarno {config['seed']}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- dataset tables

PLAYER_CLASSES = ("Player_team_left", "Player_team_right", "Goalkeeper_team_left", "Goalkeeper_team_right")


def dataset_tables(catalog) -> list[Table]:
    splits = []
    for split, label in (("train", "treningowy"), ("valid", "walidacyjny"), ("test", "testowy"),
                         ("challenge", "konkursowy")):
        part = catalog[catalog["split"] == split]
        queries, gallery = int((part["role"] == "query").sum()), int((part["role"] == "gallery").sum())
        games = part[["championship", "season", "game"]].dropna().drop_duplicates().shape[0]
        splits.append([
            label, num(len(part)), num(queries) if queries else DASH, num(gallery) if gallery else DASH,
            num(part["action_idx"].nunique()), num(games) if games else DASH,
        ])

    train = catalog[(catalog["split"] == "train") & catalog["clazz"].isin(PLAYER_CLASSES)]
    sizes = train.groupby(["action_idx", "person_uid"]).size()
    per_class = []
    for label, low, high in (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), (f"5{DASH}{sizes.max()}", 5, sizes.max())):
        count = int(((sizes >= low) & (sizes <= high)).sum())
        per_class.append([label, num(count), percent(count / len(sizes))])
    per_class += [None, ["razem", num(len(sizes)), percent(1.0)]]

    pairs = sizes[sizes >= 2]                                   # classes a P x K sampler with K = 2 can draw
    per_action = pairs.groupby(level=0).size()
    one_action = pairs[pairs.index.get_level_values(0).isin(per_action[per_action >= 8].index)]
    pools = [
        ["losowy", num(len(sizes)), num(int(sizes.sum())), percent(1.0)],
        ["PK ($16 \\times 2$)", num(len(pairs)), num(int(pairs.sum())), percent(pairs.sum() / sizes.sum())],
        ["PK-SA ($8 \\times 2$, jedna akcja)", num(len(one_action)), num(int(one_action.sum())),
         percent(one_action.sum() / sizes.sum())],
    ]
    return [
        Table("podzialy-zbioru", "Wielkość podziałów zbioru SoccerNet Re-Identification", "lrrrrr",
              [["Podział", "Wycinki", "Zapytania", "Galeria", "Akcje", "Mecze"]], splits),
        Table("wycinki-na-klase",
              "Liczba wycinków przypadających na klasę w zbiorze treningowym (klasy zawodników i bramkarzy)", "crr",
              [["Wycinki w klasie", "Liczba klas", "Udział klas"]], per_class),
        Table("pule-samplerow", "Klasy i wycinki zbioru treningowego dostępne dla poszczególnych samplerów", "lrrr",
              [["Sampler", "Klasy", "Wycinki", "Udział wycinków"]], pools),
    ]


# --------------------------------------------------------------------------- method tables

def method_tables(speed: dict) -> list[Table]:
    selection = Table(
        "warianty-doboru", "Badane warianty doboru przykładów", "lllll",
        [["Oznaczenie", "Sampler", "Rozmiar batcha", "Wybór trójek", "Pamięć XBM"]],
        [
            ["RAND", "losowy", "32", "wszystkie trójki", "nie"],
            ["PK-SH", "PK ($16 \\times 2$)", "32", "semi-hard", "nie"],
            ["PK-BH", "PK ($16 \\times 2$)", "32", "batch-hard", "nie"],
            ["PK-BH-XBM", "PK ($16 \\times 2$)", "32", "batch-hard", "1024 embeddingi"],
            ["PK-SA-SH", "PK-SA ($8 \\times 2$)", "16", "semi-hard", "nie"],
            ["PK-SA-BH", "PK-SA ($8 \\times 2$)", "16", "batch-hard", "nie"],
            ["PK-SA-BH-XBM", "PK-SA ($8 \\times 2$)", "16", "batch-hard", "1024 embeddingi"],
        ],
    )
    losses = Table(
        "funkcje-straty", "Badane funkcje straty i ich ustawienia",
        par(0.19) + par(0.42) + par(0.29),
        [["Funkcja straty", "Ustawienia", "Przykłady w batchu"]],
        [
            ["trójkowa", "margines 0,3", "według wariantu doboru (tabela~\\ref{tab:warianty-doboru})"],
            ["kontrastywna", "margines par pozytywnych 0, par negatywnych 0,5 albo 1,0", "wszystkie pary"],
            ["multi-similarity", "$\\alpha = 2$, $\\beta = 50$, $\\lambda = 1$ albo $\\lambda = 0{,}5$",
             "pary wybrane minerem multi-similarity ($\\varepsilon = 0{,}1$)"],
            ["Circle", "$m = 0{,}25$ i $\\gamma = 64$ albo $m = 0{,}4$ i $\\gamma = 80$", "wszystkie pary"],
            ["entropia krzyżowa", "wygładzanie etykiet 0,1", "sampler losowy, batch 32"],
            ["ArcFace", "skala 30, margines 0,5 rad", "sampler losowy, batch 32"],
        ],
    )
    yes = "tak"
    augment = Table(
        "zestawy-augmentacji", "Operacje wchodzące w skład zestawów augmentacji",
        par(0.40) + "cccc",
        [["Operacja", "AUG-MIN", "AUG-MED", "AUG-STRONG", "AUG-COLOR"]],
        [
            ["odbicie poziome ($p = 0{,}5$)", yes, yes, yes, yes],
            ["losowa zmiana jasności, kontrastu i~nasycenia (zakres)", DASH, "0,2", "0,2", "0,4"],
            ["losowa zmiana odcienia (zakres)", DASH, "0,05", "0,05", "0,1"],
            ["zamiana na skalę szarości ($p = 0{,}2$)", DASH, DASH, DASH, yes],
            ["dopełnienie o~10 pikseli i~losowe przycięcie do $256 \\times 128$", DASH, yes, yes, yes],
            ["RandAugment (2 operacje, siła 9)", DASH, DASH, yes, DASH],
            ["rozmycie Gaussa (jądro 3, $\\sigma$ od 0,1 do 2,0, $p = 0{,}5$)", DASH, DASH, yes, DASH],
            ["zniekształcenie perspektywiczne (siła 0,2, $p = 0{,}3$)", DASH, DASH, yes, DASH],
            ["Random Erasing ($p = 0{,}5$, od 2\\% do 40\\% pola)", DASH, yes, yes, yes],
        ],
    )
    parameters = {r["backbone"]: r["parameters"] for r in speed["results"]}
    backbones = Table(
        "ekstraktory-cech",
        "Badane ekstraktory cech (liczba parametrów razem z głową projekcyjną o~wymiarze 512)", "llrr",
        [["Sieć", "Wagi wstępne w bibliotece timm", "Wymiar cech", "Parametry [mln]"]],
        [[name, tt(weights), num(dim), pl(parameters[code] / 1e6, 2)] for code, (name, weights, dim) in BACKBONES.items()],
    )
    plan = Table(
        "plan-etapow", "Plan eksperymentów",
        par(0.19) + par(0.36) + par(0.27) + "r",
        [["Etap", "Porównywane warianty", "Ustawienia stałe", "Treningi"]],
        [
            ["Punkty odniesienia", "strata trójkowa z samplerem PK oraz entropia krzyżowa bez weight decay i z weight decay",
             "ResNet-18, AUG-MIN, 60 epok", "3"],
            ["Etap 1: dobór przykładów", "siedem wariantów z tabeli~\\ref{tab:warianty-doboru}",
             "ResNet-18, strata trójkowa, AUG-MIN, 40 epok", "7"],
            ["Etap 2: funkcje straty", "sześć funkcji z tabeli~\\ref{tab:funkcje-straty}, w tym trzy w dwóch ustawieniach "
             "i dwie bez weight decay oraz z weight decay", "ResNet-18, AUG-MIN, 40 epok", "10"],
            ["Etap 3: augmentacja", "cztery zestawy z tabeli~\\ref{tab:zestawy-augmentacji}",
             "ResNet-18, strata trójkowa, PK-SA-BH, 40 epok", "3"],
            ["Etap 4: ekstraktory cech", "osiem sieci z tabeli~\\ref{tab:ekstraktory-cech}",
             "strata trójkowa, PK-SA-BH, AUG-MED, 40 epok", "7"],
            ["Etap 5: weryfikacja wyborów", "konfiguracja bazowa (trzy treningi) i sześć wariantów różniących się od niej "
             "jednym elementem", "ResNet-34, 60 epok", "9"],
            ["Konfiguracja końcowa", "dwa dodatkowe treningi z innym ziarnem", "ResNet-34, 60 epok", "2"],
            ["Badania uzupełniające", "cztery wymiary embeddingu i dwa treningi bez wag ImageNet",
             "ResNet-34, 60 epok", "6"],
            None,
            ["Razem", "", "", "47"],
        ],
    )
    training = Table(
        "ustawienia-treningu", "Ustawienia wspólne dla wszystkich treningów",
        par(0.30) + par(0.60),
        [["Parametr", "Wartość"]],
        [
            ["optymalizator", "Adam"],
            ["współczynnik uczenia", "$3{,}5 \\cdot 10^{-4}$"],
            ["harmonogram", "liniowa rozgrzewka przez 5 epok, potem spadek kosinusowy do zera"],
            ["weight decay", "0 dla strat uczenia metryki, 0 albo $5 \\cdot 10^{-4}$ dla strat klasyfikacyjnych"],
            ["długość epoki", "5000 iteracji niezależnie od rozmiaru batcha"],
            ["liczba epok", "40 w etapach 1--4, 60 w pozostałych treningach"],
            ["precyzja obliczeń", "pojedyncza (32 bity)"],
            ["rozmiar wejścia", "$256 \\times 128$ pikseli"],
            ["wymiar embeddingu", "512"],
            ["ocena w trakcie treningu", "zbiór walidacyjny co 5 epok (straty klasyfikacyjne w etapie 2 co epokę), "
             "zapisywany model o najwyższym mAP"],
            ["ziarno", "0, a dla konfiguracji bazowej i końcowej także 1 i 2"],
        ],
    )
    return [selection, losses, augment, backbones, plan, training]


# --------------------------------------------------------------------------- result tables

def result_tables(runs: dict[str, dict], runs_dir: Path, bootstrap: dict[str, dict], speed: dict) -> list[Table]:
    def table_of_runs(name: str, caption: str, spec: str, header: list[str], labelled: list[tuple[list[str], str]],
                      extra: list[list[str]] | None = None) -> Table:
        """Rows sorted by mAP, best first; ``extra`` are columns placed after the scores."""
        order = sorted(range(len(labelled)), key=lambda i: -scores(runs_dir, labelled[i][1])[0])
        cells = metric_cells([scores(runs_dir, labelled[i][1]) for i in order])
        rows = [[*labelled[i][0], *cells[k], *(extra[i] if extra else [])] for k, i in enumerate(order)]
        return Table(name, caption, spec, [header], rows)

    tables: list[Table] = []

    # reference points: scores without training, then the three reference runs
    reference_runs = [
        ("strata trójkowa, sampler PK, batch-hard", "G0a_reference"),
        ("entropia krzyżowa bez weight decay", "G0b_variant_K"),
        ("entropia krzyżowa z weight decay", "G0c_variant_K_WD"),
    ]
    values = [v for _, v in UNTRAINED] + [scores(runs_dir, run) for _, run in reference_runs]
    labels = [label for label, _ in UNTRAINED] + [label for label, _ in reference_runs]
    cells = metric_cells(values)
    rows: list[list[str] | None] = [[label, *c] for label, c in zip(labels, cells, strict=True)]
    rows.insert(len(UNTRAINED), None)
    tables.append(Table(
        "wyniki-odniesienie",
        "Punkty odniesienia na zbiorze walidacyjnym: wartości uzyskane bez treningu oraz trzy treningi "
        "sieci ResNet-18 z~zestawem AUG-MIN (60 epok)", "lrrrr", [["Wariant", *METRIC_HEADER]], rows))

    # stage 1
    stage1 = []
    for run in STAGES[1][1]:
        config = run_config(runs_dir, run)
        sampler = {"random": "losowy", "pk": "PK", "pk_sa": "PK-SA"}[config["sampler"]["name"]]
        miner = {"batch-hard": "batch-hard", "semi-hard": "semi-hard", "none": "wszystkie trójki"}[
            config["loss"].get("miner", "batch-hard")]
        stage1.append(([selection_code(config), sampler, miner, "tak" if config["loss"].get("xbm") else "nie"], run))
    tables.append(table_of_runs(
        "wyniki-dobor",
        "Wyniki wariantów doboru przykładów na zbiorze walidacyjnym (ResNet-18, strata trójkowa, AUG-MIN, 40 epok)",
        "llllrrrr", ["Wariant", "Sampler", "Wybór trójek", "XBM", *METRIC_HEADER], stage1))

    # stage 2: every loss in the better of its computed settings, then the effect of the settings
    stage2 = [
        (["Circle", "$m = 0{,}4$, $\\gamma = 80$"], "G2_CIRCLE_M04"),
        (["trójkowa", "margines 0,3"], "G1_PK_SA_BH"),
        (["multi-similarity", "$\\lambda = 0{,}5$"], "G2_MS_B05"),
        (["kontrastywna", "margines 1,0"], "G2_CONT_M1"),
        (["ArcFace", "z weight decay"], "G2_ARC_WD"),
        (["entropia krzyżowa", "z weight decay"], "G2_CE_WD"),
    ]
    tables.append(table_of_runs(
        "wyniki-straty",
        "Wyniki funkcji straty na zbiorze walidacyjnym, każda w~lepszym z~dwóch zbadanych ustawień "
        "(ResNet-18, AUG-MIN, 40 epok)", "llrrrr", ["Funkcja straty", "Ustawienie", *METRIC_HEADER], stage2))

    pairs = [
        ("kontrastywna", "margines 0,5", "G2_CONT", "margines 1,0", "G2_CONT_M1"),
        ("multi-similarity", "$\\lambda = 1$", "G2_MS", "$\\lambda = 0{,}5$", "G2_MS_B05"),
        ("Circle", "$m = 0{,}25$, $\\gamma = 64$", "G2_CIRCLE", "$m = 0{,}4$, $\\gamma = 80$", "G2_CIRCLE_M04"),
        ("entropia krzyżowa", "bez weight decay", "G2_CE", "z weight decay", "G2_CE_WD"),
        ("ArcFace", "bez weight decay", "G2_ARC", "z weight decay", "G2_ARC_WD"),
    ]
    setting_rows: list[list[str] | None] = []
    for loss, first_label, first, second_label, second in pairs:
        delta, low, high = difference(bootstrap, second, first)
        for k, (label, run) in enumerate(((first_label, first), (second_label, second))):
            cosine, dimension = embedding_geometry(runs_dir, run)
            values = scores(runs_dir, run)
            setting_rows.append([
                loss if k == 0 else "", label, pl(values[0]), pl(cosine, 2), pl(dimension, 1),
                "" if k == 0 else f"{points(delta)} {interval(low, high)}",
            ])
        setting_rows.append(None)
    tables.append(Table(
        "wyniki-ustawienia-strat",
        "Wpływ ustawień funkcji straty na mAP na zbiorze walidacyjnym i~na rozkład embeddingów (kosinus: średnie "
        "podobieństwo kosinusowe embeddingów różnych wycinków, wymiarowość: wymiarowość efektywna, różnica mAP "
        "z~95-procentowym przedziałem ufności)",
        "llrrrl", [["Funkcja straty", "Ustawienie", "mAP", "Kosinus", "Wymiarowość", "Różnica mAP [pp]"]],
        setting_rows[:-1]))

    # stage 3
    stage3_runs = ["G1_PK_SA_BH", "G3_AUG_MED", "G3_AUG_STRONG", "G3_AUG_BOT"]
    stage3 = [([AUGMENT_LABELS[run_config(runs_dir, run)["transform"]["level"]]], run) for run in stage3_runs]
    zero = [[pl(100 * runs[run]["zero_loss_frac"], 1)] for run in stage3_runs]
    tables.append(table_of_runs(
        "wyniki-augmentacje",
        "Wyniki zestawów augmentacji na zbiorze walidacyjnym (ResNet-18, strata trójkowa, PK-SA-BH, 40 epok) "
        "oraz udział kroków treningu, w~których strata była równa zero",
        "lrrrrr", ["Zestaw", *METRIC_HEADER, "Zerowa strata [\\%]"], stage3, zero))

    # stage 4
    by_code = {r["backbone"]: r for r in speed["results"]}
    stage4 = [([BACKBONES[code][0], pl(by_code[code]["parameters"] / 1e6, 1),
                num(round(by_code[code]["inference_images_per_s"]))], run) for code, run in BACKBONE_RUNS.items()]
    tables.append(table_of_runs(
        "wyniki-ekstraktory",
        "Wyniki ekstraktorów cech na zbiorze walidacyjnym (strata trójkowa, PK-SA-BH, AUG-MED, 40 epok) "
        "oraz ich koszt: liczba parametrów i~liczba wycinków przetwarzanych w~ciągu sekundy przy batchu 64 "
        "na karcie RTX 4090",
        "lrrrrrr", ["Sieć", "Parametry [mln]", "Wycinki/s", *METRIC_HEADER], stage4))

    # stage 5: the base and the variants that change one element of it, all trained with seed 0
    variants = [
        ("zestaw AUG-STRONG", "G5_STRONG"), ("pamięć XBM", "G5_XBM"), ("strata Circle", "G5_CIRCLE"),
        ("strata multi-similarity", "G5_MS"), ("sampler PK", "G5_PK"), ("strata ArcFace z~weight decay", "G5_ARC_WD"),
    ]
    cells = metric_cells([scores(runs_dir, BASE_SEEDS[0])] + [scores(runs_dir, run) for _, run in variants])
    rows = [["konfiguracja bazowa", *cells[0], DASH], None]
    for (label, run), row_cells in zip(variants, cells[1:], strict=True):
        delta, low, high = difference(bootstrap, run, BASE_SEEDS[0])
        rows.append([label, *row_cells, f"{points(delta)} {interval(low, high)}"])
    tables.append(Table(
        "wyniki-weryfikacja",
        "Wyniki wariantów różniących się od konfiguracji bazowej jednym elementem na zbiorze walidacyjnym "
        "(ResNet-34, 60 epok, to samo ziarno, różnica mAP względem konfiguracji bazowej z~95-procentowym "
        "przedziałem ufności)",
        "lrrrrl", [["Zmieniony element", *METRIC_HEADER, "Różnica mAP [pp]"]], rows))

    def seeds_table(name: str, caption: str, seeds: tuple[str, ...]) -> Table:
        values = [scores(runs_dir, run) for run in seeds]
        rows = [[f"ziarno {run_config(runs_dir, run)['seed']}", *[pl(v) for v in row]]
                for run, row in zip(seeds, values, strict=True)]
        columns = list(zip(*values, strict=True))
        rows += [None, ["średnia", *[pl(statistics.mean(c)) for c in columns]],
                 ["odchylenie standardowe", *[pl(statistics.stdev(c)) for c in columns]]]
        return Table(name, caption, "lrrrr", [["Trening", *METRIC_HEADER]], rows)

    tables.append(seeds_table(
        "wyniki-rozrzut",
        "Rozrzut wyników między trzema treningami konfiguracji bazowej, zbiór walidacyjny", BASE_SEEDS))

    # final configuration on validation and test
    test = [tuple(float(json.loads((runs_dir / run / "eval_test.json").read_text())[m]) for m in METRICS)
            for run in FINAL_SEEDS]
    valid = [scores(runs_dir, run) for run in FINAL_SEEDS]
    rows = [[f"ziarno {run_config(runs_dir, run)['seed']}", *[pl(v) for v in v_row], *[pl(v) for v in t_row]]
            for run, v_row, t_row in zip(FINAL_SEEDS, valid, test, strict=True)]
    columns = [*zip(*valid, strict=True), *zip(*test, strict=True)]
    rows += [None, ["średnia", *[bold(pl(statistics.mean(c))) for c in columns]],
             ["odch. std.", *[pl(statistics.stdev(c)) for c in columns]]]
    tables.append(Table(
        "wyniki-koncowa",
        "Wyniki konfiguracji końcowej (ResNet-34, strata trójkowa, PK-SA-BH, AUG-STRONG, 60 epok) "
        "na zbiorze walidacyjnym i~testowym", "lrrrrrrrr",
        [["", "\\multicolumn{4}{c}{Zbiór walidacyjny}", "\\multicolumn{4}{c}{Zbiór testowy}"],
         ["Trening", *METRIC_HEADER, *METRIC_HEADER]], rows, size="\\footnotesize"))

    # re-ranking: mean and standard deviation over the three final trainings
    methods = (("bez re-rankingu", None), ("k-reciprocal", "k_reciprocal"),
               ("normalizacja po zapytaniach", "dual_softmax"), ("obie metody", "both"))
    rows = []
    for split, label, file_name, key in (("valid", "walidacyjny", "rerank_valid.json", "best"),
                                         ("test", "testowy", "rerank_test.json", "methods")):
        files = [json.loads((runs_dir / run / file_name).read_text()) for run in FINAL_SEEDS]
        baseline = statistics.mean(f["baseline"]["mAP"] for f in files)
        block = []
        for method_label, method in methods:
            per_seed = [f["baseline"] if method is None else f[key][method]["metrics"] for f in files]
            mean_map, sd_map = mean_sd([m["mAP"] for m in per_seed])
            mean_r1, sd_r1 = mean_sd([m["rank-1"] for m in per_seed])
            block.append((method_label, mean_map, sd_map, mean_r1, sd_r1,
                          statistics.mean(m["rank-5"] for m in per_seed),
                          statistics.mean(m["rank-10"] for m in per_seed)))
        top = max(b[1] for b in block)
        for k, (method_label, mean_map, sd_map, mean_r1, sd_r1, r5, r10) in enumerate(block):
            emphasise = bold if mean_map == top else (lambda text: text)
            rows.append([
                label if k == 0 else "", method_label,
                emphasise(pl(mean_map)) + f" $\\pm$ {pl(sd_map)}", pl(mean_r1) + f" $\\pm$ {pl(sd_r1)}",
                pl(r5), pl(r10), DASH if k == 0 else points(mean_map - baseline),
            ])
        if split == "valid":
            rows.append(None)
    tables.append(Table(
        "wyniki-reranking",
        "Wyniki re-rankingu dla konfiguracji końcowej, średnia i~odchylenie standardowe z~trzech treningów "
        "(parametry dobrane na zbiorze walidacyjnym i~zastosowane bez zmian na zbiorze testowym)",
        "llrrrrr", [["Zbiór", "Wariant", "mAP", "Rank-1", "Rank-5", "Rank-10", "Zysk mAP [pp]"]], rows,
        size="\\footnotesize"))

    # embedding size
    final_mean = tuple(statistics.mean(col) for col in zip(*valid, strict=True))
    by_size = [(64, scores(runs_dir, "G6_D64")), (128, scores(runs_dir, "G6_D128")), (256, scores(runs_dir, "G6_D256")),
               (512, final_mean), (1024, scores(runs_dir, "G6_D1024"))]
    cells = metric_cells([v for _, v in by_size])
    tables.append(Table(
        "wyniki-wymiar",
        "Wynik konfiguracji końcowej na zbiorze walidacyjnym w~zależności od wymiaru embeddingu "
        "(wiersz dla wymiaru 512 jest średnią z~trzech treningów, pozostałe pochodzą z~jednego treningu)",
        "rrrrr", [["Wymiar embeddingu", *METRIC_HEADER]],
        [[num(size), *c] for (size, _), c in zip(by_size, cells, strict=True)]))

    # the challenge leaderboard next to the final configuration; a different split, so only for orientation
    test_files = [json.loads((runs_dir / run / "rerank_test.json").read_text()) for run in FINAL_SEEDS]
    plain = [f["baseline"] for f in test_files]
    reranked = [f["methods"]["both"]["metrics"] for f in test_files]
    rows = [[team.replace("_", "\\_"), pl(m_ap, 2), pl(rank1, 2)] for team, m_ap, rank1 in CHALLENGE_2023]
    rows.append(None)
    for label, per_seed in (("konfiguracja końcowa", plain), ("konfiguracja końcowa z~re-rankingiem", reranked)):
        rows.append([label, pl(100 * statistics.mean(m["mAP"] for m in per_seed), 2),
                     pl(100 * statistics.mean(m["rank-1"] for m in per_seed), 2)])
    tables.append(Table(
        "wyniki-konkursu",
        "Wyniki konkursu SoccerNet Re-Identification 2023 na zbiorze konkursowym oraz wynik konfiguracji końcowej "
        "na zbiorze testowym, średnia z~trzech treningów (wartości w~procentach)",
        "lrr", [["Rozwiązanie", "mAP", "Rank-1"]], rows))
    return tables


# --------------------------------------------------------------------------- appendix

def appendix_table(runs: dict[str, dict], runs_dir: Path) -> Table:
    rows: list[list[str] | None] = []
    for heading, names in STAGES:
        if rows:
            rows.append(None)
        rows.append([f"\\multicolumn{{8}}{{l}}{{{bold(heading)}}}"])
        for name in names:
            config = run_config(runs_dir, name)
            epoch, values = best_scores(runs_dir, name)
            planned = config["num_epochs"]
            done = len(runs[name]["train_loss_epoch"]) if name in runs else logged_evals(runs_dir, name)[-1][0]
            epochs = str(planned) if done == planned else f"{done} z {planned}"
            rows.append([tt(name), describe(config), epochs, str(epoch), *[DASH if v is None else pl(v) for v in values]])
    return Table(
        "zestawienie-treningow",
        "Wszystkie treningi wykonane w~ramach pracy: konfiguracja, liczba epok, epoka o~najwyższym mAP "
        "na zbiorze walidacyjnym i~wynik z~tej epoki",
        "l" + par(0.29) + "rrrrrr",
        [["Identyfikator", "Konfiguracja", "Epoki", "Najlepsza", "mAP", "Rank-1", "Rank-5", "Rank-10"]],
        rows, size="\\footnotesize", long=True)


# --------------------------------------------------------------------------- entry point

GROUPS = ("dataset", "method", "results", "appendix")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=PROJECT_ROOT / "outputs" / "_g")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--speed", type=Path, default=PROJECT_ROOT / "outputs" / "_g" / "speed.json")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "tables")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()

    tables: list[Table] = []
    if "dataset" in args.only:
        import pyarrow.dataset  # noqa: F401  (before anything that may load torch, see scripts/eval_checkpoint.py)

        from soccernet_reid.data.catalog import load_catalog

        tables += dataset_tables(load_catalog(args.catalog))
    speed = json.loads(args.speed.read_text())
    if "method" in args.only:
        tables += method_tables(speed)
    runs = load_reviews(args.reviews)
    if "results" in args.only:
        tables += result_tables(runs, args.runs, load_bootstrap(args.reviews), speed)
    if "appendix" in args.only:
        tables.append(appendix_table(runs, args.runs))

    args.out.mkdir(parents=True, exist_ok=True)
    for table in tables:
        (args.out / f"{table.name}.tex").write_text(render(table), encoding="utf-8")
        print(f"wrote {args.out / table.name}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
