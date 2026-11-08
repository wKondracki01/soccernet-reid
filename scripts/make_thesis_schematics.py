"""Draw the explanatory figures of the thesis: the task, one action, preprocessing, the model,
the samplers and query-wise normalization.

Unlike scripts/make_thesis_figures.py these show no results. Two of them show crops of the
dataset; which crops is fixed by the arguments below, so a figure can be redrawn.

Figures (``--only`` picks a subset; each is saved as .pdf and .png)
    task            one query and the gallery of its action, before and after ranking by the final model
    action          the queries and the full gallery of one validation action, matches marked by letters
    preprocessing   the preparation of a crop for the network, with the training-only operations
    model           how crops pass through the model in training and in evaluation
    samplers        the composition of one PK and one PK-SA batch
    normalization   a synthetic numerical example of query-wise normalization (no dataset)

The figures of chapters 3 and 4 (all but ``task``) share one look: DejaVu Sans, slate lines,
pastel fills, and a dashed line for what happens only during training. Each of them is drawn at
the size it is printed at, so a 9-point label prints at 9 points; the LaTeX width to use is given
in the docstring of each function.

Inputs
    outputs/catalog.parquet and the dataset images   (only for ``task`` and ``action``)
    outputs/runs/G5_STRONG/embeddings_valid.npz      (only for ``task``)

Usage
-----
    python scripts/make_thesis_schematics.py --out outputs/figures
    python scripts/make_thesis_schematics.py --only model samplers
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_thesis_figures import BLUE, GRID, INK, ORANGE, _save  # noqa: E402

GREEN, RED = "#1a9641", "#d7191c"          # the frame colours of scripts/visualize_retrieval.py

# The example of ``task``: a validation query whose action has a gallery of eight crops, two of
# them of the same person, both ranked first by the final model (the only such query among
# those with every crop at least 90 pixels high).
TASK_RUN, TASK_QUERY = "G5_STRONG", 10601
# The example of ``action``: a validation action with two queries, each with one match in a
# gallery of six crops.
ACTION = 1697

# The common look of the chapter 3 and 4 figures.
TEXT_WIDTH_PT = 154 / 25.4 * 72            # the text block of the EiTI template: A4, 25 mm margins, 6 mm binding
SLATE = "#26323D"                          # text, outlines and arrows
FILL_GREY, FILL_BLUE, FILL_GREEN = "#F2F4F6", "#E5EEF5", "#E6F0E9"
FILL_TRAIN = "#F6F1E5"                     # boxes of training-only steps (with a dashed outline)
RULE, MUTED = "#BCC4CB", "#8A949E"         # separator lines; crops without a match
SIZE_TEXT, SIZE_NOTE, SIZE_TITLE = 9, 8, 10
LINE = 0.8                                 # points
DASH = (0, (4.5, 3))                       # in units of the line width
ROUND, HEAD_LENGTH, HEAD_HALF_WIDTH = 3, 5, 2.2   # points
STYLE = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42}


def _show(ax, path: str, colour: str = GRID, width: float = 1.0, label: str = "") -> None:
    """A crop at the network's input size with a frame and an optional label below."""
    from PIL import Image

    with Image.open(path) as im:
        ax.imshow(im.convert("RGB").resize((128, 256)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(width)
    ax.set_xlabel(label, fontsize=8, labelpad=2, color=INK)


def task_example(runs_dir: Path, catalog, run: str = TASK_RUN, query: int = TASK_QUERY) -> tuple[dict, list[str]]:
    """(ranked query, gallery paths in the order of the dataset) of the example query."""
    import numpy as np
    from visualize_retrieval import pick_queries, rank_queries

    with np.load(runs_dir / run / "embeddings_valid.npz", allow_pickle=False) as data:
        emb = {k: data[k] for k in data.files}
    row = pick_queries(rank_queries(emb, catalog, "valid"), [query])[0]
    valid = catalog[(catalog["split"] == "valid") & (catalog["role"] == "gallery")]
    unordered = valid[valid["action_idx"] == row["action"]].sort_values("bbox_idx")["path"].tolist()
    return row, unordered


def draw_task(runs_dir: Path, catalog, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    row, unordered = task_example(runs_dir, catalog)
    n = len(unordered)
    fig = plt.figure(figsize=(0.78 * (n + 1.5), 3.9))
    grid = fig.add_gridspec(2, n + 2, width_ratios=[1, 0.5, *([1] * n)], hspace=0.38, wspace=0.12)
    query_ax = fig.add_subplot(grid[:, 0])
    _show(query_ax, row["path"], BLUE, 3.0, "zapytanie")
    for c, path in enumerate(unordered):
        ax = fig.add_subplot(grid[0, c + 2])
        _show(ax, path)
        if c == 0:
            ax.set_title("galeria akcji", fontsize=9, color=INK, loc="left")
    for c, item in enumerate(row["gallery"]):
        ax = fig.add_subplot(grid[1, c + 2])
        _show(ax, item["path"], GREEN if item["correct"] else RED, 3.0, str(c + 1))
        if c == 0:
            ax.set_title("galeria uporządkowana według podobieństwa do zapytania", fontsize=9, color=INK, loc="left")
    _save(fig, out_dir, "schemat_zadania")
    plt.close(fig)


def action_example(catalog, action: int = ACTION) -> tuple[list[str], list[tuple[str, int | None]]]:
    """(query paths, gallery crops as (path, 1-based number of the matching query or None))."""
    part = catalog[(catalog["split"] == "valid") & (catalog["action_idx"] == action)]
    queries = part[part["role"] == "query"].sort_values("bbox_idx")
    gallery = part[part["role"] == "gallery"].sort_values("bbox_idx")
    number = {int(person): i + 1 for i, person in enumerate(queries["person_uid"])}
    return queries["path"].tolist(), [(path, number.get(int(person))) for path, person in
                                      zip(gallery["path"], gallery["person_uid"], strict=True)]


class _Sketch:
    """A figure printed ``printed_width`` points wide, drawn in the caller's own units.

    Positions are given in drawing units (``width`` of them across the figure); text sizes,
    line widths, rounding and arrow heads are given in printed points.
    """

    def __init__(self, width: float, height: float, printed_width: float = TEXT_WIDTH_PT) -> None:
        import matplotlib.pyplot as plt

        self.unit = printed_width / width                  # printed points per drawing unit
        self.fig = plt.figure(figsize=(printed_width / 72, height * self.unit / 72))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, width), ylim=(0, height))
        self.ax.axis("off")

    def text(self, x: float, y: float, value: str, size: float = SIZE_TEXT, bold: bool = False,
             colour: str = SLATE, ha: str = "center", background: bool = False) -> None:
        self.ax.text(x, y, value, ha=ha, va="center_baseline", fontsize=size, color=colour,
                     fontweight="bold" if bold else "normal",
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 1} if background else None)

    def box(self, x: float, y: float, w: float, h: float, lines: tuple[str, ...] = (), title: str = "",
            fill: str = FILL_GREY, dashed: bool = False, size: float = SIZE_TEXT) -> None:
        """A rounded box with an optional bold title above plain lines, all centred."""
        from matplotlib.patches import FancyBboxPatch

        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={ROUND / self.unit}",
                                         facecolor=fill, edgecolor=SLATE, linewidth=LINE,
                                         linestyle=DASH if dashed else "solid"))
        rows = ([(title, True)] if title else []) + [(line, False) for line in lines]
        step = size * 1.35 / self.unit
        top = y + h / 2 + (len(rows) - 1) * step / 2
        for i, (value, bold) in enumerate(rows):
            self.text(x + w / 2, top - i * step, value, size, bold)

    def arrow(self, points: list[tuple[float, float]], dashed: bool = False) -> None:
        """A line through the points with a filled head at the last one."""
        from matplotlib.patches import Polygon

        (x0, y0), (x1, y1) = points[-2], points[-1]
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        head, half = HEAD_LENGTH / self.unit, HEAD_HALF_WIDTH / self.unit
        bx, by = x1 - head * ux, y1 - head * uy             # the line stops where the head begins
        xs = [p[0] for p in points[:-1]] + [bx]
        ys = [p[1] for p in points[:-1]] + [by]
        self.ax.plot(xs, ys, color=SLATE, linewidth=LINE, linestyle=DASH if dashed else "solid",
                     solid_capstyle="butt", dash_capstyle="butt")
        self.ax.add_patch(Polygon([(x1, y1), (bx - half * uy, by + half * ux), (bx + half * uy, by - half * ux)],
                                  closed=True, facecolor=SLATE, edgecolor="none"))

    def rule(self, x0: float, x1: float, y: float) -> None:
        self.ax.plot([x0, x1], [y, y], color=RULE, linewidth=LINE)

    def save(self, out_dir: Path, name: str) -> None:
        """At the drawn size: a tight bounding box would change the printed size of the text."""
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            self.fig.savefig(out_dir / f"{name}.{ext}", dpi=300, facecolor="white")
        plt.close(self.fig)
        print(f"wrote {out_dir / name}.pdf / .png")


def draw_action(catalog, out_dir: Path) -> None:
    """The queries of one action and its full gallery; a letter marks the query a crop matches.

    LaTeX width: 0.8\\linewidth.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    queries, gallery = action_example(catalog, ACTION)
    colours = [BLUE, ORANGE]
    # In points, from the bottom up. Each title sits the same distance above its row of cells (the
    # tallest crops of both rows fill their cells); the wider gap above the gallery title keeps it
    # with the row below it.
    cell_w, cell_h, title_gap, letter_gap, section_gap = 52, 80, 15, 10, 26
    gallery_bottom = 18
    gallery_title = gallery_bottom + cell_h + title_gap
    query_bottom = gallery_title + section_gap + letter_gap
    query_title = query_bottom + cell_h + title_gap
    with plt.rc_context(STYLE):
        width, height = 0.8 * TEXT_WIDTH_PT, query_title + 8
        s = _Sketch(width, height, printed_width=width)

        def crop(path: str, centre: float, bottom: float, number: int | None) -> None:
            """The crop with its proportions kept, fitted into its cell, its letter below the cell."""
            colour = MUTED if number is None else colours[(number - 1) % len(colours)]
            with Image.open(path) as im:
                image = im.convert("RGB")
            scale = min(cell_w / image.width, cell_h / image.height)
            w, h = image.width * scale, image.height * scale
            x0, y0 = centre - w / 2, bottom + (cell_h - h) / 2
            s.ax.imshow(image, extent=(x0, x0 + w, y0, y0 + h), aspect="auto", interpolation="antialiased", zorder=1)
            s.ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=colour, linewidth=1.5, zorder=2))
            s.text(centre, bottom - letter_gap, "-" if number is None else chr(64 + number), SIZE_NOTE, True, colour)

        # The titles only name the two rows of crops, so they are a point smaller than the headings
        # of the diagrams.
        s.text(width / 2, query_title, "Zapytania z klatki akcji", SIZE_TEXT, True)
        for i, path in enumerate(queries):
            crop(path, width / 2 + (i - (len(queries) - 1) / 2) * 66, query_bottom, i + 1)
        s.text(width / 2, gallery_title, "Pełna galeria tej samej akcji", SIZE_TEXT, True)
        step = (width - 8) / len(gallery)
        for i, (path, number) in enumerate(gallery):
            crop(path, 4 + step * (i + 0.5), gallery_bottom, number)
        s.save(out_dir, "zapytania-galeria-akcja")


def draw_preprocessing(out_dir: Path) -> None:
    """The preparation of a crop: the evaluation path in solid lines, training-only steps dashed.

    LaTeX width: \\linewidth.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        s = _Sketch(425, 262)
        mx, mw, sx, sw = 230, 185, 9, 181
        mid, side = mx + mw / 2, sx + sw / 2
        s.text(side, 250, "Tylko podczas uczenia", SIZE_TITLE, True)
        s.text(side, 237, "(zależnie od wariantu)", SIZE_NOTE)
        s.text(mid, 250, "Przygotowanie wejścia", SIZE_TITLE, True)
        # The boxes carry bold titles of their own, so their text is a point smaller than in the
        # other diagrams: the titles then stay clearly below the 10-point column headings.

        def main_box(y: float, h: float, title: str, *lines: str) -> None:
            s.box(mx, y, mw, h, lines, title, size=SIZE_NOTE)

        def training_box(y: float, h: float, title: str, *lines: str) -> None:
            s.box(sx, y, sw, h, lines, title, FILL_TRAIN, dashed=True, size=SIZE_NOTE)

        main_box(198, 34, "Wczytanie obrazu jako RGB", "3 kanały, wartości od 0 do 255")
        main_box(153, 33, "Zmiana rozmiaru", "wysokość 256, szerokość 128")
        main_box(97, 44, "Przeliczenie wartości", "dzielenie przez 255", "liczby zmiennoprzecinkowe w [0, 1]")
        main_box(53, 32, "Normalizacja każdego kanału", r"$z = (x - \mu)\,/\,\sigma$")
        main_box(8, 33, "Wejście modelu", "3 kanały × 256 × 128")
        training_box(145, 37, "Losowe przekształcenia", "np. odbicie lub zmiana barw")
        training_box(44, 33, "Losowe wymazywanie", "fragmentu obrazu")
        for top, bottom in ((198, 186), (153, 141), (97, 85), (53, 41)):
            s.arrow([(mid, top), (mid, bottom)])
        s.arrow([(mx, 169), (210, 169), (210, 163.5), (sx + sw, 163.5)], dashed=True)
        s.arrow([(side, 145), (side, 130), (211, 130), (211, 119), (mx, 119)], dashed=True)
        s.arrow([(mx, 69), (210, 69), (210, 60.5), (sx + sw, 60.5)], dashed=True)
        s.arrow([(side, 44), (side, 24.5), (mx, 24.5)], dashed=True)
        s.save(out_dir, "przetwarzanie-wycinkow")


def draw_model(out_dir: Path) -> None:
    """How crops pass through the model in training and in evaluation; training-only steps dashed.

    LaTeX width: \\linewidth.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        s = _Sketch(600, 675)
        s.ax.plot([15, 52], [655, 655], color=SLATE, linewidth=LINE, linestyle=DASH, dash_capstyle="butt")
        s.text(60, 655, "tylko podczas uczenia", SIZE_NOTE, ha="left")
        s.box(205, 615, 310, 34, ("Dobór wycinków do partii uczącej",), fill=FILL_TRAIN, dashed=True)
        s.arrow([(360, 615), (360, 594)], dashed=True)
        s.box(205, 550, 310, 44, ("Przygotowanie wycinków", "z losowymi augmentacjami"), fill=FILL_TRAIN, dashed=True)
        s.arrow([(360, 550), (360, 494)], dashed=True)
        s.box(15, 550, 155, 44, ("Wycinki", "do oceny"))
        s.arrow([(92.5, 550), (92.5, 472), (125, 472)])
        s.text(160, 523, "przygotowanie", SIZE_NOTE)
        s.text(160, 508, "bez augmentacji", SIZE_NOTE)
        s.box(125, 450, 350, 44, ("Wejście modelu", "wycinek RGB, 256 × 128 pikseli"))
        s.text(155, 409, "Uczenie metryki i ArcFace", SIZE_TITLE, True)
        s.text(445, 409, "Entropia krzyżowa", SIZE_TITLE, True)
        s.arrow([(285, 450), (285, 385), (155, 385), (155, 362)])
        s.arrow([(315, 450), (315, 385), (445, 385), (445, 362)])
        for x in (35, 325):
            s.box(x, 314, 240, 48, ("Ekstraktor cech", "i uśrednianie przestrzenne"))
            s.arrow([(x + 120, 314), (x + 120, 294)])
        s.box(35, 232, 240, 62, ("Głowa projekcyjna", "BN → FC(512) → BN → L2"), fill=FILL_BLUE)
        s.box(325, 232, 240, 62, ("Cechy", "bez dodatkowej projekcji"), fill=FILL_BLUE)
        for x in (35, 325):
            s.arrow([(x + 120, 232), (x + 120, 209)])
            s.box(x, 177, 240, 32, ("Wektor 512-elementowy",), fill=FILL_GREEN)
            s.arrow([(x + 120, 177), (x + 120, 137)], dashed=True)
            s.text(x + 156, 157, "uczenie", SIZE_NOTE)
        s.box(35, 77, 240, 60, ("Strata metryczna", "albo klasyfikator ArcFace"), fill=FILL_TRAIN, dashed=True)
        s.box(325, 77, 240, 60, ("Klasyfikator treningowy", "i entropia krzyżowa"), fill=FILL_TRAIN, dashed=True)
        # Evaluation takes the vectors before the training-only losses and classifiers.
        s.arrow([(275, 193), (296, 193), (296, 51)])
        s.arrow([(325, 193), (304, 193), (304, 51)])
        s.text(300, 64, "wyszukiwanie", SIZE_NOTE, background=True)
        s.box(143, 8, 314, 43, ("L2 i ranking podobieństwa", "podczas oceny"), fill=FILL_GREEN)
        s.save(out_dir, "metodyka-model")


def draw_samplers(out_dir: Path) -> None:
    """The composition of one PK batch (classes from any actions) and one PK-SA batch (one action).

    LaTeX width: \\linewidth.
    """
    import matplotlib.pyplot as plt

    fills = ("#D9E8F5", "#E0EBD7", "#F5E1CC", "#E9DDF0")     # one per action: A, B, C, D
    with plt.rc_context(STYLE):
        s = _Sketch(600, 290)
        s.text(300, 270, "PK: 16 klas × 2 wycinki = 32 wycinki", SIZE_TITLE, True)
        for i in range(16):
            x, action = 29 + i * 34, i % 4
            s.text(x + 15, 241, f"c{i + 1}", SIZE_NOTE)
            for row in range(2):
                s.box(x, 203 - row * 31, 30, 27, ("ABCD"[action],), fill=fills[action])
        s.text(300, 152, "Klasy mogą pochodzić z różnych akcji", SIZE_TEXT)
        s.rule(25, 575, 137)
        s.text(300, 118, "PK-SA: 8 klas × 2 wycinki = 16 wycinków", SIZE_TITLE, True)
        for i in range(8):
            x = 55 + i * 62
            s.text(x + 20, 93, f"c{i + 1}", SIZE_NOTE)
            for row in range(2):
                s.box(x, 55 - row * 30, 40, 26, ("A",), fill=fills[0])
        s.text(300, 9, "Wszystkie klasy pochodzą z jednej wylosowanej akcji A", SIZE_TEXT)
        s.save(out_dir, "metodyka-partie")


def draw_query_normalization(out_dir: Path) -> None:
    """Illustrate the implemented weighting on synthetic, realizable cosine similarities.

    No identity labels or retrieval-quality claims are assigned to this example.
    Its five unit vectors can be embedded in any feature space of dimension >= 3.
    LaTeX width: \\linewidth.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from soccernet_reid.eval.rerank import dual_softmax_scores, dual_softmax_shares

    temperature = 0.1
    queries = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(0.75), 0.0]])
    similarity = np.array([[0.8, 0.7, 0.4], [0.9, 0.2, 0.1]])
    gallery_x = similarity[0]
    gallery_y = (similarity[1] - 0.5 * gallery_x) / np.sqrt(0.75)
    gallery = np.column_stack((gallery_x, gallery_y,
                               np.sqrt(1.0 - gallery_x**2 - gallery_y**2)))
    np.testing.assert_allclose(np.linalg.norm(gallery, axis=1), 1.0)
    np.testing.assert_allclose(queries @ gallery.T, similarity)
    shares = dual_softmax_shares(similarity, temperature)
    scores = ((similarity + 1.0) / 2.0) * shares
    np.testing.assert_allclose(shares.sum(axis=0), 1.0)
    np.testing.assert_allclose(np.log(scores), dual_softmax_scores(similarity, temperature))
    before = np.argsort(-similarity[0], kind="stable")
    after = np.argsort(-scores[0], kind="stable")
    np.testing.assert_array_equal(before, [0, 1, 2])
    np.testing.assert_array_equal(after, [1, 2, 0])

    with plt.rc_context(STYLE):
        # The caption carries the title, so none is drawn here.
        s = _Sketch(620, 276)

        def matrix(x: float, values: np.ndarray, title: str, formula: str, digits: int, fill: str) -> None:
            # The display fraction in the third formula needs the extra height between rows.
            s.text(x + 87, 263, title, SIZE_TITLE, True)
            s.text(x + 87, 235, formula, SIZE_TEXT)
            for j in range(3):
                s.text(x + 55 + 47 * j, 210, f"$g_{j + 1}$", SIZE_TEXT)
            for i in range(2):
                y = 171 - 33 * i
                s.text(x + 13, y + 15, f"$q_{i + 1}$", SIZE_TEXT, i == 0)
                for j in range(3):
                    value = f"{values[i, j]:.{digits}f}".replace(".", ",")
                    s.box(x + 32 + 47 * j, y, 46, 30, (value,), fill=fill if i == 0 else FILL_GREY)

        matrix(7, similarity, "1. Podobieństwa", "$s_{ij}$", 1, FILL_BLUE)
        matrix(219, shares, "2. Udziały zapytań", "$p_{ij}(T),\\quad T=0{,}1$", 3, FILL_BLUE)
        matrix(431, scores, "3. Końcowe wyniki", "$S_{\\mathrm{N}}(i,j)=\\dfrac{s_{ij}+1}{2}\\,p_{ij}(T)$", 3, FILL_GREEN)
        s.arrow([(187, 184), (213, 184)])
        s.arrow([(399, 184), (425, 184)])
        for x, first, second in ((94, "Wiersze: zapytania", "Kolumny: wycinki galerii"),
                                 (306, "Softmax w każdej kolumnie", "Suma udziałów w kolumnie: 1"),
                                 (518, "Wyniki ważenia podobieństw", "wartościami udziałów")):
            s.text(x, 122, first, SIZE_NOTE)
            s.text(x, 108, second, SIZE_NOTE)
        s.rule(7, 611, 91)
        s.text(310, 76, "Kolejność dla zapytania $q_1$ (sortowanie malejąco)", SIZE_TITLE, True)
        s.text(148, 54, "Przed normalizacją", SIZE_TEXT)
        s.text(472, 54, "Po normalizacji", SIZE_TEXT)
        for order, left, fill in ((before, 35, FILL_BLUE), (after, 359, FILL_GREEN)):
            for rank, index in enumerate(order):
                s.box(left + 77 * rank, 8, 70, 31, (f"$g_{index + 1}$",), fill=fill, size=SIZE_TITLE)
        s.arrow([(280, 23), (340, 23)])
        s.save(out_dir, "normalizacja_po_zapytaniach")


FIGURES = ("task", "action", "preprocessing", "model", "samplers", "normalization")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    for name, draw in (("preprocessing", draw_preprocessing), ("model", draw_model),
                       ("samplers", draw_samplers), ("normalization", draw_query_normalization)):
        if name in args.only:
            draw(args.out)
    if not any(name in args.only for name in ("task", "action")):
        return 0

    import pyarrow.dataset  # noqa: F401  (before anything that may load torch)

    from soccernet_reid.data.catalog import load_catalog

    catalog = load_catalog(args.catalog)
    if "task" in args.only:
        draw_task(args.runs, catalog, args.out)
    if "action" in args.only:
        draw_action(catalog, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
