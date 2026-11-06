"""Draw the explanatory figures of the thesis: the task, one action, the model, the samplers.

Unlike scripts/make_thesis_figures.py these show no results. Three of them show crops
of the dataset; which crops is fixed by the arguments below, so a figure can be redrawn.

Figures (``--only`` picks a subset; each is saved as .pdf and .png)
    task       one query and the gallery of its action, before and after ranking by the final model
    action     all queries and gallery crops of one validation action, matches marked by numbers
    model      block diagram: crop -> backbone -> pooling -> projection head -> embedding -> use
    samplers   one batch drawn by the PK sampler and one by the PK-SA sampler, as in training
    normalization   a synthetic numerical example of query-wise normalization (no dataset)

Inputs
    outputs/catalog.parquet and the dataset images
    outputs/runs/G5_STRONG/embeddings_valid.npz   (only for ``task``)

Usage
-----
    python scripts/make_thesis_schematics.py --out outputs/figures
    python scripts/make_thesis_schematics.py --only model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_thesis_figures import BLUE, GRID, INK, INK_SECONDARY, _save  # noqa: E402

GREEN, RED = "#1a9641", "#d7191c"          # the frame colours of scripts/visualize_retrieval.py
FILL = "#f3f2ee"

# The example of ``task``: a validation query whose action has a gallery of eight crops, two of
# them of the same person, both ranked first by the final model (the only such query among
# those with every crop at least 90 pixels high).
TASK_RUN, TASK_QUERY = "G5_STRONG", 10601
# The example of ``action``: a validation action of the median size (5 queries, 18 gallery crops).
ACTION = 6


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


def _blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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


def draw_action(catalog, out_dir: Path, per_row: int = 9) -> None:
    import matplotlib.pyplot as plt

    queries, gallery = action_example(catalog)
    gallery_rows = -(-len(gallery) // per_row)
    fig = plt.figure(figsize=(0.74 * per_row, 1.95 * (1 + gallery_rows)))
    grid = fig.add_gridspec(1 + gallery_rows, per_row, hspace=0.42, wspace=0.12)
    for c, path in enumerate(queries):
        ax = fig.add_subplot(grid[0, c])
        _show(ax, path, BLUE, 3.0, str(c + 1))
        if c == 0:
            ax.set_title("zapytania", fontsize=9, color=INK, loc="left")
    for i, (path, match) in enumerate(gallery):
        ax = fig.add_subplot(grid[1 + i // per_row, i % per_row])
        _show(ax, path, BLUE if match else GRID, 3.0 if match else 1.0, str(match) if match else "")
        if i == 0:
            ax.set_title("galeria", fontsize=9, color=INK, loc="left")
    _save(fig, out_dir, "przykladowa_akcja")
    plt.close(fig)


def draw_model(catalog, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fill: str = FILL, edge: str = INK_SECONDARY) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                    edgecolor=edge, linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK, linespacing=1.35)

    def arrow(points: list[tuple[float, float]]) -> None:
        """A line through the points with an arrow head at the last one."""
        xs, ys = zip(*points[:-1], strict=True) if len(points) > 2 else ((), ())
        if len(points) > 2:
            ax.plot(xs, ys, color=INK_SECONDARY, linewidth=0.9, solid_capstyle="butt")
        ax.annotate("", xy=points[-1], xytext=points[-2], arrowprops={
            "arrowstyle": "-|>", "color": INK_SECONDARY, "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0})

    query = action_example(catalog)[0][0]
    with Image.open(query) as im:
        ax.imshow(im.convert("RGB").resize((128, 256)), extent=(1, 8, 23, 37), aspect="auto", zorder=2)
    ax.text(4.5, 20.5, "wycinek\n256 × 128", ha="center", va="top", fontsize=8, color=INK, linespacing=1.35)

    top, height = 24, 12
    blocks = [(11, 17, "ekstraktor cech\n(sieć splotowa)"), (31, 14, "uśrednienie\nmapy cech"),
              (48, 14, "głowa\nprojekcyjna\n(BN, FC, BN)"), (65, 16, "normalizacja\nL2")]
    for x, w, text in blocks:
        box(x, top, w, height, text)
    box(84, top, 15, height, "embedding\n(512 liczb)", fill="#dce9f9", edge=BLUE)
    middle = top + height / 2
    for start, end in ((8, 11), (28, 31), (45, 48), (62, 65), (81, 84)):
        arrow([(start + 0.4, middle), (end - 0.4, middle)])

    box(46, 3, 22, 12, "trening:\nfunkcja straty\nliczona na batchu")
    box(72, 3, 27, 12, "ocena:\nranking wycinków galerii\ntej samej akcji")
    arrow([(95, top), (95, 15.4)])
    arrow([(89, top), (89, 19.5), (57, 19.5), (57, 15.4)])
    _save(fig, out_dir, "schemat_modelu")
    plt.close(fig)


def sampler_batches(catalog, seed: int = 0) -> tuple[list[str], list[str]]:
    """Paths of the first batch of the PK sampler (16 x 2) and of the PK-SA sampler (8 x 2)."""
    import pyarrow.dataset  # noqa: F401  (before torch, see scripts/eval_checkpoint.py)

    from soccernet_reid.data.catalog import assign_class_ids, filter_to_player_classes
    from soccernet_reid.samplers.pk import PKBatchSampler, PKPerActionBatchSampler

    train = assign_class_ids(filter_to_player_classes(catalog[catalog["split"] == "train"].copy())).reset_index(drop=True)
    classes, actions, paths = train["class_id"].tolist(), train["action_idx"].tolist(), train["path"].tolist()
    pk = next(iter(PKBatchSampler(classes, P=16, K=2, num_batches=1, seed=seed)))
    pk_sa = next(iter(PKPerActionBatchSampler(classes, actions, P=8, K=2, num_batches=1, seed=seed)))
    return [paths[i] for i in pk], [paths[i] for i in pk_sa]


def draw_samplers(catalog, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pk, pk_sa = sampler_batches(catalog)
    fig = plt.figure(figsize=(0.46 * 16, 4.5))
    grid = fig.add_gridspec(5, 16, height_ratios=[1, 1, 0.36, 1, 1], hspace=0.06, wspace=0.10)
    for batch, first_row, title in (
        (pk, 0, "PK: 16 klas po 2 wycinki, klasy z dowolnych akcji"),
        (pk_sa, 3, "PK-SA: 8 klas po 2 wycinki, wszystkie klasy z jednej akcji"),
    ):
        for i, path in enumerate(batch):           # the sampler yields the two crops of a class one after the other
            ax = fig.add_subplot(grid[first_row + i % 2, i // 2])
            _show(ax, path)
            if i == 0:
                ax.set_title(title, fontsize=9, color=INK, loc="left")
    _save(fig, out_dir, "samplery_pk")
    plt.close(fig)


def draw_query_normalization(out_dir: Path) -> None:
    """Illustrate the implemented weighting on synthetic, realizable cosine similarities.

    No identity labels or retrieval-quality claims are assigned to this example.
    Its five unit vectors can be embedded in any feature space of dimension >= 3.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

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

    # Match the neutral outlines and pastel fills of the chapter-4 model diagram.
    ink, grey, blue, green = "#26323D", "#F2F4F6", "#E5EEF5", "#E6F0E9"
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 620), ylim=(0, 300))
    ax.axis("off")

    def text(x: float, y: float, value: str, size: float = 10,
             bold: bool = False) -> None:
        ax.text(x, y, value, ha="center", va="center", color=ink, fontsize=size,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def box(x: float, y: float, w: float, h: float, value: str,
            fill: str = grey, size: float = 11) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=3", facecolor=fill,
                     edgecolor=ink, linewidth=0.8))
        text(x + w / 2, y + h / 2, value, size)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={
                    "arrowstyle": "-|>", "color": ink, "linewidth": 1.0,
                    "shrinkA": 0, "shrinkB": 0})

    def matrix(x: float, values: np.ndarray, title: str, formula: str,
               digits: int, fill: str) -> None:
        text(x + 87, 257, title, 11, True)
        text(x + 87, 234, formula, 11)
        for j in range(3):
            text(x + 55 + 47 * j, 211, f"$g_{j + 1}$", 10)
        for i in range(2):
            y = 171 - 33 * i
            text(x + 13, y + 15, f"$q_{i + 1}$", 10, i == 0)
            for j in range(3):
                value = f"{values[i, j]:.{digits}f}".replace(".", ",")
                box(x + 32 + 47 * j, y, 46, 30, value, fill if i == 0 else grey)

    text(310, 289, "Przykład poglądowy dla jednej akcji", 12, True)
    matrix(7, similarity, "1. Podobieństwa", "$s_{ij}$", 1, blue)
    matrix(219, shares, "2. Udziały zapytań", "$p_{ij}(T),\\quad T=0{,}1$", 3, blue)
    matrix(431, scores, "3. Końcowe wartości", "$S_{\\mathrm{N}}(i,j)=\\frac{s_{ij}+1}{2}p_{ij}(T)$", 3, green)
    arrow(187, 184, 213, 184)
    arrow(399, 184, 425, 184)
    text(94, 115, "Wiersze: zapytania\nKolumny: wycinki galerii", 9)
    text(306, 115, "Softmax w każdej kolumnie\nSuma udziałów w kolumnie: 1", 9)
    text(518, 115, "Wyniki ważenia podobieństw\nwartościami udziałów", 9)
    ax.plot([7, 611], [91, 91], color="#BCC4CB", linewidth=0.8)
    text(310, 76, "Kolejność dla zapytania $q_1$ (sortowanie malejąco)", 11, True)
    text(148, 54, "Przed normalizacją", 10)
    text(472, 54, "Po normalizacji", 10)
    for order, left, fill in ((before, 35, blue), (after, 359, green)):
        for rank, index in enumerate(order):
            box(left + 77 * rank, 8, 70, 31, f"$g_{index + 1}$", fill, 12)
    arrow(280, 23, 340, 23)
    _save(fig, out_dir, "normalizacja_po_zapytaniach")
    plt.close(fig)


FIGURES = ("task", "action", "model", "samplers", "normalization")


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
    if "normalization" in args.only:
        draw_query_normalization(args.out)
    if not any(name in args.only for name in ("task", "action", "model", "samplers")):
        return 0

    import pyarrow.dataset  # noqa: F401  (before anything that may load torch)

    from soccernet_reid.data.catalog import load_catalog

    catalog = load_catalog(args.catalog)
    if "task" in args.only:
        draw_task(args.runs, catalog, args.out)
    if "action" in args.only:
        draw_action(catalog, args.out)
    if "model" in args.only:
        draw_model(catalog, args.out)
    if "samplers" in args.only:
        draw_samplers(catalog, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
