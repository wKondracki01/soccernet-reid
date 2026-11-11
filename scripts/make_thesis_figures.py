"""Draw the result figures of the thesis from the saved review files of series G.

Inputs (all written by earlier steps, nothing is trained or evaluated here)
    outputs/_g/REVIEW_*.json   per-run validation curves and best epochs
    outputs/_g/speed.json      scripts/benchmark_speed.py (only for ``cost``)
    outputs/runs/RUN/rerank_valid.json   scripts/eval_rerank.py (only for ``ladder``)

Figures (``--only`` picks a subset; each is saved as .pdf and .png)
    ladder      validation mAP after each step from the starting point to the final model
    curves      mAP against the epoch, five separate figures: augmentation on ResNet-18,
                cross-entropy with and without weight decay, final configuration against AUG-MED,
                Multi-Similarity and Circle in their two settings, and the two cross-entropy
                reference runs with their training loss
    dimension   validation mAP against the embedding size
    cost        validation mAP against the inference time of the backbone
    augment     the four augmentation presets applied to the same training crops
    cmc         share of queries answered within the first k results, for three models

Text on the figures is Polish, numbers use a decimal comma.

Usage
-----
    python scripts/make_thesis_figures.py --out outputs/figures
    python scripts/make_thesis_figures.py --only cost dimension
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Categorical slots 1-3 and the neutrals of the palette used for every figure.
# The three hues pass the colour-vision checks for adjacent and all-pairs use;
# aqua is light on white, so every series is also labelled at its line.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# The look of the chapter 3 and 4 schematics (scripts/make_thesis_schematics.py), used by the learning
# curves: dark slate text, thin lines and figures drawn at their printed size, so that an 8 pt label
# is 8 pt in the thesis. The text is one point smaller than on the schematics, at the author's request.
TEXT_WIDTH_IN = 154 / 25.4                 # the text block of the EiTI template
SLATE, RULE = "#26323D", "#BCC4CB"
SIZE_TEXT, SIZE_NOTE, SIZE_TITLE = 8, 7, 8.7
DASH = (0, (4.5, 3))                       # in units of the line width
# Series colours of the learning curves: muted blue, copper and teal in the spirit of the schematics
# (a 15% tint of the blue is close to their blue fill). Checked with the colour-vision and contrast
# checks of the dataviz palette validator against white: all pairs pass.
CURVE_1, CURVE_2, CURVE_3 = "#356fa8", "#cf6f3c", "#2c9b84"

FINAL_SEEDS = ("G5_STRONG", "G5_STRONG_S1", "G5_STRONG_S2")
BASE_SEEDS = ("G5_REF", "G5_REF_S1", "G5_REF_S2")


def pl(value: float, digits: int = 4) -> str:
    """A number with a decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


def load_reviews(review_dir: Path) -> dict[str, dict]:
    """All runs found in REVIEW_*.json; a later file wins if a run appears twice."""
    runs: dict[str, dict] = {}
    for path in sorted(review_dir.glob("REVIEW_*.json")):
        runs.update(json.loads(path.read_text()))
    return runs


def curve(run: dict) -> tuple[list[int], list[float]]:
    """(epochs, validation mAP) of one run."""
    return [int(e[0]) for e in run["evals"]], [float(e[1]) for e in run["evals"]]


def best_map(run: dict) -> float:
    return float(run["best"][1])


def mean_best(runs: dict[str, dict], names: tuple[str, ...]) -> float:
    return statistics.mean(best_map(runs[n]) for n in names)


def _style(ax) -> None:
    """Recessive axes and a solid hairline grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def _comma_axis(ax, digits: int = 2, step: float | None = None) -> None:
    """Decimal commas on the y axis; ``step`` puts the ticks on round values the labels can show."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pl(v, digits)))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_dir / name}.pdf / .png")


def ladder_steps(runs: dict[str, dict], runs_dir: Path) -> list[tuple[str, float]]:
    """(label, validation mAP) for every step of the ladder; each step changes one thing."""
    steps = [
        ("punkt wyjścia\nR18, PK, AUG-MIN", best_map(runs["G1_PK_BH"])),
        ("sampler\nPK-SA", best_map(runs["G1_PK_SA_BH"])),
        ("augmentacje\nAUG-MED", best_map(runs["G3_AUG_MED"])),
        ("sieć\nResNet-34", best_map(runs["G4_R34"])),
        ("AUG-STRONG,\n60 epok", mean_best(runs, FINAL_SEEDS)),
    ]
    reranked = []
    for name in FINAL_SEEDS:
        path = runs_dir / name / "rerank_valid.json"
        if path.exists():
            reranked.append(json.loads(path.read_text())["best"]["both"]["metrics"]["mAP"])
    if len(reranked) == len(FINAL_SEEDS):
        steps.append(("re-ranking", statistics.mean(reranked)))
    return steps


def draw_ladder(runs: dict[str, dict], runs_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    steps = ladder_steps(runs, runs_dir)
    xs, ys = list(range(len(steps))), [v for _, v in steps]
    width = 0.9 * TEXT_WIDTH_IN                  # printed at 0.9 of the text width
    fig, ax = plt.subplots(figsize=(width, 0.5 * width))
    _curve_style(ax, xlabel="", ylabel="mAP")
    ax.plot(xs, ys, **_measured(CURVE_1, len(xs)))
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if i == 0:   # the line leaves this point upwards: the label goes underneath
            ax.annotate(pl(y), (x, y), textcoords="offset points", xytext=(0, -8), ha="center",
                        va="top", fontsize=SIZE_NOTE, color=SLATE)
            continue
        text = f"{pl(y)}\n(+{pl(100 * (y - ys[i - 1]), 1)} pp)"
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                    va="bottom", fontsize=SIZE_NOTE, color=SLATE, linespacing=1.15)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _ in steps], fontsize=SIZE_NOTE, color=SLATE)
    ax.set_ylim(min(ys) - 0.03, max(ys) + 0.035)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    _comma_axis(ax, step=0.05)
    fig.tight_layout()
    _save(fig, out_dir, "narastajace_zyski")
    plt.close(fig)

# One file per learning-curve figure, so that each can sit in the section it belongs to.
CURVE_FIGURES = ("krzywe_augmentacja", "krzywe_klasyfikacja", "krzywe_konfiguracja_koncowa",
                 "krzywe_ustawienia_strat", "krzywe_odniesienie")


def _curve_style(ax, xlabel: str = "epoka", ylabel: str | None = "mAP",
                 title: str | None = None) -> None:
    """Axes of a learning-curve figure in the look of the chapter 3 and 4 schematics."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SLATE, labelsize=SIZE_NOTE, length=0)
    ax.set_xlabel(xlabel, fontsize=SIZE_TEXT, color=SLATE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=SIZE_TEXT, color=SLATE)
    if title:
        ax.set_title(title, fontsize=SIZE_TITLE, color=SLATE)


def _curve_axes(step: float, epochs: range, share: float = 0.75):
    """A figure of mAP against the epoch, printed at ``share`` of the text width; ``epochs`` are the
    x ticks, ``step`` the spacing of the y ticks."""
    import matplotlib.pyplot as plt

    width = share * TEXT_WIDTH_IN
    fig, ax = plt.subplots(figsize=(width, 0.62 * width))
    _curve_style(ax)
    _comma_axis(ax, step=step)
    ax.set_xticks(list(epochs))
    return fig, ax


def _measured(colour: str, points: int, linewidth: float = 0.8) -> dict:
    """Plot style of values measured at separate epochs: markers joined by a thin dashed line."""
    return dict(color=colour, linewidth=linewidth, linestyle=DASH, marker="o",
                markersize=4.2 if points <= 15 else 2.8, markeredgecolor="white", markeredgewidth=0.5)


BEST_LABEL = "wybrany model (najwyższe mAP)"


def _best_ring(markersize: float) -> dict:
    """An open circle drawn around the evaluation whose saved model gives the reported result."""
    return dict(linestyle="none", marker="o", markersize=markersize + 4.5, markerfacecolor="none",
                markeredgewidth=1.0)


def _curve_line(ax, run: dict, colour: str, from_epoch: int = 0, mark_best: bool = True, **kw):
    x, y = curve(run)
    x, y = zip(*[(a, b) for a, b in zip(x, y, strict=True) if a >= from_epoch], strict=True)
    style = {**_measured(colour, len(x)), **kw}
    ax.plot(x, y, **style)
    best_epoch = int(run["best"][0])
    if mark_best and best_epoch in x:
        i = x.index(best_epoch)
        ax.plot([x[i]], [y[i]], markeredgecolor=colour, zorder=4, **_best_ring(style["markersize"]))
    return x, y


def _legend_items(entries: list[tuple[str, str]], best: bool = False, extra: list | None = None):
    """Handles and labels of the series; ``best`` adds the open circle of the selected model and
    ``extra`` holds further (handle, label) pairs."""
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], **_measured(colour, 1)) for colour, _ in entries]
    labels = [label for _, label in entries]
    if best:
        handles.append(Line2D([0], [0], markeredgecolor=SLATE, **_best_ring(_measured(SLATE, 1)["markersize"])))
        labels.append(BEST_LABEL)
    for handle, label in extra or []:
        handles.append(handle)
        labels.append(label)
    return handles, labels


LEGEND = dict(frameon=False, fontsize=SIZE_NOTE, labelcolor=SLATE, handlelength=2.4)


def _curve_legend(ax, entries: list[tuple[str, str]], loc: str = "lower right", best: bool = False,
                  extra: list | None = None) -> None:
    """Legend inside the axes."""
    handles, labels = _legend_items(entries, best, extra)
    ax.legend(handles, labels, loc=loc, **LEGEND)


def _legend_below(fig, entries: list[tuple[str, str]], best: bool = False, extra: list | None = None,
                  ncol: int | None = None) -> None:
    """Legend centred under the axes: the series in one row (or ``ncol`` columns), and in a second row
    the open circle of the selected model and any ``extra`` (handle, label) pairs. Drawn after
    tight_layout, so that it never covers a line."""
    rows = []
    if entries:
        rows.append(_legend_items(entries))
    if best or extra:
        rows.append(_legend_items([], best, extra))
    for i, (handles, labels) in enumerate(rows):
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.075 * i),
                   ncol=(ncol if i == 0 and ncol else len(labels)), columnspacing=1.5, **LEGEND)


def draw_curves(runs: dict[str, dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def finish(fig, ax, name: str, left: float, right: float, legend) -> None:
        ax.set_xlim(left, right)
        fig.tight_layout()
        legend(fig)
        _save(fig, out_dir, name)
        plt.close(fig)

    # augmentation on ResNet-18: training with flips only stops improving early
    fig, ax = _curve_axes(0.05, range(0, 41, 10))
    augment = (("G1_PK_SA_BH", CURVE_1, "AUG-MIN"), ("G3_AUG_MED", CURVE_2, "AUG-MED"), ("G3_AUG_STRONG", CURVE_3, "AUG-STRONG"))
    for name, colour, _ in augment:
        _curve_line(ax, runs[name], colour)
    finish(fig, ax, CURVE_FIGURES[0], 0, 42,
           lambda f: _legend_below(f, [(colour, label) for _, colour, label in augment], best=True))

    # cross-entropy on ResNet-18 with and without weight decay, evaluated after every epoch
    fig, ax = _curve_axes(0.05, range(0, 41, 10))
    for name, colour in (("G2_CE", CURVE_1), ("G2_CE_WD", CURVE_2)):
        _curve_line(ax, runs[name], colour)
    finish(fig, ax, CURVE_FIGURES[1], 0, 41,
           lambda f: _legend_below(f, [(CURVE_1, "bez regularyzacji wag"), (CURVE_2, "z regularyzacją wag")],
                                   best=True))

    # final configuration against the same setup with AUG-MED on ResNet-34 from epoch 10: the mean of
    # the three seeds of each, with a band from the lowest to the highest of them at every evaluation;
    # the seeds select their models at different epochs, so no single point is marked
    fig, ax = _curve_axes(0.02, range(10, 61, 10))
    for names, colour in ((BASE_SEEDS, CURVE_1), (FINAL_SEEDS, CURVE_2)):
        seeds = [dict(zip(*curve(runs[name]), strict=True)) for name in names]
        epochs = sorted(e for e in seeds[0] if e >= 10 and all(e in s for s in seeds))
        values = [[s[e] for s in seeds] for e in epochs]
        ax.fill_between(epochs, [min(v) for v in values], [max(v) for v in values], color=colour, alpha=0.15,
                        linewidth=0)
        ax.plot(epochs, [statistics.mean(v) for v in values], **_measured(colour, len(epochs)))
    ax.set_ylim(0.72, 0.82)
    band = Patch(facecolor=SLATE, alpha=0.15, linewidth=0)
    finish(fig, ax, CURVE_FIGURES[2], 8, 62,
           lambda f: _legend_below(f, [(CURVE_1, "AUG-MED, średnia"), (CURVE_2, "AUG-STRONG, średnia")],
                                   extra=[(band, "zakres trzech treningów")]))

    # Multi-Similarity and Circle on ResNet-18, each in both settings of stage 2, side by side on one
    # mAP scale; each panel names its settings, the open circle is explained once under both
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7), sharey=True)
    panels = (
        ("Multi-Similarity", (("G2_MS", CURVE_1, "$\\lambda = 1$"), ("G2_MS_B05", CURVE_2, "$\\lambda = 0{,}5$"))),
        ("Circle", (("G2_CIRCLE", CURVE_1, "$(m, \\gamma) = (0{,}25;\\,64)$"),
                    ("G2_CIRCLE_M04", CURVE_2, "$(m, \\gamma) = (0{,}4;\\,80)$"))),
    )
    for ax, (title, series) in zip(axes, panels, strict=True):
        _curve_style(ax, ylabel="mAP" if ax is axes[0] else None, title=title)
        _comma_axis(ax, step=0.05)
        ax.set_xticks(list(range(0, 41, 10)))
        ax.set_xlim(0, 42)
        for name, colour, _ in series:
            _curve_line(ax, runs[name], colour)
        _curve_legend(ax, [(colour, label) for _, colour, label in series])
    axes[0].set_ylim(0.52, 0.77)
    fig.tight_layout()
    _legend_below(fig, [], best=True)
    _save(fig, out_dir, CURVE_FIGURES[3])
    plt.close(fig)

    # the two cross-entropy reference runs (60 epochs): validation mAP, measured every five epochs,
    # next to the mean training loss of every epoch, which is drawn as a continuous line
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7))
    series = (("G0b_variant_K", CURVE_1, "bez regularyzacji wag"), ("G0c_variant_K_WD", CURVE_2, "z regularyzacją wag"))
    _curve_style(left, ylabel=None, title="mAP")
    _curve_style(right, ylabel=None, title="Strata treningowa (średnia w epoce)")
    for ax in (left, right):
        ax.set_xticks(list(range(0, 61, 10)))
        ax.set_xlim(0, 62)
    _comma_axis(left, step=0.05)
    for name, colour, _ in series:
        _curve_line(left, runs[name], colour)
    left.set_ylim(0.33, 0.58)
    _comma_axis(right, digits=0, step=2)
    for name, colour, _ in series:
        loss = [float(v) for v in runs[name]["train_loss_epoch"]]
        right.plot(range(1, len(loss) + 1), loss, color=colour, linewidth=1.0, solid_capstyle="round")
    right.set_ylim(0, 12.5)
    fig.tight_layout()
    _legend_below(fig, [(colour, label) for _, colour, label in series], best=True)
    _save(fig, out_dir, CURVE_FIGURES[4])
    plt.close(fig)


def dimension_points(runs: dict[str, dict]) -> list[tuple[int, float, float, float]]:
    """(D, mAP, lowest, highest); D = 512 is the mean of the three final seeds with their range."""
    points = []
    for d, name in ((64, "G6_D64"), (128, "G6_D128"), (256, "G6_D256"), (1024, "G6_D1024")):
        v = best_map(runs[name])
        points.append((d, v, v, v))
    final = [best_map(runs[n]) for n in FINAL_SEEDS]
    points.append((512, statistics.mean(final), min(final), max(final)))
    return sorted(points)


def draw_dimension(runs: dict[str, dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pts = dimension_points(runs)
    xs = list(range(len(pts)))
    ys = [p[1] for p in pts]
    width = 0.7 * TEXT_WIDTH_IN                  # printed at 0.7 of the text width
    fig, ax = plt.subplots(figsize=(width, 0.62 * width))
    _curve_style(ax, xlabel="wymiar wektora cech", ylabel="mAP")
    for x, (_, mean, low, high) in zip(xs, pts, strict=True):
        if high > low:                           # the range of the three seeds of D = 512
            ax.plot([x, x], [low, high], color=CURVE_1, linewidth=1.0, alpha=0.45, solid_capstyle="round")
    ax.plot(xs, ys, **_measured(CURVE_1, len(xs)))
    for x, (_, mean, _, high) in zip(xs, pts, strict=True):   # above the top of the seed range
        ax.annotate(pl(mean), (x, high), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=SIZE_NOTE, color=SLATE)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(p[0]) for p in pts], fontsize=SIZE_NOTE, color=SLATE)
    ax.set_ylim(0.79, 0.83)
    _comma_axis(ax, step=0.01)
    fig.tight_layout()
    _save(fig, out_dir, "wymiar_embeddingu")
    plt.close(fig)

# backbone code in speed.json -> (run with the 40-epoch result of the backbone axis, label on the figure)
BACKBONE_RUNS: dict[str, tuple[str, str]] = {
    "R18": ("G3_AUG_MED", "ResNet-18"),
    "R34": ("G4_R34", "ResNet-34"),
    "EB1": ("G4_EB1", "EfficientNet-B1"),
    "EB2": ("G4_EB2", "EfficientNet-B2"),
    "VGG11-BN-CONV": ("G4_VGG11_BN_CONV", "VGG11-BN"),
    "VGG16-BN-CONV": ("G4_VGG16_BN_CONV", "VGG16-BN"),
    "VGG11-BN": ("G4_VGG11_BN", "VGG11-BN + fc"),
    "VGG16-BN": ("G4_VGG16_BN", "VGG16-BN + fc"),
}


def cost_points(runs: dict[str, dict], speed: dict) -> list[dict]:
    """One point per backbone: inference time, validation mAP, parameters, label."""
    points = []
    for row in speed["results"]:
        run, label = BACKBONE_RUNS[row["backbone"]]
        points.append({
            "label": label, "ms": row["inference_ms_per_image"], "mAP": best_map(runs[run]),
            "parameters": row["parameters"],
        })
    return points


def draw_cost(runs: dict[str, dict], speed: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    pts = cost_points(runs, speed)
    width = 0.8 * TEXT_WIDTH_IN                  # printed at 0.8 of the text width
    fig, ax = plt.subplots(figsize=(width, 0.6 * width))
    _curve_style(ax, xlabel="czas przetworzenia jednego wycinka [ms]", ylabel="mAP")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.scatter([p["ms"] for p in pts], [p["mAP"] for p in pts], s=26, color=CURVE_1,
               edgecolors="white", linewidths=0.6, zorder=3)
    for p in pts:
        ax.annotate(f"{p['label']}\n{pl(p['parameters'] / 1e6, 1)} mln", (p["ms"], p["mAP"]),
                    textcoords="offset points", xytext=(5, 0), ha="left", va="center",
                    fontsize=SIZE_NOTE, color=SLATE, linespacing=1.1)
    ax.set_xlim(0, max(p["ms"] for p in pts) * 1.35)
    _comma_axis(ax, step=0.01)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pl(v, 2)))
    fig.tight_layout()
    _save(fig, out_dir, "dokladnosc_wobec_kosztu")
    plt.close(fig)

# preset key in the code -> name used in the thesis ("aug-bot" is called AUG-COLOR there)
AUGMENT_LABELS: dict[str, str] = {
    "aug-min": "AUG-MIN", "aug-med": "AUG-MED", "aug-strong": "AUG-STRONG", "aug-bot": "AUG-COLOR",
}


def draw_augmentations(
    catalog_path: Path, out_dir: Path, n_crops: int = 2, n_draws: int = 6, seed: int = 3
) -> None:
    """Each preset applied ``n_draws`` times to the same training crops; first column: the crop itself."""
    import pyarrow.dataset  # noqa: F401  (before torch, see scripts/eval_checkpoint.py)
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    from soccernet_reid.data.catalog import load_catalog
    from soccernet_reid.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transform

    catalog = load_catalog(catalog_path)
    train = catalog[(catalog["split"] == "train") & (catalog["height"] >= 150)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    paths = [train.loc[int(i), "path"] for i in sorted(rng.choice(len(train), size=n_crops, replace=False))]
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    plain = build_transform("eval")

    torch.manual_seed(seed)
    n_rows, n_cols = n_crops * len(AUGMENT_LABELS), n_draws + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(0.95 * n_cols + 0.5, 1.85 * n_rows))
    for c, path in enumerate(paths):
        with Image.open(path) as im:
            im = im.convert("RGB")
            for r, (level, label) in enumerate(AUGMENT_LABELS.items()):
                row = c * len(AUGMENT_LABELS) + r
                transform = build_transform(level)
                for col in range(n_cols):
                    ax = axes[row][col]
                    x = plain(im) if col == 0 else transform(im)
                    ax.imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                axes[row][0].set_ylabel(label, fontsize=8, color=INK)
    axes[0][0].set_title("oryginał", fontsize=8, color=INK)
    axes[0][(n_cols + 1) // 2].set_title("losowe przekształcenia tego samego wycinka", fontsize=8, color=INK)
    fig.tight_layout(pad=0.3)
    _save(fig, out_dir, "przyklady_augmentacji")
    plt.close(fig)


def cmc_curve(emb: dict, catalog, split: str, max_rank: int = 10) -> list[float]:
    """CMC(k) for k = 1..max_rank: share of queries with a correct crop among the first k results.

    Uses the ranking of the evaluation (cosine, gallery of the query's action), so
    CMC(1), CMC(5) and CMC(10) are the Rank-1 / 5 / 10 of the result tables.
    """
    import numpy as np

    from soccernet_reid.eval.ranking import compute_rankings

    sub = catalog[catalog["split"] == split]
    q_person = sub[sub["role"] == "query"].set_index("bbox_idx")["person_uid"]
    g_person = sub[sub["role"] == "gallery"].set_index("bbox_idx")["person_uid"].to_dict()
    q_ids = [int(b) for b in emb["query_bbox_idx"]]
    rankings = compute_rankings(
        query_feats=emb["query_feats"], gallery_feats=emb["gallery_feats"],
        query_bbox_idx=q_ids, gallery_bbox_idx=[int(b) for b in emb["gallery_bbox_idx"]],
        query_actions=[int(a) for a in emb["query_action_idx"]],
        gallery_actions=[int(a) for a in emb["gallery_action_idx"]],
        distance="cosine",
    )
    hits = np.zeros(max_rank, dtype=np.int64)
    for qb in q_ids:
        person = int(q_person.at[qb])
        first = next((r for r, gb in enumerate(rankings[str(qb)][:max_rank]) if int(g_person[gb]) == person), None)
        if first is not None:
            hits[first:] += 1
    return (hits / len(q_ids)).tolist()


# (label on the figure, colour, runs whose curves are averaged)
CMC_SERIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("konfiguracja końcowa (średnia z trzech treningów)", CURVE_1, FINAL_SEEDS),
    ("punkt odniesienia (R18, PK, AUG-MIN)", CURVE_2, ("G0a_reference",)),
    ("klasyfikacja: ArcFace (R18)", CURVE_3, ("G2_ARC_WD",)),
)


def cmc_series(runs_dir: Path, catalog, max_rank: int = 10) -> list[tuple[str, str, list[float]]]:
    """(label, colour, CMC values) per series; each curve is checked against the saved Rank-1 / 5 / 10."""
    import numpy as np

    out = []
    for label, colour, names in CMC_SERIES:
        curves = []
        for name in names:
            with np.load(runs_dir / name / "embeddings_valid.npz", allow_pickle=False) as data:
                emb = {k: data[k] for k in data.files}
            values = cmc_curve(emb, catalog, str(emb["split"]), max_rank)
            saved = json.loads((runs_dir / name / "eval_valid.json").read_text())
            for k in (1, 5, 10):
                if k <= max_rank and abs(values[k - 1] - saved[f"rank-{k}"]) > 1e-6:
                    raise SystemExit(f"{name}: CMC({k}) = {values[k - 1]:.6f}, saved Rank-{k} = {saved[f'rank-{k}']:.6f}")
            curves.append(values)
        out.append((label, colour, np.mean(curves, axis=0).tolist()))
    return out


def draw_cmc(runs_dir: Path, catalog_path: Path, out_dir: Path, max_rank: int = 10) -> None:
    import pyarrow.dataset  # noqa: F401  (before anything that may load torch)
    import matplotlib.pyplot as plt

    from soccernet_reid.data.catalog import load_catalog

    series = cmc_series(runs_dir, load_catalog(catalog_path), max_rank)
    ks = list(range(1, max_rank + 1))
    width = 0.75 * TEXT_WIDTH_IN                 # printed at 0.75 of the text width
    fig, ax = plt.subplots(figsize=(width, 0.66 * width))
    _curve_style(ax, xlabel="k (liczba pierwszych wyników)",
                 ylabel="odsetek zapytań z trafieniem\nwśród pierwszych k wyników")
    for label, colour, values in series:
        ax.plot(ks, values, **_measured(colour, len(ks)))
        # the curves meet on the right, so the values are written where they differ: at k = 1
        ax.annotate(pl(values[0], 3), (1, values[0]), textcoords="offset points", xytext=(-6, 0),
                    ha="right", va="center", fontsize=SIZE_NOTE, color=SLATE)
        print(f"{label}: " + " ".join(pl(v, 4) for v in values))
    ax.set_xticks(ks)
    ax.set_xlim(-0.2, max_rank + 0.4)      # room on the left for the values at k = 1
    ax.set_ylim(0.5, 1.0)
    _comma_axis(ax, step=0.1, digits=1)
    fig.tight_layout()
    _legend_below(fig, [(colour, label) for label, colour, _ in series], ncol=1)
    _save(fig, out_dir, "krzywe_cmc")
    plt.close(fig)

FIGURES = ("ladder", "curves", "dimension", "cost", "augment", "cmc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=PROJECT_ROOT / "outputs" / "_g")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--speed", type=Path, default=PROJECT_ROOT / "outputs" / "_g" / "speed.json")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    runs = load_reviews(args.reviews)
    if "ladder" in args.only:
        draw_ladder(runs, args.runs, args.out)
    if "curves" in args.only:
        draw_curves(runs, args.out)
    if "dimension" in args.only:
        draw_dimension(runs, args.out)
    if "cost" in args.only:
        if not args.speed.exists():
            raise SystemExit(f"{args.speed} is missing: run scripts/benchmark_speed.py first")
        draw_cost(runs, json.loads(args.speed.read_text()), args.out)
    if "augment" in args.only:
        draw_augmentations(args.catalog, args.out)
    if "cmc" in args.only:
        draw_cmc(args.runs, args.catalog, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
