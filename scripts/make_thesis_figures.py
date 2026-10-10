"""Draw the result figures of the thesis from the saved review files of series G.

Inputs (all written by earlier steps, nothing is trained or evaluated here)
    outputs/_g/REVIEW_*.json   per-run validation curves and best epochs
    outputs/_g/speed.json      scripts/benchmark_speed.py (only for ``cost``)
    outputs/runs/RUN/rerank_valid.json   scripts/eval_rerank.py (only for ``ladder``)

Figures (``--only`` picks a subset; each is saved as .pdf and .png)
    ladder      validation mAP after each step from the starting point to the final model
    curves      three panels of mAP against the epoch
    dimension   validation mAP against the embedding size
    cost        validation mAP against the inference time of the backbone
    augment     the four augmentation presets applied to the same training crops

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
        ("augmentacja\nAUG-MED", best_map(runs["G3_AUG_MED"])),
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
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _style(ax)
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2, solid_capstyle="round")
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if i == 0:   # the line leaves this point upwards: the label goes underneath
            ax.annotate(pl(y), (x, y), textcoords="offset points", xytext=(0, -10), ha="center",
                        va="top", fontsize=8, color=INK)
            continue
        text = f"{pl(y)}\n(+{pl(100 * (y - ys[i - 1]), 1)} pp)"
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                    va="bottom", fontsize=8, color=INK, linespacing=1.15)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _ in steps], fontsize=8, color=INK)
    ax.set_ylabel("mAP (zbiór walidacyjny)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(min(ys) - 0.03, max(ys) + 0.035)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    _comma_axis(ax, step=0.05)
    _save(fig, out_dir, "narastajace_zyski")
    plt.close(fig)


def _end_label(ax, x: float, y: float, text: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, dy), ha="left", va="center",
                fontsize=8, color=INK)


def draw_curves(runs: dict[str, dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, step in zip(axes, (0.05, 0.05, 0.02), strict=True):
        _style(ax)
        _comma_axis(ax, step=step)
        ax.set_xlabel("epoka", fontsize=9, color=INK_SECONDARY)
    axes[0].set_ylabel("mAP (zbiór walidacyjny)", fontsize=9, color=INK_SECONDARY)

    def line(ax, name: str, colour: str, from_epoch: int = 0, **kw):
        x, y = curve(runs[name])
        x, y = zip(*[(a, b) for a, b in zip(x, y, strict=True) if a >= from_epoch], strict=True)
        ax.plot(x, y, color=colour, linewidth=kw.pop("linewidth", 2), solid_capstyle="round", **kw)
        return x, y

    def legend(ax, entries: list[tuple[str, str]]) -> None:
        handles = [Line2D([0], [0], color=colour, linewidth=2) for colour, _ in entries]
        ax.legend(handles, [label for _, label in entries], loc="lower right", frameon=False,
                  fontsize=8, labelcolor=INK, handlelength=1.6)

    # (a) augmentation on ResNet-18: training with flips only stops improving early
    ax = axes[0]
    ax.set_title("(a) augmentacja, ResNet-18", fontsize=9, color=INK, loc="left")
    for name, colour, label, dy in (("G1_PK_SA_BH", BLUE, "AUG-MIN", 0), ("G3_AUG_MED", ORANGE, "AUG-MED", 5),
                                    ("G3_AUG_STRONG", AQUA, "AUG-STRONG", -5)):
        x, y = line(ax, name, colour, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1)
        _end_label(ax, x[-1], y[-1], label, dy)
    legend(ax, [(BLUE, "AUG-MIN"), (ORANGE, "AUG-MED"), (AQUA, "AUG-STRONG")])
    ax.set_xlim(0, 52)

    # (b) cross-entropy with and without weight decay
    ax = axes[1]
    ax.set_title("(b) klasyfikacja CE, ResNet-18", fontsize=9, color=INK, loc="left")
    for name, colour, label in (("G2_CE", BLUE, "bez weight decay"), ("G2_CE_WD", ORANGE, "z weight decay")):
        x, y = line(ax, name, colour)
        _end_label(ax, x[-1], y[-1], label)
    legend(ax, [(BLUE, "bez weight decay"), (ORANGE, "z weight decay")])
    ax.set_xlim(0, 58)

    # (c) final configuration against the same setup with AUG-MED, three seeds each
    ax = axes[2]
    ax.set_title("(c) ResNet-34, po trzy treningi, od 10. epoki", fontsize=9, color=INK, loc="left")
    for names, colour, label, dy in ((BASE_SEEDS, BLUE, "AUG-MED", -4), (FINAL_SEEDS, ORANGE, "AUG-STRONG", 4)):
        ends = []
        for name in names:
            x, y = line(ax, name, colour, from_epoch=10, linewidth=1.2, alpha=0.9)
            ends.append(y[-1])
        _end_label(ax, x[-1], statistics.mean(ends), label, dy)
    legend(ax, [(BLUE, "AUG-MED"), (ORANGE, "AUG-STRONG")])
    ax.set_xlim(0, 78)
    ax.set_ylim(0.72, 0.82)

    fig.tight_layout()
    _save(fig, out_dir, "krzywe_uczenia")
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
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    _style(ax)
    for x, (_, mean, low, high) in zip(xs, pts, strict=True):
        if high > low:
            ax.plot([x, x], [low, high], color=BLUE, linewidth=2, alpha=0.35, solid_capstyle="round")
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor="white",
            markeredgewidth=2, solid_capstyle="round")
    for x, (_, mean, _, high) in zip(xs, pts, strict=True):   # above the top of the seed range
        ax.annotate(pl(mean), (x, high), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(p[0]) for p in pts], fontsize=8, color=INK)
    ax.set_xlabel("wymiar embeddingu", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("mAP (zbiór walidacyjny)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0.79, 0.83)
    _comma_axis(ax, step=0.01)
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

    pts = cost_points(runs, speed)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.scatter([p["ms"] for p in pts], [p["mAP"] for p in pts], s=64, color=BLUE,
               edgecolors="white", linewidths=2, zorder=3)
    for p in pts:
        ax.annotate(f"{p['label']}\n{pl(p['parameters'] / 1e6, 1)} mln", (p["ms"], p["mAP"]),
                    textcoords="offset points", xytext=(7, 0), ha="left", va="center",
                    fontsize=7.5, color=INK, linespacing=1.1)
    ax.set_xlabel("czas przetworzenia jednego zdjęcia [ms]", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("mAP (zbiór walidacyjny)", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, max(p["ms"] for p in pts) * 1.35)
    _comma_axis(ax, step=0.01)
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pl(v, 2)))
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


FIGURES = ("ladder", "curves", "dimension", "cost", "augment")


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
