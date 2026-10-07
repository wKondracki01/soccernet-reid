"""Draw retrieval results: each query with its nearest gallery crops, right and wrong marked.

One row per query. Left: the query crop. Right: the first ``--top-k`` gallery
crops of the same action in the order the evaluation ranks them (cosine
similarity, the same code path as the metrics). A green frame means the crop
shows the query's person, a red frame a different person; the number under each
crop is the cosine distance to the query (0 = identical direction, 2 = opposite).
The row label gives the query's average precision and the rank of the first
correct crop.

Input is an .npz written by ``scripts/eval_checkpoint.py --save-embeddings``, so
no model is needed here.

Usage
-----
    python scripts/visualize_retrieval.py outputs/runs/RUN/embeddings_valid.npz \\
        --out outputs/figures/retrieval_RUN.png

    # only failures (first correct crop not at rank 1), 8 rows, another draw
    python scripts/visualize_retrieval.py EMB.npz --out fig.pdf --select failures \\
        --num-queries 8 --seed 1

``--select``:
    mixed      half queries answered correctly at rank 1, half not (default)
    random     a plain random sample of all queries
    successes  first correct crop at rank 1
    failures   first correct crop at rank 2 or later
The sample is drawn with ``--seed``, so a figure can be regenerated, and
``mixed`` / ``random`` avoid picking examples by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Must precede anything that may load torch (see scripts/eval_checkpoint.py).
import pyarrow.dataset  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.data.catalog import load_catalog  # noqa: E402
from soccernet_reid.eval.ranking import compute_rankings  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, BLUE = "#1a9641", "#d7191c", "#2b6cb0"


def rank_queries(emb: dict, catalog, split: str) -> list[dict]:
    """Per query: ranked gallery crops of its action with distance and correctness.

    Returns one dict per query with keys ``bbox_idx``, ``path``, ``action``,
    ``person``, ``ap``, ``first_correct`` (1-based rank, None if absent) and
    ``gallery`` — a list of ``{"bbox_idx", "path", "distance", "correct"}`` in
    ranking order.
    """
    sub = catalog[catalog["split"] == split]
    q_meta = sub[sub["role"] == "query"].set_index("bbox_idx")
    g_meta = sub[sub["role"] == "gallery"].set_index("bbox_idx")

    q_ids = [int(b) for b in emb["query_bbox_idx"]]
    g_ids = [int(b) for b in emb["gallery_bbox_idx"]]
    rankings = compute_rankings(
        query_feats=emb["query_feats"], gallery_feats=emb["gallery_feats"],
        query_bbox_idx=q_ids, gallery_bbox_idx=g_ids,
        query_actions=[int(a) for a in emb["query_action_idx"]],
        gallery_actions=[int(a) for a in emb["gallery_action_idx"]],
        distance="cosine",
    )
    q_feat = emb["query_feats"].astype(np.float64)
    g_feat = emb["gallery_feats"].astype(np.float64)
    q_feat /= np.maximum(np.linalg.norm(q_feat, axis=1, keepdims=True), 1e-12)
    g_feat /= np.maximum(np.linalg.norm(g_feat, axis=1, keepdims=True), 1e-12)
    g_pos = {b: i for i, b in enumerate(g_ids)}

    out = []
    for qi, qb in enumerate(q_ids):
        person = int(q_meta.at[qb, "person_uid"])
        gallery, hits, precisions = [], 0, []
        for rank, gb in enumerate(rankings[str(qb)], start=1):
            correct = int(g_meta.at[gb, "person_uid"]) == person
            if correct:
                hits += 1
                precisions.append(hits / rank)
            gallery.append({
                "bbox_idx": gb, "path": g_meta.at[gb, "path"], "correct": correct,
                "distance": float(1.0 - q_feat[qi] @ g_feat[g_pos[gb]]),
            })
        first = next((r for r, g in enumerate(gallery, start=1) if g["correct"]), None)
        out.append({
            "bbox_idx": qb, "path": q_meta.at[qb, "path"], "action": int(q_meta.at[qb, "action_idx"]),
            "person": person, "ap": float(np.mean(precisions)) if precisions else 0.0,
            "first_correct": first, "gallery": gallery,
        })
    return out


def select_queries(ranked: list[dict], mode: str, n: int, seed: int) -> list[dict]:
    """Seeded sample of ``n`` queries according to ``mode`` (see module docstring)."""
    rng = np.random.default_rng(seed)
    ok = [r for r in ranked if r["first_correct"] == 1]
    bad = [r for r in ranked if r["first_correct"] != 1]

    def draw(pool: list[dict], k: int) -> list[dict]:
        k = min(k, len(pool))
        return [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))] if k else []

    if mode == "random":
        return draw(ranked, n)
    if mode == "successes":
        return draw(ok, n)
    if mode == "failures":
        return draw(bad, n)
    if mode == "mixed":
        return draw(ok, n - n // 2) + draw(bad, n // 2)
    raise ValueError(f"Unknown selection mode {mode!r}")


def draw_figure(rows: list[dict], top_k: int, out_path: Path, title: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_cols = top_k + 1
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.15 * n_cols + 0.6, 2.55 * len(rows)), squeeze=False)

    def show(ax, path: str, colour: str, caption: str) -> None:
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB").resize((128, 256)))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3.5)
        ax.set_xlabel(caption, fontsize=8, labelpad=2)

    for r, row in enumerate(rows):
        first = row["first_correct"]
        show(axes[r][0], row["path"], BLUE, "query")
        axes[r][0].set_ylabel(
            f"AP {row['ap']:.2f}\nfirst correct: {first if first is not None else '–'}", fontsize=8
        )
        shown = row["gallery"][:top_k]
        for c in range(1, n_cols):
            ax = axes[r][c]
            if c - 1 < len(shown):
                g = shown[c - 1]
                show(ax, g["path"], GREEN if g["correct"] else RED, f"{g['distance']:.2f}")
            else:  # this action's gallery is shorter than top-k: leave an empty cell
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if r == 0:
                ax.set_title(str(c), fontsize=9)
    axes[0][0].set_title("query", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("embeddings", type=Path, help=".npz from eval_checkpoint.py --save-embeddings")
    parser.add_argument("--out", type=Path, required=True, help="output image (.png / .pdf)")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--select", default="mixed", choices=("mixed", "random", "successes", "failures"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    with np.load(args.embeddings, allow_pickle=False) as data:
        emb = {k: data[k] for k in data.files}
    split = str(emb["split"])
    ranked = rank_queries(emb, load_catalog(args.catalog), split)
    rows = select_queries(ranked, args.select, args.num_queries, args.seed)
    rank1 = np.mean([r["first_correct"] == 1 for r in ranked])
    print(f"{split}: {len(ranked):,} queries, mAP {np.mean([r['ap'] for r in ranked]):.4f}, rank-1 {rank1:.4f}")
    print(f"drawing {len(rows)} queries ({args.select}, seed {args.seed}): bbox_idx {[r['bbox_idx'] for r in rows]}")
    draw_figure(rows, args.top_k, args.out, args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
