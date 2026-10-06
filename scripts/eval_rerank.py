"""Tune and apply per-action re-ranking on cached embeddings.

Re-ranking is pure post-processing, so the network runs once per checkpoint and
split; every parameter setting is then scored from the saved embeddings.

Workflow
--------
1. Save the embeddings (needs the checkpoint, ideally the training machine)::

       python scripts/eval_checkpoint.py outputs/runs/RUN/best.pt --split valid \\
           --save-embeddings outputs/runs/RUN/embeddings_valid.npz
       python scripts/eval_checkpoint.py outputs/runs/RUN/best.pt --split test \\
           --save-embeddings outputs/runs/RUN/embeddings_test.npz

2. Choose the parameters on VALID::

       python scripts/eval_rerank.py tune outputs/runs/RUN/embeddings_valid.npz \\
           --out outputs/runs/RUN/rerank_valid.json

3. Apply the frozen parameters to TEST::

       python scripts/eval_rerank.py apply outputs/runs/RUN/embeddings_test.npz \\
           --params outputs/runs/RUN/rerank_valid.json \\
           --out outputs/runs/RUN/rerank_test.json

``tune`` refuses any split other than valid and ``apply`` only accepts parameters
that were tuned on valid, so the test split never influences a choice.

Methods (see soccernet_reid/eval/rerank.py): ``k_reciprocal``, ``dual_softmax``
and ``both``. The plain ranking is always scored through the same code path as
evaluation during training, and on valid it is compared with the best mAP stored
in the checkpoint as a sanity check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Must precede torch (pulled in by soccernet_reid.training): on Windows, loading
# pyarrow's dataset DLLs after torch crashes the process (see eval_checkpoint.py).
import pyarrow.dataset  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.data.catalog import load_catalog  # noqa: E402
from soccernet_reid.eval.metrics import compute_metrics  # noqa: E402
from soccernet_reid.eval.rerank import (  # noqa: E402
    dual_softmax_scores,
    dual_softmax_shares,
    group_positions_by_action,
    k_reciprocal_components,
    rankings_from_action_scores,
)
from soccernet_reid.training import metrics_from_embeddings, split_groundtruth  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Search space. An action holds ~24 crops and most queries have one positive,
# so k1 / k2 stay far below the paper's 20 / 6.
K1_GRID = (1, 2, 3, 4, 5, 6, 8, 10)
K2_GRID = (1, 2, 3)
LAMBDA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TEMPERATURE_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
RANKS = (1, 5, 10)
METRIC_KEYS = ("mAP", "rank-1", "rank-5", "rank-10")


def load_embeddings(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        emb = {k: data[k] for k in data.files}
    for key in ("query_feats", "gallery_feats", "query_bbox_idx", "gallery_bbox_idx",
                "query_action_idx", "gallery_action_idx", "split"):
        if key not in emb:
            raise KeyError(f"{path} has no {key!r}; was it written by eval_checkpoint.py --save-embeddings?")
    emb["split"] = str(emb["split"])
    return emb


class Actions:
    """Per-action views of the embeddings, in the order the ranking code expects."""

    def __init__(self, emb: dict) -> None:
        q = emb["query_feats"].astype(np.float64)
        g = emb["gallery_feats"].astype(np.float64)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        self.emb = emb
        q_groups = group_positions_by_action(emb["query_action_idx"])
        g_groups = group_positions_by_action(emb["gallery_action_idx"])
        self.ids = [a for a in q_groups if a in g_groups and len(g_groups[a]) > 0]
        self.query = {a: q[q_groups[a]] for a in self.ids}
        self.gallery = {a: g[g_groups[a]] for a in self.ids}
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS flags
            self.similarity = {a: self.query[a] @ self.gallery[a].T for a in self.ids}
        if not all(np.isfinite(s).all() for s in self.similarity.values()):
            raise ValueError("embeddings contain NaN or Inf; cannot re-rank")
        self._components: dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def components(self, k1: int, k2: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        key = (int(k1), int(k2))
        if key not in self._components:
            self._components[key] = {
                a: k_reciprocal_components(self.query[a], self.gallery[a], k1, k2) for a in self.ids
            }
        return self._components[key]

    def scores(self, method: str, params: dict) -> dict[int, np.ndarray]:
        if method == "dual_softmax":
            t = params["temperature"]
            return {a: dual_softmax_scores(self.similarity[a], t) for a in self.ids}
        comps = self.components(params["k1"], params["k2"])
        lam = params["lambda_value"]
        if method == "k_reciprocal":
            return {a: -((1 - lam) * j + lam * o) for a, (j, o) in comps.items()}
        if method == "both":
            t = params["temperature"]
            return {
                a: (1 - ((1 - lam) * j + lam * o)) * dual_softmax_shares(self.similarity[a], t)
                for a, (j, o) in comps.items()
            }
        raise ValueError(f"Unknown method {method!r}")

    def metrics(self, method: str, params: dict, gt: dict, validate: bool) -> dict[str, float]:
        emb = self.emb
        rankings = rankings_from_action_scores(
            self.scores(method, params),
            emb["query_bbox_idx"], emb["gallery_bbox_idx"],
            emb["query_action_idx"], emb["gallery_action_idx"],
        )
        return compute_metrics(rankings, gt["query"], gt["gallery"], ranks=RANKS, validate=validate)


def _best(rows: list[dict], tie_keys: tuple[str, ...]) -> dict:
    """Highest mAP; ties go to the smaller parameter values (simpler setting first)."""
    return min(rows, key=lambda r: (-r["mAP"], *(r[k] for k in tie_keys)))


def _fmt(m: dict) -> str:
    return "  ".join(f"{k}={m[k]:.4f}" for k in METRIC_KEYS)


def _delta(m: dict, base: dict) -> str:
    return "  ".join(f"{k} {100 * (m[k] - base[k]):+.2f}" for k in METRIC_KEYS)


def _baseline(emb: dict, catalog, split: str) -> dict[str, float]:
    return metrics_from_embeddings(emb, catalog, split=split, distance="cosine", ranks=RANKS)


def tune(args: argparse.Namespace) -> int:
    emb = load_embeddings(args.embeddings)
    split = emb["split"]
    if split != "valid":
        print(f"ERROR: tuning is allowed on the valid split only, got {split!r}", file=sys.stderr)
        return 2
    catalog = load_catalog(args.catalog)
    gt = split_groundtruth(catalog, split)
    actions = Actions(emb)
    print(f"{args.embeddings}: split={split}, {len(emb['query_bbox_idx']):,} queries, "
          f"{len(emb['gallery_bbox_idx']):,} gallery crops, {len(actions.ids):,} actions")

    baseline = _baseline(emb, catalog, split)
    print(f"plain ranking      {_fmt(baseline)}")
    stored = float(emb["best_mAP"]) if "best_mAP" in emb else float("nan")
    if np.isfinite(stored):
        diff = abs(baseline["mAP"] - stored)
        print(f"  checkpoint best_mAP = {stored:.4f}, difference {diff:.2e}"
              + ("" if diff < 1e-4 else "   <-- DOES NOT MATCH the checkpoint"))

    t0 = time.perf_counter()
    kr_rows: list[dict] = []
    first = True
    for k1 in args.k1:
        for k2 in args.k2:
            for lam in args.lambdas:
                p = {"k1": k1, "k2": k2, "lambda_value": lam}
                kr_rows.append({**p, **actions.metrics("k_reciprocal", p, gt, validate=first)})
                first = False
    best_kr = _best(kr_rows, ("k1", "k2", "lambda_value"))
    print(f"k_reciprocal grid: {len(kr_rows)} settings in {time.perf_counter() - t0:.0f}s")

    ds_rows = [
        {"temperature": t, **actions.metrics("dual_softmax", {"temperature": t}, gt, validate=(i == 0))}
        for i, t in enumerate(args.temperatures)
    ]
    best_ds = _best(ds_rows, ("temperature",))

    both_rows: list[dict] = []
    for lam in args.lambdas:
        for t in args.temperatures:
            p = {"k1": best_kr["k1"], "k2": best_kr["k2"], "lambda_value": lam, "temperature": t}
            both_rows.append({**p, **actions.metrics("both", p, gt, validate=not both_rows)})
    best_both = _best(both_rows, ("lambda_value", "temperature"))
    print(f"all grids done in {time.perf_counter() - t0:.0f}s")

    def pack(row: dict, keys: tuple[str, ...]) -> dict:
        params = {k: row[k] for k in keys}
        return {"params": params, "metrics": {k: row[k] for k in METRIC_KEYS}}

    best = {
        "k_reciprocal": pack(best_kr, ("k1", "k2", "lambda_value")),
        "dual_softmax": pack(best_ds, ("temperature",)),
        "both": pack(best_both, ("k1", "k2", "lambda_value", "temperature")),
    }
    # Re-score the winners with full validation of the rankings.
    for method, entry in best.items():
        entry["metrics"] = actions.metrics(method, entry["params"], gt, validate=True)
        print(f"{method:18s} {_fmt(entry['metrics'])}   {entry['params']}")
        print(f"{'':18s} vs plain (pp): {_delta(entry['metrics'], baseline)}")

    result = {
        "mode": "tune",
        "embeddings": str(args.embeddings),
        "checkpoint": str(emb.get("checkpoint", "")),
        "split": split,
        "n_queries": int(len(emb["query_bbox_idx"])),
        "n_gallery": int(len(emb["gallery_bbox_idx"])),
        "n_actions": len(actions.ids),
        "baseline": baseline,
        "best": best,
        "grid": {"k_reciprocal": kr_rows, "dual_softmax": ds_rows, "both": both_rows},
        "search_space": {"k1": list(args.k1), "k2": list(args.k2),
                         "lambda_value": list(args.lambdas), "temperature": list(args.temperatures)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    return 0


def apply(args: argparse.Namespace) -> int:
    tuned = json.loads(args.params.read_text())
    if tuned.get("mode") != "tune" or tuned.get("split") != "valid":
        print(f"ERROR: {args.params} is not the result of tuning on valid", file=sys.stderr)
        return 2
    emb = load_embeddings(args.embeddings)
    split = emb["split"]
    catalog = load_catalog(args.catalog)
    gt = split_groundtruth(catalog, split)
    actions = Actions(emb)
    print(f"{args.embeddings}: split={split}, {len(emb['query_bbox_idx']):,} queries, "
          f"{len(emb['gallery_bbox_idx']):,} gallery crops, {len(actions.ids):,} actions")
    print(f"parameters frozen from {args.params} (tuned on {tuned['split']})")

    baseline = _baseline(emb, catalog, split)
    print(f"plain ranking      {_fmt(baseline)}")
    methods = {}
    for method, entry in tuned["best"].items():
        metrics = actions.metrics(method, entry["params"], gt, validate=True)
        methods[method] = {"params": entry["params"], "metrics": metrics}
        print(f"{method:18s} {_fmt(metrics)}   {entry['params']}")
        print(f"{'':18s} vs plain (pp): {_delta(metrics, baseline)}")

    result = {
        "mode": "apply",
        "embeddings": str(args.embeddings),
        "checkpoint": str(emb.get("checkpoint", "")),
        "split": split,
        "params_from": str(args.params),
        "n_queries": int(len(emb["query_bbox_idx"])),
        "n_gallery": int(len(emb["gallery_bbox_idx"])),
        "n_actions": len(actions.ids),
        "baseline": baseline,
        "methods": methods,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("embeddings", type=Path, help=".npz written by eval_checkpoint.py --save-embeddings")
        p.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "outputs" / "catalog.parquet")
        p.add_argument("--out", type=Path, required=True, help="JSON file for the results")

    p_tune = sub.add_parser("tune", help="search the parameters on the valid split")
    common(p_tune)
    p_tune.add_argument("--k1", type=int, nargs="+", default=list(K1_GRID))
    p_tune.add_argument("--k2", type=int, nargs="+", default=list(K2_GRID))
    p_tune.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDA_GRID))
    p_tune.add_argument("--temperatures", type=float, nargs="+", default=list(TEMPERATURE_GRID))
    p_tune.set_defaults(func=tune)

    p_apply = sub.add_parser("apply", help="score a split with parameters tuned on valid")
    common(p_apply)
    p_apply.add_argument("--params", type=Path, required=True, help="JSON written by `tune`")
    p_apply.set_defaults(func=apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
