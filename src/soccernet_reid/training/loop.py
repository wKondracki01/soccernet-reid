"""Train and evaluate loops — backend-agnostic, used by the Hydra entrypoint.

Both functions are pure helpers (no Hydra, no W&B); the entrypoint script
wires them together with config and logging.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from soccernet_reid.data.dataset import ReIDImageDataset
from soccernet_reid.eval.metrics import compute_metrics
from soccernet_reid.eval.official import catalog_to_groundtruth_dict
from soccernet_reid.eval.ranking import compute_rankings
from soccernet_reid.losses.factory import LossModule
from soccernet_reid.models.model import ReIDModel
from soccernet_reid.transforms import build_transform


def _embedding_for_metric_loss(out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Pick the embedding tensor that metric losses (Triplet/Contrastive/...) consume.

    For BNNeck this is the PRE-BN feature (BoT-ReID convention). For all other
    heads the model output IS the embedding tensor.
    """
    if isinstance(out, dict):
        return out["embedding_metric"]
    return out


def _embedding_for_classifier(out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """For CE / ArcFace: use POST-BN L2-normed embedding from BNNeck, raw output otherwise."""
    if isinstance(out, dict):
        return out["embedding_retrieval"]
    return out


def _embedding_for_retrieval(out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(out, dict):
        return out["embedding_retrieval"]
    return out


def train_one_epoch(
    *,
    model: ReIDModel,
    loss_module: LossModule,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    log_every: int = 50,
    epoch: int = 0,
    on_step: Any | None = None,
) -> dict[str, float]:
    """Run one training epoch over `loader`.

    Returns a dict with mean training loss and final LR.
    The optional `on_step` callable (signature: (step_idx, loss_value, lr)) is
    invoked once per gradient step — useful for W&B logging without coupling
    the loop to W&B itself.
    """
    model.train()
    losses: list[float] = []
    step = 0
    use_amp = scaler is not None

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["class_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                out = model(images)
                if loss_module.requires_classes:
                    emb = _embedding_for_classifier(out)
                else:
                    emb = _embedding_for_metric_loss(out)
                loss = loss_module.call(emb, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            if loss_module.requires_classes:
                emb = _embedding_for_classifier(out)
            else:
                emb = _embedding_for_metric_loss(out)
            loss = loss_module.call(emb, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)
        lr = optimizer.param_groups[0]["lr"]
        if on_step is not None:
            on_step(step, loss_val, lr)
        if (step + 1) % log_every == 0 or step == 0:
            print(
                f"  epoch {epoch:>3d} step {step + 1:>5d} | loss {loss_val:.4f} | lr {lr:.2e}"
            )
        step += 1

    return {"train_loss_mean": float(np.mean(losses)), "lr_final": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def _extract_split_features(
    df: pd.DataFrame,
    model: ReIDModel,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    transform = build_transform("eval", height=256, width=128)
    ds = ReIDImageDataset(df, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    feats: list[np.ndarray] = []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        out = model(imgs)
        emb = _embedding_for_retrieval(out)
        feats.append(emb.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _query_gallery_frames(
    catalog: pd.DataFrame, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rows of ``split`` plus its query and gallery frames, as every evaluation uses them.

    Queries whose identity has no crop in the gallery of the same action are
    dropped (the official evaluator raises on them). On the official valid and
    test splits this removes nothing.
    """
    sub = catalog[catalog["split"] == split]
    if sub.empty:
        raise ValueError(f"No rows in catalog for split={split!r}")
    query_df = sub[sub["role"] == "query"].reset_index(drop=True)
    gallery_df = sub[sub["role"] == "gallery"].reset_index(drop=True)

    gallery_pairs = set(
        map(tuple, gallery_df[["action_idx", "person_uid"]].itertuples(index=False, name=None))
    )
    keep = query_df.apply(
        lambda r: (int(r["action_idx"]), int(r["person_uid"])) in gallery_pairs, axis=1
    )
    query_df = query_df[keep].reset_index(drop=True)
    return sub, query_df, gallery_df


def extract_split_embeddings(
    *,
    model: ReIDModel,
    catalog: pd.DataFrame,
    split: str = "valid",
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, np.ndarray]:
    """Retrieval embeddings of a split's queries and gallery, with their identifiers.

    Exactly the features and the query / gallery selection that
    :func:`evaluate_model` ranks, so metrics computed later from the returned
    arrays reproduce its result. Meant for post-processing that needs the
    embeddings themselves (re-ranking, visualisation) without re-running the
    network for every variant.
    """
    model.eval()
    _, query_df, gallery_df = _query_gallery_frames(catalog, split)
    return {
        "query_feats": _extract_split_features(query_df, model, device, batch_size, num_workers),
        "gallery_feats": _extract_split_features(gallery_df, model, device, batch_size, num_workers),
        "query_bbox_idx": query_df["bbox_idx"].to_numpy(dtype=np.int64),
        "gallery_bbox_idx": gallery_df["bbox_idx"].to_numpy(dtype=np.int64),
        "query_action_idx": query_df["action_idx"].to_numpy(dtype=np.int64),
        "gallery_action_idx": gallery_df["action_idx"].to_numpy(dtype=np.int64),
    }


def evaluate_model(
    *,
    model: ReIDModel,
    catalog: pd.DataFrame,
    split: str = "valid",
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    distance: str = "cosine",
    ranks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Extract features for query+gallery, rank, return mAP / Rank-k.

    Filters out queries whose person_uid is not represented in gallery of the
    same action (the official evaluator's precondition).
    """
    embeddings = extract_split_embeddings(
        model=model,
        catalog=catalog,
        split=split,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return metrics_from_embeddings(embeddings, catalog, split=split, distance=distance, ranks=ranks)


def split_groundtruth(
    catalog: pd.DataFrame, split: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Ground truth of ``split`` in the official ``bbox_info.json`` layout.

    Restricted to the queries and gallery crops that :func:`evaluate_model`
    ranks, so it pairs with rankings built from :func:`extract_split_embeddings`.
    """
    sub, query_df, gallery_df = _query_gallery_frames(catalog, split)
    keep_bbox = set(query_df["bbox_idx"].astype(int)) | set(gallery_df["bbox_idx"].astype(int))
    subset = sub[sub["bbox_idx"].isin(keep_bbox)]
    gt = catalog_to_groundtruth_dict(subset, split=split)
    gt["query"] = {k: v for k, v in gt["query"].items() if int(k) in keep_bbox}
    return gt


def metrics_from_embeddings(
    embeddings: Mapping[str, np.ndarray],
    catalog: pd.DataFrame,
    *,
    split: str = "valid",
    distance: str = "cosine",
    ranks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """mAP / Rank-k of the plain ranking for arrays from :func:`extract_split_embeddings`.

    This is the second half of :func:`evaluate_model`; cached embeddings scored
    here give the same numbers as an evaluation during training.
    """
    rankings = compute_rankings(
        query_feats=embeddings["query_feats"],
        gallery_feats=embeddings["gallery_feats"],
        query_bbox_idx=[int(b) for b in embeddings["query_bbox_idx"]],
        gallery_bbox_idx=[int(b) for b in embeddings["gallery_bbox_idx"]],
        query_actions=[int(a) for a in embeddings["query_action_idx"]],
        gallery_actions=[int(a) for a in embeddings["gallery_action_idx"]],
        distance=distance,
    )
    gt = split_groundtruth(catalog, split)
    return compute_metrics(rankings, gt["query"], gt["gallery"], ranks=ranks)


def cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay LR schedule with linear warmup. Step-wise (call per iteration)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)
        if total_iters <= warmup_iters:
            return 1.0
        progress = (step - warmup_iters) / (total_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
