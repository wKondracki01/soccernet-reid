"""Hydra entrypoint for SoccerNet ReID training.

Defaults to the reference config (`R18 + TRI + PK-BH + AUG-MIN`, embedding D=512,
60 epochs, Adam lr=3.5e-4, cosine LR with warmup 5).

Usage:
    # Reference run (Faza 0a):
    uv run python scripts/train.py
    # Smoke (2 epochs × 30 iters):
    uv run python scripts/train.py +experiment=smoke
    # Different backbone + loss:
    uv run python scripts/train.py backbone=eb1 loss=ms
    # Override individual fields:
    uv run python scripts/train.py num_epochs=40 optimizer.lr=1e-4

The Hydra defaults list lives in `configs/config.yaml`; per-component options
are in `configs/{backbone,head,loss,sampler,transform}/`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler

# Allow running directly without `uv run -m ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.data.catalog import (  # noqa: E402
    assign_class_ids,
    build_catalog,
    filter_to_player_classes,
    load_catalog,
)
from soccernet_reid.data.dataset import ReIDImageDataset  # noqa: E402
from soccernet_reid.losses import build_loss  # noqa: E402
from soccernet_reid.models import build_model  # noqa: E402
from soccernet_reid.samplers import (  # noqa: E402
    PKBatchSampler,
    PKPerActionBatchSampler,
)
from soccernet_reid.training import (  # noqa: E402
    enable_determinism,
    evaluate_model,
    pick_device,
    seed_everything,
    train_one_epoch,
)
from soccernet_reid.training.loop import cosine_lr_with_warmup  # noqa: E402
from soccernet_reid.training.state import amp_supported  # noqa: E402
from soccernet_reid.transforms import build_transform  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")


def _load_or_build_catalog(cfg: DictConfig) -> pd.DataFrame:
    cat_path = Path(cfg.catalog_path)
    if cat_path.exists():
        print(f"Loading catalog from {cat_path}")
        return load_catalog(cat_path)
    print(f"Catalog not found at {cat_path}; building from {cfg.reid_root}")
    return build_catalog(cfg.reid_root)


def _build_train_loader(cfg: DictConfig, train_df: pd.DataFrame, num_iters: int) -> DataLoader:
    transform = build_transform(
        level=cfg.transform.level,
        height=cfg.transform.height,
        width=cfg.transform.width,
    )
    ds = ReIDImageDataset(train_df, transform=transform)
    sampler_cfg = cfg.sampler

    if sampler_cfg.name == "pk":
        # `num_batches` defines an "epoch" worth of iterations
        sampler = PKBatchSampler(
            class_ids=train_df["class_id"].tolist(),
            P=sampler_cfg.P,
            K=sampler_cfg.K,
            num_batches=num_iters,
            seed=cfg.seed,
        )
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers)
    elif sampler_cfg.name == "pk_sa":
        sampler = PKPerActionBatchSampler(
            class_ids=train_df["class_id"].tolist(),
            action_ids=train_df["action_idx"].tolist(),
            P=sampler_cfg.P,
            K=sampler_cfg.K,
            num_batches=num_iters,
            seed=cfg.seed,
        )
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers)
    elif sampler_cfg.name == "random":
        # Use a torch RandomSampler; cap the per-epoch iteration count via
        # `num_samples = batch_size * num_iters`.
        rs = RandomSampler(
            ds,
            replacement=True,
            num_samples=sampler_cfg.batch_size * num_iters,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        loader = DataLoader(
            ds,
            sampler=rs,
            batch_size=sampler_cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler_cfg.name!r}")

    return loader


def _wandb_init(cfg: DictConfig, output_dir: Path):
    if not cfg.wandb.enabled:
        return None
    import wandb  # local import to avoid hard dep when wandb not used

    if cfg.wandb.mode == "disabled":
        return None
    os.environ["WANDB_MODE"] = cfg.wandb.mode
    return wandb.init(
        project=cfg.wandb.project,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        notes=cfg.wandb.notes,
        dir=str(output_dir),
    )


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    if cfg.deterministic:
        enable_determinism(deterministic=True, warn_only=True)

    device = pick_device(cfg.device)
    print(f"Device: {device}")

    use_amp = bool(cfg.amp) and amp_supported(device)
    if cfg.amp and not use_amp:
        print(f"AMP disabled (not supported on device {device.type})")

    # 1. Catalog
    full = _load_or_build_catalog(cfg)
    train_df = full[full["split"] == "train"].copy()
    train_df = filter_to_player_classes(train_df)
    train_df = assign_class_ids(train_df)
    num_classes = int(train_df["class_id"].nunique())
    print(f"Train: {len(train_df):,} samples, {num_classes:,} classes")

    # 2. Model
    model = build_model(
        backbone_name=cfg.backbone.name,
        head_name=cfg.head.name,
        embedding_dim=cfg.embedding_dim,
        pretrained=cfg.backbone.pretrained,
    ).to(device)
    embedding_dim = model.embedding_dim
    print(
        f"Model: {cfg.backbone.name} + {cfg.head.name} -> D={embedding_dim} "
        f"({sum(p.numel() for p in model.parameters()):,} params)"
    )

    # 3. Loss
    loss_kwargs = {k: v for k, v in cfg.loss.items() if k != "name"}
    loss_module = build_loss(
        name=cfg.loss.name,
        embedding_dim=embedding_dim,
        num_classes=num_classes if cfg.loss.name in ("ce", "arc") else None,
        **loss_kwargs,
    )
    loss_module.call.to(device)

    # 4. Loader
    train_loader = _build_train_loader(cfg, train_df, num_iters=cfg.num_iters_per_epoch)

    # 5. Optimizer + scheduler
    params = list(model.parameters()) + [
        p for p in loss_module.call.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        params, lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay)
    )
    total_iters = cfg.num_iters_per_epoch * cfg.num_epochs
    warmup_iters = cfg.num_iters_per_epoch * cfg.scheduler.warmup_epochs
    scheduler = cosine_lr_with_warmup(
        optimizer, warmup_iters=warmup_iters, total_iters=total_iters,
        min_lr_ratio=cfg.scheduler.min_lr_ratio,
    )

    scaler = torch.amp.GradScaler() if use_amp else None

    # 6. W&B
    wandb_run = _wandb_init(cfg, output_dir)

    def on_step(step: int, loss: float, lr: float) -> None:
        if wandb_run is not None:
            wandb_run.log({"train/loss_step": loss, "train/lr": lr})

    # 7. Train + eval loop
    best_map = -1.0
    for epoch in range(cfg.num_epochs):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            loss_module=loss_module,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            log_every=cfg.log_every,
            epoch=epoch,
            on_step=on_step,
        )
        dt = time.perf_counter() - t0
        print(f"Epoch {epoch}: train_loss={train_metrics['train_loss_mean']:.4f} "
              f"lr={train_metrics['lr_final']:.2e} time={dt:.1f}s")
        epoch_log = {
            "train/loss_epoch_mean": train_metrics["train_loss_mean"],
            "train/lr_epoch_end": train_metrics["lr_final"],
            "train/epoch_time_s": dt,
            "epoch": epoch,
        }

        # Eval cadence
        if (epoch + 1) % cfg.eval.every_n_epochs == 0 or (epoch + 1) == cfg.num_epochs:
            print(f"  Evaluating on {cfg.eval.split} ...")
            t0 = time.perf_counter()
            eval_metrics = evaluate_model(
                model=model,
                catalog=full,
                split=cfg.eval.split,
                device=device,
                batch_size=cfg.eval.batch_size,
                num_workers=cfg.num_workers,
                distance=cfg.eval.distance,
                ranks=tuple(cfg.eval.ranks),
            )
            dt = time.perf_counter() - t0
            print(f"  Eval: mAP={eval_metrics['mAP']:.4f} "
                  f"R-1={eval_metrics['rank-1']:.4f} time={dt:.1f}s")
            for k, v in eval_metrics.items():
                epoch_log[f"valid/{k}"] = v

            if eval_metrics["mAP"] > best_map:
                best_map = eval_metrics["mAP"]
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "loss_module": loss_module.call.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_mAP": best_map,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                ckpt_path = output_dir / "best.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  Saved best checkpoint -> {ckpt_path} (mAP={best_map:.4f})")

                # W&B Artifact upload (versioned checkpoint, recoverable from any machine).
                # Wrapped in try/except so a network blip doesn't crash the training run.
                if wandb_run is not None:
                    try:
                        import wandb
                        artifact = wandb.Artifact(
                            name=f"{cfg.experiment_name}-best",
                            type="model",
                            description=(
                                f"Best valid mAP={best_map:.4f} at epoch {epoch} "
                                f"({cfg.backbone.name}+{cfg.head.name}+{cfg.loss.name})"
                            ),
                            metadata={
                                "epoch": epoch,
                                "valid_mAP": best_map,
                                "valid_rank_1": eval_metrics.get("rank-1"),
                                "valid_rank_5": eval_metrics.get("rank-5"),
                                "valid_rank_10": eval_metrics.get("rank-10"),
                                "backbone": cfg.backbone.name,
                                "head": cfg.head.name,
                                "loss": cfg.loss.name,
                                "embedding_dim": embedding_dim,
                            },
                        )
                        artifact.add_file(str(ckpt_path))
                        wandb_run.log_artifact(
                            artifact,
                            aliases=[f"epoch-{epoch}", f"map-{best_map:.4f}", "best"],
                        )
                        print(f"  Uploaded checkpoint to W&B as '{cfg.experiment_name}-best'")
                    except Exception as e:
                        print(f"  W&B artifact upload failed (non-fatal): {e}")

        if wandb_run is not None:
            wandb_run.log(epoch_log)

    print(f"\nDone. Best valid mAP: {best_map:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
