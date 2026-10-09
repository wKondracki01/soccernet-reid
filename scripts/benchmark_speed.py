"""Measure how fast each backbone turns crops into embeddings, and how fast it trains.

The timing covers the network alone: the input is a random tensor that already
sits on the device, so image decoding, augmentation and data loading are left
out. The numbers therefore compare the feature extractors themselves and do not
depend on the disk or on the number of CPU cores. Epoch times printed during
training are not a substitute: several runs usually share one GPU there.

Run it on an otherwise idle GPU.

For every backbone the script reports
    parameters        backbone + head, without any classifier
    inference         images per second and milliseconds per image at
                      ``--batch-size`` (model in eval mode, no gradients)
    latency           milliseconds for a single image (batch of 1)
    training          optimisation steps per second for a triplet step on a
                      ``--train-batch-size`` batch (forward, batch-hard triplet
                      loss, backward, Adam update), unless ``--no-train``

Each figure is the median of ``--repeats`` measurements of ``--iters`` batches,
taken after ``--warmup`` batches that are not timed.

Usage
-----
    python scripts/benchmark_speed.py --out outputs/_g/speed.json

    # only two backbones, CPU, quick look
    python scripts/benchmark_speed.py --backbones R18 R34 --device cpu --iters 5 --repeats 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Must precede anything that may load torch (see scripts/eval_checkpoint.py).
import pyarrow.dataset  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.losses import build_loss  # noqa: E402
from soccernet_reid.models import build_model  # noqa: E402
from soccernet_reid.training import pick_device  # noqa: E402

ALL_BACKBONES: tuple[str, ...] = (
    "R18", "R34", "EB1", "EB2", "VGG11-BN-CONV", "VGG16-BN-CONV", "VGG11-BN", "VGG16-BN",
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work, so that the clock measures finished batches."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_seconds_per_iter(step, device: torch.device, warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        step()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        _sync(device)
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times)


def time_inference(
    model: torch.nn.Module, batch: torch.Tensor, device: torch.device,
    warmup: int, iters: int, repeats: int,
) -> float:
    """Seconds per batch for a forward pass in eval mode."""
    model.eval()

    def step() -> None:
        with torch.inference_mode():
            model(batch)

    return _median_seconds_per_iter(step, device, warmup, iters, repeats)


def time_training(
    model: torch.nn.Module, batch: torch.Tensor, labels: torch.Tensor, embedding_dim: int,
    device: torch.device, warmup: int, iters: int, repeats: int,
) -> float:
    """Seconds per optimisation step: forward, batch-hard triplet loss, backward, Adam update."""
    model.train()
    loss_module = build_loss("tri", embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_module.call(model(batch), labels)
        loss.backward()
        optimizer.step()

    return _median_seconds_per_iter(step, device, warmup, iters, repeats)


def benchmark_backbone(
    code: str,
    device: torch.device,
    *,
    head: str = "projection",
    embedding_dim: int = 512,
    height: int = 256,
    width: int = 128,
    batch_size: int = 64,
    train_batch_size: int = 16,
    warmup: int = 20,
    iters: int = 100,
    repeats: int = 3,
    train: bool = True,
) -> dict:
    """All measurements for one backbone code (weights are random: speed does not depend on them)."""
    model = build_model(code, head, embedding_dim=embedding_dim, pretrained=False).to(device)
    result: dict = {
        "backbone": code, "head": head, "embedding_dim": embedding_dim,
        "input": [height, width], "parameters": count_parameters(model),
        "batch_size": batch_size,
    }

    batch = torch.randn(batch_size, 3, height, width, device=device)
    sec = time_inference(model, batch, device, warmup, iters, repeats)
    result["inference_images_per_s"] = batch_size / sec
    result["inference_ms_per_image"] = 1000.0 * sec / batch_size

    single = torch.randn(1, 3, height, width, device=device)
    result["latency_ms_single_image"] = 1000.0 * time_inference(model, single, device, warmup, iters, repeats)

    if train:
        # P identities x 2 crops, as the PK samplers produce
        train_batch = torch.randn(train_batch_size, 3, height, width, device=device)
        labels = torch.arange(train_batch_size // 2, device=device).repeat_interleave(2)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sec = time_training(model, train_batch, labels, embedding_dim, device, warmup, iters, repeats)
        result["train_batch_size"] = train_batch_size
        result["train_steps_per_s"] = 1.0 / sec
        result["train_images_per_s"] = train_batch_size / sec
        if device.type == "cuda":
            result["train_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def format_table(rows: list[dict]) -> str:
    head = f"{'backbone':15s} {'params [M]':>10s} {'img/s':>9s} {'ms/img':>8s} {'1 img [ms]':>10s}"
    with_train = any("train_steps_per_s" in r for r in rows)
    if with_train:
        head += f" {'train it/s':>10s}"
    lines = [head]
    for r in rows:
        line = (f"{r['backbone']:15s} {r['parameters'] / 1e6:10.2f} {r['inference_images_per_s']:9.0f} "
                f"{r['inference_ms_per_image']:8.3f} {r['latency_ms_single_image']:10.2f}")
        if with_train:
            line += f" {r.get('train_steps_per_s', float('nan')):10.1f}"
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbones", nargs="+", default=list(ALL_BACKBONES))
    parser.add_argument("--head", default="projection")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch (64 = the evaluation batch)")
    parser.add_argument("--train-batch-size", type=int, default=16, help="training batch (16 = PK-SA 8x2)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-train", action="store_true", help="skip the training-step measurement")
    parser.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    parser.add_argument("--out", type=Path, default=None, help="write the results as JSON")
    args = parser.parse_args()

    device = pick_device(args.device)
    environment = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "torch": torch.__version__,
        "precision": "float32",
    }
    print(f"device: {environment['device_name']} | torch {environment['torch']} | float32")

    rows = []
    for code in args.backbones:
        rows.append(benchmark_backbone(
            code, device, head=args.head, embedding_dim=args.embedding_dim,
            height=args.height, width=args.width, batch_size=args.batch_size,
            train_batch_size=args.train_batch_size, warmup=args.warmup, iters=args.iters,
            repeats=args.repeats, train=not args.no_train,
        ))
        print(format_table(rows[-1:]).splitlines()[-1] if len(rows) > 1 else format_table(rows), flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"environment": environment, "results": rows}, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
