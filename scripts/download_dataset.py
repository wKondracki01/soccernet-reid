"""Download (or verify) SoccerNet ReID 2023 dataset.

Usage examples
--------------

Verify / repair existing download (skips files with matching MD5, fast):
    uv run python scripts/download_dataset.py --target dataSoccerNet

Force a clean re-download from scratch (deletes existing zips, ~18 GB transfer):
    uv run python scripts/download_dataset.py --target dataSoccerNet --force

Download + extract zips into split directories:
    uv run python scripts/download_dataset.py --target dataSoccerNet --extract

Full clean reinstall (delete + re-download + re-extract + verify file counts):
    uv run python scripts/download_dataset.py --target dataSoccerNet --force --extract --verify

Notes
-----
- SoccerNetDownloader uses MD5 hashes to skip already-downloaded files. So a no-flag
  re-run is a cheap integrity check — broken/missing files get re-downloaded, the rest
  is left alone.
- ReID 2023 is public; no NDA password is required.
- Final layout (under --target):
    dataSoccerNet/reid-2023/
        train.zip, valid.zip, test.zip, challenge.zip   (always — downloaded)
        train/, valid/, test/, challenge/                (only with --extract)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Files we expect to download. Sizes from observed values, in GB, approximate.
EXPECTED_ZIPS: dict[str, float] = {
    "train.zip": 11.0,
    "valid.zip": 2.2,
    "test.zip": 2.2,
    "challenge.zip": 1.6,
}

# After extraction, expected PNG counts per split (from official bbox_info.json).
EXPECTED_FILE_COUNTS: dict[str, int] = {
    "train": 248_234,
    "valid": 11_638 + 34_355,   # query + gallery
    "test": 11_777 + 34_989,
    "challenge": 9_021 + 26_082,
}


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(target: Path, splits: list[str], force: bool) -> None:
    """Run SoccerNetDownloader for the given splits."""
    from SoccerNet.Downloader import SoccerNetDownloader  # imported lazily

    reid_dir = target / "reid-2023"

    if force:
        for split in splits:
            zip_path = reid_dir / f"{split}.zip"
            if zip_path.exists():
                print(f"[force] Removing {zip_path}")
                zip_path.unlink()

    target.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(target))
    print(f"\nDownloading task='reid-2023' splits={splits} → {target.resolve()}")
    print("(SoccerNetDownloader skips files whose MD5 already matches; this can be fast.)")
    t0 = time.perf_counter()
    downloader.downloadDataTask(task="reid-2023", split=splits)
    print(f"\nDownload phase done in {time.perf_counter() - t0:.0f}s")

    # Confirm zip files now exist
    missing = [s for s in splits if not (reid_dir / f"{s}.zip").exists()]
    if missing:
        print(f"ERROR: expected zip(s) missing after download: {missing}", file=sys.stderr)
        sys.exit(2)


def extract(target: Path, splits: list[str], force: bool) -> None:
    """Extract the downloaded zips into per-split directories."""
    reid_dir = target / "reid-2023"

    for split in splits:
        zip_path = reid_dir / f"{split}.zip"
        out_dir = reid_dir / split

        if not zip_path.is_file():
            print(f"  skip {split}: {zip_path} not found", file=sys.stderr)
            continue

        if out_dir.is_dir() and any(out_dir.iterdir()):
            if force:
                print(f"[force] Removing existing {out_dir} ...")
                shutil.rmtree(out_dir)
            else:
                print(f"  skip extraction of {split}: {out_dir} already exists (use --force to re-extract)")
                continue

        print(f"\nExtracting {zip_path.name} ({_human_bytes(zip_path.stat().st_size)}) → {out_dir} ...")
        t0 = time.perf_counter()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(reid_dir)
        print(f"  done in {time.perf_counter() - t0:.0f}s")


def verify(target: Path, splits: list[str]) -> int:
    """Count PNG files per split and compare to expected. Returns 0 if all match."""
    reid_dir = target / "reid-2023"
    failures = 0
    print("\nVerifying file counts:")
    for split in splits:
        split_dir = reid_dir / split
        if not split_dir.is_dir():
            print(f"  {split}: SKIP (directory missing)")
            continue
        actual = sum(1 for _ in split_dir.rglob("*.png"))
        expected = EXPECTED_FILE_COUNTS.get(split)
        status = "OK" if expected is None or actual == expected else "MISMATCH"
        if status == "MISMATCH":
            failures += 1
        print(f"  {split:10s} actual={actual:>7,d}  expected={expected:>7,d}  [{status}]")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("dataSoccerNet"),
        help="Directory where SoccerNet creates reid-2023/ subfolder (default: dataSoccerNet)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test", "challenge"],
        choices=["train", "valid", "test", "challenge"],
        help="Which splits to download (default: all four)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing zips/extracts before re-downloading (~18 GB transfer)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Unzip into per-split directories after download",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After extraction, count PNG files per split and compare to expected",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download phase (useful with --extract --verify on existing zips)",
    )
    args = parser.parse_args()

    print(f"Target: {args.target.resolve()}")
    print(f"Splits: {args.splits}")
    print(
        f"Estimated download size: ~{sum(EXPECTED_ZIPS[f'{s}.zip'] for s in args.splits):.1f} GB"
    )

    if not args.no_download:
        download(args.target, args.splits, args.force)

    if args.extract:
        extract(args.target, args.splits, args.force)

    if args.verify:
        if not args.extract and not all(
            (args.target / "reid-2023" / s).is_dir() for s in args.splits
        ):
            print(
                "WARNING: --verify needs extracted directories. "
                "Either pass --extract too, or extract manually first.",
                file=sys.stderr,
            )
        failures = verify(args.target, args.splits)
        if failures:
            print(f"\n{failures} split(s) failed verification.", file=sys.stderr)
            return 4

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
