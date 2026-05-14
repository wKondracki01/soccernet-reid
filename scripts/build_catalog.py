"""Build the SoccerNet ReID catalog DataFrame and cache it to Parquet.

Usage:
    uv run python scripts/build_catalog.py \\
        --reid-root dataSoccerNet/reid-2023 \\
        --out outputs/catalog.parquet \\
        [--no-verify]   # skip on-disk verification
        [--full-verify] # check that every catalog path exists (slow)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running this script directly (without `uv run -m ...`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_reid.data.catalog import (  # noqa: E402
    build_catalog,
    save_catalog,
    summarize,
    verify_filename_roundtrip,
    verify_image_dimensions,
    verify_paths_exist,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reid-root", type=Path, required=True, help="Path to reid-2023/ directory")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--no-verify", action="store_true", help="Skip on-disk verification")
    parser.add_argument(
        "--full-verify",
        action="store_true",
        help="Check existence of EVERY file (slow, ~30s for 376k files)",
    )
    parser.add_argument(
        "--dimension-sample",
        type=int,
        default=100,
        help="How many images to open and verify dimensions (default: 100)",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop catalog rows whose path does NOT exist on disk (instead of "
        "failing). Useful on Windows where some files extract with mangled "
        "names due to non-ASCII chars (rare — ~9/376k for SoccerNet ReID 2023).",
    )
    args = parser.parse_args()

    if not args.reid_root.is_dir():
        print(f"ERROR: --reid-root {args.reid_root} is not a directory", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    print(f"Building catalog from {args.reid_root} ...")
    df = build_catalog(args.reid_root)
    print(f"  built {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    print("\nSummary:")
    print(json.dumps(summarize(df), indent=2, default=str))

    if not args.no_verify:
        print("\nVerifying filename round-trip (sample 500) ...")
        rt_errors = verify_filename_roundtrip(df, sample_size=500)
        if rt_errors:
            print(f"  FAIL: {len(rt_errors)} mismatches; first 5:", file=sys.stderr)
            for e in rt_errors[:5]:
                print(f"    {e}", file=sys.stderr)
            return 2
        print("  OK")

        if args.full_verify:
            print("\nVerifying ALL paths exist on disk (this may take ~30s) ...")
            missing = verify_paths_exist(df, sample_size=None)
        else:
            print("\nVerifying random sample of paths exist on disk ...")
            missing = verify_paths_exist(df, sample_size=200)
        if missing:
            if args.drop_missing:
                missing_set = set(missing)
                before = len(df)
                df = df[~df["path"].isin(missing_set)].reset_index(drop=True)
                print(
                    f"  {len(missing)} files missing; dropped from catalog "
                    f"({before:,} -> {len(df):,} rows)"
                )
                for m in missing[:5]:
                    print(f"    dropped: {m}")
            else:
                print(f"  FAIL: {len(missing)} missing files; first 5:", file=sys.stderr)
                for m in missing[:5]:
                    print(f"    {m}", file=sys.stderr)
                print(
                    "  (re-run with --drop-missing to filter these out of the catalog)",
                    file=sys.stderr,
                )
                return 3
        else:
            print("  OK")

        print(f"\nVerifying image dimensions (sample {args.dimension_sample}) ...")
        dim_errors = verify_image_dimensions(df, sample_size=args.dimension_sample)
        if dim_errors:
            print(f"  FAIL: {len(dim_errors)} mismatches; first 5:", file=sys.stderr)
            for path, err in dim_errors[:5]:
                print(f"    {path}: {err}", file=sys.stderr)
            return 4
        print("  OK")

    save_catalog(df, args.out)
    print(f"\nSaved catalog to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
