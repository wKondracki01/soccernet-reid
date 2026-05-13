"""Catalog builder for SoccerNet ReID 2023.

Builds a single pandas DataFrame indexing every image across train/valid/test/challenge,
using `bbox_info.json` files as the source of truth and reconstructing filenames from
the metadata. The DataFrame can be cached to Parquet for fast reload.

Expected layout (under reid_root):
    train/    train_bbox_info.json + <champ>/<season>/<game>/<action>/<file>.png
    valid/    bbox_info.json + query/<champ>/.../<file>.png + gallery/<champ>/.../<file>.png
    test/     bbox_info.json + query/... + gallery/...
    challenge/  query/<file>.png + gallery/<file>.png  (flat, no annotations)
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from soccernet_reid.data.filename_parser import (
    parse_annotated_filename,
    parse_challenge_filename,
)

SPLIT_NAMES: tuple[str, ...] = ("train", "valid", "test", "challenge")

# Per §4.3 of PLAN_EKSPERYMENTOW.md: training uses only "player-like" classes.
PLAYER_CLASSES: frozenset[str] = frozenset(
    {
        "Player_team_left",
        "Player_team_right",
        "Goalkeeper_team_left",
        "Goalkeeper_team_right",
    }
)

# Columns in the catalog DataFrame, in stable order.
_COLUMNS: tuple[str, ...] = (
    "split",          # train | valid | test | challenge
    "role",           # query | gallery | <NA> (train has no role)
    "bbox_idx",       # int64 — unique within (split, role) only; NOT globally unique
    "action_idx",     # int64
    "person_uid",     # Int64 nullable — None for challenge
    "frame_idx",      # Int64 nullable — None for challenge
    "clazz",          # str — None for challenge
    "id_",            # str — None for challenge; can be literal "None" string in train
    "uai",            # str — None for challenge
    "championship",   # str — None for challenge (flat layout)
    "season",         # str — None for challenge
    "game",           # str — None for challenge
    "relative_path",  # str — directory relative to split root, "" for challenge
    "filename",       # str — reconstructed image filename
    "path",           # str — absolute path to image
    "height",         # int64
    "width",          # int64
)


def _annotated_filename(entry: dict[str, Any]) -> str:
    """Reconstruct the annotated filename from a bbox_info entry."""
    return (
        f"{entry['bbox_idx']}-"
        f"{entry['action_idx']}-"
        f"{entry['person_uid']}-"
        f"{entry['frame_idx']}-"
        f"{entry['clazz']}-"
        f"{entry['id']}-"
        f"{entry['UAI']}-"
        f"{entry['height']}x{entry['width']}.png"
    )


def expected_filename(entry: dict[str, Any]) -> str:
    """Public alias for :func:`_annotated_filename` for use in tests/scripts."""
    return _annotated_filename(entry)


def _parse_relative_path(rel: str) -> tuple[str, str, str]:
    """Split `championship/season/game/action` → (championship, season, game).

    The trailing path segment is the action index, redundant with `action_idx` in
    the bbox entry; we discard it here.
    """
    parts = rel.split("/")
    if len(parts) < 4:
        raise ValueError(f"relative_path has too few segments: {rel!r}")
    return parts[0], parts[1], parts[2]


def load_bbox_info(reid_root: Path, split: str) -> dict[str, Any]:
    """Load the raw bbox_info JSON for an annotated split.

    Returns the deserialized JSON. Structure differs between splits:
        train:        {bbox_idx_str: entry, ...}                (flat)
        valid / test: {"query": {...}, "gallery": {...}}        (split by role)
    """
    if split == "train":
        path = reid_root / "train" / "train_bbox_info.json"
    elif split in ("valid", "test"):
        path = reid_root / split / "bbox_info.json"
    else:
        raise ValueError(f"bbox_info exists only for train/valid/test, got {split!r}")
    with path.open() as f:
        return json.load(f)


def _row_from_annotated_entry(
    entry: dict[str, Any],
    split: str,
    role: str | None,
    reid_root: Path,
) -> dict[str, Any]:
    championship, season, game = _parse_relative_path(entry["relative_path"])
    filename = _annotated_filename(entry)
    split_root = reid_root / split
    if role is not None:
        full_dir = split_root / role / entry["relative_path"]
    else:
        full_dir = split_root / entry["relative_path"]
    return {
        "split": split,
        "role": role,
        "bbox_idx": entry["bbox_idx"],
        "action_idx": entry["action_idx"],
        "person_uid": entry["person_uid"],
        "frame_idx": entry["frame_idx"],
        "clazz": entry["clazz"],
        "id_": entry["id"],
        "uai": entry["UAI"],
        "championship": championship,
        "season": season,
        "game": game,
        "relative_path": entry["relative_path"],
        "filename": filename,
        "path": str(full_dir / filename),
        "height": entry["height"],
        "width": entry["width"],
    }


def _rows_from_train(reid_root: Path) -> Iterable[dict[str, Any]]:
    data = load_bbox_info(reid_root, "train")
    for entry in data.values():
        yield _row_from_annotated_entry(entry, split="train", role=None, reid_root=reid_root)


def _rows_from_valid_or_test(reid_root: Path, split: str) -> Iterable[dict[str, Any]]:
    data = load_bbox_info(reid_root, split)
    for role in ("query", "gallery"):
        if role not in data:
            raise ValueError(f"{split} bbox_info missing role {role!r}")
        for entry in data[role].values():
            yield _row_from_annotated_entry(entry, split=split, role=role, reid_root=reid_root)


def _rows_from_challenge(reid_root: Path) -> Iterable[dict[str, Any]]:
    """Challenge has no annotations — enumerate the filesystem directly."""
    ch_root = reid_root / "challenge"
    for role in ("query", "gallery"):
        role_dir = ch_root / role
        if not role_dir.is_dir():
            raise ValueError(f"Expected directory {role_dir}")
        for png_path in sorted(role_dir.iterdir()):
            if png_path.suffix != ".png":
                continue
            parsed = parse_challenge_filename(png_path.name)
            yield {
                "split": "challenge",
                "role": role,
                "bbox_idx": parsed.bbox_idx,
                "action_idx": parsed.action_idx,
                "person_uid": None,
                "frame_idx": None,
                "clazz": None,
                "id_": None,
                "uai": None,
                "championship": None,
                "season": None,
                "game": None,
                "relative_path": "",
                "filename": png_path.name,
                "path": str(png_path),
                "height": parsed.height,
                "width": parsed.width,
            }


def build_catalog(
    reid_root: Path | str,
    splits: Iterable[str] = SPLIT_NAMES,
) -> pd.DataFrame:
    """Build the full catalog DataFrame from `bbox_info.json` files and challenge files.

    Args:
        reid_root: path to the `reid-2023/` directory containing train/valid/test/challenge.
        splits: which splits to include. Defaults to all four.

    Returns:
        DataFrame with the columns defined in `_COLUMNS`. Nullable ints use Int64 dtype.
    """
    reid_root = Path(reid_root)
    rows: list[dict[str, Any]] = []
    for split in splits:
        if split == "train":
            rows.extend(_rows_from_train(reid_root))
        elif split in ("valid", "test"):
            rows.extend(_rows_from_valid_or_test(reid_root, split))
        elif split == "challenge":
            rows.extend(_rows_from_challenge(reid_root))
        else:
            raise ValueError(f"Unknown split: {split!r}")

    df = pd.DataFrame(rows, columns=list(_COLUMNS))
    # Nullable int columns (challenge has None for person_uid/frame_idx)
    df["person_uid"] = df["person_uid"].astype("Int64")
    df["frame_idx"] = df["frame_idx"].astype("Int64")
    df["bbox_idx"] = df["bbox_idx"].astype("int64")
    df["action_idx"] = df["action_idx"].astype("int64")
    df["height"] = df["height"].astype("int64")
    df["width"] = df["width"].astype("int64")
    return df


def save_catalog(df: pd.DataFrame, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def load_catalog(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


# --- Verification helpers ---------------------------------------------------


def verify_paths_exist(df: pd.DataFrame, sample_size: int | None = 200) -> list[str]:
    """Check that file paths in the catalog exist on disk.

    Args:
        df: catalog DataFrame.
        sample_size: if None, checks every row (slow on full catalog).
                     Otherwise samples this many rows uniformly at random.

    Returns:
        List of missing paths (empty if all OK).
    """
    if sample_size is None or sample_size >= len(df):
        sample = df
    else:
        sample = df.sample(n=sample_size, random_state=42)
    missing = [p for p in sample["path"].tolist() if not Path(p).is_file()]
    return missing


def verify_image_dimensions(df: pd.DataFrame, sample_size: int = 50) -> list[tuple[str, str]]:
    """Open a sample of images and verify that PIL-reported size matches the catalog.

    Returns list of (path, error_message) for mismatches.
    """
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    mismatches: list[tuple[str, str]] = []
    for _, row in sample.iterrows():
        path = Path(row["path"])
        try:
            with Image.open(path) as im:
                actual_w, actual_h = im.size  # PIL returns (width, height)
        except Exception as e:
            mismatches.append((str(path), f"open failed: {e}"))
            continue
        if actual_h != row["height"] or actual_w != row["width"]:
            mismatches.append(
                (
                    str(path),
                    f"expected {row['height']}x{row['width']}, got {actual_h}x{actual_w}",
                )
            )
    return mismatches


def verify_filename_roundtrip(df: pd.DataFrame, sample_size: int = 500) -> list[str]:
    """Parse a sample of filenames and verify fields match the catalog.

    This is the sanity check that confirms our filename parser agrees with bbox_info.
    Challenge rows are skipped (they don't have person_uid/clazz/etc.).
    """
    annotated = df[df["split"] != "challenge"]
    if len(annotated) == 0:
        return []
    sample = annotated.sample(n=min(sample_size, len(annotated)), random_state=42)
    errors: list[str] = []
    for _, row in sample.iterrows():
        try:
            parsed = parse_annotated_filename(row["filename"])
        except ValueError as e:
            errors.append(f"{row['filename']}: parse failed ({e})")
            continue
        mismatches = []
        if parsed.bbox_idx != row["bbox_idx"]:
            mismatches.append(f"bbox_idx {parsed.bbox_idx}!={row['bbox_idx']}")
        if parsed.action_idx != row["action_idx"]:
            mismatches.append(f"action_idx {parsed.action_idx}!={row['action_idx']}")
        if parsed.person_uid != row["person_uid"]:
            mismatches.append(f"person_uid {parsed.person_uid}!={row['person_uid']}")
        if parsed.frame_idx != row["frame_idx"]:
            mismatches.append(f"frame_idx {parsed.frame_idx}!={row['frame_idx']}")
        if parsed.clazz != row["clazz"]:
            mismatches.append(f"clazz {parsed.clazz!r}!={row['clazz']!r}")
        if parsed.id_ != row["id_"]:
            mismatches.append(f"id {parsed.id_!r}!={row['id_']!r}")
        if parsed.uai != row["uai"]:
            mismatches.append(f"uai {parsed.uai!r}!={row['uai']!r}")
        if parsed.height != row["height"] or parsed.width != row["width"]:
            mismatches.append(f"size {parsed.height}x{parsed.width}!={row['height']}x{row['width']}")
        if mismatches:
            errors.append(f"{row['filename']}: " + "; ".join(mismatches))
    return errors


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Return a dictionary of summary statistics suitable for printing or logging."""
    summary: dict[str, Any] = {"total_rows": len(df)}
    for split in SPLIT_NAMES:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        per_split = {"rows": len(sub)}
        if "role" in sub.columns:
            roles = sub["role"].value_counts(dropna=False).to_dict()
            per_split["by_role"] = {str(k): int(v) for k, v in roles.items()}
        if split != "challenge":
            per_split["unique_actions"] = int(sub["action_idx"].nunique())
            per_split["unique_action_uid_pairs"] = int(
                sub.groupby(["action_idx", "person_uid"]).ngroups
            )
            per_split["unique_classes"] = int(sub["clazz"].nunique())
        summary[split] = per_split
    return summary


def _set_seed(seed: int) -> None:
    """Convenience: seed the stdlib random module used by verify_* sampling fallbacks."""
    random.seed(seed)


# --- Training-side helpers --------------------------------------------------


def filter_to_player_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose `clazz` is one of :data:`PLAYER_CLASSES`.

    Per the plan (§4.3) this is the standard training filter — it removes the
    `Main_referee`, `Side_referee`, and `Staff_members` rows which represent
    out-of-domain identities and would only add noise to player re-identification.
    The filter is applied to TRAINING only; evaluation always uses the full
    `valid`/`test` query/gallery split for parity with the official leaderboard.
    """
    return df[df["clazz"].isin(PLAYER_CLASSES)].reset_index(drop=True)


def assign_class_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `class_id` column: contiguous int ID per unique (action_idx, person_uid) pair.

    The label space treats each (action, uid) as its own class — required by the
    fact that SoccerNet identity labels are valid only within an action (§1.1).
    `class_id` ranges 0..C-1 where C is the number of unique pairs in `df`.

    Returns a new DataFrame with the column appended. Rows where `person_uid` is
    NA (e.g., challenge split) get `class_id = -1`.
    """
    out = df.copy()
    has_label = out["person_uid"].notna()
    pair_keys = list(
        zip(
            out.loc[has_label, "action_idx"].astype("int64"),
            out.loc[has_label, "person_uid"].astype("int64"),
            strict=False,
        )
    )
    # Build deterministic mapping in sorted order so two runs over the same df
    # produce identical class_id columns (no Python dict-insertion-order surprises).
    unique_pairs = sorted(set(pair_keys))
    pair_to_id = {pair: i for i, pair in enumerate(unique_pairs)}
    class_id_col = pd.Series(-1, index=out.index, dtype="int64")
    class_id_col.loc[has_label] = [pair_to_id[p] for p in pair_keys]
    out["class_id"] = class_id_col
    return out
