"""Filename parser for SoccerNet ReID 2023.

The README claims `_` as field separator, but real files use `-`. Class names
(`Player_team_left`, `Goalkeeper_team_right`) and UAI tokens contain `_` internally,
so splitting by `-` cleanly separates exactly 8 fields.

Annotated splits (train / valid / test):
    <bbox_idx>-<action_idx>-<person_uid>-<frame_idx>-<class>-<id>-<UAI>-<HxW>.png

Challenge:
    <bbox_idx>-<action_idx>-<HxW>.png
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedAnnotatedName:
    bbox_idx: int
    action_idx: int
    person_uid: int
    frame_idx: int
    clazz: str
    id_: str
    uai: str
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ParsedChallengeName:
    bbox_idx: int
    action_idx: int
    height: int
    width: int


def _split_hw(token: str) -> tuple[int, int]:
    h_str, w_str = token.split("x", maxsplit=1)
    return int(h_str), int(w_str)


def parse_annotated_filename(filename: str) -> ParsedAnnotatedName:
    """Parse a train/valid/test filename. Raises ValueError on malformed input."""
    if not filename.endswith(".png"):
        raise ValueError(f"Expected .png suffix: {filename!r}")
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) != 8:
        raise ValueError(f"Expected 8 hyphen-separated fields, got {len(parts)}: {filename!r}")
    try:
        bbox_idx = int(parts[0])
        action_idx = int(parts[1])
        person_uid = int(parts[2])
        frame_idx = int(parts[3])
        height, width = _split_hw(parts[7])
    except ValueError as e:
        raise ValueError(f"Bad integer field in {filename!r}: {e}") from e
    return ParsedAnnotatedName(
        bbox_idx=bbox_idx,
        action_idx=action_idx,
        person_uid=person_uid,
        frame_idx=frame_idx,
        clazz=parts[4],
        id_=parts[5],
        uai=parts[6],
        height=height,
        width=width,
    )


def parse_challenge_filename(filename: str) -> ParsedChallengeName:
    """Parse a challenge filename (no person_uid, frame_idx, class, id, UAI)."""
    if not filename.endswith(".png"):
        raise ValueError(f"Expected .png suffix: {filename!r}")
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 hyphen-separated fields, got {len(parts)}: {filename!r}")
    try:
        bbox_idx = int(parts[0])
        action_idx = int(parts[1])
        height, width = _split_hw(parts[2])
    except ValueError as e:
        raise ValueError(f"Bad integer field in {filename!r}: {e}") from e
    return ParsedChallengeName(
        bbox_idx=bbox_idx,
        action_idx=action_idx,
        height=height,
        width=width,
    )
