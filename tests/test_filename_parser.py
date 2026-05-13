"""Unit tests for the filename parser. No dataset required."""
from __future__ import annotations

import pytest

from soccernet_reid.data.filename_parser import (
    parse_annotated_filename,
    parse_challenge_filename,
)


class TestAnnotatedFilenames:
    def test_typical_player_left(self) -> None:
        # Real example from train (Paris SG vs Toulouse, action 20)
        name = "116652-4313-75403-11351-Player_team_left-10-07aa00020045669b0007-130x63.png"
        p = parse_annotated_filename(name)
        assert p.bbox_idx == 116652
        assert p.action_idx == 4313
        assert p.person_uid == 75403
        assert p.frame_idx == 11351
        assert p.clazz == "Player_team_left"
        assert p.id_ == "10"
        assert p.uai == "07aa00020045669b0007"
        assert p.height == 130
        assert p.width == 63

    def test_id_can_be_literal_None(self) -> None:
        # First entry of train has id 'None' (string)
        name = "0-0-0-0-Player_team_left-None-000r000_004597cb000d-74x32.png"
        p = parse_annotated_filename(name)
        assert p.id_ == "None"
        assert p.bbox_idx == 0
        assert p.action_idx == 0

    def test_player_team_right(self) -> None:
        name = "54404-1940-33397-5089-Player_team_right-9-014r002_00858c3b0003-775x449.png"
        p = parse_annotated_filename(name)
        assert p.clazz == "Player_team_right"
        assert p.id_ == "9"
        assert p.uai == "014r002_00858c3b0003"
        assert p.height == 775
        assert p.width == 449

    def test_uai_with_underscores_preserved(self) -> None:
        name = "5183-746-174491-1998-Player_team_left-23-02ba00013034513b0009-107x60.png"
        p = parse_annotated_filename(name)
        assert p.uai == "02ba00013034513b0009"

    @pytest.mark.parametrize(
        "bad",
        [
            "not-a-png-file.jpg",  # wrong suffix
            "1-2-3.png",  # too few fields
            "1-2-3-4-5-6-7-8-9.png",  # too many fields
            "abc-2-3-4-Player_team_left-x-uai-12x34.png",  # non-int bbox_idx
            "1-2-3-4-Player_team_left-x-uai-axb.png",  # non-int dimensions
        ],
    )
    def test_malformed_raises(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_annotated_filename(bad)


class TestChallengeFilenames:
    def test_typical(self) -> None:
        p = parse_challenge_filename("340-17-115x51.png")
        assert p.bbox_idx == 340
        assert p.action_idx == 17
        assert p.height == 115
        assert p.width == 51

    def test_large_indices(self) -> None:
        p = parse_challenge_filename("16977-880-624x185.png")
        assert p.bbox_idx == 16977
        assert p.action_idx == 880
        assert p.height == 624
        assert p.width == 185

    @pytest.mark.parametrize(
        "bad",
        [
            "1-2.png",  # too few fields
            "1-2-3-4.png",  # too many
            "1-2-12345.png",  # missing 'x' in dims
            "1-2-12x.png",  # bad width
            "1-2-12x34.jpg",  # wrong extension
        ],
    )
    def test_malformed_raises(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_challenge_filename(bad)
