"""Integration tests for the catalog builder.

These run against the real local dataset at dataSoccerNet/reid-2023/ and are skipped
if it isn't present (e.g. on a fresh clone before download).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from soccernet_reid.data.catalog import (
    SPLIT_NAMES,
    build_catalog,
    expected_filename,
    load_bbox_info,
    summarize,
    verify_filename_roundtrip,
    verify_image_dimensions,
    verify_paths_exist,
)

REID_ROOT = Path(__file__).resolve().parent.parent / "dataSoccerNet" / "reid-2023"

# Counts derived from the real dataset (verified at catalog construction time).
EXPECTED_COUNTS = {
    "train": 248_234,
    "valid_query": 11_638,
    "valid_gallery": 34_355,
    "test_query": 11_777,
    "test_gallery": 34_989,
    "challenge_query": 9_021,
    "challenge_gallery": 26_082,
}


pytestmark = pytest.mark.skipif(
    not REID_ROOT.is_dir(),
    reason=f"SoccerNet ReID dataset not found at {REID_ROOT}",
)


@pytest.fixture(scope="module")
def catalog():
    return build_catalog(REID_ROOT)


class TestRowCounts:
    def test_total_matches_official_size(self, catalog) -> None:
        # 340 993 annotated + 35 103 challenge = 376 096 total
        assert len(catalog) == 376_096

    def test_train_size(self, catalog) -> None:
        assert (catalog["split"] == "train").sum() == EXPECTED_COUNTS["train"]

    @pytest.mark.parametrize("split", ["valid", "test"])
    def test_query_gallery_split(self, catalog, split: str) -> None:
        sub = catalog[catalog["split"] == split]
        q = (sub["role"] == "query").sum()
        g = (sub["role"] == "gallery").sum()
        assert q == EXPECTED_COUNTS[f"{split}_query"]
        assert g == EXPECTED_COUNTS[f"{split}_gallery"]

    def test_challenge_size(self, catalog) -> None:
        sub = catalog[catalog["split"] == "challenge"]
        assert (sub["role"] == "query").sum() == EXPECTED_COUNTS["challenge_query"]
        assert (sub["role"] == "gallery").sum() == EXPECTED_COUNTS["challenge_gallery"]


class TestSchema:
    def test_train_role_is_null(self, catalog) -> None:
        train = catalog[catalog["split"] == "train"]
        assert train["role"].isna().all()

    def test_challenge_has_null_annotations(self, catalog) -> None:
        ch = catalog[catalog["split"] == "challenge"]
        assert ch["person_uid"].isna().all()
        assert ch["frame_idx"].isna().all()
        assert ch["clazz"].isna().all()
        assert ch["uai"].isna().all()

    def test_annotated_have_complete_fields(self, catalog) -> None:
        annotated = catalog[catalog["split"].isin(["train", "valid", "test"])]
        assert annotated["person_uid"].notna().all()
        assert annotated["frame_idx"].notna().all()
        assert annotated["clazz"].notna().all()
        assert annotated["uai"].notna().all()
        assert annotated["championship"].notna().all()
        assert annotated["season"].notna().all()
        assert annotated["game"].notna().all()


class TestPathsAndFilenames:
    def test_random_paths_exist(self, catalog) -> None:
        missing = verify_paths_exist(catalog, sample_size=300)
        assert missing == [], f"Missing files (first 5): {missing[:5]}"

    def test_filename_roundtrip(self, catalog) -> None:
        errors = verify_filename_roundtrip(catalog, sample_size=500)
        assert errors == [], f"Roundtrip errors (first 5): {errors[:5]}"

    def test_image_dimensions(self, catalog) -> None:
        errors = verify_image_dimensions(catalog, sample_size=30)
        assert errors == [], f"Dimension mismatches (first 5): {errors[:5]}"

    def test_expected_filename_helper(self) -> None:
        # Round-trip the first train entry through the helper
        data = load_bbox_info(REID_ROOT, "train")
        entry = data["0"]
        fn = expected_filename(entry)
        # Field 4 is class — must contain 'Player_' or 'Goalkeeper_' or 'referee'
        assert fn.endswith(".png")
        assert "-" in fn  # uses hyphen separator
        # File must exist on disk
        full_path = REID_ROOT / "train" / entry["relative_path"] / fn
        assert full_path.is_file(), f"Reconstructed path does not exist: {full_path}"


class TestSummary:
    def test_summary_contains_all_splits(self, catalog) -> None:
        s = summarize(catalog)
        assert s["total_rows"] == 376_096
        for split in SPLIT_NAMES:
            assert split in s
        # Train should have many unique (action, uid) pairs — at least 50k
        assert s["train"]["unique_action_uid_pairs"] >= 50_000
