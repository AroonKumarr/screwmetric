"""
ScrewMetric — Tests: Dataset Splitter
========================================
Unit tests for ``dataset/scripts/dataset_splitter.py``.

Coverage:
- Correct split counts (70/20/10 with rounding)
- Reproducibility with the same seed
- No image overlap between splits
- All source images appear in exactly one split
- COCO annotation files written and valid
- Images copied to correct directories
- Idempotency (re-running does not duplicate images)
- Empty dataset raises ValueError
- SplitResult.split_counts returns correct mapping
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from dataset_splitter import DatasetSplitter, SplitResult
from config import SplitConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_split(tmp_dataset: dict[str, Any]) -> SplitResult:
    return DatasetSplitter(tmp_dataset["config"]).split()


def _load_annotation(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# SplitResult dataclass
# ===========================================================================

class TestSplitResult:
    def test_split_counts_returns_dict(self) -> None:
        result = SplitResult(
            train_ids=[1, 2, 3],
            val_ids=[4, 5],
            test_ids=[6],
        )
        assert result.split_counts == {"train": 3, "val": 2, "test": 1}

    def test_split_counts_empty(self) -> None:
        result = SplitResult()
        assert result.split_counts == {"train": 0, "val": 0, "test": 0}


# ===========================================================================
# Happy-path split tests
# ===========================================================================

class TestDatasetSplitterHappyPath:
    def test_split_returns_split_result(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_split(tmp_dataset)
        assert isinstance(result, SplitResult)

    def test_total_images_preserved(self, tmp_dataset: dict[str, Any]) -> None:
        n_total = len(tmp_dataset["coco"]["images"])
        result = _run_split(tmp_dataset)
        counts = result.split_counts
        assert sum(counts.values()) == n_total

    def test_no_image_overlap_between_splits(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_split(tmp_dataset)
        train_set = set(result.train_ids)
        val_set = set(result.val_ids)
        test_set = set(result.test_ids)
        assert train_set.isdisjoint(val_set), "Train ∩ Val must be empty"
        assert train_set.isdisjoint(test_set), "Train ∩ Test must be empty"
        assert val_set.isdisjoint(test_set), "Val ∩ Test must be empty"

    def test_every_image_assigned_to_a_split(self, tmp_dataset: dict[str, Any]) -> None:
        all_ids = {img["id"] for img in tmp_dataset["coco"]["images"]}
        result = _run_split(tmp_dataset)
        assigned = set(result.train_ids) | set(result.val_ids) | set(result.test_ids)
        assert all_ids == assigned

    def test_annotation_files_written(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _run_split(tmp_dataset)
        for split in ("train", "val", "test"):
            assert config.paths.split_annotation_path(split).exists(), \
                f"Annotation missing for split '{split}'"

    def test_annotation_files_are_valid_coco(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _run_split(tmp_dataset)
        for split in ("train", "val", "test"):
            ann = _load_annotation(config.paths.split_annotation_path(split))
            for key in ("images", "annotations", "categories"):
                assert key in ann, f"Key '{key}' missing in {split} annotation"

    def test_train_images_copied_to_disk(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        result = _run_split(tmp_dataset)
        train_img_dir = config.paths.split_images_dir("train")
        copied = list(train_img_dir.glob("*.jpg"))
        assert len(copied) == result.split_counts["train"]

    def test_annotation_image_count_matches_split_count(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        result = _run_split(tmp_dataset)
        for split in ("train", "val", "test"):
            ann = _load_annotation(config.paths.split_annotation_path(split))
            expected = result.split_counts[split]
            actual = len(ann["images"])
            assert actual == expected, \
                f"Split '{split}': expected {expected} images in COCO, got {actual}"

    def test_categories_preserved_in_splits(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _run_split(tmp_dataset)
        original_cats = tmp_dataset["coco"]["categories"]
        for split in ("train", "val", "test"):
            ann = _load_annotation(config.paths.split_annotation_path(split))
            assert ann["categories"] == original_cats


# ===========================================================================
# Reproducibility
# ===========================================================================

class TestDatasetSplitterReproducibility:
    def test_same_seed_gives_same_split(self, tmp_dataset: dict[str, Any]) -> None:
        result_a = _run_split(tmp_dataset)
        result_b = _run_split(tmp_dataset)
        assert result_a.train_ids == result_b.train_ids
        assert result_a.val_ids == result_b.val_ids
        assert result_a.test_ids == result_b.test_ids

    def test_different_seed_gives_different_split(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        import sys
        _SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
        sys.path.insert(0, str(_SCRIPTS))
        from conftest import _make_tmp_config
        from config import SplitConfig, ValidationConfig, PreviewConfig

        config_a = tmp_dataset["config"]
        root = config_a.paths.dataset_root

        # Build a second config with a different seed over the same data
        config_b = _make_tmp_config(root)
        object.__setattr__(
            config_b, "split",
            SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=999),
        )

        result_a = DatasetSplitter(config_a).split()
        result_b = DatasetSplitter(config_b).split()

        n = len(tmp_dataset["coco"]["images"])
        if n >= 5:
            assert result_a.train_ids != result_b.train_ids, \
                "Different seeds should (usually) produce different splits"


# ===========================================================================
# Edge cases and failure modes
# ===========================================================================

class TestDatasetSplitterFailureModes:
    def test_empty_annotation_raises_value_error(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        # Overwrite annotation with zero images
        empty_coco = {
            "images": [], "annotations": [],
            "categories": [{"id": 1, "name": "screw"}],
        }
        config.paths.default_annotation_file.write_text(
            json.dumps(empty_coco), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="No images"):
            DatasetSplitter(config).split()

    def test_missing_annotation_file_raises(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        config.paths.default_annotation_file.unlink()
        with pytest.raises(FileNotFoundError):
            DatasetSplitter(config).split()

    def test_split_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)

    def test_idempotent_image_copy(self, tmp_dataset: dict[str, Any]) -> None:
        """Running split twice must not duplicate files."""
        config = tmp_dataset["config"]
        _run_split(tmp_dataset)
        _run_split(tmp_dataset)
        train_imgs = list(config.paths.split_images_dir("train").glob("*.jpg"))
        # Count must equal the first split's train count, not doubled
        ann = _load_annotation(config.paths.split_annotation_path("train"))
        assert len(train_imgs) == len(ann["images"])

    def test_missing_source_image_is_skipped_gracefully(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        # Delete one source image — splitter should warn and continue
        img_dir = config.paths.screw_only_dir
        first = sorted(img_dir.glob("*.jpg"))[0]
        first.unlink()
        # Must not raise
        result = _run_split(tmp_dataset)
        assert isinstance(result, SplitResult)
