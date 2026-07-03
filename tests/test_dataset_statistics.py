"""
ScrewMetric — Tests: Dataset Statistics
=========================================
Unit tests for ``dataset/scripts/dataset_statistics.py``.

Coverage:
- Statistics computed from clean synthetic COCO data
- Resolution statistics (mean, min, max, std)
- Bounding-box statistics
- Polygon statistics
- Category distribution
- Missing-label count
- Multi-annotation count
- Split statistics loaded from existing split files
- Dataset size computation
- Serialisation (to_dict)
- Failure when annotation file is missing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from dataset_statistics import (
    DatasetStatisticsComputer,
    DatasetStatistics,
    ResolutionStats,
    BBoxStats,
    PolygonStats,
    SplitStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute(tmp_dataset: dict[str, Any]) -> DatasetStatistics:
    return DatasetStatisticsComputer(tmp_dataset["config"]).compute()


# ===========================================================================
# DatasetStatistics dataclass
# ===========================================================================

class TestDatasetStatisticsDataclass:
    def test_to_dict_keys(self) -> None:
        s = DatasetStatistics()
        d = s.to_dict()
        required = {
            "total_images", "total_annotations", "category_distribution",
            "resolution", "bbox", "polygon", "splits",
            "dataset_size_bytes", "dataset_size_human",
            "missing_labels", "images_with_multiple_annotations",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_is_json_serialisable(self) -> None:
        s = DatasetStatistics()
        json.dumps(s.to_dict())  # must not raise

    def test_splits_in_to_dict_are_dicts(self) -> None:
        s = DatasetStatistics(splits={"train": SplitStats(count=5)})
        d = s.to_dict()
        assert isinstance(d["splits"]["train"], dict)
        assert d["splits"]["train"]["count"] == 5


# ===========================================================================
# Core statistics computation
# ===========================================================================

class TestDatasetStatisticsComputer:
    def test_compute_returns_statistics_object(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        assert isinstance(stats, DatasetStatistics)

    def test_total_images_correct(self, tmp_dataset: dict[str, Any]) -> None:
        n = len(tmp_dataset["coco"]["images"])
        stats = _compute(tmp_dataset)
        assert stats.total_images == n

    def test_total_annotations_correct(self, tmp_dataset: dict[str, Any]) -> None:
        n = len(tmp_dataset["coco"]["annotations"])
        stats = _compute(tmp_dataset)
        assert stats.total_annotations == n

    def test_category_distribution_contains_screw(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        assert "screw" in stats.category_distribution
        assert stats.category_distribution["screw"] > 0

    def test_category_distribution_sum_equals_annotations(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        total_from_dist = sum(stats.category_distribution.values())
        assert total_from_dist == stats.total_annotations

    def test_no_missing_labels_in_clean_dataset(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        assert stats.missing_labels == 0

    def test_missing_labels_detected(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        coco = tmp_dataset["coco"]
        # Add an image with no annotation
        coco["images"].append({
            "id": 9999, "file_name": "extra.jpg",
            "width": 640, "height": 480, "license": 0,
            "flickr_url": "", "coco_url": "", "date_captured": 0,
        })
        config.paths.default_annotation_file.write_text(
            json.dumps(coco), encoding="utf-8"
        )
        stats = _compute(tmp_dataset)
        assert stats.missing_labels == 1

    def test_stats_file_written_to_disk(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _compute(tmp_dataset)
        assert config.paths.dataset_stats_path.exists()

    def test_stats_file_is_valid_json(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _compute(tmp_dataset)
        raw = config.paths.dataset_stats_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert "total_images" in parsed


# ===========================================================================
# Resolution statistics
# ===========================================================================

class TestResolutionStats:
    def test_avg_width_correct(self, tmp_dataset: dict[str, Any]) -> None:
        # All synthetic images are 640×480
        stats = _compute(tmp_dataset)
        assert stats.resolution.avg_width == 640.0

    def test_avg_height_correct(self, tmp_dataset: dict[str, Any]) -> None:
        stats = _compute(tmp_dataset)
        assert stats.resolution.avg_height == 480.0

    def test_min_max_width_equal_for_uniform_images(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        assert stats.resolution.min_width == stats.resolution.max_width == 640

    def test_empty_images_returns_default_stats(self) -> None:
        from dataset_statistics import DatasetStatisticsComputer
        # Use the private method directly
        computer = object.__new__(DatasetStatisticsComputer)
        result = DatasetStatisticsComputer._compute_resolution_stats(computer, [])
        assert result.avg_width == 0.0

    def test_std_dev_zero_for_uniform_images(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        assert stats.resolution.std_width == 0.0
        assert stats.resolution.std_height == 0.0


# ===========================================================================
# Bounding-box statistics
# ===========================================================================

class TestBBoxStats:
    def test_bbox_count_equals_annotations(self, tmp_dataset: dict[str, Any]) -> None:
        n_ann = len(tmp_dataset["coco"]["annotations"])
        stats = _compute(tmp_dataset)
        assert stats.bbox.count == n_ann

    def test_bbox_avg_width_positive(self, tmp_dataset: dict[str, Any]) -> None:
        stats = _compute(tmp_dataset)
        assert stats.bbox.avg_width > 0

    def test_bbox_avg_area_is_width_times_height(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        # Synthetic bbox: [10,10,40,70] → area = 40*70 = 2800
        assert abs(stats.bbox.avg_area - 2800.0) < 1.0

    def test_empty_annotations_returns_default_bbox(self) -> None:
        computer = object.__new__(DatasetStatisticsComputer)
        result = DatasetStatisticsComputer._compute_bbox_stats(computer, [])
        assert result.count == 0


# ===========================================================================
# Polygon statistics
# ===========================================================================

class TestPolygonStats:
    def test_polygon_count_equals_annotations(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset)
        # One polygon per annotation
        assert stats.polygon.count == stats.total_annotations

    def test_avg_points_per_polygon(self, tmp_dataset: dict[str, Any]) -> None:
        stats = _compute(tmp_dataset)
        # Synthetic segmentation: 4 points (8 coords / 2)
        assert stats.polygon.avg_points_per_polygon == 4.0

    def test_empty_annotations_returns_default_polygon(self) -> None:
        computer = object.__new__(DatasetStatisticsComputer)
        result = DatasetStatisticsComputer._compute_polygon_stats(computer, [])
        assert result.count == 0


# ===========================================================================
# Split statistics
# ===========================================================================

class TestSplitStatistics:
    def test_split_stats_populated_after_split(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset_with_split)
        assert stats.splits["train"].count > 0
        assert stats.splits["val"].count > 0
        assert stats.splits["test"].count > 0

    def test_split_ratios_sum_to_one(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        stats = _compute(tmp_dataset_with_split)
        total_ratio = sum(s.ratio for s in stats.splits.values())
        assert abs(total_ratio - 1.0) < 0.02  # allow rounding

    def test_split_stats_fallback_for_missing_split(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        # No split has been run → SplitStats should default to zeros
        stats = _compute(tmp_dataset)
        for name in ("train", "val", "test"):
            assert stats.splits[name].count == 0


# ===========================================================================
# Dataset size
# ===========================================================================

class TestDatasetSize:
    def test_dataset_size_bytes_positive(self, tmp_dataset: dict[str, Any]) -> None:
        stats = _compute(tmp_dataset)
        assert stats.dataset_size_bytes > 0

    def test_dataset_size_human_is_string(self, tmp_dataset: dict[str, Any]) -> None:
        stats = _compute(tmp_dataset)
        assert isinstance(stats.dataset_size_human, str)
        assert len(stats.dataset_size_human) > 0


# ===========================================================================
# Failure modes
# ===========================================================================

class TestDatasetStatisticsFailures:
    def test_missing_annotation_file_raises(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        config.paths.default_annotation_file.unlink()
        with pytest.raises(FileNotFoundError):
            DatasetStatisticsComputer(config).compute()
