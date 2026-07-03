"""
ScrewMetric — Tests: Dataset Validator
========================================
Unit tests for ``dataset/scripts/dataset_validator.py``.

Coverage:
- Happy-path validation on a clean synthetic dataset
- Corrupted annotation file detection (missing file)
- Detection of missing images (in COCO but not on disk)
- Detection of extra images (on disk but not in COCO)
- Detection of duplicate image IDs
- Detection of orphan annotations
- Detection of invalid category IDs
- Detection of unannotated images
- Validation report serialisation (to_dict / is_valid)
- Report saved to disk
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure dataset/scripts is importable
_SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from dataset_validator import DatasetValidator, ValidationReport
from config import PipelineConfig, ValidationConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_validator(tmp_dataset: dict[str, Any]) -> DatasetValidator:
    return DatasetValidator(tmp_dataset["config"])


# ===========================================================================
# ValidationReport dataclass tests
# ===========================================================================

class TestValidationReport:
    """Tests for the :class:`ValidationReport` dataclass."""

    def test_is_valid_with_no_errors(self) -> None:
        report = ValidationReport()
        assert report.is_valid is True

    def test_is_invalid_with_errors(self) -> None:
        report = ValidationReport(errors=["something went wrong"])
        assert report.is_valid is False

    def test_to_dict_contains_all_fields(self) -> None:
        report = ValidationReport()
        d = report.to_dict()
        expected_keys = {
            "total_images_in_filesystem",
            "total_images_in_coco",
            "valid_images",
            "missing_images",
            "extra_images",
            "duplicate_filenames",
            "duplicate_image_ids",
            "corrupted_images",
            "unannotated_images",
            "orphan_annotations",
            "invalid_category_ids",
            "unsupported_extensions",
            "warnings",
            "errors",
            "is_valid",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_is_valid_is_computed(self) -> None:
        report = ValidationReport(errors=["e1"])
        assert report.to_dict()["is_valid"] is False

    def test_to_dict_is_json_serialisable(self) -> None:
        report = ValidationReport(warnings=["w1"], errors=["e1"])
        json_str = json.dumps(report.to_dict())
        assert "w1" in json_str


# ===========================================================================
# Happy-path validation
# ===========================================================================

class TestDatasetValidatorHappyPath:
    """Tests for a clean synthetic dataset."""

    def test_validate_returns_report(self, tmp_dataset: dict[str, Any]) -> None:
        report = _build_validator(tmp_dataset).validate()
        assert isinstance(report, ValidationReport)

    def test_validate_is_valid_for_clean_dataset(self, tmp_dataset: dict[str, Any]) -> None:
        report = _build_validator(tmp_dataset).validate()
        assert report.is_valid, f"Expected valid, got errors: {report.errors}"

    def test_validate_counts_images_correctly(self, tmp_dataset: dict[str, Any]) -> None:
        coco = tmp_dataset["coco"]
        report = _build_validator(tmp_dataset).validate()
        assert report.total_images_in_coco == len(coco["images"])

    def test_validate_no_missing_images(self, tmp_dataset: dict[str, Any]) -> None:
        report = _build_validator(tmp_dataset).validate()
        assert report.missing_images == []

    def test_validate_no_corrupted_images(self, tmp_dataset: dict[str, Any]) -> None:
        report = _build_validator(tmp_dataset).validate()
        assert report.corrupted_images == []

    def test_validate_no_orphan_annotations(self, tmp_dataset: dict[str, Any]) -> None:
        report = _build_validator(tmp_dataset).validate()
        assert report.orphan_annotations == []

    def test_validate_no_unannotated_images(self, tmp_dataset: dict[str, Any]) -> None:
        report = _build_validator(tmp_dataset).validate()
        assert report.unannotated_images == []

    def test_validate_report_saved_to_disk(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _build_validator(tmp_dataset).validate()
        assert config.paths.validation_report_path.exists()

    def test_validate_report_is_valid_json(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        _build_validator(tmp_dataset).validate()
        raw = config.paths.validation_report_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert "is_valid" in parsed


# ===========================================================================
# Failure-mode tests: missing structure
# ===========================================================================

class TestDatasetValidatorStructureFailures:
    """Tests for missing directories / files."""

    def test_missing_dataset_root_returns_errors(self, tmp_path: Path) -> None:
        import sys
        _SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
        sys.path.insert(0, str(_SCRIPTS))
        from conftest import _make_tmp_config

        # Use a non-existent subdirectory as the root
        bad_root = tmp_path / "does_not_exist"
        config = _make_tmp_config(bad_root)
        report = DatasetValidator(config).validate()
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_missing_annotation_file_returns_error(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        # Remove the annotation file
        config.paths.default_annotation_file.unlink()
        report = DatasetValidator(config).validate()
        assert not report.is_valid

    def test_corrupted_annotation_json_returns_error(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        config.paths.default_annotation_file.write_text("not valid json!!!",
                                                         encoding="utf-8")
        report = DatasetValidator(config).validate()
        assert not report.is_valid


# ===========================================================================
# Failure-mode tests: image-level problems
# ===========================================================================

class TestDatasetValidatorImageProblems:
    """Tests for per-image problems."""

    def test_missing_image_file_detected(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        # Delete one image from disk
        img_dir = config.paths.screw_only_dir
        first_img = sorted(img_dir.glob("*.jpg"))[0]
        first_img.unlink()
        report = DatasetValidator(config).validate()
        assert len(report.missing_images) == 1
        assert not report.is_valid

    def test_extra_image_on_disk_is_a_warning(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        # Write an extra image not in COCO
        extra = config.paths.screw_only_dir / "extra_not_in_coco.jpg"
        from PIL import Image as PILImage
        PILImage.new("RGB", (10, 10)).save(str(extra), "JPEG")
        report = DatasetValidator(config).validate()
        assert "extra_not_in_coco.jpg" in report.extra_images
        # Extra images produce warnings, not errors
        assert report.is_valid

    def test_unsupported_extension_produces_warning(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        txt_file = config.paths.screw_only_dir / "readme.txt"
        txt_file.write_text("not an image")
        report = DatasetValidator(config).validate()
        assert "readme.txt" in report.unsupported_extensions

    def test_corrupted_image_detected(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        img_dir = config.paths.screw_only_dir
        first_img = sorted(img_dir.glob("*.jpg"))[0]
        first_img.write_bytes(b"this is not valid jpeg data at all")
        report = DatasetValidator(config).validate()
        assert first_img.name in report.corrupted_images
        assert not report.is_valid


# ===========================================================================
# Failure-mode tests: annotation-level problems
# ===========================================================================

class TestDatasetValidatorAnnotationProblems:
    """Tests for annotation-level integrity checks."""

    def _write_coco(self, config: PipelineConfig, coco: dict[str, Any]) -> None:
        config.paths.default_annotation_file.write_text(
            json.dumps(coco), encoding="utf-8"
        )

    def test_duplicate_image_ids_detected(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        coco = tmp_dataset["coco"]
        # Duplicate the first image entry
        coco["images"].append({**coco["images"][0]})
        self._write_coco(config, coco)
        report = DatasetValidator(config).validate()
        assert coco["images"][0]["id"] in report.duplicate_image_ids

    def test_orphan_annotation_detected(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        coco = tmp_dataset["coco"]
        coco["annotations"].append({
            "id": 9999,
            "image_id": 99999,  # unknown
            "category_id": 1,
            "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            "area": 100.0,
            "bbox": [0, 0, 10, 10],
            "iscrowd": 0,
        })
        self._write_coco(config, coco)
        report = DatasetValidator(config).validate()
        assert 9999 in report.orphan_annotations
        assert not report.is_valid

    def test_invalid_category_id_detected(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        coco = tmp_dataset["coco"]
        coco["annotations"].append({
            "id": 8888,
            "image_id": coco["images"][0]["id"],
            "category_id": 999,  # non-existent
            "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            "area": 100.0,
            "bbox": [0, 0, 10, 10],
            "iscrowd": 0,
        })
        self._write_coco(config, coco)
        report = DatasetValidator(config).validate()
        assert 8888 in report.invalid_category_ids
        assert not report.is_valid

    def test_unannotated_image_produces_warning(self, tmp_dataset: dict[str, Any]) -> None:
        config = tmp_dataset["config"]
        coco = tmp_dataset["coco"]
        # Add an image but no matching annotation
        new_img = {"id": 9001, "file_name": "orphan_img.jpg",
                   "width": 640, "height": 480, "license": 0,
                   "flickr_url": "", "coco_url": "", "date_captured": 0}
        coco["images"].append(new_img)
        # Write a placeholder image on disk
        from PIL import Image as PILImage
        (config.paths.screw_only_dir / "orphan_img.jpg").write_bytes(b"")
        self._write_coco(config, coco)
        report = DatasetValidator(config).validate()
        assert 9001 in report.unannotated_images
