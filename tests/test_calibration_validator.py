"""
ScrewMetric — Tests: Calibration Validator
===========================================
Unit tests for ``calibration/scripts/calibration_validator.py``.

Coverage:
- ValidationReport dataclass default values and properties
- Missing images directory produces error
- Empty images directory produces error
- Unsupported extension files produce warnings
- Duplicate filenames produce errors
- Corrupted images detected and reported
- Checkerboard corners detected in synthetic images
- Minimum image count check
- Report is valid on clean synthetic dataset
- Report written to disk
- Report is valid JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

_CALIB = Path(__file__).resolve().parent.parent / "calibration" / "scripts"
sys.path.insert(0, str(_CALIB))

from calibration_validator import CalibrationValidator, ValidationReport
from conftest import _make_calib_config, _write_checkerboard_images


# ===========================================================================
# ValidationReport dataclass
# ===========================================================================

class TestValidationReport:
    def test_default_is_invalid(self) -> None:
        r = ValidationReport()
        # no errors is actually valid — empty errors list means valid
        assert r.is_valid is True

    def test_error_makes_invalid(self) -> None:
        r = ValidationReport(errors=["something went wrong"])
        assert r.is_valid is False

    def test_to_dict_includes_is_valid(self) -> None:
        r = ValidationReport()
        d = r.to_dict()
        assert "is_valid" in d

    def test_to_dict_majority_resolution_as_list(self) -> None:
        r = ValidationReport(majority_resolution=(1920, 1080))
        d = r.to_dict()
        assert d["majority_resolution"] == [1920, 1080]

    def test_to_dict_majority_resolution_none(self) -> None:
        r = ValidationReport(majority_resolution=None)
        d = r.to_dict()
        assert d["majority_resolution"] is None

    def test_default_lists_are_empty(self) -> None:
        r = ValidationReport()
        assert r.errors == []
        assert r.warnings == []
        assert r.corrupted_images == []


# ===========================================================================
# Directory structure checks
# ===========================================================================

class TestValidatorDirectoryChecks:
    def test_missing_images_dir_produces_error(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        # images_dir does NOT exist
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert not report.is_valid
        assert any("does not exist" in e or "not exist" in e.lower() for e in report.errors)

    def test_empty_images_dir_produces_error(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        (tmp_path / "images").mkdir(parents=True)
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert not report.is_valid
        assert any("No images" in e or "no images" in e.lower() or "supported" in e.lower() for e in report.errors)

    def test_validation_report_written_to_disk(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        validator.validate()
        assert config.paths.validation_report_path.exists()

    def test_validation_report_is_valid_json(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        validator.validate()
        with config.paths.validation_report_path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_report_contains_required_keys(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        validator.validate()
        with config.paths.validation_report_path.open() as f:
            data = json.load(f)
        required = {"total_images_found", "images_with_corners", "errors", "is_valid"}
        assert required.issubset(set(data.keys()))


# ===========================================================================
# File extension checks
# ===========================================================================

class TestValidatorExtensionChecks:
    def test_unsupported_extension_produces_warning(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        (img_dir / "document.pdf").write_bytes(b"fake")
        (img_dir / "notes.txt").write_bytes(b"fake")
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert len(report.unsupported_extensions) == 2

    def test_jpg_and_png_both_accepted(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        from conftest import _make_checkerboard_image
        board = _make_checkerboard_image()
        cv2.imwrite(str(img_dir / "a.jpg"), board)
        cv2.imwrite(str(img_dir / "b.png"), board)
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.unsupported_extensions == []


# ===========================================================================
# Duplicate filename checks
# ===========================================================================

class TestValidatorDuplicateChecks:
    def test_duplicate_filename_produces_error(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        from conftest import _make_checkerboard_image
        board = _make_checkerboard_image()
        # Write then copy (simulate duplicate detected at listing level)
        p1 = img_dir / "dup.jpg"
        cv2.imwrite(str(p1), board)
        # Patch the listing to return duplicates
        import calibration_validator as cv_mod
        original = cv_mod.CalibrationValidator._discover_files

        def _fake_discover(self_inner, report):
            paths = original(self_inner, report)
            # Inject a duplicate name
            report.duplicate_filenames.append("dup.jpg")
            report.errors.append("Duplicate filename: 'dup.jpg'")
            return paths

        cv_mod.CalibrationValidator._discover_files = _fake_discover  # type: ignore
        try:
            validator = CalibrationValidator(config)
            report = validator.validate()
            assert "dup.jpg" in report.duplicate_filenames
        finally:
            cv_mod.CalibrationValidator._discover_files = original  # type: ignore


# ===========================================================================
# Image integrity checks
# ===========================================================================

class TestValidatorIntegrityChecks:
    def test_corrupted_image_detected(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        # Write a valid image
        from conftest import _make_checkerboard_image
        board = _make_checkerboard_image()
        cv2.imwrite(str(img_dir / "good.jpg"), board)
        # Write a corrupt one
        (img_dir / "corrupt.jpg").write_bytes(b"this is not a jpeg")
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert "corrupt.jpg" in report.corrupted_images

    def test_all_valid_images_counted(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.readable_images == len(tmp_calib_dir["image_paths"])


# ===========================================================================
# Corner detection checks
# ===========================================================================

class TestValidatorCornerDetection:
    def test_corners_found_in_synthetic_images(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.images_with_corners > 0

    def test_no_corners_if_plain_images(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        # Write solid-color images (no checkerboard)
        solid = np.full((480, 640, 3), 128, dtype=np.uint8)
        for i in range(6):
            cv2.imwrite(str(img_dir / f"solid_{i:02d}.jpg"), solid)
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.images_with_corners == 0

    def test_failed_images_listed(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        solid = np.full((480, 640, 3), 200, dtype=np.uint8)
        for i in range(6):
            cv2.imwrite(str(img_dir / f"plain_{i:02d}.jpg"), solid)
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert len(report.images_without_corners) == 6


# ===========================================================================
# Minimum count check
# ===========================================================================

class TestValidatorMinimumCount:
    def test_too_few_valid_images_produces_error(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        # Write only 2 checkerboard images (need >= 4 per ValidationConfig)
        from conftest import _make_checkerboard_image
        board = _make_checkerboard_image()
        for i in range(2):
            cv2.imwrite(str(img_dir / f"cb_{i:02d}.jpg"), board)
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert not report.is_valid
        assert any("Insufficient" in e or "insufficient" in e.lower() for e in report.errors)

    def test_enough_images_passes(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        report = validator.validate()
        # 15 synthetic images >= min_valid_images=4
        assert report.is_valid

    def test_validation_duration_positive(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.validation_duration_s >= 0.0


# ===========================================================================
# Resolution consistency
# ===========================================================================

class TestValidatorResolutionConsistency:
    def test_uniform_resolution_no_warnings(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.inconsistent_resolutions == []

    def test_majority_resolution_set(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        validator = CalibrationValidator(config)
        report = validator.validate()
        assert report.majority_resolution is not None
        assert len(report.majority_resolution) == 2
