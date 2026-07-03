"""
ScrewMetric — Tests: Camera Calibration
=========================================
Unit tests for ``calibration/scripts/calibrate_camera.py``.

Coverage:
- CheckerboardCornerDetector: object points shape and values
- CheckerboardCornerDetector: detects corners in synthetic images
- CheckerboardCornerDetector: returns failure result for corrupt image
- CheckerboardCornerDetector: detects all images in a list
- CameraCalibrator: produces CalibrationResult with camera matrix
- CameraCalibrator: produces 3x3 camera matrix
- CameraCalibrator: produces correct distortion coefficients shape
- CameraCalibrator: mean reprojection error is finite and positive
- CameraCalibrator: saves .npy files to disk
- CameraCalibrator: saves reprojection_error.json
- CameraCalibrator: raises ValueError for too few images
- CalibrationResult: to_dict serialisation
- CalibrationResult: default field types correct
- CornerDetectionResult: filename property
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

_CALIB = Path(__file__).resolve().parent.parent / "calibration" / "scripts"
sys.path.insert(0, str(_CALIB))

from calibrate_camera import (
    CalibrationResult,
    CameraCalibrator,
    CheckerboardCornerDetector,
    CornerDetectionResult,
)
from conftest import _make_calib_config, _make_checkerboard_image, _write_checkerboard_images


# ===========================================================================
# CornerDetectionResult dataclass
# ===========================================================================

class TestCornerDetectionResult:
    def test_default_success_is_false(self, tmp_path: Path) -> None:
        r = CornerDetectionResult(image_path=tmp_path / "img.jpg")
        assert r.success is False

    def test_filename_property(self, tmp_path: Path) -> None:
        r = CornerDetectionResult(image_path=tmp_path / "checkerboard_001.jpg")
        assert r.filename == "checkerboard_001.jpg"

    def test_default_corners_is_none(self, tmp_path: Path) -> None:
        r = CornerDetectionResult(image_path=tmp_path / "img.jpg")
        assert r.corners is None

    def test_default_image_size(self, tmp_path: Path) -> None:
        r = CornerDetectionResult(image_path=tmp_path / "img.jpg")
        assert r.image_size == (0, 0)


# ===========================================================================
# CalibrationResult dataclass
# ===========================================================================

class TestCalibrationResult:
    def test_default_camera_matrix_is_identity(self) -> None:
        r = CalibrationResult()
        assert r.camera_matrix.shape == (3, 3)
        assert np.allclose(r.camera_matrix, np.eye(3))

    def test_default_dist_coeffs_shape(self) -> None:
        r = CalibrationResult()
        assert r.dist_coeffs.shape == (1, 5)

    def test_to_dict_camera_matrix_is_list(self) -> None:
        r = CalibrationResult()
        d = r.to_dict()
        assert isinstance(d["camera_matrix"], list)
        assert len(d["camera_matrix"]) == 3

    def test_to_dict_dist_coeffs_is_list(self) -> None:
        r = CalibrationResult()
        d = r.to_dict()
        assert isinstance(d["dist_coeffs"], list)

    def test_to_dict_image_size_is_list(self) -> None:
        r = CalibrationResult(image_size=(1920, 1440))
        d = r.to_dict()
        assert d["image_size"] == [1920, 1440]

    def test_to_dict_failed_images_is_list(self) -> None:
        r = CalibrationResult(failed_images=["bad.jpg"])
        d = r.to_dict()
        assert d["failed_images"] == ["bad.jpg"]


# ===========================================================================
# CheckerboardCornerDetector: object points
# ===========================================================================

class TestObjectPoints:
    def test_object_points_shape(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        objp = detector.build_object_points()
        expected_n = config.checkerboard.total_inner_corners
        assert objp.shape == (expected_n, 1, 3)

    def test_object_points_z_is_zero(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        objp = detector.build_object_points()
        assert np.all(objp[:, :, 2] == 0.0)

    def test_object_points_dtype_float32(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        objp = detector.build_object_points()
        assert objp.dtype == np.float32

    def test_object_points_first_element_is_origin(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        objp = detector.build_object_points()
        assert objp[0, 0, 0] == 0.0
        assert objp[0, 0, 1] == 0.0

    def test_object_points_scaled_by_square_size(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        sq = config.checkerboard.square_size_mm
        detector = CheckerboardCornerDetector(config)
        objp = detector.build_object_points()
        # Second point along x should be (sq, 0, 0)
        assert abs(float(objp[1, 0, 0]) - sq) < 1e-5


# ===========================================================================
# CheckerboardCornerDetector: detection
# ===========================================================================

class TestCornerDetection:
    def test_detects_corners_in_synthetic_image(self, tmp_path: Path) -> None:
        from conftest import _make_calib_config
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        paths = _write_checkerboard_images(img_dir, n=1)
        detector = CheckerboardCornerDetector(config)
        result = detector.detect_single(paths[0])
        assert result.success is True
        assert result.corners is not None

    def test_failure_for_corrupt_image(self, tmp_path: Path) -> None:
        from conftest import _make_calib_config
        config = _make_calib_config(tmp_path)
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        detector = CheckerboardCornerDetector(config)
        result = detector.detect_single(bad)
        assert result.success is False
        assert result.error_message != ""

    def test_failure_for_plain_image(self, tmp_path: Path) -> None:
        from conftest import _make_calib_config
        config = _make_calib_config(tmp_path)
        solid = np.full((480, 640, 3), 128, dtype=np.uint8)
        p = tmp_path / "solid.jpg"
        cv2.imwrite(str(p), solid)
        detector = CheckerboardCornerDetector(config)
        result = detector.detect_single(p)
        assert result.success is False

    def test_image_size_populated_on_success(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        result = detector.detect_single(tmp_calib_dir["image_paths"][0])
        assert result.image_size != (0, 0)

    def test_detect_all_returns_one_per_image(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        paths = tmp_calib_dir["image_paths"]
        results = detector.detect_all(paths)
        assert len(results) == len(paths)

    def test_detect_all_majority_succeed(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        results = detector.detect_all(tmp_calib_dir["image_paths"])
        successful = [r for r in results if r.success]
        assert len(successful) >= len(results) * 0.8

    def test_corners_array_correct_shape(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        detector = CheckerboardCornerDetector(config)
        result = detector.detect_single(tmp_calib_dir["image_paths"][0])
        if result.success:
            expected_n = config.checkerboard.total_inner_corners
            assert result.corners.shape[0] == expected_n


# ===========================================================================
# CameraCalibrator: full calibration
# ===========================================================================

class TestCameraCalibrator:
    def test_calibration_returns_result(self, tmp_calib_with_results: dict) -> None:
        assert isinstance(
            tmp_calib_with_results["calibration_result"], CalibrationResult
        )

    def test_camera_matrix_is_3x3(self, tmp_calib_with_results: dict) -> None:
        K = tmp_calib_with_results["calibration_result"].camera_matrix
        assert K.shape == (3, 3)

    def test_camera_matrix_has_positive_focal_length(
        self, tmp_calib_with_results: dict
    ) -> None:
        K = tmp_calib_with_results["calibration_result"].camera_matrix
        assert K[0, 0] > 0  # fx
        assert K[1, 1] > 0  # fy

    def test_dist_coeffs_shape(self, tmp_calib_with_results: dict) -> None:
        d = tmp_calib_with_results["calibration_result"].dist_coeffs
        # cv2.calibrateCamera returns (1, 5) or (5, 1) or (1, 14)
        assert d.size >= 4

    def test_mean_reprojection_error_is_finite(
        self, tmp_calib_with_results: dict
    ) -> None:
        err = tmp_calib_with_results["calibration_result"].mean_reprojection_error
        assert np.isfinite(err)

    def test_mean_reprojection_error_is_positive(
        self, tmp_calib_with_results: dict
    ) -> None:
        err = tmp_calib_with_results["calibration_result"].mean_reprojection_error
        assert err >= 0.0

    def test_per_image_errors_match_successful_count(
        self, tmp_calib_with_results: dict
    ) -> None:
        result = tmp_calib_with_results["calibration_result"]
        assert len(result.per_image_errors) == len(result.successful_images)

    def test_successful_images_list_non_empty(
        self, tmp_calib_with_results: dict
    ) -> None:
        result = tmp_calib_with_results["calibration_result"]
        assert len(result.successful_images) > 0

    def test_image_size_populated(self, tmp_calib_with_results: dict) -> None:
        result = tmp_calib_with_results["calibration_result"]
        assert result.image_size != (0, 0)
        assert result.image_size[0] > 0

    def test_calibration_duration_non_negative(
        self, tmp_calib_with_results: dict
    ) -> None:
        result = tmp_calib_with_results["calibration_result"]
        assert result.calibration_duration_s >= 0



# ===========================================================================
# CameraCalibrator: artefact saving
# ===========================================================================

class TestCalibrationArtefacts:
    def test_camera_matrix_npy_saved(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        assert config.paths.camera_matrix_path.exists()

    def test_dist_coeffs_npy_saved(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        assert config.paths.dist_coeffs_path.exists()

    def test_rotation_vectors_npy_saved(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        assert config.paths.rotation_vectors_path.exists()

    def test_translation_vectors_npy_saved(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        assert config.paths.translation_vectors_path.exists()

    def test_reprojection_error_json_saved(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        assert config.paths.reprojection_error_path.exists()

    def test_camera_matrix_npy_loadable(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        K = np.load(str(config.paths.camera_matrix_path))
        assert K.shape == (3, 3)

    def test_dist_coeffs_npy_loadable(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        d = np.load(str(config.paths.dist_coeffs_path))
        assert d.size >= 4


# ===========================================================================
# CameraCalibrator: error handling
# ===========================================================================

class TestCalibrationErrorHandling:
    def test_too_few_images_raises_value_error(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        # Provide failed detections only
        from calibrate_camera import CornerDetectionResult
        bad = [
            CornerDetectionResult(
                image_path=tmp_calib_dir["root"] / f"bad_{i}.jpg",
                success=False,
            )
            for i in range(3)
        ]
        calibrator = CameraCalibrator(config)
        with pytest.raises(ValueError, match="Calibration requires"):
            calibrator.calibrate(bad)

    def test_empty_detection_list_raises(self, tmp_calib_dir: dict) -> None:
        config = tmp_calib_dir["config"]
        calibrator = CameraCalibrator(config)
        with pytest.raises(ValueError):
            calibrator.calibrate([])
