"""
ScrewMetric — Tests: Report Generator
=======================================
Unit tests for ``calibration/scripts/report_generator.py``.

Coverage:
- CalibrationReport dataclass to_dict serialisation
- generate() saves calibration_report.json
- generate() saves camera_parameters.yaml
- calibration_report.json is valid JSON
- camera_parameters.yaml is valid YAML
- Report contains all required fields
- YAML contains camera_matrix, dist_coeffs, focal lengths
- Focal lengths extracted correctly from camera matrix
- Principal point extracted correctly
- Mean reprojection error round-trips correctly
- generate_from_disk() works after artefacts saved
- Calibration date is ISO format string
- Image size stored correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

_CALIB = Path(__file__).resolve().parent.parent / "calibration" / "scripts"
sys.path.insert(0, str(_CALIB))

from calibrate_camera import CalibrationResult
from report_generator import CalibrationReport, CalibrationReportGenerator
from conftest import _make_calib_config


# ===========================================================================
# CalibrationReport dataclass
# ===========================================================================

class TestCalibrationReportDataclass:
    def test_to_dict_is_dict(self) -> None:
        r = CalibrationReport()
        assert isinstance(r.to_dict(), dict)

    def test_to_dict_contains_camera_matrix(self) -> None:
        r = CalibrationReport(camera_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        d = r.to_dict()
        assert "camera_matrix" in d

    def test_to_dict_contains_dist_coeffs(self) -> None:
        r = CalibrationReport(dist_coeffs=[0.1, -0.2, 0.0, 0.0, 0.05])
        d = r.to_dict()
        assert "dist_coeffs" in d

    def test_to_dict_all_required_keys(self) -> None:
        r = CalibrationReport()
        d = r.to_dict()
        required = {
            "num_images_total", "num_images_successful", "num_images_failed",
            "image_resolution", "camera_matrix", "dist_coeffs",
            "focal_length_fx", "focal_length_fy", "principal_point_cx",
            "principal_point_cy", "mean_reprojection_error",
            "calibration_date", "execution_time_s",
        }
        assert required.issubset(set(d.keys()))

    def test_default_calibration_date_empty(self) -> None:
        r = CalibrationReport()
        assert r.calibration_date == ""


# ===========================================================================
# generate() — JSON report
# ===========================================================================

class TestReportGeneratorJSON:
    def test_json_report_saved(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        assert config.paths.calibration_report_path.exists()

    def test_json_report_is_valid_json(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.calibration_report_path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_report_contains_camera_matrix(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.calibration_report_path.open() as f:
            data = json.load(f)
        assert "camera_matrix" in data
        assert len(data["camera_matrix"]) == 3

    def test_json_report_focal_length_matches_matrix(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        K = synthetic_calib_result.camera_matrix
        assert abs(report.focal_length_fx - K[0, 0]) < 0.01
        assert abs(report.focal_length_fy - K[1, 1]) < 0.01

    def test_json_report_principal_point_matches_matrix(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        K = synthetic_calib_result.camera_matrix
        assert abs(report.principal_point_cx - K[0, 2]) < 0.01
        assert abs(report.principal_point_cy - K[1, 2]) < 0.01

    def test_json_report_mean_error_matches(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        assert abs(report.mean_reprojection_error - synthetic_calib_result.mean_reprojection_error) < 1e-5

    def test_json_report_image_count(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        assert report.num_images_failed == len(synthetic_calib_result.failed_images)

    def test_json_report_calibration_date_is_string(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        assert isinstance(report.calibration_date, str)
        assert len(report.calibration_date) > 0

    def test_json_report_image_resolution(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        assert report.image_resolution == list(synthetic_calib_result.image_size)

    def test_json_report_checkerboard_corners(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        report = gen.generate(synthetic_calib_result)
        assert report.checkerboard_inner_corners == [
            config.checkerboard.inner_corners_x,
            config.checkerboard.inner_corners_y,
        ]


# ===========================================================================
# generate() — YAML camera parameters
# ===========================================================================

class TestReportGeneratorYAML:
    def test_yaml_saved(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        assert config.paths.camera_parameters_path.exists()

    def test_yaml_is_valid(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_yaml_contains_camera_matrix(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert "camera_matrix" in data

    def test_yaml_contains_dist_coeffs(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert "dist_coeffs" in data

    def test_yaml_contains_focal_lengths(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert "focal_length_fx" in data
        assert "focal_length_fy" in data

    def test_yaml_contains_reprojection_error(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert "reprojection_error" in data

    def test_yaml_image_dimensions(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert data.get("image_width") == synthetic_calib_result.image_size[0]
        assert data.get("image_height") == synthetic_calib_result.image_size[1]

    def test_yaml_date_created_present(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        gen.generate(synthetic_calib_result)
        with config.paths.camera_parameters_path.open() as f:
            data = yaml.safe_load(f)
        assert "date_created" in data


# ===========================================================================
# generate_from_disk()
# ===========================================================================

class TestGenerateFromDisk:
    def test_generate_from_disk_works(self, tmp_calib_with_results: dict) -> None:
        config = tmp_calib_with_results["config"]
        gen = CalibrationReportGenerator(config)
        # Artefacts already saved by fixture — regenerate from disk
        report = gen.generate_from_disk()
        assert isinstance(report, CalibrationReport)

    def test_generate_from_disk_raises_if_no_npy(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        gen = CalibrationReportGenerator(config)
        with pytest.raises(FileNotFoundError):
            gen.generate_from_disk()

    def test_generate_from_disk_produces_yaml(
        self, tmp_calib_with_results: dict
    ) -> None:
        config = tmp_calib_with_results["config"]
        gen = CalibrationReportGenerator(config)
        gen.generate_from_disk()
        assert config.paths.camera_parameters_path.exists()


# ===========================================================================
# Integration: generate from full calibration
# ===========================================================================

class TestReportIntegration:
    def test_full_calibration_generates_report(
        self, tmp_calib_with_results: dict
    ) -> None:
        config = tmp_calib_with_results["config"]
        cal_result = tmp_calib_with_results["calibration_result"]
        gen = CalibrationReportGenerator(config)
        report = gen.generate(cal_result)
        assert report.num_images_successful > 0
        assert report.focal_length_fx > 0
        assert report.mean_reprojection_error >= 0
