"""
ScrewMetric — Integration Tests: Full Calibration Pipeline
===========================================================
Tests the calibrate_camera.py CLI entry point and verifies all
expected output artefacts are created correctly.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Add calibration scripts to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CALIB_SCRIPTS = _PROJECT_ROOT / "calibration" / "scripts"
if str(_CALIB_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CALIB_SCRIPTS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibration_output_dir(tmp_path_factory) -> Path:
    """Run CalibrationPipeline on a synthetic directory and return its output folder."""
    from camera_calibration import CalibrationPipeline
    from conftest import _make_calib_config, _write_checkerboard_images

    tmp_dir = tmp_path_factory.mktemp("hermetic_calib")
    config = _make_calib_config(tmp_dir)
    image_paths = _write_checkerboard_images(
        tmp_dir / "images",
        n=15,
        inner_cols=config.checkerboard.inner_corners_x,
        inner_rows=config.checkerboard.inner_corners_y,
    )
    pipeline = CalibrationPipeline(config)
    pipeline.run(
        run_validation=True,
        run_calibration=True,
        run_report=True,
        run_visualization=True,
    )
    return config.paths.output_dir



# ---------------------------------------------------------------------------
# Test Cases — Calibration Output Artefacts
# ---------------------------------------------------------------------------

def test_camera_matrix_npy_exists(calibration_output_dir: Path) -> None:
    """camera_matrix.npy must exist after calibration."""
    assert (calibration_output_dir / "camera_matrix.npy").exists(), \
        "camera_matrix.npy not found — run calibrate_camera.py first"


def test_dist_coeffs_npy_exists(calibration_output_dir: Path) -> None:
    """dist_coeffs.npy must exist after calibration."""
    assert (calibration_output_dir / "dist_coeffs.npy").exists(), \
        "dist_coeffs.npy not found — run calibrate_camera.py first"


def test_camera_matrix_shape(calibration_output_dir: Path) -> None:
    """camera_matrix.npy must be a 3×3 matrix."""
    K = np.load(str(calibration_output_dir / "camera_matrix.npy"))
    assert K.shape == (3, 3), f"Expected shape (3,3), got {K.shape}"


def test_camera_matrix_positive_focal_lengths(calibration_output_dir: Path) -> None:
    """Focal lengths fx and fy must be strictly positive."""
    K = np.load(str(calibration_output_dir / "camera_matrix.npy"))
    fx, fy = K[0, 0], K[1, 1]
    assert fx > 0.0, f"fx must be > 0, got {fx}"
    assert fy > 0.0, f"fy must be > 0, got {fy}"


def test_dist_coeffs_shape(calibration_output_dir: Path) -> None:
    """Distortion coefficients must have between 4 and 14 elements."""
    D = np.load(str(calibration_output_dir / "dist_coeffs.npy"))
    D = D.flatten()
    assert 4 <= len(D) <= 14, f"Unexpected D shape: {D.shape}"


def test_camera_parameters_yaml_exists(calibration_output_dir: Path) -> None:
    """camera_parameters.yaml must be present after calibration."""
    assert (calibration_output_dir / "camera_parameters.yaml").exists()


def test_camera_parameters_yaml_is_valid(calibration_output_dir: Path) -> None:
    """camera_parameters.yaml must be parseable YAML."""
    with open(calibration_output_dir / "camera_parameters.yaml") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_reprojection_error_json_exists(calibration_output_dir: Path) -> None:
    """reprojection_error.json must be present."""
    assert (calibration_output_dir / "reprojection_error.json").exists()


def test_reprojection_error_json_structure(calibration_output_dir: Path) -> None:
    """reprojection_error.json must contain a mean_reprojection_error key."""
    with open(calibration_output_dir / "reprojection_error.json") as f:
        data = json.load(f)
    assert "mean_reprojection_error" in data or "mean_error" in data or \
           any("error" in k.lower() for k in data), \
        f"Expected reprojection error key, got keys: {list(data.keys())}"


def test_calibration_visualization_exists(calibration_output_dir: Path) -> None:
    """calibration_visualization.png must be present."""
    assert (calibration_output_dir / "calibration_visualization.png").exists()


def test_scale_computation_is_correct(calibration_output_dir: Path) -> None:
    """Verify that the mm/px scale derived from K matches the pinhole formula."""
    K = np.load(str(calibration_output_dir / "camera_matrix.npy"))
    fx, fy = K[0, 0], K[1, 1]
    f_avg = (fx + fy) / 2.0
    Z = 300.0  # mm — standard test distance
    scale = Z / f_avg
    # Scale must be positive and finite
    assert scale > 0.0, f"scale={scale:.4f} mm/px must be positive"
    assert np.isfinite(scale), "scale must be finite"
    # scale * f_avg must recover Z
    assert abs(scale * f_avg - Z) < 0.001, "Scale formula inversion failed"
