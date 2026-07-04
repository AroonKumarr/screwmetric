"""
ScrewMetric — Unit Tests for Metrology Module
==============================================
Tests pixel-to-mm conversion logic, camera calibration file loading,
rotated rectangle fitting, and lens distortion correction.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure measurement/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "measurement") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "measurement"))

from pixel_to_mm import PixelToMMConverter, MeasurementConfig, ScrewMeasurement  # type: ignore[import]


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_calibration_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary NumPy files with valid synthetic camera parameters.

    Returns:
        Tuple of (camera_matrix_path, dist_coeffs_path).
    """
    # Standard 3x3 camera matrix
    # fx=800, fy=800, cx=320, cy=240
    K = np.array([
        [800.0,   0.0, 320.0],
        [  0.0, 800.0, 240.0],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float64)

    # 5 radial/tangential distortion coefficients
    D = np.array([-0.2, 0.1, 0.001, -0.002, 0.05], dtype=np.float64)

    K_path = tmp_path / "camera_matrix.npy"
    D_path = tmp_path / "dist_coeffs.npy"

    np.save(str(K_path), K)
    np.save(str(D_path), D)

    return K_path, D_path


@pytest.fixture
def converter(mock_calibration_files: tuple[Path, Path]) -> PixelToMMConverter:
    """Return a PixelToMMConverter configured with mock calibration paths."""
    K_path, D_path = mock_calibration_files
    cfg = MeasurementConfig(
        camera_matrix_path=K_path,
        dist_coeffs_path=D_path,
        known_distance_mm=300.0,
        min_contour_area_px=10.0,
    )
    conv = PixelToMMConverter(cfg)
    conv.load_calibration()
    return conv


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_load_calibration_loads_npy_files(
    mock_calibration_files: tuple[Path, Path]
) -> None:
    """Verify that valid .npy files are loaded correctly and scale computed."""
    K_path, D_path = mock_calibration_files
    cfg = MeasurementConfig(
        camera_matrix_path=K_path,
        dist_coeffs_path=D_path,
        known_distance_mm=300.0,
    )
    conv = PixelToMMConverter(cfg)
    K, D = conv.load_calibration()

    assert K.shape == (3, 3)
    assert D.shape == (1, 5)
    assert conv._scale is not None
    # scale = Z / f = 300 / 800 = 0.375 mm/px
    assert abs(conv._scale - 0.375) < 1e-6


def test_load_calibration_missing_files_raises() -> None:
    """FileNotFoundError must be raised when files do not exist."""
    cfg = MeasurementConfig(
        camera_matrix_path=Path("non_existent_K.npy"),
        dist_coeffs_path=Path("non_existent_D.npy"),
    )
    conv = PixelToMMConverter(cfg)
    with pytest.raises(FileNotFoundError):
        conv.load_calibration()


def test_convert_pixel_to_mm_zero_distance_raises(
    mock_calibration_files: tuple[Path, Path]
) -> None:
    """ValueError must be raised if known_distance_mm is non-positive."""
    K_path, D_path = mock_calibration_files
    with pytest.raises(ValueError):
        MeasurementConfig(
            camera_matrix_path=K_path,
            dist_coeffs_path=D_path,
            known_distance_mm=0.0,
        )
    with pytest.raises(ValueError):
        MeasurementConfig(
            camera_matrix_path=K_path,
            dist_coeffs_path=D_path,
            known_distance_mm=-150.0,
        )


def test_fit_rotated_rect_returns_major_minor(
    converter: PixelToMMConverter
) -> None:
    """Verify minAreaRect fitting always returns major >= minor."""
    # Create a synthetic contour forming a tall rectangle: width=10, height=50
    contour = np.array([
        [[100, 100]],
        [[110, 100]],
        [[110, 150]],
        [[100, 150]]
    ], dtype=np.int32)

    major, minor = converter.fit_rotated_rect(contour)
    assert major >= minor
    # OpenCV fitting may return slightly different values due to pixel centers, but major is ~50, minor is ~10
    assert abs(major - 50.0) <= 2.0
    assert abs(minor - 10.0) <= 2.0


def test_undistort_contour_shape_preserved(
    converter: PixelToMMConverter
) -> None:
    """Ensure cv2.undistortPoints does not modify contour shape/dimensions."""
    contour = np.array([
        [[100, 100]],
        [[200, 100]],
        [[200, 200]],
        [[100, 200]]
    ], dtype=np.int32)

    undistorted = converter.undistort_contour(contour)
    assert undistorted.shape == contour.shape
    assert undistorted.dtype == np.float32


def test_measure_returns_valid_measurement(
    converter: PixelToMMConverter
) -> None:
    """Verify standard measurement pipeline returns correct metrics."""
    # Create a binary mask with a clear rectangular object
    mask = np.zeros((480, 640), dtype=np.uint8)
    # Rectangle of width 40, height 120 pixels
    cv2.rectangle(mask, (100, 100), (140, 220), 255, -1)

    result = converter.measure(mask, confidence=0.88)

    assert isinstance(result, ScrewMeasurement)
    assert result.confidence == 0.88
    assert result.pixel_length >= result.pixel_diameter
    assert result.length_mm > 0.0
    assert result.diameter_mm > 0.0
    assert result.scale_mm_per_px > 0.0
    assert result.method == "focal_length_pinhole"


def test_zero_distortion_produces_same_result(
    mock_calibration_files: tuple[Path, Path]
) -> None:
    """Ensure D = 0 gives consistent results without distortion displacement."""
    K_path, D_path = mock_calibration_files
    # Overwrite distortion file with zeros
    np.save(str(D_path), np.zeros((1, 5), dtype=np.float64))

    cfg = MeasurementConfig(
        camera_matrix_path=K_path,
        dist_coeffs_path=D_path,
        known_distance_mm=300.0,
        min_contour_area_px=10.0,
    )
    conv = PixelToMMConverter(cfg)
    conv.load_calibration()

    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(mask, (200, 200), (220, 300), 255, -1)

    res = conv.measure(mask)
    # Without distortion, length_px should be exactly 100 pixels
    # scale = 300 / 800 = 0.375
    # length_mm = 100 * 0.375 = 37.5
    assert abs(res.pixel_length - 100.0) <= 1.0
    assert abs(res.length_mm - 37.5) <= 0.5


def test_invalid_mask_raises(converter: PixelToMMConverter) -> None:
    """ValueError must be raised if the mask has no non-zero contour."""
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="No contours found"):
        converter.measure(empty_mask)

    # Noise mask below min_contour_area_px
    noisy_mask = np.zeros((100, 100), dtype=np.uint8)
    noisy_mask[50, 50] = 255  # area = 1 px² < min_contour_area_px (10.0)
    with pytest.raises(ValueError, match="below minimum threshold"):
        converter.measure(noisy_mask)
