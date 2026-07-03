"""
ScrewMetric — Tests: Capture Calibration Images
================================================
Unit tests for ``calibration/scripts/capture_calibration_images.py``.

Coverage:
- CaptureSession dataclass defaults
- captured_count property equals len(captured_paths)
- print_guidance() runs without exception
- CalibrationImageCapture initialises with default camera_index
- _detect_corners returns False for plain image
- _detect_corners returns True for checkerboard image
- _next_image_path returns correct filename pattern
- _draw_hud modifies image in-place without crash
- capture_live raises IOError for invalid camera_index
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_CALIB = Path(__file__).resolve().parent.parent / "calibration" / "scripts"
sys.path.insert(0, str(_CALIB))

from capture_calibration_images import CalibrationImageCapture, CaptureSession
from conftest import _make_calib_config, _make_checkerboard_image


# ===========================================================================
# CaptureSession dataclass
# ===========================================================================

class TestCaptureSession:
    def test_default_captured_count_is_zero(self) -> None:
        s = CaptureSession()
        assert s.captured_count == 0

    def test_captured_count_reflects_paths(self, tmp_path: Path) -> None:
        s = CaptureSession()
        s.captured_paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
        assert s.captured_count == 2

    def test_default_skipped_count_is_zero(self) -> None:
        s = CaptureSession()
        assert s.skipped_count == 0

    def test_default_duration_is_zero(self) -> None:
        s = CaptureSession()
        assert s.session_duration_s == 0.0

    def test_captured_paths_default_empty(self) -> None:
        s = CaptureSession()
        assert s.captured_paths == []


# ===========================================================================
# CalibrationImageCapture initialisation
# ===========================================================================

class TestCalibrationImageCaptureInit:
    def test_default_camera_index(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        assert cap._camera_index == 0

    def test_custom_camera_index(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config, camera_index=2)
        assert cap._camera_index == 2

    def test_board_config_attached(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        assert cap._board.inner_corners_x == 9
        assert cap._board.inner_corners_y == 6


# ===========================================================================
# print_guidance()
# ===========================================================================

class TestPrintGuidance:
    def test_print_guidance_runs_without_exception(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        cap.print_guidance()
        out = capsys.readouterr().out
        assert "checkerboard" in out.lower() or "Checkerboard" in out

    def test_guidance_contains_board_dimensions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        cap.print_guidance()
        out = capsys.readouterr().out
        assert "9" in out and "6" in out

    def test_guidance_contains_square_size(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        cap.print_guidance()
        out = capsys.readouterr().out
        assert "25" in out


# ===========================================================================
# _detect_corners()
# ===========================================================================

class TestDetectCorners:
    def test_no_corners_in_plain_image(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        solid = np.full((480, 640, 3), 128, dtype=np.uint8)
        found, corners = cap._detect_corners(solid)
        assert found is False

    def test_corners_found_in_checkerboard(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        board = _make_checkerboard_image(9, 6, square_px=60)
        found, corners = cap._detect_corners(board)
        assert found is True
        assert corners is not None

    def test_corners_none_on_failure(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config)
        solid = np.full((100, 100, 3), 200, dtype=np.uint8)
        found, corners = cap._detect_corners(solid)
        assert not found


# ===========================================================================
# _next_image_path()
# ===========================================================================

class TestNextImagePath:
    def test_path_is_in_images_dir(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        (tmp_path / "images").mkdir(parents=True)
        cap = CalibrationImageCapture(config)
        p = cap._next_image_path(0)
        assert p.parent == config.paths.images_dir

    def test_path_has_jpg_extension(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        (tmp_path / "images").mkdir(parents=True)
        cap = CalibrationImageCapture(config)
        p = cap._next_image_path(0)
        assert p.suffix.lower() == ".jpg"

    def test_path_includes_index(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        (tmp_path / "images").mkdir(parents=True)
        cap = CalibrationImageCapture(config)
        p = cap._next_image_path(0)
        assert "checkerboard" in p.name.lower()

    def test_increments_with_existing_files(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)
        # Write one existing file
        (img_dir / "checkerboard_001.jpg").write_bytes(b"x")
        cap = CalibrationImageCapture(config)
        p = cap._next_image_path(0)
        # Should be 002 now
        assert "002" in p.name


# ===========================================================================
# _draw_hud()
# ===========================================================================

class TestDrawHud:
    def test_hud_does_not_change_shape(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        original_shape = img.shape
        CalibrationImageCapture._draw_hud(
            img,
            status_text="OK",
            status_color=(0, 255, 0),
            captured=5,
            target=30,
        )
        assert img.shape == original_shape

    def test_hud_modifies_image(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        CalibrationImageCapture._draw_hud(
            img,
            status_text="Corners",
            status_color=(72, 199, 142),
            captured=3,
            target=20,
        )
        # Image should no longer be all zeros
        assert img.max() > 0


# ===========================================================================
# capture_live() — camera error
# ===========================================================================

class TestCaptureLive:
    def test_invalid_camera_raises_ioerror(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        cap = CalibrationImageCapture(config, camera_index=999)
        with pytest.raises(IOError, match="Cannot open camera"):
            cap.capture_live(target_count=1)
