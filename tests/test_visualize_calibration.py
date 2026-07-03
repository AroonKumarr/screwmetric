"""
ScrewMetric — Tests: Calibration Visualizer
=============================================
Unit tests for ``calibration/scripts/visualize_calibration.py``.

Coverage:
- VisualizationResult dataclass defaults
- generate_all() saves calibration_visualization.png
- visualization PNG is a valid image file
- generate_all() saves undistortion preview images
- undistortion previews are valid JPEG images
- generate_all() works with empty successful_images list
- generate_all() accepts detection_results for corner sheet
- Stats panel generated without crash
- Before/after pair generated without crash
- generate_from_disk() raises if no artefacts present
- generate_from_disk() works after calibration
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

_CALIB = Path(__file__).resolve().parent.parent / "calibration" / "scripts"
sys.path.insert(0, str(_CALIB))

from calibrate_camera import CalibrationResult
from visualize_calibration import CalibrationVisualizer, VisualizationResult
from conftest import _make_calib_config, _make_checkerboard_image


# ===========================================================================
# VisualizationResult dataclass
# ===========================================================================

class TestVisualizationResult:
    def test_default_visualization_path_is_none(self) -> None:
        r = VisualizationResult()
        assert r.visualization_path is None

    def test_default_previews_list_empty(self) -> None:
        r = VisualizationResult()
        assert r.undistortion_preview_paths == []

    def test_default_corner_overlays_empty(self) -> None:
        r = VisualizationResult()
        assert r.corner_overlay_paths == []


# ===========================================================================
# generate_all() — composite visualization
# ===========================================================================

class TestVisualizerGenerateAll:
    def test_visualization_png_saved(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        result = viz.generate_all(synthetic_calib_result)
        assert result.visualization_path is not None
        assert result.visualization_path.exists()

    def test_visualization_is_valid_image(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        result = viz.generate_all(synthetic_calib_result)
        with Image.open(result.visualization_path) as im:
            assert im.width > 0
            assert im.height > 0

    def test_visualization_is_png(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        result = viz.generate_all(synthetic_calib_result)
        assert result.visualization_path.suffix.lower() == ".png"

    def test_visualization_nonempty(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        result = viz.generate_all(synthetic_calib_result)
        assert result.visualization_path.stat().st_size > 0

    def test_works_with_empty_successful_images(
        self, tmp_path: Path
    ) -> None:
        config = _make_calib_config(tmp_path)
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 500.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        result = CalibrationResult(
            camera_matrix=K,
            dist_coeffs=np.zeros((1, 5)),
            image_size=(640, 480),
            successful_images=[],
        )
        viz = CalibrationVisualizer(config)
        vr = viz.generate_all(result)
        assert vr.visualization_path is not None

    def test_works_with_detection_results(
        self,
        tmp_path: Path,
        synthetic_calib_result: CalibrationResult,
    ) -> None:
        """Generate composite with corner overlay using local detection results."""
        config = _make_calib_config(tmp_path)
        from calibrate_camera import CornerDetectionResult  # type: ignore

        # Build simple detection results pointing to non-existent paths
        # so render_corner_thumb falls back to the placeholder branch
        det_results = [
            CornerDetectionResult.__new__(CornerDetectionResult)
            for _ in range(3)
        ]
        for i, r in enumerate(det_results):
            r.image_path = tmp_path / f"cb_{i:03d}.jpg"
            r.success = False
            r.corners = None
            r.image_size = (0, 0)
            r.error_message = "placeholder"

        viz = CalibrationVisualizer(config)
        vr = viz.generate_all(synthetic_calib_result, det_results)
        assert vr.visualization_path is not None
        assert vr.visualization_path.exists()



# ===========================================================================
# generate_all() — undistortion previews
# ===========================================================================

class TestVisualizerUndistortionPreviews:
    def _make_result_with_images(
        self, tmp_path: Path, n: int = 3
    ) -> tuple[CalibrationResult, Any]:
        """Create a CalibrationResult with real image paths."""
        from conftest import _write_checkerboard_images
        img_dir = tmp_path / "images"
        paths = _write_checkerboard_images(img_dir, n=n)
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 500.0
        K[0, 2] = 300.0
        K[1, 2] = 210.0
        result = CalibrationResult(
            camera_matrix=K,
            dist_coeffs=np.zeros((1, 5)),
            image_size=(paths[0] and cv2.imread(str(paths[0])).shape[1], 0)
            if paths else (600, 420),
            successful_images=paths,
        )
        return result, _make_calib_config(tmp_path)

    def test_previews_saved_when_images_exist(self, tmp_path: Path) -> None:
        result, config = self._make_result_with_images(tmp_path, n=3)
        viz = CalibrationVisualizer(config)
        vr = viz.generate_all(result)
        assert len(vr.undistortion_preview_paths) > 0

    def test_previews_are_valid_images(self, tmp_path: Path) -> None:
        result, config = self._make_result_with_images(tmp_path, n=3)
        viz = CalibrationVisualizer(config)
        vr = viz.generate_all(result)
        for p in vr.undistortion_preview_paths:
            assert p.exists()
            img = cv2.imread(str(p))
            assert img is not None

    def test_max_previews_respected(self, tmp_path: Path) -> None:
        result, config = self._make_result_with_images(tmp_path, n=5)
        viz = CalibrationVisualizer(config)
        vr = viz.generate_all(result)
        # max_preview_images=3 in test config
        assert len(vr.undistortion_preview_paths) <= 3

    def test_previews_saved_in_correct_dir(self, tmp_path: Path) -> None:
        result, config = self._make_result_with_images(tmp_path, n=2)
        viz = CalibrationVisualizer(config)
        vr = viz.generate_all(result)
        for p in vr.undistortion_preview_paths:
            assert p.parent == config.paths.undistortion_preview_dir


# ===========================================================================
# Internal helper methods
# ===========================================================================

class TestVisualizerHelpers:
    def test_stats_panel_correct_shape(
        self, tmp_path: Path, synthetic_calib_result: CalibrationResult
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        panel = viz._build_stats_panel(synthetic_calib_result)
        assert panel.ndim == 3
        assert panel.shape[2] == 3  # BGR

    def test_tile_thumbnails_produces_correct_columns(
        self, tmp_path: Path
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        thumbs = [np.zeros((60, 80, 3), dtype=np.uint8)] * 6
        sheet = viz._tile_thumbnails(thumbs, title="Test", cols=3)
        assert sheet.ndim == 3
        # Width should be 3 * 80
        assert sheet.shape[1] == 3 * 80

    def test_add_title_bar_increases_height(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        base = np.zeros((100, 200, 3), dtype=np.uint8)
        result = viz._add_title_bar(base, "Title")
        assert result.shape[0] > base.shape[0]

    def test_render_corner_thumb_returns_correct_size(
        self,
        tmp_path: Path,
        synthetic_detection_results: list,
    ) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        det = synthetic_detection_results[0]
        thumb = viz._render_corner_thumb(det)
        assert thumb.shape[:2] == (viz._THUMB_H, viz._THUMB_W)


# ===========================================================================
# generate_from_disk()
# ===========================================================================

class TestVisualizerFromDisk:
    def test_generate_from_disk_raises_if_no_npy(self, tmp_path: Path) -> None:
        config = _make_calib_config(tmp_path)
        viz = CalibrationVisualizer(config)
        with pytest.raises(FileNotFoundError):
            viz.generate_from_disk()

    def test_generate_from_disk_works_after_calibration(
        self, tmp_calib_with_results: dict
    ) -> None:
        config = tmp_calib_with_results["config"]
        viz = CalibrationVisualizer(config)
        vr = viz.generate_from_disk()
        assert vr.visualization_path is not None
        assert vr.visualization_path.exists()
