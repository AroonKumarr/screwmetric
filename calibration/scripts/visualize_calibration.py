"""
ScrewMetric — Calibration Visualizer
=======================================
Generates visual diagnostics for the camera calibration pipeline.

Responsibility (Single Responsibility Principle):
    Visualization only.  No calibration computation or report writing here.

Outputs:
    - Corner detection overlays (drawn on a contact sheet)
    - Before/after undistortion comparison grid
    - ``calibration_visualization.png`` — composite summary figure
    - ``undistortion_preview/`` — individual undistorted sample images

Authors: ScrewMetric Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibrate_camera import CalibrationResult, CornerDetectionResult
from calib_config import CalibrationConfig, get_logger
from calib_utils import ensure_dir, load_numpy, save_image

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VisualizationResult:
    """Paths to all generated visualization artefacts.

    Attributes:
        visualization_path: Main composite visualization PNG.
        undistortion_preview_paths: Individual undistorted image paths.
        corner_overlay_paths: Per-image corner detection overlay paths.
    """

    visualization_path: Optional[Path] = None
    undistortion_preview_paths: list[Path] = field(default_factory=list)
    corner_overlay_paths: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class CalibrationVisualizer:
    """Generates all calibration visualizations.

    Creates:

    1. A **corner detection contact sheet** showing detected corners on
       each calibration image.
    2. A **before/after undistortion comparison** for sample images.
    3. A **composite ``calibration_visualization.png``** combining all
       the above into a single shareable figure.
    4. Individual **undistorted preview images** in
       ``undistortion_preview/``.

    Args:
        config: Full calibration configuration.
    """

    _THUMB_W: int = 320
    _THUMB_H: int = 240
    _GRID_COLS: int = 4
    _TITLE_H: int = 40
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _BG_COLOR: tuple[int, int, int] = (30, 30, 30)
    _TEXT_COLOR: tuple[int, int, int] = (240, 240, 240)
    _ACCENT_COLOR: tuple[int, int, int] = (72, 199, 142)   # green
    _ERROR_COLOR: tuple[int, int, int] = (100, 100, 255)   # red-ish BGR

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._paths = config.paths
        self._board = config.checkerboard
        self._proc = config.process

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        result: CalibrationResult,
        detection_results: Optional[list[CornerDetectionResult]] = None,
    ) -> VisualizationResult:
        """Generate all calibration visualizations.

        Args:
            result: Completed calibration result.
            detection_results: Optional list of per-image corner detections.
                If provided, corner overlays are included in the composite.

        Returns:
            :class:`VisualizationResult` with paths to all generated files.
        """
        ensure_dir(self._paths.output_dir)
        ensure_dir(self._paths.undistortion_preview_dir)

        viz_result = VisualizationResult()

        # 1. Corner detection contact sheet
        corner_sheet: Optional[np.ndarray] = None
        if detection_results:
            corner_sheet = self._build_corner_sheet(detection_results)

        # 2. Undistortion comparison strip
        undist_strip = self._build_undistortion_strip(result)

        # 3. Stats panel
        stats_panel = self._build_stats_panel(result)

        # 4. Composite image
        composite = self._assemble_composite(corner_sheet, undist_strip, stats_panel)
        save_image(composite, self._paths.visualization_path)
        viz_result.visualization_path = self._paths.visualization_path
        logger.info("Composite visualization saved → %s", self._paths.visualization_path)

        # 5. Individual undistortion previews
        preview_paths = self._save_undistortion_previews(result)
        viz_result.undistortion_preview_paths = preview_paths

        return viz_result

    def generate_from_disk(self) -> VisualizationResult:
        """Re-generate visualizations by loading artefacts from disk.

        Returns:
            :class:`VisualizationResult` with paths to generated files.

        Raises:
            FileNotFoundError: If required ``.npy`` artefacts are missing.
        """
        logger.info("Loading calibration artefacts from disk for visualization…")
        camera_matrix = load_numpy(self._paths.camera_matrix_path)
        dist_coeffs = load_numpy(self._paths.dist_coeffs_path)

        result = CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            successful_images=sorted(
                self._paths.images_dir.glob("*.jpg")
            )[:self._proc.max_preview_images],
        )
        return self.generate_all(result)

    # ------------------------------------------------------------------
    # Corner sheet
    # ------------------------------------------------------------------

    def _build_corner_sheet(
        self,
        detections: list[CornerDetectionResult],
    ) -> np.ndarray:
        """Build a contact sheet showing corner overlays for every image.

        Args:
            detections: Per-image corner detection results.

        Returns:
            BGR contact sheet image.
        """
        thumbs: list[np.ndarray] = []
        for det in detections:
            thumb = self._render_corner_thumb(det)
            thumbs.append(thumb)

        return self._tile_thumbnails(
            thumbs,
            title="Corner Detection",
            cols=self._GRID_COLS,
        )

    def _render_corner_thumb(self, det: CornerDetectionResult) -> np.ndarray:
        """Render a single corner-detection thumbnail.

        Args:
            det: Detection result for one image.

        Returns:
            BGR thumbnail with corners drawn (or a red placeholder).
        """
        w, h = self._THUMB_W, self._THUMB_H
        img = cv2.imread(str(det.image_path))

        if img is None:
            thumb = np.full((h, w, 3), 60, dtype=np.uint8)
            cv2.putText(
                thumb, "UNREADABLE", (10, h // 2),
                self._FONT, 0.5, self._ERROR_COLOR, 1,
            )
            return thumb

        if det.success and det.corners is not None:
            cv2.drawChessboardCorners(
                img, self._board.pattern_size, det.corners, True
            )
            border_color = self._ACCENT_COLOR
        else:
            border_color = self._ERROR_COLOR

        thumb = cv2.resize(img, (w, h))
        cv2.rectangle(thumb, (0, 0), (w - 1, h - 1), border_color, 3)

        label = "OK" if det.success else "FAIL"
        cv2.putText(
            thumb, label, (6, h - 8),
            self._FONT, 0.45,
            self._ACCENT_COLOR if det.success else self._ERROR_COLOR, 1,
        )
        return thumb

    # ------------------------------------------------------------------
    # Undistortion strip
    # ------------------------------------------------------------------

    def _build_undistortion_strip(
        self,
        result: CalibrationResult,
    ) -> Optional[np.ndarray]:
        """Build a before/after undistortion comparison strip.

        Args:
            result: Calibration result containing camera matrix.

        Returns:
            BGR strip image, or ``None`` if no source images are available.
        """
        sample_paths = result.successful_images[: self._proc.max_preview_images]
        if not sample_paths:
            sample_paths = list(
                self._paths.images_dir.glob("*.jpg")
            )[: self._proc.max_preview_images]

        if not sample_paths:
            return None

        pairs: list[np.ndarray] = []
        for path in sample_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            pair = self._make_before_after_pair(img, result)
            pairs.append(pair)

        if not pairs:
            return None

        strip = np.vstack(pairs)
        return self._add_title_bar(strip, "Before / After Undistortion")

    def _make_before_after_pair(
        self,
        img: np.ndarray,
        result: CalibrationResult,
    ) -> np.ndarray:
        """Create a side-by-side before/after undistortion comparison.

        Args:
            img: Original BGR image.
            result: Calibration result.

        Returns:
            Side-by-side BGR comparison image.
        """
        h, w = img.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            result.camera_matrix,
            result.dist_coeffs,
            (w, h),
            self._proc.undistortion_alpha,
            (w, h),
        )
        undistorted = cv2.undistort(
            img, result.camera_matrix, result.dist_coeffs, None, new_K
        )

        th, tw = self._THUMB_H * 2, self._THUMB_W * 2
        before = cv2.resize(img, (tw, th))
        after = cv2.resize(undistorted, (tw, th))

        # Labels
        cv2.putText(
            before, "ORIGINAL", (10, 28),
            self._FONT, 0.7, (200, 200, 200), 2,
        )
        cv2.putText(
            after, "UNDISTORTED", (10, 28),
            self._FONT, 0.7, self._ACCENT_COLOR, 2,
        )

        divider = np.full((th, 4, 3), 80, dtype=np.uint8)
        return np.hstack([before, divider, after])

    # ------------------------------------------------------------------
    # Stats panel
    # ------------------------------------------------------------------

    def _build_stats_panel(self, result: CalibrationResult) -> np.ndarray:
        """Build a dark panel displaying key calibration parameters.

        Args:
            result: Calibration result.

        Returns:
            BGR stats panel image.
        """
        panel_w = self._THUMB_W * self._GRID_COLS
        panel_h = 220
        panel = np.full((panel_h, panel_w, 3), self._BG_COLOR, dtype=np.uint8)

        K = result.camera_matrix
        d = result.dist_coeffs.flatten()

        lines = [
            ("CAMERA CALIBRATION PARAMETERS", (0.65, self._ACCENT_COLOR, 2)),
            (f"Board: {self._board.inner_corners_x}x{self._board.inner_corners_y} inner corners  |  Square: {self._board.square_size_mm} mm", (0.5, self._TEXT_COLOR, 1)),
            (f"Images: {len(result.successful_images)} successful  /  {len(result.failed_images)} failed", (0.5, self._TEXT_COLOR, 1)),
            (f"Image size: {result.image_size[0]} x {result.image_size[1]} px", (0.5, self._TEXT_COLOR, 1)),
            (f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}", (0.5, (180, 220, 255), 1)),
            (f"k1={d[0]:.4f}  k2={d[1]:.4f}  p1={d[2]:.4f}  p2={d[3]:.4f}  k3={d[4] if len(d)>4 else 0:.4f}", (0.45, (180, 220, 255), 1)),
            (f"Mean reprojection error: {result.mean_reprojection_error:.4f} px", (0.6, self._ACCENT_COLOR, 2)),
        ]

        y = 28
        for text, (scale, color, thickness) in lines:
            cv2.putText(panel, text, (16, y), self._FONT, scale, color, thickness)
            y += int(scale * 60) + 6

        return panel

    # ------------------------------------------------------------------
    # Composite assembly
    # ------------------------------------------------------------------

    def _assemble_composite(
        self,
        corner_sheet: Optional[np.ndarray],
        undist_strip: Optional[np.ndarray],
        stats_panel: np.ndarray,
    ) -> np.ndarray:
        """Stack all panels into a single composite image.

        Args:
            corner_sheet: Corner detection contact sheet, or ``None``.
            undist_strip: Undistortion comparison strip, or ``None``.
            stats_panel: Calibration stats panel.

        Returns:
            Composite BGR image.
        """
        parts: list[np.ndarray] = []

        target_w = stats_panel.shape[1]

        def _pad_to_width(img: np.ndarray, w: int) -> np.ndarray:
            ih, iw = img.shape[:2]
            if iw == 0 or ih == 0 or w == 0:
                return np.full((1, max(w, 1), 3), self._BG_COLOR, dtype=np.uint8)
            if iw == w:
                return img
            # Use PIL instead of cv2.resize to avoid macOS OpenCV threading segfault
            from PIL import Image as _PILImage
            new_h = max(1, int(ih * w / iw))
            pil_img = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_resized = pil_img.resize((w, new_h), _PILImage.BILINEAR)
            return cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)



        parts.append(stats_panel)

        if corner_sheet is not None:
            parts.append(_pad_to_width(corner_sheet, target_w))

        if undist_strip is not None:
            parts.append(_pad_to_width(undist_strip, target_w))

        separator = np.full((4, target_w, 3), 60, dtype=np.uint8)
        stacked = parts[0]
        for part in parts[1:]:
            stacked = np.vstack([stacked, separator, part])

        return stacked

    # ------------------------------------------------------------------
    # Undistortion previews
    # ------------------------------------------------------------------

    def _save_undistortion_previews(
        self,
        result: CalibrationResult,
    ) -> list[Path]:
        """Save individual undistorted images to ``undistortion_preview/``.

        Args:
            result: Calibration result.

        Returns:
            List of paths to saved preview images.
        """
        sample_paths = result.successful_images[: self._proc.max_preview_images]
        if not sample_paths:
            sample_paths = sorted(self._paths.images_dir.glob("*.jpg"))[
                : self._proc.max_preview_images
            ]

        saved: list[Path] = []
        for i, path in enumerate(sample_paths):
            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                result.camera_matrix,
                result.dist_coeffs,
                (w, h),
                self._proc.undistortion_alpha,
                (w, h),
            )
            undistorted = cv2.undistort(
                img, result.camera_matrix, result.dist_coeffs, None, new_K
            )

            out_path = (
                self._paths.undistortion_preview_dir
                / f"undistorted_{i + 1:03d}_{path.stem}.jpg"
            )
            save_image(undistorted, out_path)
            saved.append(out_path)
            logger.info("Undistortion preview saved → %s", out_path.name)

        return saved

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _tile_thumbnails(
        self,
        thumbs: list[np.ndarray],
        title: str,
        cols: int,
    ) -> np.ndarray:
        """Arrange thumbnails in a grid with a title bar.

        Args:
            thumbs: List of equal-size BGR thumbnails.
            title: Title text for the contact sheet.
            cols: Number of grid columns.

        Returns:
            BGR grid image.
        """
        if not thumbs:
            placeholder = np.full(
                (self._THUMB_H, self._THUMB_W, 3), self._BG_COLOR, dtype=np.uint8
            )
            thumbs = [placeholder]

        rows = math.ceil(len(thumbs) / cols)
        tw, th = thumbs[0].shape[1], thumbs[0].shape[0]
        grid_w = cols * tw
        grid_h = rows * th

        grid = np.full((grid_h, grid_w, 3), self._BG_COLOR, dtype=np.uint8)

        for idx, thumb in enumerate(thumbs):
            r, c = divmod(idx, cols)
            y0, x0 = r * th, c * tw
            grid[y0 : y0 + th, x0 : x0 + tw] = thumb

        return self._add_title_bar(grid, title)

    def _add_title_bar(self, img: np.ndarray, title: str) -> np.ndarray:
        """Prepend a dark title bar above ``img``.

        Args:
            img: Base image.
            title: Title text.

        Returns:
            Image with title bar prepended.
        """
        w = img.shape[1]
        bar = np.full((self._TITLE_H, w, 3), (20, 20, 20), dtype=np.uint8)
        cv2.putText(
            bar, title, (12, 28),
            self._FONT, 0.7, self._ACCENT_COLOR, 2,
        )
        return np.vstack([bar, img])


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate visualizations from existing artefacts or synthetic data."""
    print("=" * 62)
    print("  ScrewMetric — Calibration Visualizer Module")
    print("=" * 62)

    try:
        config = CalibrationConfig.default()
        visualizer = CalibrationVisualizer(config)

        cam_path = config.paths.camera_matrix_path
        if cam_path.exists():
            print("\nLoading artefacts from disk and regenerating visualization…")
            viz_result = visualizer.generate_from_disk()
        else:
            print("\n⚠ No camera_matrix.npy found — generating with synthetic data…")
            K = np.array([
                [1500.0, 0.0, 960.0],
                [0.0, 1500.0, 540.0],
                [0.0, 0.0, 1.0],
            ])
            d = np.array([[0.1, -0.2, 0.001, 0.0005, 0.05]])
            result = CalibrationResult(
                camera_matrix=K,
                dist_coeffs=d,
                mean_reprojection_error=0.312,
                image_size=(1920, 1440),
                successful_images=[],
                failed_images=[],
            )
            viz_result = visualizer.generate_all(result)

        print(f"\n  Composite visualization : {viz_result.visualization_path}")
        print(f"  Undistortion previews   : {len(viz_result.undistortion_preview_paths)}")
        print(f"  Corner overlays         : {len(viz_result.corner_overlay_paths)}")

        print("\n✅ visualize_calibration.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ visualize_calibration.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
