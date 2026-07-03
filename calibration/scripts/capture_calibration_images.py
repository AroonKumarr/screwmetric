"""
ScrewMetric — Calibration Image Capture
=========================================
Provides a live camera capture interface for collecting checkerboard
calibration images via OpenCV, and displays guidance for iPhone capture.

Responsibility (Single Responsibility Principle):
    Image acquisition only.  No calibration computation lives here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calib_config import CalibrationConfig, get_logger
from calib_utils import ensure_dir, list_image_files

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaptureSession:
    """Summary of a completed image capture session.

    Attributes:
        captured_paths: Paths to successfully saved images.
        skipped_count: Number of frames discarded (poor corners).
        session_duration_s: Total session wall-clock time in seconds.
    """

    captured_paths: list[Path] = field(default_factory=list)
    skipped_count: int = 0
    session_duration_s: float = 0.0

    @property
    def captured_count(self) -> int:
        """Number of successfully saved images."""
        return len(self.captured_paths)


# ---------------------------------------------------------------------------
# Capture engine
# ---------------------------------------------------------------------------

class CalibrationImageCapture:
    """Captures checkerboard calibration images from a connected camera.

    Provides two modes:

    - **Live capture** (``capture_live``): Opens an OpenCV VideoCapture
      window and auto-detects checkerboard corners before saving.
    - **Guidance** (``print_guidance``): Prints step-by-step instructions
      for capturing images using an iPhone (or any camera) manually.

    Args:
        config: Full calibration configuration.
        camera_index: OpenCV camera device index (default ``0``).
    """

    _WINDOW_NAME: str = "ScrewMetric — Calibration Capture  |  Q=quit  SPACE=capture"
    _OVERLAY_FOUND_COLOR: tuple[int, int, int] = (72, 199, 142)
    _OVERLAY_MISSING_COLOR: tuple[int, int, int] = (255, 100, 67)
    _MIN_CAPTURE_INTERVAL_S: float = 0.5

    def __init__(
        self,
        config: CalibrationConfig,
        camera_index: int = 0,
    ) -> None:
        self._config = config
        self._camera_index = camera_index
        self._paths = config.paths
        self._board = config.checkerboard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_live(
        self,
        target_count: int = 30,
        require_corners: bool = True,
    ) -> CaptureSession:
        """Open a live preview window and capture calibration images.

        Images are saved to the configured ``images_dir``.  If
        ``require_corners`` is ``True``, only frames where the checkerboard
        pattern is fully detected are accepted.

        Args:
            target_count: Desired number of images to capture.
            require_corners: If ``True``, reject frames where OpenCV cannot
                detect all checkerboard corners.

        Returns:
            A :class:`CaptureSession` summarising the session.

        Raises:
            IOError: If the camera cannot be opened.
        """
        ensure_dir(self._paths.images_dir)
        session = CaptureSession()
        t_start = time.perf_counter()

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise IOError(
                f"Cannot open camera at index {self._camera_index}. "
                "Ensure a camera is connected and not in use by another process."
            )

        logger.info(
            "Live capture started — target=%d  require_corners=%s",
            target_count, require_corners,
        )

        try:
            last_capture_time = 0.0
            while session.captured_count < target_count:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame — retrying.")
                    continue

                display = frame.copy()
                corners_found, corners = self._detect_corners(frame)

                if corners_found and corners is not None:
                    cv2.drawChessboardCorners(
                        display, self._board.pattern_size, corners, corners_found
                    )
                    status_text = "Corners detected"
                    status_color = self._OVERLAY_FOUND_COLOR
                else:
                    status_text = "No corners"
                    status_color = self._OVERLAY_MISSING_COLOR

                self._draw_hud(
                    display, status_text, status_color,
                    session.captured_count, target_count,
                )
                cv2.imshow(self._WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("Capture terminated by user.")
                    break

                now = time.perf_counter()
                if key == ord(" ") and (now - last_capture_time) >= self._MIN_CAPTURE_INTERVAL_S:
                    if require_corners and not corners_found:
                        session.skipped_count += 1
                        logger.debug("Frame rejected — corners not detected.")
                    else:
                        save_path = self._next_image_path(session.captured_count)
                        cv2.imwrite(str(save_path), frame)
                        session.captured_paths.append(save_path)
                        last_capture_time = now
                        logger.info(
                            "Captured %d/%d → %s",
                            session.captured_count, target_count, save_path.name,
                        )

        finally:
            cap.release()
            cv2.destroyAllWindows()
            session.session_duration_s = round(time.perf_counter() - t_start, 2)

        logger.info(
            "Session complete — captured=%d  skipped=%d  duration=%.1fs",
            session.captured_count, session.skipped_count, session.session_duration_s,
        )
        return session

    def print_guidance(self) -> None:
        """Print step-by-step instructions for manual iPhone capture."""
        board = self._board
        img_dir = self._paths.images_dir
        print("\n" + "=" * 62)
        print("  ScrewMetric — Calibration Image Capture Guidance")
        print("=" * 62)
        print(f"""
CHECKERBOARD DETAILS
  Inner corners : {board.inner_corners_x} x {board.inner_corners_y}
  Square size   : {board.square_size_mm} mm
  Total squares : {board.inner_corners_x + 1} x {board.inner_corners_y + 1}

OUTPUT DIRECTORY
  {img_dir}

CAPTURE INSTRUCTIONS
  1. Print the checkerboard on a flat, rigid surface.
     Avoid glossy finishes (causes reflections).

  2. Ensure even lighting. Avoid direct sunlight or bright spots.

  3. Capture at least 20 images from varied positions:
     - Angles: tilt left/right/up/down 30-45 degrees
     - Distances: near, medium, far
     - All four board rotations
     - Fill the frame without clipping edges

  4. On iPhone:
     - Open Camera -> Photo
     - Tap board to focus, then shoot
     - Transfer via AirDrop or cable
     - Rename: checkerboard_001.jpg, checkerboard_002.jpg ...
     - Place in: {img_dir}

  5. Run: python camera_calibration.py

TIPS
  Use 20-30 good images. Vary both position AND angle.
  Avoid blur (use Timer). Ensure all corners are visible.
""")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_corners(
        self,
        frame: np.ndarray,
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Detect checkerboard corners in a BGR frame.

        Args:
            frame: BGR image from the camera.

        Returns:
            Tuple of ``(found, corners)``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(
            gray, self._board.pattern_size, flags
        )
        return bool(found), corners

    def _next_image_path(self, index: int) -> Path:
        """Return the next image save path.

        Args:
            index: Zero-based capture index.

        Returns:
            Path like ``images/checkerboard_001.jpg``.
        """
        existing = list_image_files(
            self._paths.images_dir,
            frozenset({".jpg", ".jpeg", ".png"}),
        )
        n = len(existing) + 1
        return self._paths.images_dir / f"checkerboard_{n:03d}.jpg"

    @staticmethod
    def _draw_hud(
        img: np.ndarray,
        status_text: str,
        status_color: tuple[int, int, int],
        captured: int,
        target: int,
    ) -> None:
        """Overlay a HUD on the live preview frame.

        Args:
            img: Frame to annotate in-place.
            status_text: Corner detection status.
            status_color: BGR colour for status text.
            captured: Images captured so far.
            target: Total target count.
        """
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(
            img, status_text, (10, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
        )
        cv2.putText(
            img,
            f"Captured: {captured}/{target}  |  SPACE=capture  Q=quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Print capture guidance and demonstrate configuration loading."""
    print("=" * 62)
    print("  ScrewMetric — Capture Calibration Images Module")
    print("=" * 62)

    try:
        config = CalibrationConfig.default()
        capture = CalibrationImageCapture(config)
        capture.print_guidance()

        existing = list_image_files(
            config.paths.images_dir,
            config.validation.supported_extensions,
        )
        print(f"Existing images in images/ : {len(existing)}")
        print(f"Board pattern              : {config.checkerboard.pattern_size}")
        print(f"Square size                : {config.checkerboard.square_size_mm} mm")

        print("\nTo start live capture, run:")
        print("  capture.capture_live(target_count=30)")

        print("\n✅ capture_calibration_images.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ capture_calibration_images.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
