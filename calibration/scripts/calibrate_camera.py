"""
ScrewMetric — Camera Calibrator
==================================
Implements the OpenCV camera calibration pipeline: corner detection,
sub-pixel refinement, and camera matrix estimation.

Responsibility (Single Responsibility Principle):
    Two classes — one detects corners, one performs calibration.
    No I/O of calibration images or report generation here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from calib_config import CalibrationConfig, get_logger
from calib_utils import ensure_dir, list_image_files, save_json, save_numpy

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CornerDetectionResult:
    """Outcome of checkerboard corner detection for a single image.

    Attributes:
        image_path: Path to the source image.
        success: Whether all corners were found.
        corners: Sub-pixel refined corner array, or ``None`` on failure.
        image_size: ``(width, height)`` of the image.
        error_message: Description of the failure, if any.
    """

    image_path: Path
    success: bool = False
    corners: Optional[np.ndarray] = None
    image_size: tuple[int, int] = (0, 0)
    error_message: str = ""

    @property
    def filename(self) -> str:
        """Filename component of :attr:`image_path`."""
        return self.image_path.name


@dataclass
class CalibrationResult:
    """Complete output of a successful camera calibration.

    Attributes:
        camera_matrix: 3x3 intrinsic camera matrix (K).
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3).
        rotation_vectors: Per-image rotation vectors.
        translation_vectors: Per-image translation vectors.
        mean_reprojection_error: Mean reprojection error across all images.
        per_image_errors: Per-image reprojection error values.
        image_size: Image resolution used ``(width, height)``.
        successful_images: Paths of images used for calibration.
        failed_images: Filenames that could not contribute.
        calibration_duration_s: Wall-clock time for calibration.
    """

    camera_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 5), dtype=np.float64)
    )
    rotation_vectors: list[np.ndarray] = field(default_factory=list)
    translation_vectors: list[np.ndarray] = field(default_factory=list)
    mean_reprojection_error: float = 0.0
    per_image_errors: list[float] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    successful_images: list[Path] = field(default_factory=list)
    failed_images: list[str] = field(default_factory=list)
    calibration_duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-compatible dictionary.

        Returns:
            Dict with all fields; numpy arrays converted to nested lists.
        """
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "mean_reprojection_error": self.mean_reprojection_error,
            "per_image_errors": self.per_image_errors,
            "image_size": list(self.image_size),
            "successful_images": [p.name for p in self.successful_images],
            "failed_images": self.failed_images,
            "calibration_duration_s": self.calibration_duration_s,
        }


# ---------------------------------------------------------------------------
# Corner detector
# ---------------------------------------------------------------------------

class CheckerboardCornerDetector:
    """Detects and sub-pixel-refines checkerboard corners in images.

    Single responsibility: transform raw images into precise corner arrays.

    Args:
        config: Full calibration configuration.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._board = config.checkerboard
        self._proc = config.process
        self._subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self._proc.subpix_max_iterations,
            self._proc.subpix_epsilon,
        )

    def detect_all(
        self,
        image_paths: list[Path],
    ) -> list[CornerDetectionResult]:
        """Run corner detection on a list of image paths.

        Args:
            image_paths: Sorted list of image file paths.

        Returns:
            List of :class:`CornerDetectionResult`, one per image.
        """
        results: list[CornerDetectionResult] = []
        for path in tqdm(image_paths, desc="Detecting corners", unit="img"):
            results.append(self.detect_single(path))
        return results

    def detect_single(self, path: Path) -> CornerDetectionResult:
        """Detect corners in a single image.

        Args:
            path: Path to the calibration image.

        Returns:
            :class:`CornerDetectionResult` for this image.
        """
        result = CornerDetectionResult(image_path=path)
        img = cv2.imread(str(path))

        if img is None:
            result.error_message = f"OpenCV could not decode: {path.name}"
            logger.warning("Skipping unreadable image: %s", path.name)
            return result

        h, w = img.shape[:2]
        result.image_size = (w, h)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            self._board.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found or corners is None:
            result.error_message = "Checkerboard pattern not found"
            logger.debug("Corners not found: %s", path.name)
            return result

        refined = cv2.cornerSubPix(
            gray,
            corners,
            self._proc.subpix_window_size,
            self._proc.subpix_zero_zone,
            self._subpix_criteria,
        )

        result.success = True
        result.corners = refined
        logger.debug("Corners detected: %s  (%d pts)", path.name, len(refined))
        return result

    def build_object_points(self) -> np.ndarray:
        """Build the 3-D object point grid for one checkerboard view.

        Returns a float32 array shaped ``(total_corners, 1, 3)`` where each
        point is ``(col * square_mm, row * square_mm, 0.0)``.

        Returns:
            Object points array shaped ``(total_corners, 1, 3)``.
        """
        cx = self._board.inner_corners_x
        cy = self._board.inner_corners_y
        sq = self._board.square_size_mm
        objp = np.zeros((self._board.total_inner_corners, 1, 3), dtype=np.float32)
        for idx in range(self._board.total_inner_corners):
            objp[idx, 0, 0] = (idx % cx) * sq
            objp[idx, 0, 1] = (idx // cx) * sq
        return objp


# ---------------------------------------------------------------------------
# Camera calibrator
# ---------------------------------------------------------------------------

class CameraCalibrator:
    """Estimates camera intrinsics from a set of corner detections.

    Accepts :class:`CornerDetectionResult` objects and calls
    ``cv2.calibrateCamera``.  Saves all artefacts to disk.

    Args:
        config: Full calibration configuration.
    """

    _MIN_IMAGES: int = 4

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._paths = config.paths
        self._detector = CheckerboardCornerDetector(config)

    def calibrate(
        self,
        detection_results: list[CornerDetectionResult],
    ) -> CalibrationResult:
        """Run camera calibration from pre-computed corner detections.

        Args:
            detection_results: Output of
                :meth:`~CheckerboardCornerDetector.detect_all`.

        Returns:
            A fully populated :class:`CalibrationResult`.

        Raises:
            ValueError: If fewer than ``_MIN_IMAGES`` images have corners.
        """
        t_start = time.perf_counter()
        result = CalibrationResult()

        successful = [r for r in detection_results if r.success]
        result.failed_images = [r.filename for r in detection_results if not r.success]

        if len(successful) < self._MIN_IMAGES:
            raise ValueError(
                f"Calibration requires >= {self._MIN_IMAGES} images with "
                f"detectable corners; only {len(successful)} available."
            )

        obj_template = self._detector.build_object_points()
        object_points = [obj_template] * len(successful)
        image_points = [r.corners for r in successful]
        image_size = successful[0].image_size

        logger.info(
            "Running cv2.calibrateCamera — %d images  size=%dx%d",
            len(successful), *image_size,
        )

        _rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,  # type: ignore[arg-type]
            image_points,   # type: ignore[arg-type]
            image_size,
            None,
            None,
        )

        per_errors = self._compute_per_image_errors(
            object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
        )

        result.camera_matrix = camera_matrix
        result.dist_coeffs = dist_coeffs
        result.rotation_vectors = list(rvecs)
        result.translation_vectors = list(tvecs)
        result.mean_reprojection_error = float(np.mean(per_errors))
        result.per_image_errors = per_errors
        result.image_size = image_size
        result.successful_images = [r.image_path for r in successful]
        result.calibration_duration_s = round(time.perf_counter() - t_start, 2)

        logger.info(
            "Calibration done — mean_error=%.4f px  time=%.2fs",
            result.mean_reprojection_error, result.calibration_duration_s,
        )
        self._save_artefacts(result)
        return result

    def calibrate_from_images(self, image_paths: list[Path]) -> CalibrationResult:
        """Convenience: detect corners then calibrate.

        Args:
            image_paths: Sorted list of calibration image paths.

        Returns:
            A fully populated :class:`CalibrationResult`.
        """
        detections = self._detector.detect_all(image_paths)
        return self.calibrate(detections)

    def _compute_per_image_errors(
        self,
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        rvecs: tuple,
        tvecs: tuple,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> list[float]:
        """Compute per-image mean reprojection error in pixels.

        Args:
            object_points: 3-D object points per image.
            image_points: Detected 2-D corner points per image.
            rvecs: Rotation vectors.
            tvecs: Translation vectors.
            camera_matrix: Intrinsic matrix.
            dist_coeffs: Distortion coefficients.

        Returns:
            List of per-image mean errors.
        """
        errors: list[float] = []
        for objp, imgp, rvec, tvec in zip(
            object_points, image_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(
                objp, rvec, tvec, camera_matrix, dist_coeffs
            )
            # Reshape both to (N, 2) for reliable comparison across OpenCV versions
            imgp_2d = imgp.reshape(-1, 2)
            proj_2d = projected.reshape(-1, 2)
            diff = imgp_2d - proj_2d
            err = float(np.sqrt((diff ** 2).sum(axis=1)).mean())
            errors.append(round(err, 6))
        return errors


    def _save_artefacts(self, result: CalibrationResult) -> None:
        """Persist all calibration artefacts to the output directory.

        Args:
            result: Completed calibration result.
        """
        ensure_dir(self._paths.output_dir)
        save_numpy(result.camera_matrix, self._paths.camera_matrix_path)
        save_numpy(result.dist_coeffs, self._paths.dist_coeffs_path)
        save_numpy(
            np.array(result.rotation_vectors),
            self._paths.rotation_vectors_path,
        )
        save_numpy(
            np.array(result.translation_vectors),
            self._paths.translation_vectors_path,
        )
        reprojection_data = {
            "mean_reprojection_error": result.mean_reprojection_error,
            "per_image_errors": {
                p.name: err
                for p, err in zip(result.successful_images, result.per_image_errors)
            },
        }
        save_json(reprojection_data, self._paths.reprojection_error_path)
        logger.info("Artefacts saved → %s", self._paths.output_dir)


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run camera calibration on images in the configured directory."""
    print("=" * 62)
    print("  ScrewMetric — Camera Calibrator Module")
    print("=" * 62)

    try:
        config = CalibrationConfig.default()
        image_files = list_image_files(
            config.paths.images_dir,
            config.validation.supported_extensions,
        )

        print(f"\nImages dir    : {config.paths.images_dir}")
        print(f"Images found  : {len(image_files)}")
        print(f"Board pattern : {config.checkerboard.pattern_size}")

        if not image_files:
            print(
                "\n⚠ No images found. Place checkerboard images in:\n"
                f"  {config.paths.images_dir}\n"
                "\n✅ calibrate_camera.py executed successfully (no images)."
            )
            return

        print("\nRunning calibration...\n")
        calibrator = CameraCalibrator(config)
        result = calibrator.calibrate_from_images(image_files)

        print(f"\n{'─' * 45}")
        print(f"  Successful images     : {len(result.successful_images)}")
        print(f"  Failed images         : {len(result.failed_images)}")
        print(f"  Image size            : {result.image_size}")
        print(f"  Mean reprojection err : {result.mean_reprojection_error:.4f} px")
        print(f"  Calibration time      : {result.calibration_duration_s:.2f}s")
        print(f"\n  Camera Matrix (K):")
        for row in result.camera_matrix:
            print(f"    [{row[0]:12.4f}  {row[1]:12.4f}  {row[2]:12.4f}]")
        print(f"\n  Distortion Coefficients:")
        print(f"    {result.dist_coeffs.flatten().tolist()}")
        print(f"{'─' * 45}")

        print("\n✅ calibrate_camera.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ calibrate_camera.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
