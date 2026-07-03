"""
ScrewMetric — Camera Calibration Configuration
===============================================
Centralised configuration for the entire camera calibration pipeline.
All paths, constants, and tuneable parameters live here.

Responsibility (Single Responsibility Principle):
    Only configuration.  No I/O, no OpenCV calls, no business logic.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger for the calibration pipeline.

    Creates a new :class:`logging.StreamHandler` only when none is attached,
    preventing duplicate handlers on repeated imports.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------

def _calibration_root() -> Path:
    """Resolve the calibration module root relative to this config file.

    Returns:
        Absolute path to ``calibration/``.
    """
    # config.py lives in calibration/scripts/ → parent.parent == calibration/
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Checkerboard configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckerboardConfig:
    """Physical and geometric properties of the calibration checkerboard.

    Attributes:
        inner_corners_x: Number of inner corners along the horizontal axis
            (total horizontal squares minus 1).
        inner_corners_y: Number of inner corners along the vertical axis
            (total vertical squares minus 1).
        square_size_mm: Physical size of one square in millimetres.

    Raises:
        ValueError: If corners are < 2 or square size is non-positive.
    """

    inner_corners_x: int = 9
    inner_corners_y: int = 6
    square_size_mm: float = 25.0

    def __post_init__(self) -> None:
        if self.inner_corners_x < 2:
            raise ValueError(
                f"inner_corners_x must be >= 2, got {self.inner_corners_x}"
            )
        if self.inner_corners_y < 2:
            raise ValueError(
                f"inner_corners_y must be >= 2, got {self.inner_corners_y}"
            )
        if self.square_size_mm <= 0.0:
            raise ValueError(
                f"square_size_mm must be positive, got {self.square_size_mm}"
            )

    @property
    def pattern_size(self) -> tuple[int, int]:
        """OpenCV pattern size ``(width, height)`` of inner corners."""
        return (self.inner_corners_x, self.inner_corners_y)

    @property
    def total_inner_corners(self) -> int:
        """Total number of inner corner points on the board."""
        return self.inner_corners_x * self.inner_corners_y


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationPathConfig:
    """Filesystem paths used across the entire calibration pipeline.

    All paths are absolute so scripts work regardless of the current
    working directory.

    Attributes:
        calibration_root: Root of the ``calibration/`` directory.
    """

    calibration_root: Path = field(default_factory=_calibration_root)

    @property
    def images_dir(self) -> Path:
        """Directory containing raw calibration images."""
        return self.calibration_root / "images"

    @property
    def output_dir(self) -> Path:
        """Root output directory for all generated artefacts."""
        return self.calibration_root / "output"

    @property
    def undistortion_preview_dir(self) -> Path:
        """Directory for individual undistorted sample images."""
        return self.output_dir / "undistortion_preview"

    @property
    def camera_matrix_path(self) -> Path:
        """Path to the saved camera matrix numpy array."""
        return self.output_dir / "camera_matrix.npy"

    @property
    def dist_coeffs_path(self) -> Path:
        """Path to the saved distortion coefficients numpy array."""
        return self.output_dir / "dist_coeffs.npy"

    @property
    def rotation_vectors_path(self) -> Path:
        """Path to the saved rotation vectors numpy array."""
        return self.output_dir / "rotation_vectors.npy"

    @property
    def translation_vectors_path(self) -> Path:
        """Path to the saved translation vectors numpy array."""
        return self.output_dir / "translation_vectors.npy"

    @property
    def reprojection_error_path(self) -> Path:
        """Path to the per-image reprojection error JSON."""
        return self.output_dir / "reprojection_error.json"

    @property
    def validation_report_path(self) -> Path:
        """Path to the pre-calibration validation report JSON."""
        return self.output_dir / "validation_report.json"

    @property
    def calibration_report_path(self) -> Path:
        """Path to the post-calibration full report JSON."""
        return self.output_dir / "calibration_report.json"

    @property
    def camera_parameters_path(self) -> Path:
        """Path to the YAML human-readable camera parameters file."""
        return self.output_dir / "camera_parameters.yaml"

    @property
    def visualization_path(self) -> Path:
        """Path to the calibration visualization PNG."""
        return self.output_dir / "calibration_visualization.png"


# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds and options for the pre-calibration validator.

    Attributes:
        min_valid_images: Minimum number of images in which corners must be
            successfully detected before calibration is allowed.
        supported_extensions: File extensions considered valid images.
        verify_image_integrity: Whether to attempt decoding every image
            to confirm it is not corrupted (slower, but thorough).
    """

    min_valid_images: int = 10
    supported_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        )
    )
    verify_image_integrity: bool = True


# ---------------------------------------------------------------------------
# Calibration algorithm configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationProcessConfig:
    """Tuneable parameters for the OpenCV calibration algorithm.

    Attributes:
        subpix_max_iterations: Max iterations for sub-pixel corner refinement
            (``cv2.TERM_CRITERIA_MAX_ITER``).
        subpix_epsilon: Convergence threshold for sub-pixel refinement
            (``cv2.TERM_CRITERIA_EPS``).
        subpix_window_size: Half-size of the search window for sub-pixel
            corner refinement.
        subpix_zero_zone: Half-size of the dead-zone in the centre of the
            search window.  (-1, -1) disables the dead zone.
        undistortion_alpha: Alpha for ``cv2.getOptimalNewCameraMatrix``.
            ``0.0`` retains only valid pixels (crops); ``1.0`` keeps all
            pixels (adds black borders).
        max_preview_images: Maximum number of undistortion preview images
            to generate in ``undistortion_preview/``.
    """

    subpix_max_iterations: int = 30
    subpix_epsilon: float = 0.001
    subpix_window_size: tuple[int, int] = (11, 11)
    subpix_zero_zone: tuple[int, int] = (-1, -1)
    undistortion_alpha: float = 0.0
    max_preview_images: int = 5


# ---------------------------------------------------------------------------
# Top-level aggregate configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationConfig:
    """Aggregates all sub-configurations into a single pipeline object.

    Pass this object to every class constructor to keep function signatures
    clean and enable full dependency injection.

    Attributes:
        paths: Filesystem path configuration.
        checkerboard: Checkerboard geometry parameters.
        validation: Validation thresholds and parameters.
        process: Calibration algorithm parameters.
    """

    paths: CalibrationPathConfig = field(default_factory=CalibrationPathConfig)
    checkerboard: CheckerboardConfig = field(default_factory=CheckerboardConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    process: CalibrationProcessConfig = field(default_factory=CalibrationProcessConfig)

    @classmethod
    def default(cls) -> "CalibrationConfig":
        """Return a pipeline config with all default values.

        Returns:
            A fully-initialised :class:`CalibrationConfig`.
        """
        return cls(
            paths=CalibrationPathConfig(),
            checkerboard=CheckerboardConfig(),
            validation=ValidationConfig(),
            process=CalibrationProcessConfig(),
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate configuration loading and print all key settings."""
    print("=" * 62)
    print("  ScrewMetric — Camera Calibration Configuration Module")
    print("=" * 62)

    try:
        cfg = CalibrationConfig.default()

        print("\n[CheckerboardConfig]")
        print(f"  inner_corners_x    : {cfg.checkerboard.inner_corners_x}")
        print(f"  inner_corners_y    : {cfg.checkerboard.inner_corners_y}")
        print(f"  pattern_size       : {cfg.checkerboard.pattern_size}")
        print(f"  total_corners      : {cfg.checkerboard.total_inner_corners}")
        print(f"  square_size_mm     : {cfg.checkerboard.square_size_mm} mm")

        print("\n[CalibrationPathConfig]")
        print(f"  calibration_root   : {cfg.paths.calibration_root}")
        print(f"  images_dir         : {cfg.paths.images_dir}")
        print(f"  output_dir         : {cfg.paths.output_dir}")
        print(f"  camera_matrix      : {cfg.paths.camera_matrix_path}")
        print(f"  dist_coeffs        : {cfg.paths.dist_coeffs_path}")
        print(f"  calibration_report : {cfg.paths.calibration_report_path}")
        print(f"  camera_parameters  : {cfg.paths.camera_parameters_path}")
        print(f"  visualization      : {cfg.paths.visualization_path}")

        print("\n[ValidationConfig]")
        print(f"  min_valid_images   : {cfg.validation.min_valid_images}")
        print(f"  extensions         : {sorted(cfg.validation.supported_extensions)}")
        print(f"  verify_integrity   : {cfg.validation.verify_image_integrity}")

        print("\n[CalibrationProcessConfig]")
        print(f"  subpix_iterations  : {cfg.process.subpix_max_iterations}")
        print(f"  subpix_epsilon     : {cfg.process.subpix_epsilon}")
        print(f"  subpix_window      : {cfg.process.subpix_window_size}")
        print(f"  undist_alpha       : {cfg.process.undistortion_alpha}")
        print(f"  max_previews       : {cfg.process.max_preview_images}")

        # Validate checkerboard constraints
        bad = CheckerboardConfig(inner_corners_x=2, inner_corners_y=2, square_size_mm=25.0)
        assert bad.total_inner_corners == 4

        print("\n✅ config.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ config.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
