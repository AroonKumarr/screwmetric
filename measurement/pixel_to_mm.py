"""
ScrewMetric — Pixel-to-Millimetre Converter
============================================
Converts segmentation mask geometry into real-world screw dimensions
(in millimetres) using calibrated camera intrinsics and lens distortion
correction.

Mathematical Model
------------------
We use the **pinhole camera model** with radial + tangential distortion
correction (OpenCV 5-coefficient model).

**Scale derivation:**

For a camera observing a flat object at known depth Z (mm) along the
optical axis:

    u = fx * (X / Z) + cx
    →  X_mm = (u - cx) * Z / fx

Therefore the pixel-to-mm scale factor at distance Z is:

    scale [mm/px] = Z / f_avg

where f_avg = (fx + fy) / 2 is the average focal length in pixels.

**Pipeline per measurement:**

1. Find the largest contour in the binary segmentation mask.
2. Undistort the contour points using K and D (cv2.undistortPoints).
3. Fit a minimum-area rotated rectangle to the undistorted contour.
4. Extract major axis (→ screw length) and minor axis (→ screw diameter).
5. Multiply both axes by scale [mm/px].

Responsibility (Single Responsibility Principle):
    Only metrology math. No model loading, no config-file I/O.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import sys
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow importing model_config for paths when run standalone
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helper (consistent with rest of project)
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    return log


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasurementConfig:
    """Configuration for the pixel-to-mm converter.

    Attributes:
        camera_matrix_path: Path to the NumPy ``.npy`` file containing
            the 3×3 camera intrinsic matrix ``K``.
        dist_coeffs_path: Path to the NumPy ``.npy`` file containing
            the 1×5 distortion coefficients ``D``.
        known_distance_mm: Perpendicular distance from the camera's
            optical centre to the object (screw) plane in millimetres.
            This is the Z used in the scale formula: scale = Z / f.
        min_contour_area_px: Minimum contour area (px²) required to
            consider a mask valid.  Protects against noise masks.

    Note:
        ``known_distance_mm`` is required for correct mm scaling.
        Measure the actual camera-to-screw distance in your capture rig.
        A typical close-up smartphone setup is 150–400 mm.
    """

    camera_matrix_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "calibration" / "output" / "camera_matrix.npy"
    )
    dist_coeffs_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "calibration" / "output" / "dist_coeffs.npy"
    )
    known_distance_mm: float = 300.0
    min_contour_area_px: float = 100.0

    def __post_init__(self) -> None:
        if self.known_distance_mm <= 0.0:
            raise ValueError(
                f"known_distance_mm must be positive, got {self.known_distance_mm}"
            )
        if self.min_contour_area_px < 0.0:
            raise ValueError(
                f"min_contour_area_px must be non-negative, got {self.min_contour_area_px}"
            )

    @classmethod
    def default(cls) -> "MeasurementConfig":
        """Return config with project-default paths."""
        return cls()


# ---------------------------------------------------------------------------
# Measurement result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScrewMeasurement:
    """Computed real-world dimensions of a detected screw.

    Attributes:
        length_mm: Estimated screw length in millimetres (major axis).
        diameter_mm: Estimated screw diameter in millimetres (minor axis).
        confidence: Detection confidence forwarded from inference.
        pixel_length: Major axis in pixels (before mm conversion).
        pixel_diameter: Minor axis in pixels (before mm conversion).
        scale_mm_per_px: Applied mm-per-pixel scale factor.
        method: Description of the scale estimation method used.
        rect_angle_deg: Orientation of the fitted rectangle (degrees).
    """

    length_mm: float
    diameter_mm: float
    confidence: float
    pixel_length: float
    pixel_diameter: float
    scale_mm_per_px: float
    method: str
    rect_angle_deg: float = 0.0

    def to_dict(self) -> dict:
        """Serialise the measurement to a plain dictionary."""
        return {
            "length_mm": round(self.length_mm, 3),
            "diameter_mm": round(self.diameter_mm, 3),
            "confidence": round(self.confidence, 4),
            "pixel_length": round(self.pixel_length, 2),
            "pixel_diameter": round(self.pixel_diameter, 2),
            "scale_mm_per_px": round(self.scale_mm_per_px, 6),
            "method": self.method,
            "rect_angle_deg": round(self.rect_angle_deg, 2),
        }


# ---------------------------------------------------------------------------
# Core converter class
# ---------------------------------------------------------------------------

class PixelToMMConverter:
    """Converts segmentation mask geometry into millimetre measurements.

    Args:
        config: Measurement configuration.

    Example::

        cfg = MeasurementConfig(known_distance_mm=300.0)
        converter = PixelToMMConverter(cfg)
        measurement = converter.measure(mask, confidence=0.92)
        print(measurement.length_mm, measurement.diameter_mm)
    """

    def __init__(self, config: MeasurementConfig) -> None:
        self._config = config
        self._K: Optional[np.ndarray] = None     # (3, 3) camera matrix
        self._D: Optional[np.ndarray] = None     # (1, 5) distortion coeffs
        self._scale: Optional[float] = None      # mm/px

    # ------------------------------------------------------------------
    # Public: calibration loading
    # ------------------------------------------------------------------

    def load_calibration(self) -> tuple[np.ndarray, np.ndarray]:
        """Load camera intrinsics and distortion coefficients from disk.

        Returns:
            Tuple ``(K, D)`` where K is shape (3, 3) and D is shape (1, 5).

        Raises:
            FileNotFoundError: If either ``.npy`` file is missing.
            ValueError: If the loaded arrays have unexpected shapes.
        """
        K_path = self._config.camera_matrix_path
        D_path = self._config.dist_coeffs_path

        if not K_path.exists():
            raise FileNotFoundError(
                f"Camera matrix not found: {K_path}\n"
                "Run the calibration pipeline first: "
                "python calibration/scripts/camera_calibration.py"
            )
        if not D_path.exists():
            raise FileNotFoundError(
                f"Distortion coefficients not found: {D_path}\n"
                "Run the calibration pipeline first."
            )

        K = np.load(str(K_path))
        D = np.load(str(D_path))

        if K.shape != (3, 3):
            raise ValueError(f"camera_matrix.npy must be (3,3), got {K.shape}")
        if D.ndim == 1:
            D = D.reshape(1, -1)
        if D.shape[1] not in {4, 5, 8, 12, 14}:
            raise ValueError(
                f"dist_coeffs.npy must have 4/5/8/12/14 coefficients, "
                f"got {D.shape[1]}"
            )

        self._K = K
        self._D = D
        self._scale = self._compute_scale()

        logger.info(
            "Calibration loaded — fx=%.2f fy=%.2f | scale=%.6f mm/px @ Z=%.1f mm",
            K[0, 0], K[1, 1], self._scale, self._config.known_distance_mm,
        )
        return K, D

    # ------------------------------------------------------------------
    # Public: measurement API
    # ------------------------------------------------------------------

    def compute_real_world_length(self, mask: np.ndarray) -> float:
        """Compute the screw length (major axis) in millimetres.

        Args:
            mask: Binary uint8 mask of shape (H, W).

        Returns:
            Screw length in millimetres.

        Raises:
            RuntimeError: If calibration has not been loaded.
            ValueError: If the mask contains no valid contour.
        """
        self._ensure_calibrated()
        major_px, _ = self.fit_rotated_rect(self._get_contour(mask))
        return self.convert_pixel_to_mm(major_px)

    def compute_real_world_diameter(self, mask: np.ndarray) -> float:
        """Compute the screw diameter (minor axis) in millimetres.

        Args:
            mask: Binary uint8 mask of shape (H, W).

        Returns:
            Screw diameter in millimetres.

        Raises:
            RuntimeError: If calibration has not been loaded.
            ValueError: If the mask contains no valid contour.
        """
        self._ensure_calibrated()
        _, minor_px = self.fit_rotated_rect(self._get_contour(mask))
        return self.convert_pixel_to_mm(minor_px)

    def convert_pixel_to_mm(self, pixel_distance: float) -> float:
        """Convert a pixel distance to millimetres using the current scale.

        Args:
            pixel_distance: Distance in pixels.

        Returns:
            Distance in millimetres.

        Raises:
            RuntimeError: If calibration has not been loaded.
        """
        self._ensure_calibrated()
        assert self._scale is not None
        return pixel_distance * self._scale

    def undistort_contour(self, contour: np.ndarray) -> np.ndarray:
        """Correct lens distortion on a contour's pixel coordinates.

        Uses ``cv2.undistortPoints`` with the full camera matrix to map
        distorted image-plane coordinates to undistorted coordinates.

        Args:
            contour: Contour array of shape (N, 1, 2) as returned by
                ``cv2.findContours``.

        Returns:
            Undistorted contour of the same shape.

        Raises:
            RuntimeError: If calibration has not been loaded.
        """
        self._ensure_calibrated()
        assert self._K is not None and self._D is not None

        pts = contour.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(pts, self._K, self._D, P=self._K)
        return undistorted.reshape(-1, 1, 2)

    def fit_rotated_rect(
        self, contour: np.ndarray
    ) -> tuple[float, float]:
        """Fit a minimum-area rotated rectangle and return axis lengths.

        The major axis corresponds to screw **length**; the minor axis
        to screw **diameter**.

        Args:
            contour: Undistorted contour array of shape (N, 1, 2).

        Returns:
            Tuple ``(major_px, minor_px)`` in pixels, with major >= minor.
        """
        if len(contour) < 5:
            # Fallback for degenerate contours: use bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            major = float(max(w, h))
            minor = float(min(w, h))
            return major, minor

        _, (w, h), angle = cv2.minAreaRect(contour)
        major = float(max(w, h))
        minor = float(min(w, h))
        return major, minor

    def measure(
        self,
        mask: np.ndarray,
        confidence: float = 1.0,
    ) -> ScrewMeasurement:
        """Full measurement pipeline: mask → ScrewMeasurement.

        Steps:
            1. Extract contour from mask.
            2. Undistort contour points.
            3. Fit rotated rectangle → major/minor axes.
            4. Convert to mm using pinhole scale.

        Args:
            mask: Binary uint8 mask of shape (H, W).
            confidence: Detection confidence to embed in the result.

        Returns:
            :class:`ScrewMeasurement` with all computed fields.

        Raises:
            RuntimeError: If calibration has not been loaded.
            ValueError: If the mask contains no valid contour.
        """
        self._ensure_calibrated()

        contour = self._get_contour(mask)
        undistorted = self.undistort_contour(contour)
        major_px, minor_px = self.fit_rotated_rect(undistorted)

        # Angle of the fitted rectangle
        _, _, angle = cv2.minAreaRect(undistorted) if len(undistorted) >= 5 else (
            (0, 0), (major_px, minor_px), 0.0
        )

        assert self._scale is not None
        length_mm = major_px * self._scale
        diameter_mm = minor_px * self._scale

        logger.info(
            "Measurement — length=%.2f mm  diameter=%.2f mm  "
            "(major=%.1f px  minor=%.1f px  scale=%.6f mm/px)",
            length_mm, diameter_mm, major_px, minor_px, self._scale,
        )

        return ScrewMeasurement(
            length_mm=round(length_mm, 3),
            diameter_mm=round(diameter_mm, 3),
            confidence=round(float(confidence), 4),
            pixel_length=round(major_px, 2),
            pixel_diameter=round(minor_px, 2),
            scale_mm_per_px=round(self._scale, 6),
            method="focal_length_pinhole",
            rect_angle_deg=round(float(angle), 2),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_scale(self) -> float:
        """Compute the mm-per-pixel scale at the configured object distance.

        Formula: scale = Z / f_avg
        where Z is the known camera-to-object distance and
        f_avg = (fx + fy) / 2.

        Returns:
            Scale in mm per pixel.
        """
        assert self._K is not None
        fx = float(self._K[0, 0])
        fy = float(self._K[1, 1])
        f_avg = (fx + fy) / 2.0
        if f_avg <= 0.0:
            raise ValueError(
                f"Focal length must be positive, got fx={fx}, fy={fy}"
            )
        Z = self._config.known_distance_mm
        return Z / f_avg

    def _ensure_calibrated(self) -> None:
        """Raise RuntimeError if calibration has not been loaded."""
        if self._K is None or self._D is None:
            raise RuntimeError(
                "Calibration not loaded. Call load_calibration() first."
            )

    def _get_contour(self, mask: np.ndarray) -> np.ndarray:
        """Extract the largest contour from a binary mask.

        Args:
            mask: Binary uint8 mask.

        Returns:
            Largest contour array of shape (N, 1, 2).

        Raises:
            ValueError: If no contour with sufficient area is found.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError(
                "No contours found in mask — is the mask non-empty?"
            )

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self._config.min_contour_area_px:
            raise ValueError(
                f"Largest contour area ({area:.1f} px²) is below "
                f"minimum threshold ({self._config.min_contour_area_px} px²). "
                "The mask may be too noisy or the screw too small."
            )

        return largest


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate pixel-to-mm conversion with real calibration data."""
    print("=" * 64)
    print("  ScrewMetric — Pixel-to-MM Converter Module")
    print("=" * 64)

    try:
        cfg = MeasurementConfig.default()

        print(f"\n[MeasurementConfig]")
        print(f"  camera_matrix_path : {cfg.camera_matrix_path}")
        print(f"  dist_coeffs_path   : {cfg.dist_coeffs_path}")
        print(f"  known_distance_mm  : {cfg.known_distance_mm} mm")

        converter = PixelToMMConverter(cfg)

        # Attempt to load real calibration; fall back to synthetic if missing
        K, D = None, None
        try:
            K, D = converter.load_calibration()
            print(f"\n[load_calibration]")
            print(f"  K =\n{K}")
            print(f"  D = {D}")
            print(f"  scale = {converter._scale:.6f} mm/px")
        except FileNotFoundError as exc:
            print(f"\n[load_calibration] SKIPPED — {exc}")
            print("  → Using synthetic K/D for demonstration...")
            # Inject synthetic calibration for demo
            K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
            D = np.zeros((1, 5))
            converter._K = K
            converter._D = D
            converter._scale = converter._compute_scale()
            print(f"  scale (synthetic) = {converter._scale:.6f} mm/px")

        # Build a synthetic screw-like mask (tall narrow rectangle)
        mask = np.zeros((600, 400), dtype=np.uint8)
        # Screw body: ~400px tall, ~80px wide
        cv2.rectangle(mask, (160, 100), (240, 500), 255, -1)

        measurement = converter.measure(mask, confidence=0.95)

        print(f"\n[measure — synthetic screw mask]")
        print(f"  pixel_length    = {measurement.pixel_length:.1f} px")
        print(f"  pixel_diameter  = {measurement.pixel_diameter:.1f} px")
        print(f"  length_mm       = {measurement.length_mm:.2f} mm")
        print(f"  diameter_mm     = {measurement.diameter_mm:.2f} mm")
        print(f"  confidence      = {measurement.confidence}")
        print(f"  scale_mm_per_px = {measurement.scale_mm_per_px:.6f}")

        # Verify scaling: pixel_length * scale ≈ length_mm
        expected = measurement.pixel_length * measurement.scale_mm_per_px
        assert abs(measurement.length_mm - expected) < 0.01, "Scale mismatch"

        # Verify no-distortion stability
        D_zero = np.zeros((1, 5))
        converter._D = D_zero
        m2 = converter.measure(mask, confidence=0.9)
        assert m2.length_mm > 0, "Zero distortion should still produce valid result"
        print(f"\n[zero distortion]   length_mm = {m2.length_mm:.2f} mm  ✓")

        # Test invalid mask raises
        bad_mask = np.zeros((100, 100), dtype=np.uint8)
        try:
            converter.measure(bad_mask)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            print("[invalid mask]      ValueError raised correctly  ✓")

        print("\n✅ pixel_to_mm.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ pixel_to_mm.py failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
