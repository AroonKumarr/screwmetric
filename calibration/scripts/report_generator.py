"""
ScrewMetric — Calibration Report Generator
============================================
Generates human-readable and machine-readable calibration reports.

Responsibility (Single Responsibility Principle):
    Report serialisation only.  No calibration computation here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from calibrate_camera import CalibrationResult
from calib_config import CalibrationConfig, get_logger
from calib_utils import ensure_dir, load_numpy, save_json, save_yaml

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Structured calibration report ready for serialisation.

    Attributes:
        num_images_total: Total images processed.
        num_images_successful: Images that contributed to calibration.
        num_images_failed: Images that were rejected.
        image_resolution: ``[width, height]`` in pixels.
        checkerboard_inner_corners: ``[x, y]`` inner corners.
        square_size_mm: Physical square size in millimetres.
        camera_matrix: 3x3 intrinsic matrix as nested list.
        dist_coeffs: Distortion coefficients as a flat list.
        focal_length_fx: Focal length in pixels (x-axis).
        focal_length_fy: Focal length in pixels (y-axis).
        principal_point_cx: Principal point x-coordinate in pixels.
        principal_point_cy: Principal point y-coordinate in pixels.
        mean_reprojection_error: Mean reprojection error in pixels.
        per_image_errors: Per-image errors keyed by filename.
        calibration_date: ISO-8601 UTC timestamp of this report.
        execution_time_s: Time taken by the calibration run.
    """

    num_images_total: int = 0
    num_images_successful: int = 0
    num_images_failed: int = 0
    image_resolution: list[int] = field(default_factory=list)
    checkerboard_inner_corners: list[int] = field(default_factory=list)
    square_size_mm: float = 0.0
    camera_matrix: list[list[float]] = field(default_factory=list)
    dist_coeffs: list[float] = field(default_factory=list)
    focal_length_fx: float = 0.0
    focal_length_fy: float = 0.0
    principal_point_cx: float = 0.0
    principal_point_cy: float = 0.0
    mean_reprojection_error: float = 0.0
    per_image_errors: dict[str, float] = field(default_factory=dict)
    calibration_date: str = ""
    execution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Returns:
            Flat dictionary representation of the report.
        """
        return {
            "num_images_total": self.num_images_total,
            "num_images_successful": self.num_images_successful,
            "num_images_failed": self.num_images_failed,
            "image_resolution": self.image_resolution,
            "checkerboard_inner_corners": self.checkerboard_inner_corners,
            "square_size_mm": self.square_size_mm,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "focal_length_fx": self.focal_length_fx,
            "focal_length_fy": self.focal_length_fy,
            "principal_point_cx": self.principal_point_cx,
            "principal_point_cy": self.principal_point_cy,
            "mean_reprojection_error": self.mean_reprojection_error,
            "per_image_errors": self.per_image_errors,
            "calibration_date": self.calibration_date,
            "execution_time_s": self.execution_time_s,
        }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class CalibrationReportGenerator:
    """Generates ``calibration_report.json`` and ``camera_parameters.yaml``.

    Accepts a :class:`~calibrate_camera.CalibrationResult` and serialises
    all relevant fields into two formats:

    - **JSON** — detailed report for programmatic consumption
    - **YAML** — human-readable camera parameters for downstream use
      (e.g. ROS, measurement module)

    Args:
        config: Full calibration configuration.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._paths = config.paths
        self._board = config.checkerboard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, result: CalibrationResult) -> CalibrationReport:
        """Generate and save both the JSON report and the YAML parameters.

        Args:
            result: Completed calibration result from
                :class:`~calibrate_camera.CameraCalibrator`.

        Returns:
            The populated :class:`CalibrationReport` dataclass.
        """
        t_start = time.perf_counter()
        logger.info("Generating calibration report…")

        report = self._build_report(result)

        ensure_dir(self._paths.output_dir)
        self._write_json_report(report)
        self._write_yaml_parameters(result)

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info("Report generation complete in %.3fs", elapsed)
        return report

    def generate_from_disk(self) -> CalibrationReport:
        """Re-generate the report by loading artefacts from disk.

        Useful when the calibration was run previously and only the
        report needs to be regenerated.

        Returns:
            The populated :class:`CalibrationReport`.

        Raises:
            FileNotFoundError: If required numpy artefacts are missing.
        """
        logger.info("Loading calibration artefacts from disk…")
        camera_matrix = load_numpy(self._paths.camera_matrix_path)
        dist_coeffs = load_numpy(self._paths.dist_coeffs_path)

        # Reconstruct a minimal CalibrationResult
        result = CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        return self.generate(result)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_report(self, result: CalibrationResult) -> CalibrationReport:
        """Populate a :class:`CalibrationReport` from a calibration result.

        Args:
            result: Source calibration data.

        Returns:
            Populated report dataclass.
        """
        K = result.camera_matrix
        d = result.dist_coeffs.flatten().tolist()
        total = len(result.successful_images) + len(result.failed_images)
        per_errors = {
            p.name: err
            for p, err in zip(result.successful_images, result.per_image_errors)
        }

        return CalibrationReport(
            num_images_total=total,
            num_images_successful=len(result.successful_images),
            num_images_failed=len(result.failed_images),
            image_resolution=list(result.image_size),
            checkerboard_inner_corners=[
                self._board.inner_corners_x,
                self._board.inner_corners_y,
            ],
            square_size_mm=self._board.square_size_mm,
            camera_matrix=K.tolist(),
            dist_coeffs=d,
            focal_length_fx=round(float(K[0, 0]), 4),
            focal_length_fy=round(float(K[1, 1]), 4),
            principal_point_cx=round(float(K[0, 2]), 4),
            principal_point_cy=round(float(K[1, 2]), 4),
            mean_reprojection_error=round(result.mean_reprojection_error, 6),
            per_image_errors=per_errors,
            calibration_date=datetime.now(tz=timezone.utc).isoformat(),
            execution_time_s=result.calibration_duration_s,
        )

    def _write_json_report(self, report: CalibrationReport) -> None:
        """Serialise the report to ``calibration_report.json``.

        Args:
            report: Completed report to serialise.
        """
        save_json(report.to_dict(), self._paths.calibration_report_path)
        logger.info(
            "calibration_report.json saved → %s",
            self._paths.calibration_report_path,
        )

    def _write_yaml_parameters(self, result: CalibrationResult) -> None:
        """Serialise camera parameters to ``camera_parameters.yaml``.

        Produces a YAML file compatible with OpenCV's ``FileStorage``
        convention and the ROS ``camera_info`` message format.

        Args:
            result: Source calibration data.
        """
        K = result.camera_matrix
        d = result.dist_coeffs.flatten().tolist()

        yaml_data: dict[str, Any] = {
            "# ScrewMetric Camera Parameters": None,
            "# Generated by calibration/scripts/report_generator.py": None,
            "image_width": result.image_size[0] if result.image_size else 0,
            "image_height": result.image_size[1] if result.image_size else 0,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": K.flatten().tolist(),
            },
            "dist_coeffs": {
                "rows": 1,
                "cols": len(d),
                "data": d,
            },
            "focal_length_fx": round(float(K[0, 0]), 4),
            "focal_length_fy": round(float(K[1, 1]), 4),
            "principal_point_cx": round(float(K[0, 2]), 4),
            "principal_point_cy": round(float(K[1, 2]), 4),
            "reprojection_error": round(result.mean_reprojection_error, 6),
            "checkerboard_inner_corners_x": self._board.inner_corners_x,
            "checkerboard_inner_corners_y": self._board.inner_corners_y,
            "square_size_mm": self._board.square_size_mm,
            "date_created": datetime.now(tz=timezone.utc).isoformat(),
        }

        # Remove comment pseudo-keys
        clean_data = {k: v for k, v in yaml_data.items() if not k.startswith("#") and v is not None}
        save_yaml(clean_data, self._paths.camera_parameters_path)
        logger.info(
            "camera_parameters.yaml saved → %s",
            self._paths.camera_parameters_path,
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate report from existing artefacts, or show parameter preview."""
    print("=" * 62)
    print("  ScrewMetric — Calibration Report Generator Module")
    print("=" * 62)

    try:
        config = CalibrationConfig.default()
        generator = CalibrationReportGenerator(config)

        cam_matrix_path = config.paths.camera_matrix_path
        if not cam_matrix_path.exists():
            print(
                f"\n⚠ No camera_matrix.npy found at:\n  {cam_matrix_path}\n"
                "Run camera_calibration.py first to generate calibration artefacts.\n"
                "\nDemonstrating with synthetic data..."
            )
            # Demonstrate with synthetic data
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
                per_image_errors=[0.30, 0.31, 0.32],
                image_size=(1920, 1440),
                successful_images=[
                    config.paths.images_dir / "checkerboard_001.jpg",
                    config.paths.images_dir / "checkerboard_002.jpg",
                    config.paths.images_dir / "checkerboard_003.jpg",
                ],
                failed_images=[],
                calibration_duration_s=2.14,
            )
            report = generator.generate(result)
        else:
            report = generator.generate_from_disk()

        print(f"\n{'─' * 45}")
        print(f"  Calibration date        : {report.calibration_date[:19]}")
        print(f"  Images (total/ok/fail)  : {report.num_images_total}/{report.num_images_successful}/{report.num_images_failed}")
        print(f"  Image resolution        : {report.image_resolution}")
        print(f"  Checkerboard corners    : {report.checkerboard_inner_corners}")
        print(f"  Square size             : {report.square_size_mm} mm")
        print(f"  Focal length (fx, fy)   : ({report.focal_length_fx}, {report.focal_length_fy})")
        print(f"  Principal point (cx,cy) : ({report.principal_point_cx}, {report.principal_point_cy})")
        print(f"  Mean reprojection error : {report.mean_reprojection_error:.4f} px")
        print(f"  Execution time          : {report.execution_time_s:.2f}s")
        print(f"{'─' * 45}")
        print(f"\n  Report saved   → {config.paths.calibration_report_path}")
        print(f"  YAML saved     → {config.paths.camera_parameters_path}")

        print("\n✅ report_generator.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ report_generator.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
