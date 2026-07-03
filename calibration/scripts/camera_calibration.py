"""
ScrewMetric — Camera Calibration Pipeline (Orchestrator)
==========================================================
Composition root for the entire camera calibration module.
Wires together all sub-modules and exposes a single CLI entry point.

Responsibility (Single Responsibility Principle):
    Orchestration only.  All business logic lives in the sub-modules.

Usage::

    # Full pipeline
    python camera_calibration.py

    # Skip specific stages
    python camera_calibration.py --skip-validation
    python camera_calibration.py --skip-visualization

    # Custom checkerboard
    python camera_calibration.py --corners-x 8 --corners-y 5 --square-mm 30.0

Authors: ScrewMetric Team
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from calibrate_camera import CalibrationResult, CameraCalibrator
from calibration_validator import CalibrationValidator, ValidationReport
from calib_config import (
    CalibrationConfig,
    CalibrationPathConfig,
    CalibrationProcessConfig,
    CheckerboardConfig,
    ValidationConfig,
    get_logger,
)
from report_generator import CalibrationReport, CalibrationReportGenerator
from calib_utils import list_image_files
from visualize_calibration import CalibrationVisualizer, VisualizationResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Aggregated result from a full calibration pipeline run.

    Attributes:
        success: Whether the pipeline completed without fatal errors.
        validation_report: Result of the pre-calibration validation stage.
        calibration_result: Result of the OpenCV calibration stage.
        calibration_report: Generated report from the report stage.
        visualization_result: Paths to generated visual artefacts.
        elapsed_seconds: Total wall-clock time for the pipeline.
        error_message: Set when ``success`` is ``False``.
    """

    success: bool = False
    validation_report: Optional[ValidationReport] = None
    calibration_result: Optional[CalibrationResult] = None
    calibration_report: Optional[CalibrationReport] = None
    visualization_result: Optional[VisualizationResult] = None
    elapsed_seconds: float = 0.0
    error_message: str = ""


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class CalibrationPipeline:
    """Orchestrates all stages of the camera calibration pipeline.

    Stage order:

    1. Validate calibration images (``CalibrationValidator``)
    2. Detect checkerboard corners and calibrate (``CameraCalibrator``)
    3. Generate reports (``CalibrationReportGenerator``)
    4. Generate visualizations (``CalibrationVisualizer``)

    Args:
        config: Full calibration configuration.
    """

    _STAGE_NAMES = [
        "Validation",
        "Calibration",
        "Report Generation",
        "Visualization",
    ]

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._paths = config.paths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        run_validation: bool = True,
        run_calibration: bool = True,
        run_report: bool = True,
        run_visualization: bool = True,
    ) -> PipelineResult:
        """Execute the calibration pipeline.

        Args:
            run_validation: Run the pre-calibration validation stage.
            run_calibration: Run the OpenCV calibration stage.
            run_report: Generate JSON and YAML reports.
            run_visualization: Generate visualization images.

        Returns:
            :class:`PipelineResult` summarising all stages.
        """
        t_start = time.perf_counter()
        result = PipelineResult()

        logger.info("=" * 60)
        logger.info("  ScrewMetric Camera Calibration Pipeline")
        logger.info("=" * 60)

        # ── Stage 1: Validation ───────────────────────────────────────
        if run_validation:
            report = self._run_stage(
                stage_name="Validation",
                stage_number=1,
                runner=lambda: CalibrationValidator(self._config).validate(),
            )
            result.validation_report = report
            if report is not None and not report.is_valid:
                logger.warning(
                    "Validation found %d error(s). "
                    "Review validation_report.json before proceeding.",
                    len(report.errors),
                )

        # ── Stage 2: Calibration ──────────────────────────────────────
        detection_results = None
        if run_calibration:
            image_files = list_image_files(
                self._paths.images_dir,
                self._config.validation.supported_extensions,
            )
            if not image_files:
                logger.warning(
                    "No calibration images found in %s — skipping calibration.",
                    self._paths.images_dir,
                )
            else:
                from calibrate_camera import CheckerboardCornerDetector

                detector = CheckerboardCornerDetector(self._config)
                detection_results = detector.detect_all(image_files)

                cal_result = self._run_stage(
                    stage_name="Calibration",
                    stage_number=2,
                    runner=lambda: CameraCalibrator(self._config).calibrate(
                        detection_results  # type: ignore[arg-type]
                    ),
                )
                result.calibration_result = cal_result

        # ── Stage 3: Report ───────────────────────────────────────────
        if run_report and result.calibration_result is not None:
            cal_report = self._run_stage(
                stage_name="Report Generation",
                stage_number=3,
                runner=lambda: CalibrationReportGenerator(self._config).generate(
                    result.calibration_result  # type: ignore[arg-type]
                ),
            )
            result.calibration_report = cal_report

        # ── Stage 4: Visualization ────────────────────────────────────
        if run_visualization and result.calibration_result is not None:
            viz = self._run_stage(
                stage_name="Visualization",
                stage_number=4,
                runner=lambda: CalibrationVisualizer(self._config).generate_all(
                    result.calibration_result,  # type: ignore[arg-type]
                    detection_results,
                ),
            )
            result.visualization_result = viz

        result.elapsed_seconds = round(time.perf_counter() - t_start, 2)
        result.success = True

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_stage(
        stage_name: str,
        stage_number: int,
        runner,
    ):
        """Execute one pipeline stage with error isolation.

        Args:
            stage_name: Human-readable stage name for logging.
            stage_number: Ordinal stage number.
            runner: Zero-argument callable that performs the stage work.

        Returns:
            The return value of ``runner()``, or ``None`` on failure.
        """
        logger.info("Starting stage %d: %s", stage_number, stage_name)
        t0 = time.perf_counter()
        try:
            result = runner()
            elapsed = round(time.perf_counter() - t0, 2)
            logger.info(
                "Stage %d (%s) completed in %.2fs",
                stage_number, stage_name, elapsed,
            )
            return result
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            logger.error(
                "Stage %d (%s) failed after %.2fs: %s",
                stage_number, stage_name, elapsed, exc,
            )
            logger.debug("Traceback:", exc_info=True)
            return None

    def _print_summary(self, result: PipelineResult) -> None:
        """Print a formatted pipeline summary to stdout.

        Args:
            result: Completed pipeline result.
        """
        lines = [
            "",
            "─" * 60,
            "  Camera Calibration Pipeline Summary",
            "─" * 60,
            f"  Status                 : {'✅ SUCCESS' if result.success else '❌ FAILED'}",
            f"  Elapsed time           : {result.elapsed_seconds:.2f}s",
        ]

        if result.validation_report:
            vr = result.validation_report
            lines += [
                f"  Images found           : {vr.total_images_found}",
                f"  Images with corners    : {vr.images_with_corners}",
                f"  Validation status      : {'VALID' if vr.is_valid else 'INVALID'}",
            ]

        if result.calibration_result:
            cr = result.calibration_result
            lines += [
                f"  Calibration images     : {len(cr.successful_images)} used / {len(cr.failed_images)} failed",
                f"  Image resolution       : {cr.image_size[0]}x{cr.image_size[1]}",
                f"  Mean reprojection err  : {cr.mean_reprojection_error:.4f} px",
            ]

        if result.calibration_report:
            lines.append(
                f"  Report saved           : {self._paths.calibration_report_path.name}"
            )
            lines.append(
                f"  Camera params YAML     : {self._paths.camera_parameters_path.name}"
            )

        if result.visualization_result:
            vz = result.visualization_result
            lines += [
                f"  Visualization          : {self._paths.visualization_path.name}",
                f"  Undistortion previews  : {len(vz.undistortion_preview_paths)}",
            ]

        lines.append("─" * 60)
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured :class:`~argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="camera_calibration",
        description="ScrewMetric — Camera Calibration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python camera_calibration.py\n"
            "  python camera_calibration.py --corners-x 8 --corners-y 5\n"
            "  python camera_calibration.py --skip-validation --skip-visualization\n"
        ),
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the pre-calibration validation stage.",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip the OpenCV calibration stage.",
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Skip JSON/YAML report generation.",
    )
    parser.add_argument(
        "--skip-visualization", action="store_true",
        help="Skip visualization generation.",
    )
    parser.add_argument(
        "--corners-x", type=int, default=9, metavar="N",
        help="Number of inner corners along the horizontal axis (default: 9).",
    )
    parser.add_argument(
        "--corners-y", type=int, default=6, metavar="N",
        help="Number of inner corners along the vertical axis (default: 6).",
    )
    parser.add_argument(
        "--square-mm", type=float, default=25.0, metavar="MM",
        help="Physical square size in millimetres (default: 25.0).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, metavar="A",
        help=(
            "Undistortion alpha: 0.0 = crop to valid pixels (default), "
            "1.0 = keep all pixels."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the calibration pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("=" * 62)
    print("  ScrewMetric — Camera Calibration Pipeline")
    print("=" * 62)

    try:
        config = CalibrationConfig(
            paths=CalibrationPathConfig(),
            checkerboard=CheckerboardConfig(
                inner_corners_x=args.corners_x,
                inner_corners_y=args.corners_y,
                square_size_mm=args.square_mm,
            ),
            validation=ValidationConfig(),
            process=CalibrationProcessConfig(
                undistortion_alpha=args.alpha,
            ),
        )

        print(f"\n  Images dir     : {config.paths.images_dir}")
        print(f"  Output dir     : {config.paths.output_dir}")
        print(f"  Board pattern  : {config.checkerboard.pattern_size} inner corners")
        print(f"  Square size    : {config.checkerboard.square_size_mm} mm")
        print(f"  Alpha          : {config.process.undistortion_alpha}")

        pipeline = CalibrationPipeline(config)
        result = pipeline.run(
            run_validation=not args.skip_validation,
            run_calibration=not args.skip_calibration,
            run_report=not args.skip_report,
            run_visualization=not args.skip_visualization,
        )

        sys.exit(0 if result.success else 1)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        print(f"\n❌ Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
