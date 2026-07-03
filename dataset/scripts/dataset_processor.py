#!/usr/bin/env python3
"""
ScrewMetric — Dataset Processor (CLI Entry Point)
===================================================
Orchestrates the complete dataset processing pipeline in a single command:

    1. Validate dataset integrity
    2. Split into train / val / test
    3. Compute statistics
    4. Generate Markdown report
    5. Generate contact-sheet previews

This module acts as the composition root — it wires together all
sub-components but contains no business logic itself.

Usage::

    python dataset_processor.py [--skip-validation] [--skip-split]
                                 [--skip-stats] [--skip-report]
                                 [--skip-preview]

Authors: ScrewMetric Team
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import PipelineConfig, get_logger
from dataset_validator import DatasetValidator, ValidationReport
from dataset_splitter import DatasetSplitter, SplitResult
from dataset_statistics import DatasetStatisticsComputer, DatasetStatistics
from dataset_report_generator import DatasetReportGenerator
from preview_generator import PreviewGenerator

logger = get_logger(__name__)

# ANSI colours for terminal output (degrade gracefully on Windows)
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Aggregated result of the full dataset processing pipeline.

    Attributes:
        validation_report: Result of the validation stage, or ``None`` if skipped.
        split_result: Result of the split stage, or ``None`` if skipped.
        statistics: Computed statistics, or ``None`` if skipped.
        report_path: Path to the written Markdown report, or ``None`` if skipped.
        preview_paths: Mapping of split name → preview PNG path (may be empty).
        success: ``True`` if the pipeline completed without fatal errors.
        elapsed_seconds: Total wall-clock time in seconds.
    """

    validation_report: Optional[ValidationReport] = None
    split_result: Optional[SplitResult] = None
    statistics: Optional[DatasetStatistics] = None
    report_path: Optional[Path] = None
    preview_paths: dict[str, Optional[Path]] = None  # type: ignore[assignment]
    success: bool = False
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.preview_paths is None:
            self.preview_paths = {}


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class DatasetProcessor:
    """Orchestrates the complete dataset processing pipeline.

    Args:
        config: Pipeline configuration.  Pass ``PipelineConfig.default()``
            for the standard configuration.

    Example::

        processor = DatasetProcessor(PipelineConfig.default())
        result = processor.run()
        if result.success:
            print("Pipeline completed successfully.")
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        run_validation: bool = True,
        run_split: bool = True,
        run_statistics: bool = True,
        run_report: bool = True,
        run_preview: bool = True,
    ) -> PipelineResult:
        """Execute all enabled pipeline stages in order.

        Args:
            run_validation: Enable dataset validation stage.
            run_split: Enable dataset splitting stage.
            run_statistics: Enable statistics computation stage.
            run_report: Enable Markdown report generation stage.
            run_preview: Enable contact-sheet preview generation stage.

        Returns:
            A :class:`PipelineResult` summarising every stage's outcome.
        """
        _banner("ScrewMetric Dataset Processing Pipeline")
        t_start = time.perf_counter()
        result = PipelineResult()

        # ── Stage 1: Validation ──────────────────────────────────────
        if run_validation:
            result.validation_report = self._run_stage(
                stage_name="Dataset Validation",
                stage_number=1,
                runner=lambda: DatasetValidator(self._config).validate(),
            )
            if result.validation_report and not result.validation_report.is_valid:
                logger.warning(
                    "Validation found %d error(s). "
                    "Proceeding with remaining stages — review validation_report.json.",
                    len(result.validation_report.errors),
                )
        else:
            _skip_stage(1, "Dataset Validation")

        # ── Stage 2: Split ───────────────────────────────────────────
        if run_split:
            result.split_result = self._run_stage(
                stage_name="Dataset Split",
                stage_number=2,
                runner=lambda: DatasetSplitter(self._config).split(),
            )
            if result.split_result:
                counts = result.split_result.split_counts
                logger.info(
                    "Split complete — train: %d | val: %d | test: %d",
                    counts["train"], counts["val"], counts["test"],
                )
        else:
            _skip_stage(2, "Dataset Split")

        # ── Stage 3: Statistics ──────────────────────────────────────
        if run_statistics:
            result.statistics = self._run_stage(
                stage_name="Dataset Statistics",
                stage_number=3,
                runner=lambda: DatasetStatisticsComputer(self._config).compute(),
            )
        else:
            _skip_stage(3, "Dataset Statistics")

        # ── Stage 4: Report ──────────────────────────────────────────
        if run_report:
            result.report_path = self._run_stage(
                stage_name="Report Generation",
                stage_number=4,
                runner=lambda: DatasetReportGenerator(self._config).generate(),
            )
        else:
            _skip_stage(4, "Report Generation")

        # ── Stage 5: Previews ────────────────────────────────────────
        if run_preview:
            preview_paths = self._run_stage(
                stage_name="Preview Generation",
                stage_number=5,
                runner=lambda: PreviewGenerator(self._config).generate_all(),
            )
            result.preview_paths = preview_paths or {}
        else:
            _skip_stage(5, "Preview Generation")

        # ── Summary ──────────────────────────────────────────────────
        result.elapsed_seconds = round(time.perf_counter() - t_start, 2)
        result.success = True
        _print_summary(result, self._config)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_stage(stage_name: str, stage_number: int, runner):
        """Run a single pipeline stage with consistent logging and error handling.

        Args:
            stage_name: Human-readable name of the stage.
            stage_number: Sequential stage number (for display).
            runner: Zero-argument callable that performs the stage.

        Returns:
            The return value of ``runner()``, or ``None`` on failure.
        """
        print(f"\n{_BOLD}{_CYAN}[{stage_number}/5] {stage_name}{_RESET}")
        logger.info("Starting stage %d: %s", stage_number, stage_name)
        t0 = time.perf_counter()
        try:
            result = runner()
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  {_GREEN}✓ Completed in {elapsed}s{_RESET}")
            logger.info("Stage %d completed in %.2fs", stage_number, elapsed)
            return result
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  {_RED}✗ Failed: {exc}{_RESET}")
            logger.error(
                "Stage %d (%s) failed after %.2fs: %s",
                stage_number, stage_name, elapsed, exc,
            )
            logger.debug(traceback.format_exc())
            return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    """Print a styled banner to stdout.

    Args:
        title: Text to display in the banner.
    """
    width = 60
    border = "═" * width
    print(f"\n{_BOLD}{_CYAN}╔{border}╗")
    print(f"║  {title:<{width - 2}}║")
    print(f"╚{border}╝{_RESET}\n")


def _skip_stage(number: int, name: str) -> None:
    """Print a skip notice for a pipeline stage.

    Args:
        number: Stage number.
        name: Stage name.
    """
    print(f"\n{_YELLOW}[{number}/5] {name} — SKIPPED{_RESET}")


def _print_summary(result: PipelineResult, config: PipelineConfig) -> None:
    """Print a formatted pipeline summary to stdout.

    Args:
        result: The completed pipeline result.
        config: Pipeline configuration (for path display).
    """
    dataset_root = config.paths.dataset_root
    print(f"\n{_BOLD}{'─' * 60}")
    print("  Pipeline Summary")
    print(f"{'─' * 60}{_RESET}")
    print(f"  ⏱  Elapsed time    : {result.elapsed_seconds}s")

    if result.validation_report:
        vr = result.validation_report
        v_status = f"{_GREEN}✅ Valid{_RESET}" if vr.is_valid else f"{_RED}❌ Invalid{_RESET}"
        print(f"  🔍 Validation      : {v_status} "
              f"({len(vr.errors)} errors, {len(vr.warnings)} warnings)")

    if result.split_result:
        sc = result.split_result.split_counts
        print(
            f"  ✂️  Split           : "
            f"train={sc['train']} | val={sc['val']} | test={sc['test']}"
        )

    if result.statistics:
        s = result.statistics
        print(f"  📊 Statistics      : {s.total_images} images, {s.total_annotations} annotations")

    if result.report_path:
        print(f"  📝 Report          : {result.report_path.relative_to(dataset_root)}")

    for split, path in (result.preview_paths or {}).items():
        if path:
            print(f"  🖼  Preview [{split}]   : {path.relative_to(dataset_root)}")

    print(f"{_BOLD}{'─' * 60}{_RESET}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="dataset_processor",
        description=(
            "ScrewMetric Dataset Processing Pipeline\n\n"
            "Validates, splits, analyses, and previews the annotated screw dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run the full pipeline (default)\n"
            "  python dataset_processor.py\n\n"
            "  # Validate only\n"
            "  python dataset_processor.py --skip-split --skip-stats --skip-report --skip-preview\n\n"
            "  # Regenerate statistics and report without re-splitting\n"
            "  python dataset_processor.py --skip-validation --skip-split\n"
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the dataset validation stage.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip the dataset splitting stage (useful when splits already exist).",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip statistics computation.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip Markdown report generation.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip contact-sheet preview generation.",
    )
    return parser


def main() -> None:
    """CLI entry point for the dataset processing pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = PipelineConfig.default()
    processor = DatasetProcessor(config)

    result = processor.run(
        run_validation=not args.skip_validation,
        run_split=not args.skip_split,
        run_statistics=not args.skip_stats,
        run_report=not args.skip_report,
        run_preview=not args.skip_preview,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
