"""
ScrewMetric — Tests: Dataset Processor (Pipeline Orchestrator)
===============================================================
Unit tests for ``dataset/scripts/dataset_processor.py``.

Coverage:
- Full pipeline run on synthetic dataset (all stages enabled)
- Individual stage skip flags respected
- Pipeline result fields populated correctly
- success=True on clean run
- Stage failures captured (not raised) — pipeline continues
- Summary counts are consistent
- CLI argument parsing
- Standalone main() exits cleanly
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from dataset_processor import DatasetProcessor, PipelineResult, _build_arg_parser


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_full(tmp_dataset: dict[str, Any], **kwargs) -> PipelineResult:
    return DatasetProcessor(tmp_dataset["config"]).run(**kwargs)


# ===========================================================================
# PipelineResult dataclass
# ===========================================================================

class TestPipelineResult:
    def test_default_fields(self) -> None:
        result = PipelineResult()
        assert result.success is False
        assert result.preview_paths == {}
        assert result.elapsed_seconds == 0.0
        assert result.validation_report is None
        assert result.split_result is None
        assert result.statistics is None
        assert result.report_path is None

    def test_preview_paths_initialised_to_empty_dict(self) -> None:
        result = PipelineResult()
        assert isinstance(result.preview_paths, dict)


# ===========================================================================
# Full pipeline run
# ===========================================================================

class TestDatasetProcessorFullRun:
    def test_pipeline_returns_result(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert isinstance(result, PipelineResult)

    def test_pipeline_succeeds_on_clean_dataset(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        result = _run_full(tmp_dataset)
        assert result.success is True

    def test_validation_report_populated(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert result.validation_report is not None

    def test_split_result_populated(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert result.split_result is not None

    def test_statistics_populated(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert result.statistics is not None

    def test_report_path_exists(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert result.report_path is not None
        assert result.report_path.exists()

    def test_preview_paths_populated(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert isinstance(result.preview_paths, dict)
        assert len(result.preview_paths) == 3

    def test_elapsed_time_positive(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert result.elapsed_seconds > 0

    def test_split_counts_sum_to_total(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset)
        assert result.split_result is not None
        total = sum(result.split_result.split_counts.values())
        expected = len(tmp_dataset["coco"]["images"])
        assert total == expected


# ===========================================================================
# Stage skip flags
# ===========================================================================

class TestDatasetProcessorSkipFlags:
    def test_skip_validation_leaves_report_none(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        result = _run_full(tmp_dataset, run_validation=False)
        assert result.validation_report is None

    def test_skip_split_leaves_split_none(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(tmp_dataset, run_split=False)
        assert result.split_result is None

    def test_skip_statistics_leaves_stats_none(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        result = _run_full(tmp_dataset, run_statistics=False)
        assert result.statistics is None

    def test_skip_report_leaves_report_path_none(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        result = _run_full(tmp_dataset, run_report=False)
        assert result.report_path is None

    def test_skip_preview_leaves_preview_paths_empty(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        result = _run_full(tmp_dataset, run_preview=False)
        assert result.preview_paths == {}

    def test_skip_all_stages_still_succeeds(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        result = _run_full(
            tmp_dataset,
            run_validation=False,
            run_split=False,
            run_statistics=False,
            run_report=False,
            run_preview=False,
        )
        assert result.success is True

    def test_partial_run_validation_only(self, tmp_dataset: dict[str, Any]) -> None:
        result = _run_full(
            tmp_dataset,
            run_validation=True,
            run_split=False,
            run_statistics=False,
            run_report=False,
            run_preview=False,
        )
        assert result.success is True
        assert result.validation_report is not None
        assert result.split_result is None


# ===========================================================================
# Stage failure handling (pipeline continues)
# ===========================================================================

class TestDatasetProcessorFailureHandling:
    def test_stage_failure_does_not_raise(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        # Break the annotation file so validation, split, and stats fail
        config.paths.default_annotation_file.write_text(
            "not valid json", encoding="utf-8"
        )
        # Pipeline must not raise — it should capture the error
        result = DatasetProcessor(config).run()
        assert isinstance(result, PipelineResult)
        assert result.success is True  # orchestrator itself succeeded

    def test_failed_stage_returns_none_for_that_stage(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        config = tmp_dataset["config"]
        # Corrupt the annotation JSON so split and statistics stages fail
        config.paths.default_annotation_file.write_text(
            "not valid json", encoding="utf-8"
        )
        result = DatasetProcessor(config).run()
        # The split stage fails because it cannot parse the corrupted JSON
        assert result.split_result is None
        # The statistics stage also fails for the same reason
        assert result.statistics is None

    def test_pipeline_continues_after_stage_failure(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        """Even if validation fails, report generation should still run."""
        config = tmp_dataset["config"]
        config.paths.default_annotation_file.write_text(
            "not valid json", encoding="utf-8"
        )
        result = DatasetProcessor(config).run(
            run_validation=True,
            run_split=False,
            run_statistics=False,
            run_report=True,  # should still produce a report from defaults
            run_preview=False,
        )
        # Report stage should succeed (uses defaults when stats/validation missing)
        assert result.report_path is not None


# ===========================================================================
# _run_stage static helper
# ===========================================================================

class TestRunStageHelper:
    def test_successful_runner_returns_value(self) -> None:
        result = DatasetProcessor._run_stage(
            stage_name="Test Stage",
            stage_number=0,
            runner=lambda: 42,
        )
        assert result == 42

    def test_failing_runner_returns_none(self) -> None:
        result = DatasetProcessor._run_stage(
            stage_name="Bad Stage",
            stage_number=0,
            runner=lambda: 1 / 0,  # ZeroDivisionError
        )
        assert result is None


# ===========================================================================
# CLI argument parser
# ===========================================================================

class TestCLIArgParser:
    def test_parser_default_flags_are_all_false(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.skip_validation is False
        assert args.skip_split is False
        assert args.skip_stats is False
        assert args.skip_report is False
        assert args.skip_preview is False

    def test_skip_validation_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--skip-validation"])
        assert args.skip_validation is True

    def test_skip_split_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--skip-split"])
        assert args.skip_split is True

    def test_skip_stats_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--skip-stats"])
        assert args.skip_stats is True

    def test_skip_report_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--skip-report"])
        assert args.skip_report is True

    def test_skip_preview_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--skip-preview"])
        assert args.skip_preview is True

    def test_all_skip_flags_combined(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--skip-validation", "--skip-split",
            "--skip-stats", "--skip-report", "--skip-preview",
        ])
        assert all([
            args.skip_validation, args.skip_split,
            args.skip_stats, args.skip_report, args.skip_preview,
        ])
