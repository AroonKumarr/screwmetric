"""
ScrewMetric — Dataset Report Generator
========================================
Produces the ``DATASET_REPORT.md`` Markdown document from live pipeline
artefacts (validation report, statistics, split annotations).

Responsibility (Single Responsibility Principle):
    Only report-template rendering and file writing live here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PipelineConfig, get_logger
from utils import load_json, save_json

logger = get_logger(__name__)


class DatasetReportGenerator:
    """Renders a Markdown dataset report from pipeline artefacts.

    The report draws from three JSON sources:

    * ``dataset_stats.json`` — quantitative statistics
    * ``validation_report.json`` — validation findings
    * Split annotation files — per-split details

    Args:
        config: Pipeline configuration.

    Example::

        gen = DatasetReportGenerator(PipelineConfig.default())
        gen.generate()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._paths = config.paths
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Path:
        """Render and write ``DATASET_REPORT.md``.

        Returns:
            Absolute path to the written report file.
        """
        stats = self._safe_load_json(self._paths.dataset_stats_path, {})
        validation = self._safe_load_json(self._paths.validation_report_path, {})

        report_md = self._render(stats, validation)

        self._paths.dataset_report_path.parent.mkdir(parents=True, exist_ok=True)
        self._paths.dataset_report_path.write_text(report_md, encoding="utf-8")
        logger.info("Dataset report saved → %s", self._paths.dataset_report_path)
        return self._paths.dataset_report_path

    # ------------------------------------------------------------------
    # Private rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        stats: dict[str, Any],
        validation: dict[str, Any],
    ) -> str:
        """Compose the full Markdown report string.

        Args:
            stats: Loaded ``dataset_stats.json`` dict (may be empty).
            validation: Loaded ``validation_report.json`` dict (may be empty).

        Returns:
            Complete Markdown string.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            self._section_header(now),
            self._section_overview(stats, validation),
            self._section_folder_structure(),
            self._section_image_counts(stats),
            self._section_split_ratios(stats),
            self._section_resolution(stats),
            self._section_categories(stats),
            self._section_annotation_stats(stats),
            self._section_validation(validation),
            self._section_recommendations(stats, validation),
            self._section_footer(),
        ]
        return "\n\n".join(sections)

    # ── Individual sections ──────────────────────────────────────────

    def _section_header(self, timestamp: str) -> str:
        return (
            "# ScrewMetric — Dataset Report\n\n"
            f"> **Generated:** {timestamp}  \n"
            f"> **Project:** ScrewMetric Computer Vision Pipeline  \n"
            f"> **Purpose:** Screw segmentation and dimension measurement\n\n"
            "---"
        )

    def _section_overview(
        self,
        stats: dict[str, Any],
        validation: dict[str, Any],
    ) -> str:
        total = stats.get("total_images", "N/A")
        total_ann = stats.get("total_annotations", "N/A")
        size = stats.get("dataset_size_human", "N/A")
        is_valid = validation.get("is_valid", None)
        status = "✅ Passed" if is_valid else ("❌ Failed" if is_valid is False else "⚠️ Unknown")
        n_errors = len(validation.get("errors", []))
        n_warnings = len(validation.get("warnings", []))

        return (
            "## 1. Dataset Overview\n\n"
            "| Property | Value |\n"
            "|---|---|\n"
            f"| Total annotated images | **{total}** |\n"
            f"| Total annotation instances | **{total_ann}** |\n"
            f"| Dataset size on disk | **{size}** |\n"
            f"| Validation status | {status} |\n"
            f"| Validation errors | {n_errors} |\n"
            f"| Validation warnings | {n_warnings} |\n"
        )

    def _section_folder_structure(self) -> str:
        return (
            "## 2. Folder Structure\n\n"
            "```\n"
            "dataset/\n"
            "├── raw_images/\n"
            "│   ├── screw_only/          ← Annotated images\n"
            "│   └── screw_checkerboard/  ← Calibration images\n"
            "├── annotations/\n"
            "│   └── instances_default.json  ← Master COCO annotation\n"
            "├── splits/\n"
            "│   ├── train/\n"
            "│   │   ├── images/\n"
            "│   │   └── annotations/instances_train.json\n"
            "│   ├── val/\n"
            "│   │   ├── images/\n"
            "│   │   └── annotations/instances_val.json\n"
            "│   └── test/\n"
            "│       ├── images/\n"
            "│       └── annotations/instances_test.json\n"
            "├── previews/\n"
            "│   ├── preview_train.png\n"
            "│   ├── preview_val.png\n"
            "│   └── preview_test.png\n"
            "├── scripts/                 ← Dataset processing module\n"
            "├── validation_report.json\n"
            "├── dataset_stats.json\n"
            "└── DATASET_REPORT.md\n"
            "```"
        )

    def _section_image_counts(self, stats: dict[str, Any]) -> str:
        total = stats.get("total_images", 0)
        missing = stats.get("missing_labels", 0)
        multi = stats.get("images_with_multiple_annotations", 0)

        return (
            "## 3. Image Counts\n\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            f"| Total annotated images | {total} |\n"
            f"| Images without labels | {missing} |\n"
            f"| Images with multiple annotations | {multi} |\n"
        )

    def _section_split_ratios(self, stats: dict[str, Any]) -> str:
        splits = stats.get("splits", {})
        total = stats.get("total_images", 1) or 1

        rows = ""
        for name in ("train", "val", "test"):
            raw = splits.get(name, {})
            # Handle both new dict format and old integer format
            if isinstance(raw, dict):
                count = raw.get("count", 0)
                ann_count = raw.get("annotation_count", 0)
            else:
                count = int(raw) if raw else 0
                ann_count = 0
            pct = round(count / total * 100, 1) if total else 0.0
            rows += f"| {name.capitalize()} | {count} | {pct}% | {ann_count} |\n"

        return (
            "## 4. Dataset Splits\n\n"
            "| Split | Images | Percentage | Annotations |\n"
            "|---|---|---|---|\n"
            + rows
        )

    def _section_resolution(self, stats: dict[str, Any]) -> str:
        res = stats.get("resolution", {})
        return (
            "## 5. Image Resolution Statistics\n\n"
            "| Metric | Width (px) | Height (px) |\n"
            "|---|---|---|\n"
            f"| Mean | {res.get('avg_width', 'N/A')} | {res.get('avg_height', 'N/A')} |\n"
            f"| Min | {res.get('min_width', 'N/A')} | {res.get('min_height', 'N/A')} |\n"
            f"| Max | {res.get('max_width', 'N/A')} | {res.get('max_height', 'N/A')} |\n"
            f"| Std Dev | {res.get('std_width', 'N/A')} | {res.get('std_height', 'N/A')} |\n"
        )

    def _section_categories(self, stats: dict[str, Any]) -> str:
        dist = stats.get("category_distribution", {})
        if not dist:
            return "## 6. Category Summary\n\n_No category data available._"

        rows = "\n".join(
            f"| {name} | {count} |" for name, count in dist.items()
        )
        return (
            "## 6. Category Summary\n\n"
            "| Category | Instance Count |\n"
            "|---|---|\n"
            + rows
        )

    def _section_annotation_stats(self, stats: dict[str, Any]) -> str:
        bbox = stats.get("bbox", {})
        poly = stats.get("polygon", {})
        total_ann = stats.get("total_annotations", "N/A")

        return (
            "## 7. Annotation Statistics\n\n"
            "### Bounding Boxes\n\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            f"| Total bboxes | {bbox.get('count', 'N/A')} |\n"
            f"| Avg width (px) | {bbox.get('avg_width', 'N/A')} |\n"
            f"| Avg height (px) | {bbox.get('avg_height', 'N/A')} |\n"
            f"| Avg area (px²) | {bbox.get('avg_area', 'N/A')} |\n"
            f"| Min area (px²) | {bbox.get('min_area', 'N/A')} |\n"
            f"| Max area (px²) | {bbox.get('max_area', 'N/A')} |\n\n"
            "### Polygon Segmentations\n\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            f"| Total polygons | {poly.get('count', 'N/A')} |\n"
            f"| Avg vertices/polygon | {poly.get('avg_points_per_polygon', 'N/A')} |\n"
            f"| Min vertices | {poly.get('min_points', 'N/A')} |\n"
            f"| Max vertices | {poly.get('max_points', 'N/A')} |\n"
        )

    def _section_validation(self, validation: dict[str, Any]) -> str:
        is_valid = validation.get("is_valid", None)
        status_icon = "✅" if is_valid else ("❌" if is_valid is False else "⚠️")
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])

        error_block = ""
        if errors:
            error_list = "\n".join(f"- {e}" for e in errors[:20])
            if len(errors) > 20:
                error_list += f"\n- _…and {len(errors) - 20} more_"
            error_block = f"\n### Errors\n\n{error_list}\n"

        warning_block = ""
        if warnings:
            warn_list = "\n".join(f"- {w}" for w in warnings[:20])
            if len(warnings) > 20:
                warn_list += f"\n- _…and {len(warnings) - 20} more_"
            warning_block = f"\n### Warnings\n\n{warn_list}\n"

        missing_imgs = len(validation.get("missing_images", []))
        corrupted = len(validation.get("corrupted_images", []))
        orphan_anns = len(validation.get("orphan_annotations", []))

        return (
            "## 8. Validation Results\n\n"
            f"**Overall status:** {status_icon} {'Valid' if is_valid else 'Invalid'}\n\n"
            "| Check | Result |\n"
            "|---|---|\n"
            f"| Images in COCO | {validation.get('total_images_in_coco', 'N/A')} |\n"
            f"| Valid images | {validation.get('valid_images', 'N/A')} |\n"
            f"| Missing images | {missing_imgs} |\n"
            f"| Corrupted images | {corrupted} |\n"
            f"| Orphan annotations | {orphan_anns} |\n"
            f"| Duplicate filenames | {len(validation.get('duplicate_filenames', []))} |\n"
            f"| Duplicate image IDs | {len(validation.get('duplicate_image_ids', []))} |\n"
            + error_block
            + warning_block
        )

    def _section_recommendations(
        self,
        stats: dict[str, Any],
        validation: dict[str, Any],
    ) -> str:
        recs: list[str] = []
        total = stats.get("total_images", 0)
        missing_labels = stats.get("missing_labels", 0)
        errors = validation.get("errors", [])

        if total < 100:
            recs.append(
                f"Dataset has only **{total}** images. Consider collecting more images "
                "to improve model generalisation (target ≥ 200 per class)."
            )
        if missing_labels:
            recs.append(
                f"**{missing_labels}** image(s) have no annotations. "
                "Annotate or remove them before training."
            )
        if errors:
            recs.append(
                "Resolve all **validation errors** before training. "
                "See `validation_report.json` for details."
            )
        if not recs:
            recs.append("Dataset is in good shape. Proceed to model training.")

        rec_list = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(recs))

        return (
            "## 9. Recommendations\n\n"
            + rec_list
        )

    def _section_footer(self) -> str:
        return (
            "---\n\n"
            "_Report auto-generated by the ScrewMetric Dataset Processing Pipeline._  \n"
            "_Edit `dataset/scripts/dataset_report_generator.py` to customise._"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_load_json(path: Path, default: Any) -> Any:
        """Load JSON without raising on missing or malformed files.

        Args:
            path: File to load.
            default: Value to return on failure.

        Returns:
            Loaded data or ``default``.
        """
        try:
            return load_json(path)
        except Exception as exc:
            logger.warning("Could not load '%s': %s — using defaults.", path, exc)
            return default


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the Markdown dataset report from existing artefacts.

    Reads ``dataset_stats.json`` and ``validation_report.json`` and writes
    ``DATASET_REPORT.md`` to the dataset root.
    """
    print("=" * 60)
    print("  ScrewMetric — Dataset Report Generator Module")
    print("=" * 60)

    try:
        from config import PipelineConfig
        config = PipelineConfig.default()

        print(f"\nStats source  : {config.paths.dataset_stats_path}")
        print(f"Val source    : {config.paths.validation_report_path}")
        print(f"Output report : {config.paths.dataset_report_path}")
        print("\nGenerating report…\n")

        generator = DatasetReportGenerator(config)
        report_path = generator.generate()

        size_kb = report_path.stat().st_size // 1024
        print(f"\n{'─' * 40}")
        print(f"  Report written : {report_path}")
        print(f"  File size      : {size_kb} KB")
        print(f"{'─' * 40}")

        print(f"\n✅ dataset_report_generator.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ dataset_report_generator.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
