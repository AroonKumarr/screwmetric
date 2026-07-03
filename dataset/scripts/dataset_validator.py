"""
ScrewMetric — Dataset Validator
================================
Validates a COCO-format dataset and produces a structured
``validation_report.json`` documenting every issue found.

Responsibility (Single Responsibility Principle):
    Only validation logic lives here.  No splitting, no statistics.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from config import PipelineConfig, ValidationConfig, get_logger
from utils import (
    load_coco_annotation,
    build_image_id_map,
    build_annotations_by_image,
    save_json,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured representation of all validation findings.

    Attributes:
        total_images_in_filesystem: Number of image files found on disk.
        total_images_in_coco: Number of image entries in the COCO file.
        valid_images: Images that exist on disk and are not corrupted.
        missing_images: Filenames present in COCO but absent on disk.
        extra_images: Filenames on disk but not referenced by COCO.
        duplicate_filenames: Filenames that appear more than once in COCO.
        duplicate_image_ids: Image IDs that appear more than once in COCO.
        corrupted_images: Filenames that could not be opened by Pillow.
        unannotated_images: Image IDs that have zero annotations.
        orphan_annotations: Annotation IDs whose ``image_id`` is unknown.
        invalid_category_ids: Annotation IDs referencing a non-existent category.
        unsupported_extensions: Files on disk with unsupported extensions.
        warnings: Non-fatal advisories.
        errors: Fatal or data-integrity issues.
        is_valid: ``True`` only if ``errors`` is empty.
    """

    total_images_in_filesystem: int = 0
    total_images_in_coco: int = 0
    valid_images: int = 0
    missing_images: list[str] = field(default_factory=list)
    extra_images: list[str] = field(default_factory=list)
    duplicate_filenames: list[str] = field(default_factory=list)
    duplicate_image_ids: list[int] = field(default_factory=list)
    corrupted_images: list[str] = field(default_factory=list)
    unannotated_images: list[int] = field(default_factory=list)
    orphan_annotations: list[int] = field(default_factory=list)
    invalid_category_ids: list[int] = field(default_factory=list)
    unsupported_extensions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when no fatal errors were recorded."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary.

        Returns:
            JSON-serialisable dictionary.
        """
        d = asdict(self)
        d["is_valid"] = self.is_valid
        return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """Validates a COCO-format dataset and emits a structured report.

    The validator is intentionally stateless between runs — create a new
    instance (or call :meth:`validate`) for each validation pass.

    Args:
        config: Pipeline configuration.

    Example::

        validator = DatasetValidator(PipelineConfig.default())
        report = validator.validate()
        print(report.is_valid)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._paths = config.paths
        self._val_cfg: ValidationConfig = config.validation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Run all validation checks and return a :class:`ValidationReport`.

        Steps performed (in order):

        1. Verify dataset root, annotation file, and image directory exist.
        2. Load and parse the COCO JSON.
        3. Check for duplicate image IDs and duplicate filenames in COCO.
        4. Detect missing images (in COCO but not on disk).
        5. Detect extra images (on disk but not in COCO).
        6. Check for unsupported file extensions.
        7. Optionally verify image integrity (Pillow open).
        8. Check for orphan annotations (unknown ``image_id``).
        9. Check for invalid category IDs.
        10. Check for unannotated images.

        Returns:
            A fully-populated :class:`ValidationReport`.
        """
        report = ValidationReport()

        # ── Step 1: structural checks ────────────────────────────────
        if not self._check_structure(report):
            self._save_report(report)
            return report

        # ── Step 2: load COCO ────────────────────────────────────────
        try:
            coco = load_coco_annotation(self._paths.default_annotation_file)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            report.errors.append(f"Cannot load annotation file: {exc}")
            self._save_report(report)
            return report

        images: list[dict[str, Any]] = coco["images"]
        annotations: list[dict[str, Any]] = coco["annotations"]
        categories: list[dict[str, Any]] = coco["categories"]

        report.total_images_in_coco = len(images)
        valid_cat_ids = {cat["id"] for cat in categories}

        # ── Step 3: duplicates ───────────────────────────────────────
        self._check_duplicate_ids(images, report)
        self._check_duplicate_filenames(images, report)

        # ── Step 4–6: filesystem checks ──────────────────────────────
        img_dir = self._paths.screw_only_dir
        disk_files = list(img_dir.iterdir()) if img_dir.is_dir() else []
        disk_names = {f.name for f in disk_files if f.is_file()}
        coco_names = {img["file_name"] for img in images}

        report.total_images_in_filesystem = len(
            [f for f in disk_files if f.is_file()]
        )

        # Missing (in COCO but not on disk)
        for name in sorted(coco_names - disk_names):
            report.missing_images.append(name)
            report.errors.append(f"Image referenced in COCO but missing on disk: {name}")

        # Extra (on disk but not in COCO) — warning only
        for name in sorted(disk_names - coco_names):
            report.extra_images.append(name)
            report.warnings.append(f"Image on disk but not referenced in COCO: {name}")

        # Unsupported extensions
        for f in disk_files:
            if f.is_file() and f.suffix.lower() not in self._val_cfg.supported_extensions:
                report.unsupported_extensions.append(f.name)
                report.warnings.append(f"Unsupported file extension: {f.name}")

        # ── Step 7: image integrity ──────────────────────────────────
        if self._val_cfg.verify_image_integrity:
            self._check_image_integrity(images, img_dir, report)

        # ── Step 8: orphan annotations ───────────────────────────────
        image_id_set = {img["id"] for img in images}
        ann_by_image = build_annotations_by_image(coco)

        for ann in annotations:
            if ann["image_id"] not in image_id_set:
                report.orphan_annotations.append(ann["id"])
                report.errors.append(
                    f"Annotation id={ann['id']} references unknown image_id={ann['image_id']}"
                )

        # ── Step 9: invalid category IDs ────────────────────────────
        for ann in annotations:
            if ann["category_id"] not in valid_cat_ids:
                report.invalid_category_ids.append(ann["id"])
                report.errors.append(
                    f"Annotation id={ann['id']} has invalid category_id={ann['category_id']}"
                )

        # ── Step 10: unannotated images ──────────────────────────────
        for img in images:
            if img["id"] not in ann_by_image:
                report.unannotated_images.append(img["id"])
                report.warnings.append(
                    f"Image '{img['file_name']}' (id={img['id']}) has no annotations"
                )

        # ── Compute valid count ──────────────────────────────────────
        corrupted_set = set(report.corrupted_images)
        missing_set = set(report.missing_images)
        report.valid_images = sum(
            1 for img in images
            if img["file_name"] not in corrupted_set
            and img["file_name"] not in missing_set
        )

        self._save_report(report)
        self._log_summary(report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_structure(self, report: ValidationReport) -> bool:
        """Verify that required directories and files exist.

        Args:
            report: Report object to append errors to.

        Returns:
            ``True`` if the structure is intact enough to continue.
        """
        ok = True
        checks = [
            (self._paths.dataset_root, "Dataset root directory"),
            (self._paths.annotations_dir, "Annotations directory"),
            (self._paths.default_annotation_file, "COCO annotation file"),
            (self._paths.screw_only_dir, "screw_only image directory"),
        ]
        for path, description in checks:
            if not path.exists():
                report.errors.append(f"{description} not found: {path}")
                ok = False
        return ok

    def _check_duplicate_ids(
        self,
        images: list[dict[str, Any]],
        report: ValidationReport,
    ) -> None:
        """Detect duplicate image IDs in the COCO file.

        Args:
            images: List of COCO image dicts.
            report: Report to append findings to.
        """
        seen: dict[int, int] = {}
        for img in images:
            img_id = img["id"]
            seen[img_id] = seen.get(img_id, 0) + 1
        for img_id, count in seen.items():
            if count > 1:
                report.duplicate_image_ids.append(img_id)
                report.errors.append(
                    f"Duplicate image_id={img_id} appears {count} times in COCO"
                )

    def _check_duplicate_filenames(
        self,
        images: list[dict[str, Any]],
        report: ValidationReport,
    ) -> None:
        """Detect duplicate filenames in the COCO file.

        Args:
            images: List of COCO image dicts.
            report: Report to append findings to.
        """
        seen: dict[str, int] = {}
        for img in images:
            name = img["file_name"]
            seen[name] = seen.get(name, 0) + 1
        for name, count in seen.items():
            if count > 1:
                report.duplicate_filenames.append(name)
                report.warnings.append(
                    f"Duplicate filename '{name}' appears {count} times in COCO"
                )

    def _check_image_integrity(
        self,
        images: list[dict[str, Any]],
        img_dir: Path,
        report: ValidationReport,
    ) -> None:
        """Attempt to open each referenced image with Pillow.

        Args:
            images: COCO image dicts to check.
            img_dir: Directory where images reside.
            report: Report to append findings to.
        """
        logger.info("Verifying image integrity for %d images…", len(images))
        for img_meta in tqdm(images, desc="Integrity check", unit="img"):
            img_path = img_dir / img_meta["file_name"]
            if not img_path.exists():
                continue  # already captured as missing
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, Exception) as exc:
                report.corrupted_images.append(img_meta["file_name"])
                report.errors.append(
                    f"Corrupted image '{img_meta['file_name']}': {exc}"
                )

    def _save_report(self, report: ValidationReport) -> None:
        """Persist the report to ``validation_report.json``.

        Args:
            report: The completed validation report.
        """
        try:
            save_json(report.to_dict(), self._paths.validation_report_path)
            logger.info(
                "Validation report saved → %s", self._paths.validation_report_path
            )
        except OSError as exc:
            logger.error("Could not save validation report: %s", exc)

    @staticmethod
    def _log_summary(report: ValidationReport) -> None:
        """Emit a concise summary of the report to the log.

        Args:
            report: The completed validation report.
        """
        status = "✅ PASSED" if report.is_valid else "❌ FAILED"
        logger.info(
            "%s | images=%d valid=%d missing=%d corrupted=%d "
            "orphan_anns=%d errors=%d warnings=%d",
            status,
            report.total_images_in_coco,
            report.valid_images,
            len(report.missing_images),
            len(report.corrupted_images),
            len(report.orphan_annotations),
            len(report.errors),
            len(report.warnings),
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the dataset validator against the real dataset and print a summary.

    Uses the default :class:`~config.PipelineConfig` so no arguments are
    needed.  The validation report is written to ``dataset/validation_report.json``.
    """
    print("=" * 60)
    print("  ScrewMetric — Dataset Validator Module")
    print("=" * 60)

    try:
        from config import PipelineConfig
        config = PipelineConfig.default()

        print(f"\nDataset root : {config.paths.dataset_root}")
        print(f"Annotation   : {config.paths.default_annotation_file}")
        print(f"Image dir    : {config.paths.screw_only_dir}")
        print("\nRunning validation…\n")

        validator = DatasetValidator(config)
        report = validator.validate()

        print(f"\n{'─' * 40}")
        print(f"  Total images in COCO   : {report.total_images_in_coco}")
        print(f"  Total images on disk   : {report.total_images_in_filesystem}")
        print(f"  Valid images           : {report.valid_images}")
        print(f"  Missing images         : {len(report.missing_images)}")
        print(f"  Corrupted images       : {len(report.corrupted_images)}")
        print(f"  Extra images (on disk) : {len(report.extra_images)}")
        print(f"  Orphan annotations     : {len(report.orphan_annotations)}")
        print(f"  Errors                 : {len(report.errors)}")
        print(f"  Warnings               : {len(report.warnings)}")
        print(f"  Report path            : {config.paths.validation_report_path}")
        print(f"{'─' * 40}")

        status = "✅ PASSED" if report.is_valid else "❌ FAILED"
        print(f"\n  Overall status: {status}")
        print(f"\n✅ dataset_validator.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ dataset_validator.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

