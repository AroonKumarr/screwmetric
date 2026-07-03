"""
ScrewMetric — Calibration Validator
======================================
Validates calibration images before running the expensive calibration
algorithm.  Detects problems early and produces a structured report.

Responsibility (Single Responsibility Principle):
    Validation only.  No calibration, no I/O of camera parameters.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from calib_config import CalibrationConfig, get_logger
from calib_utils import ensure_dir, is_image_readable, list_image_files, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured result from the pre-calibration validation pass.

    Attributes:
        total_images_found: Total image files found in the images directory.
        readable_images: Images that OpenCV can successfully decode.
        corrupted_images: Filenames of files that OpenCV cannot decode.
        unsupported_extensions: Filenames with unrecognised file extensions.
        duplicate_filenames: Filenames that appear more than once.
        images_with_corners: Images where all checkerboard corners are found.
        images_without_corners: Filenames where corner detection failed.
        inconsistent_resolutions: Filenames whose resolution differs from
            the majority resolution.
        majority_resolution: ``(width, height)`` of the most common image
            resolution, or ``None`` if no images are readable.
        min_images_required: Minimum valid images required for calibration.
        errors: List of blocking error messages.
        warnings: List of non-blocking warning messages.
        validation_duration_s: Wall-clock time taken by the validator.
    """

    total_images_found: int = 0
    readable_images: int = 0
    corrupted_images: list[str] = field(default_factory=list)
    unsupported_extensions: list[str] = field(default_factory=list)
    duplicate_filenames: list[str] = field(default_factory=list)
    images_with_corners: int = 0
    images_without_corners: list[str] = field(default_factory=list)
    inconsistent_resolutions: list[str] = field(default_factory=list)
    majority_resolution: tuple[int, int] | None = None
    min_images_required: int = 10
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_duration_s: float = 0.0

    @property
    def is_valid(self) -> bool:
        """``True`` when there are no blocking errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-compatible dictionary.

        Returns:
            Dict with all fields, converting tuples/dataclass members as needed.
        """
        d = asdict(self)
        d["majority_resolution"] = list(self.majority_resolution) if self.majority_resolution else None
        d["is_valid"] = self.is_valid
        return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CalibrationValidator:
    """Validates calibration images and generates a :class:`ValidationReport`.

    Performs these checks in order:

    1. Images directory exists and is non-empty.
    2. All files have supported extensions.
    3. No duplicate filenames.
    4. All images are readable by OpenCV.
    5. Consistent image resolution across the set.
    6. Checkerboard corners are detectable in enough images.
    7. Minimum valid image count.

    Args:
        config: Full calibration configuration.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._paths = config.paths
        self._board = config.checkerboard
        self._val_cfg = config.validation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Run all validation checks and write the report to disk.

        Returns:
            A :class:`ValidationReport` summarising every check performed.
        """
        t_start = time.perf_counter()
        report = ValidationReport(
            min_images_required=self._val_cfg.min_valid_images,
        )

        logger.info("Starting calibration validation…")

        if not self._check_directory_structure(report):
            report.validation_duration_s = round(time.perf_counter() - t_start, 2)
            self._save_report(report)
            return report

        all_files = self._discover_files(report)

        self._check_duplicates(all_files, report)
        valid_images = self._check_image_integrity(all_files, report)
        self._check_resolution_consistency(valid_images, report)
        self._check_corner_detection(valid_images, report)
        self._check_minimum_count(report)

        report.validation_duration_s = round(time.perf_counter() - t_start, 2)
        self._save_report(report)

        status = "VALID" if report.is_valid else "INVALID"
        logger.info(
            "Validation complete [%s] — images=%d  corners=%d  errors=%d  warnings=%d  %.2fs",
            status, report.total_images_found, report.images_with_corners,
            len(report.errors), len(report.warnings), report.validation_duration_s,
        )
        return report

    # ------------------------------------------------------------------
    # Private check methods
    # ------------------------------------------------------------------

    def _check_directory_structure(self, report: ValidationReport) -> bool:
        """Verify that the images directory exists and is not empty.

        Args:
            report: Report object to append errors to.

        Returns:
            ``True`` if the directory is usable; ``False`` to abort early.
        """
        images_dir = self._paths.images_dir
        if not images_dir.exists():
            report.errors.append(
                f"Images directory does not exist: {images_dir}"
            )
            return False
        if not images_dir.is_dir():
            report.errors.append(
                f"Images path is not a directory: {images_dir}"
            )
            return False
        return True

    def _discover_files(self, report: ValidationReport) -> list[Path]:
        """List all files and separate supported images from unsupported.

        Args:
            report: Report object to append warnings to.

        Returns:
            List of paths with supported image extensions.
        """
        images_dir = self._paths.images_dir
        all_files = sorted(p for p in images_dir.iterdir() if p.is_file())
        supported_exts = self._val_cfg.supported_extensions

        valid: list[Path] = []
        for p in all_files:
            if p.suffix.lower() in supported_exts:
                valid.append(p)
            elif p.name != ".DS_Store":
                report.unsupported_extensions.append(p.name)
                report.warnings.append(
                    f"Unsupported file extension: '{p.name}'"
                )

        report.total_images_found = len(valid)
        if not valid:
            report.errors.append(
                f"No images with supported extensions found in {images_dir}. "
                f"Supported: {sorted(supported_exts)}"
            )
        return valid

    def _check_duplicates(
        self, files: list[Path], report: ValidationReport
    ) -> None:
        """Detect duplicate filenames.

        Args:
            files: List of image paths to inspect.
            report: Report to append findings to.
        """
        seen: set[str] = set()
        for p in files:
            if p.name in seen:
                report.duplicate_filenames.append(p.name)
                report.errors.append(f"Duplicate filename: '{p.name}'")
            seen.add(p.name)

    def _check_image_integrity(
        self, files: list[Path], report: ValidationReport
    ) -> list[Path]:
        """Verify each image is decodable by OpenCV.

        Args:
            files: Image paths to check.
            report: Report to append findings to.

        Returns:
            List of paths that decoded successfully.
        """
        valid: list[Path] = []
        for p in tqdm(files, desc="Checking integrity", unit="img", leave=False):
            if self._val_cfg.verify_image_integrity:
                if is_image_readable(p):
                    valid.append(p)
                    report.readable_images += 1
                else:
                    report.corrupted_images.append(p.name)
                    report.errors.append(f"Corrupted/unreadable image: '{p.name}'")
            else:
                valid.append(p)
                report.readable_images += 1
        return valid

    def _check_resolution_consistency(
        self, files: list[Path], report: ValidationReport
    ) -> None:
        """Verify all images share the same resolution.

        Args:
            files: Readable image paths.
            report: Report to append findings to.
        """
        if not files:
            return

        resolution_counts: dict[tuple[int, int], int] = {}
        resolution_by_file: dict[str, tuple[int, int]] = {}

        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            res = (w, h)
            resolution_counts[res] = resolution_counts.get(res, 0) + 1
            resolution_by_file[p.name] = res

        if not resolution_counts:
            return

        majority_res = max(resolution_counts, key=lambda k: resolution_counts[k])
        report.majority_resolution = majority_res

        for name, res in resolution_by_file.items():
            if res != majority_res:
                report.inconsistent_resolutions.append(name)
                report.warnings.append(
                    f"Image '{name}' has resolution {res[0]}x{res[1]}, "
                    f"majority is {majority_res[0]}x{majority_res[1]}"
                )

    def _check_corner_detection(
        self, files: list[Path], report: ValidationReport
    ) -> None:
        """Attempt checkerboard corner detection on every readable image.

        Args:
            files: Readable image paths.
            report: Report to append findings to.
        """
        pattern = self._board.pattern_size
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        for p in tqdm(files, desc="Detecting corners", unit="img", leave=False):
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCorners(gray, pattern, flags)
            if found:
                report.images_with_corners += 1
            else:
                report.images_without_corners.append(p.name)
                logger.debug("Corners not found in: %s", p.name)

    def _check_minimum_count(self, report: ValidationReport) -> None:
        """Ensure enough images have detectable corners.

        Args:
            report: Report to check and append errors to.
        """
        required = self._val_cfg.min_valid_images
        found = report.images_with_corners
        if found < required:
            report.errors.append(
                f"Insufficient images with detectable corners: "
                f"{found} found, {required} required. "
                "Capture more calibration images covering varied angles."
            )
        else:
            logger.info(
                "Corner detection passed: %d/%d images meet minimum requirement.",
                found, required,
            )

    def _save_report(self, report: ValidationReport) -> None:
        """Serialise and write the validation report to disk.

        Args:
            report: Completed validation report.
        """
        ensure_dir(self._paths.output_dir)
        save_json(report.to_dict(), self._paths.validation_report_path)
        logger.info(
            "Validation report saved → %s", self._paths.validation_report_path
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run validation on the configured images directory and print summary."""
    print("=" * 62)
    print("  ScrewMetric — Calibration Validator Module")
    print("=" * 62)

    try:
        config = CalibrationConfig.default()

        print(f"\nImages dir  : {config.paths.images_dir}")
        print(f"Output dir  : {config.paths.output_dir}")
        print(f"Board       : {config.checkerboard.pattern_size} inner corners")
        print(f"Min valid   : {config.validation.min_valid_images} images")
        print("\nRunning validation...\n")

        validator = CalibrationValidator(config)
        report = validator.validate()

        print(f"\n{'─' * 45}")
        print(f"  Total images found        : {report.total_images_found}")
        print(f"  Readable images           : {report.readable_images}")
        print(f"  Corrupted images          : {len(report.corrupted_images)}")
        print(f"  Images with corners       : {report.images_with_corners}")
        print(f"  Images without corners    : {len(report.images_without_corners)}")
        print(f"  Inconsistent resolutions  : {len(report.inconsistent_resolutions)}")
        print(f"  Unsupported extensions    : {len(report.unsupported_extensions)}")
        print(f"  Errors                    : {len(report.errors)}")
        print(f"  Warnings                  : {len(report.warnings)}")
        print(f"  Majority resolution       : {report.majority_resolution}")
        print(f"  Validation duration       : {report.validation_duration_s:.2f}s")
        print(f"{'─' * 45}")

        status = "VALID ✅" if report.is_valid else "INVALID ❌"
        print(f"\n  Overall status: {status}")

        if report.errors:
            print("\n  Errors:")
            for e in report.errors:
                print(f"    • {e}")

        if report.warnings:
            print("\n  Warnings:")
            for w in report.warnings[:5]:
                print(f"    • {w}")
            if len(report.warnings) > 5:
                print(f"    … and {len(report.warnings) - 5} more")

        print(f"\n✅ calibration_validator.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ calibration_validator.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
