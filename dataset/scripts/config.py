"""
ScrewMetric — Dataset Processing Configuration
===============================================
Centralised configuration for the entire dataset processing pipeline.
All paths, constants, and tuneable parameters live here.
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
    """Return a consistently-formatted logger.

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

def _dataset_root() -> Path:
    """Resolve the dataset root relative to this config file's location.

    Returns:
        Absolute path to ``dataset/``.
    """
    # config.py lives in dataset/scripts/ → parent.parent == dataset/
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used across the entire pipeline.

    All paths are absolute so that scripts work regardless of the
    current working directory.
    """

    dataset_root: Path = field(default_factory=_dataset_root)

    @property
    def raw_images_dir(self) -> Path:
        """Root of raw images (contains ``screw_only/`` and ``screw_checkerboard/``)."""
        return self.dataset_root / "raw_images"

    @property
    def screw_only_dir(self) -> Path:
        """Annotated screw-only images."""
        return self.raw_images_dir / "screw_only"

    @property
    def screw_checkerboard_dir(self) -> Path:
        """Screw + checkerboard images (used for calibration)."""
        return self.raw_images_dir / "screw_checkerboard"

    @property
    def annotations_dir(self) -> Path:
        """COCO annotation files."""
        return self.dataset_root / "annotations"

    @property
    def default_annotation_file(self) -> Path:
        """Master COCO annotation file exported from CVAT."""
        return self.annotations_dir / "instances_default.json"

    @property
    def splits_dir(self) -> Path:
        """Root of train / val / test splits."""
        return self.dataset_root / "splits"

    @property
    def train_dir(self) -> Path:
        return self.splits_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.splits_dir / "val"

    @property
    def test_dir(self) -> Path:
        return self.splits_dir / "test"

    @property
    def undistorted_dir(self) -> Path:
        return self.dataset_root / "undistorted_images"

    @property
    def validation_report_path(self) -> Path:
        return self.dataset_root / "validation_report.json"

    @property
    def dataset_stats_path(self) -> Path:
        return self.dataset_root / "dataset_stats.json"

    @property
    def dataset_report_path(self) -> Path:
        return self.dataset_root / "DATASET_REPORT.md"

    @property
    def previews_dir(self) -> Path:
        return self.dataset_root / "previews"

    def split_annotation_path(self, split: str) -> Path:
        """Return the COCO annotation JSON path for a given split name.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            Absolute path to the annotation file.
        """
        return self.splits_dir / split / "annotations" / f"instances_{split}.json"

    def split_images_dir(self, split: str) -> Path:
        """Return the images directory path for a given split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            Absolute path to the images directory.
        """
        return self.splits_dir / split / "images"


# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    """Hyper-parameters controlling dataset splitting.

    Attributes:
        train_ratio: Fraction of data assigned to the training set.
        val_ratio: Fraction of data assigned to the validation set.
        test_ratio: Fraction of data assigned to the test set.
        random_seed: Seed for reproducible shuffling.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    random_seed: int = 42

    def __post_init__(self) -> None:
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 10)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f}. "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )


# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationConfig:
    """Parameters for dataset validation.

    Attributes:
        supported_extensions: Set of image file extensions (lower-case, with dot)
            accepted as valid.
        verify_image_integrity: If ``True``, attempt to open every image with
            Pillow to detect corruption.
    """

    supported_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})
    )
    verify_image_integrity: bool = True


# ---------------------------------------------------------------------------
# Preview configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreviewConfig:
    """Parameters for contact-sheet generation.

    Attributes:
        thumbnail_size: ``(width, height)`` of each thumbnail cell in pixels.
        max_cols: Maximum number of columns in the contact sheet grid.
        max_images: Maximum number of images to include per contact sheet
            (prevents enormous files for very large splits).
        background_colour: RGB fill colour for empty cells.
        border_px: Border width in pixels drawn around each thumbnail.
        border_colour: RGB colour of the thumbnail border.
        font_size: Approximate font size for overlay labels (may be unsupported
            without a font file; falls back gracefully).
        jpeg_quality: Output JPEG quality (1–95).
    """

    thumbnail_size: tuple[int, int] = (256, 256)
    max_cols: int = 6
    max_images: int = 54          # 6 cols × 9 rows — keeps sheets readable
    background_colour: tuple[int, int, int] = (30, 30, 30)
    border_px: int = 4
    border_colour: tuple[int, int, int] = (70, 70, 70)
    font_size: int = 14
    jpeg_quality: int = 85


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Aggregates all sub-configurations into a single object.

    Pass this around instead of individual configs to keep function
    signatures clean.

    Attributes:
        paths: Filesystem path configuration.
        split: Dataset split parameters.
        validation: Validation parameters.
        preview: Contact-sheet generation parameters.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Return a pipeline config with all default values.

        Returns:
            A fully-initialised :class:`PipelineConfig`.
        """
        return cls(
            paths=PathConfig(),
            split=SplitConfig(),
            validation=ValidationConfig(),
            preview=PreviewConfig(),
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate configuration loading and display key settings.

    Prints resolved paths and configuration values to stdout so the
    module can be verified without any external dependencies.
    """
    print("=" * 60)
    print("  ScrewMetric — Configuration Module")
    print("=" * 60)

    try:
        cfg = PipelineConfig.default()

        print("\n[PathConfig]")
        print(f"  dataset_root         : {cfg.paths.dataset_root}")
        print(f"  screw_only_dir       : {cfg.paths.screw_only_dir}")
        print(f"  default_annotation   : {cfg.paths.default_annotation_file}")
        print(f"  splits_dir           : {cfg.paths.splits_dir}")
        print(f"  validation_report    : {cfg.paths.validation_report_path}")
        print(f"  dataset_stats        : {cfg.paths.dataset_stats_path}")

        print("\n[SplitConfig]")
        print(f"  train_ratio : {cfg.split.train_ratio}")
        print(f"  val_ratio   : {cfg.split.val_ratio}")
        print(f"  test_ratio  : {cfg.split.test_ratio}")
        print(f"  random_seed : {cfg.split.random_seed}")

        print("\n[ValidationConfig]")
        print(f"  supported_extensions    : {sorted(cfg.validation.supported_extensions)}")
        print(f"  verify_image_integrity  : {cfg.validation.verify_image_integrity}")

        print("\n[PreviewConfig]")
        print(f"  thumbnail_size : {cfg.preview.thumbnail_size}")
        print(f"  max_cols       : {cfg.preview.max_cols}")
        print(f"  max_images     : {cfg.preview.max_images}")

        # Verify split ratios sum to 1.0
        total = cfg.split.train_ratio + cfg.split.val_ratio + cfg.split.test_ratio
        assert abs(total - 1.0) < 1e-9, f"Split ratios must sum to 1.0, got {total}"

        print("\n✅ config.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ config.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

