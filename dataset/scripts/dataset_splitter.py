"""
ScrewMetric — Dataset Splitter
================================
Splits a COCO-format dataset into train / val / test subsets in a
reproducible, overlap-free manner and writes the resulting images and
COCO annotation files to disk.

Responsibility (Single Responsibility Principle):
    Only splitting and file-copying logic lives here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import PipelineConfig, SplitConfig, get_logger
from utils import (
    copy_file,
    ensure_dir,
    filter_coco_for_image_ids,
    load_coco_annotation,
    save_json,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Summary of a completed dataset split operation.

    Attributes:
        train_ids: Image IDs assigned to the training split.
        val_ids: Image IDs assigned to the validation split.
        test_ids: Image IDs assigned to the test split.
        train_annotation_path: Path to the written ``instances_train.json``.
        val_annotation_path: Path to the written ``instances_val.json``.
        test_annotation_path: Path to the written ``instances_test.json``.
    """

    train_ids: list[int] = field(default_factory=list)
    val_ids: list[int] = field(default_factory=list)
    test_ids: list[int] = field(default_factory=list)
    train_annotation_path: Path | None = None
    val_annotation_path: Path | None = None
    test_annotation_path: Path | None = None

    @property
    def split_counts(self) -> dict[str, int]:
        """Return a mapping of split name → image count.

        Returns:
            Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
        """
        return {
            "train": len(self.train_ids),
            "val": len(self.val_ids),
            "test": len(self.test_ids),
        }


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Splits a COCO dataset into train / val / test subsets.

    Args:
        config: Pipeline configuration.

    Example::

        splitter = DatasetSplitter(PipelineConfig.default())
        result = splitter.split()
        print(result.split_counts)
    """

    _SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

    def __init__(self, config: PipelineConfig) -> None:
        self._paths = config.paths
        self._split_cfg: SplitConfig = config.split

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self) -> SplitResult:
        """Execute the full split pipeline.

        Steps performed:

        1. Load the master COCO annotation file.
        2. Shuffle image IDs with a fixed seed for reproducibility.
        3. Partition IDs according to configured ratios (no overlap).
        4. Copy image files to their respective split directories.
        5. Write filtered COCO JSON files for each split.

        Returns:
            A :class:`SplitResult` with IDs and output paths.

        Raises:
            FileNotFoundError: If the master annotation file is missing.
            ValueError: If the annotation file has no images.
        """
        logger.info("Loading master annotation file…")
        coco = load_coco_annotation(self._paths.default_annotation_file)
        all_images: list[dict[str, Any]] = coco["images"]

        if not all_images:
            raise ValueError("No images found in annotation file — cannot split.")

        logger.info("Splitting %d images (seed=%d)…", len(all_images), self._split_cfg.random_seed)

        # Deterministic shuffle
        rng = random.Random(self._split_cfg.random_seed)
        shuffled = all_images[:]
        rng.shuffle(shuffled)

        # Partition
        n_total = len(shuffled)
        n_train = round(n_total * self._split_cfg.train_ratio)
        n_val = round(n_total * self._split_cfg.val_ratio)
        # test gets whatever remains to avoid rounding drift
        n_test = n_total - n_train - n_val

        train_images = shuffled[:n_train]
        val_images = shuffled[n_train: n_train + n_val]
        test_images = shuffled[n_train + n_val:]

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(train_images), len(val_images), len(test_images),
        )

        split_map = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        result = SplitResult(
            train_ids=[img["id"] for img in train_images],
            val_ids=[img["id"] for img in val_images],
            test_ids=[img["id"] for img in test_images],
        )

        # Copy files and write annotations for each split
        for split_name, split_images in split_map.items():
            ann_path = self._write_split(
                split_name=split_name,
                split_images=split_images,
                coco=coco,
            )
            setattr(result, f"{split_name}_annotation_path", ann_path)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_split(
        self,
        split_name: str,
        split_images: list[dict[str, Any]],
        coco: dict[str, Any],
    ) -> Path:
        """Copy images and write a filtered COCO annotation for one split.

        Args:
            split_name: ``"train"``, ``"val"``, or ``"test"``.
            split_images: List of COCO image dicts belonging to this split.
            coco: The full COCO annotation dict (used to filter annotations).

        Returns:
            Path to the written COCO annotation JSON file.
        """
        images_dst = self._paths.split_images_dir(split_name)
        ann_dst = self._paths.split_annotation_path(split_name)

        ensure_dir(images_dst)
        ensure_dir(ann_dst.parent)

        image_ids = [img["id"] for img in split_images]

        # Copy images
        logger.info("Copying %d images to '%s' split…", len(split_images), split_name)
        for img_meta in tqdm(split_images, desc=f"Copy [{split_name}]", unit="img"):
            src = self._paths.screw_only_dir / img_meta["file_name"]
            dst = images_dst / img_meta["file_name"]
            if not dst.exists():  # skip if already present
                try:
                    copy_file(src, dst)
                except FileNotFoundError:
                    logger.warning(
                        "Skipping missing image '%s' during copy.", img_meta["file_name"]
                    )

        # Write filtered COCO annotation
        filtered_coco = filter_coco_for_image_ids(coco, image_ids)
        save_json(filtered_coco, ann_dst)
        logger.info("Annotation written → %s", ann_dst)

        return ann_dst


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the dataset splitter against the real dataset and print a summary.

    Splits are written to ``dataset/splits/{train,val,test}/``.
    If splits already exist from a previous run, image copies are skipped
    (idempotent).
    """
    print("=" * 60)
    print("  ScrewMetric — Dataset Splitter Module")
    print("=" * 60)

    try:
        from config import PipelineConfig
        config = PipelineConfig.default()

        print(f"\nDataset root   : {config.paths.dataset_root}")
        print(f"Annotation     : {config.paths.default_annotation_file}")
        print(f"Splits output  : {config.paths.splits_dir}")
        print(f"Ratios         : train={config.split.train_ratio} "
              f"val={config.split.val_ratio} "
              f"test={config.split.test_ratio}")
        print(f"Random seed    : {config.split.random_seed}")
        print("\nRunning split…\n")

        splitter = DatasetSplitter(config)
        result = splitter.split()

        counts = result.split_counts
        print(f"\n{'─' * 40}")
        print(f"  Train images : {counts['train']}")
        print(f"  Val images   : {counts['val']}")
        print(f"  Test images  : {counts['test']}")
        print(f"  Total        : {sum(counts.values())}")
        print(f"\n  Annotations written:")
        print(f"    {result.train_annotation_path}")
        print(f"    {result.val_annotation_path}")
        print(f"    {result.test_annotation_path}")
        print(f"{'─' * 40}")

        print(f"\n✅ dataset_splitter.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ dataset_splitter.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
