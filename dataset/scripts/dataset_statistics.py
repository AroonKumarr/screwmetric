"""
ScrewMetric — Dataset Statistics
==================================
Computes comprehensive statistics about the annotated dataset and each split,
then serialises the results to ``dataset_stats.json``.

Responsibility (Single Responsibility Principle):
    Only statistics computation and serialisation live here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from PIL import Image

from config import PipelineConfig, get_logger
from utils import (
    build_annotations_by_image,
    get_directory_size,
    human_readable_size,
    load_coco_annotation,
    save_json,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResolutionStats:
    """Statistics about image resolutions.

    Attributes:
        avg_width: Mean image width in pixels.
        avg_height: Mean image height in pixels.
        min_width: Minimum image width in pixels.
        min_height: Minimum image height in pixels.
        max_width: Maximum image width in pixels.
        max_height: Maximum image height in pixels.
        std_width: Standard deviation of widths.
        std_height: Standard deviation of heights.
    """

    avg_width: float = 0.0
    avg_height: float = 0.0
    min_width: int = 0
    min_height: int = 0
    max_width: int = 0
    max_height: int = 0
    std_width: float = 0.0
    std_height: float = 0.0


@dataclass
class BBoxStats:
    """Bounding-box size statistics.

    Attributes:
        count: Total bounding-box count.
        avg_width: Mean bbox width in pixels.
        avg_height: Mean bbox height in pixels.
        avg_area: Mean bbox area in square pixels.
        min_area: Minimum bbox area.
        max_area: Maximum bbox area.
    """

    count: int = 0
    avg_width: float = 0.0
    avg_height: float = 0.0
    avg_area: float = 0.0
    min_area: float = 0.0
    max_area: float = 0.0


@dataclass
class PolygonStats:
    """Polygon / segmentation statistics.

    Attributes:
        count: Total number of polygon segmentations.
        avg_points_per_polygon: Mean number of vertices per polygon.
        min_points: Minimum vertices in a single polygon.
        max_points: Maximum vertices in a single polygon.
    """

    count: int = 0
    avg_points_per_polygon: float = 0.0
    min_points: int = 0
    max_points: int = 0


@dataclass
class SplitStats:
    """Statistics for one dataset split.

    Attributes:
        count: Number of images in this split.
        annotation_count: Total annotations in this split.
        ratio: Actual fraction of the total dataset.
    """

    count: int = 0
    annotation_count: int = 0
    ratio: float = 0.0


# ---------------------------------------------------------------------------
# Top-level statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetStatistics:
    """Aggregated statistics for the entire ScrewMetric dataset.

    Attributes:
        total_images: Total annotated images in the master COCO file.
        total_annotations: Total annotation instances.
        category_distribution: Mapping of category name → instance count.
        resolution: Image resolution statistics.
        bbox: Bounding-box statistics.
        polygon: Polygon/segmentation statistics.
        splits: Per-split statistics keyed by split name.
        dataset_size_bytes: Total size of the dataset root in bytes.
        dataset_size_human: Human-readable size string.
        missing_labels: Count of images with no annotations.
        images_with_multiple_annotations: Count of images with >1 annotation.
    """

    total_images: int = 0
    total_annotations: int = 0
    category_distribution: dict[str, int] = field(default_factory=dict)
    resolution: ResolutionStats = field(default_factory=ResolutionStats)
    bbox: BBoxStats = field(default_factory=BBoxStats)
    polygon: PolygonStats = field(default_factory=PolygonStats)
    splits: dict[str, SplitStats] = field(default_factory=dict)
    dataset_size_bytes: int = 0
    dataset_size_human: str = ""
    missing_labels: int = 0
    images_with_multiple_annotations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-serialisable dictionary.

        Returns:
            Nested dictionary representation.
        """
        d = asdict(self)
        # Convert nested dataclasses that asdict may not handle as expected
        d["splits"] = {
            name: asdict(s) for name, s in self.splits.items()
        }
        return d


# ---------------------------------------------------------------------------
# Statistics computer
# ---------------------------------------------------------------------------

class DatasetStatisticsComputer:
    """Computes and persists comprehensive dataset statistics.

    Args:
        config: Pipeline configuration.

    Example::

        computer = DatasetStatisticsComputer(PipelineConfig.default())
        stats = computer.compute()
        print(stats.total_images)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._paths = config.paths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> DatasetStatistics:
        """Compute statistics and write ``dataset_stats.json``.

        Returns:
            Populated :class:`DatasetStatistics` instance.

        Raises:
            FileNotFoundError: If the master annotation file is missing.
        """
        logger.info("Computing dataset statistics…")
        stats = DatasetStatistics()

        coco = load_coco_annotation(self._paths.default_annotation_file)
        images: list[dict[str, Any]] = coco["images"]
        annotations: list[dict[str, Any]] = coco["annotations"]
        categories: list[dict[str, Any]] = coco["categories"]

        cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
        ann_by_image = build_annotations_by_image(coco)

        stats.total_images = len(images)
        stats.total_annotations = len(annotations)

        # ── Category distribution ─────────────────────────────────────
        stats.category_distribution = self._compute_category_distribution(
            annotations, cat_id_to_name
        )

        # ── Resolution stats ──────────────────────────────────────────
        stats.resolution = self._compute_resolution_stats(images)

        # ── BBox stats ────────────────────────────────────────────────
        stats.bbox = self._compute_bbox_stats(annotations)

        # ── Polygon stats ─────────────────────────────────────────────
        stats.polygon = self._compute_polygon_stats(annotations)

        # ── Missing labels ────────────────────────────────────────────
        stats.missing_labels = sum(
            1 for img in images if img["id"] not in ann_by_image
        )

        # ── Images with multiple annotations ─────────────────────────
        stats.images_with_multiple_annotations = sum(
            1 for anns in ann_by_image.values() if len(anns) > 1
        )

        # ── Per-split statistics ──────────────────────────────────────
        stats.splits = self._compute_split_stats(stats.total_images)

        # ── Dataset size ──────────────────────────────────────────────
        stats.dataset_size_bytes = get_directory_size(self._paths.dataset_root)
        stats.dataset_size_human = human_readable_size(stats.dataset_size_bytes)

        # ── Persist ───────────────────────────────────────────────────
        save_json(stats.to_dict(), self._paths.dataset_stats_path)
        logger.info("Dataset statistics saved → %s", self._paths.dataset_stats_path)

        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_category_distribution(
        self,
        annotations: list[dict[str, Any]],
        cat_id_to_name: dict[int, str],
    ) -> dict[str, int]:
        """Count annotation instances per category.

        Args:
            annotations: All COCO annotation dicts.
            cat_id_to_name: Mapping from category ID to name.

        Returns:
            ``{"category_name": count, ...}`` sorted by count descending.
        """
        dist: dict[str, int] = {}
        for ann in annotations:
            name = cat_id_to_name.get(ann["category_id"], f"unknown_{ann['category_id']}")
            dist[name] = dist.get(name, 0) + 1
        return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))

    def _compute_resolution_stats(
        self,
        images: list[dict[str, Any]],
    ) -> ResolutionStats:
        """Compute width/height statistics from COCO image metadata.

        Args:
            images: List of COCO image dicts.

        Returns:
            Populated :class:`ResolutionStats`.
        """
        if not images:
            return ResolutionStats()

        widths = [img.get("width", 0) for img in images]
        heights = [img.get("height", 0) for img in images]

        return ResolutionStats(
            avg_width=round(statistics.mean(widths), 2),
            avg_height=round(statistics.mean(heights), 2),
            min_width=min(widths),
            min_height=min(heights),
            max_width=max(widths),
            max_height=max(heights),
            std_width=round(statistics.pstdev(widths), 2) if len(widths) > 1 else 0.0,
            std_height=round(statistics.pstdev(heights), 2) if len(heights) > 1 else 0.0,
        )

    def _compute_bbox_stats(
        self,
        annotations: list[dict[str, Any]],
    ) -> BBoxStats:
        """Compute bounding-box dimension statistics.

        COCO bboxes are ``[x, y, width, height]``.

        Args:
            annotations: All COCO annotation dicts.

        Returns:
            Populated :class:`BBoxStats`.
        """
        bboxes = [ann["bbox"] for ann in annotations if "bbox" in ann]
        if not bboxes:
            return BBoxStats()

        widths = [bb[2] for bb in bboxes]
        heights = [bb[3] for bb in bboxes]
        areas = [w * h for w, h in zip(widths, heights)]

        return BBoxStats(
            count=len(bboxes),
            avg_width=round(statistics.mean(widths), 2),
            avg_height=round(statistics.mean(heights), 2),
            avg_area=round(statistics.mean(areas), 2),
            min_area=round(min(areas), 2),
            max_area=round(max(areas), 2),
        )

    def _compute_polygon_stats(
        self,
        annotations: list[dict[str, Any]],
    ) -> PolygonStats:
        """Compute statistics about polygon segmentation vertices.

        Args:
            annotations: All COCO annotation dicts.

        Returns:
            Populated :class:`PolygonStats`.
        """
        point_counts: list[int] = []
        for ann in annotations:
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list):
                    # COCO flat polygon: [x0,y0,x1,y1,...] → n_points = len/2
                    point_counts.append(len(seg) // 2)

        if not point_counts:
            return PolygonStats()

        return PolygonStats(
            count=len(point_counts),
            avg_points_per_polygon=round(statistics.mean(point_counts), 2),
            min_points=min(point_counts),
            max_points=max(point_counts),
        )

    def _compute_split_stats(self, total_images: int) -> dict[str, SplitStats]:
        """Compute per-split statistics from the written split annotation files.

        Falls back gracefully if a split has not yet been created.

        Args:
            total_images: Total image count (used to compute ratio).

        Returns:
            ``{"train": SplitStats, "val": SplitStats, "test": SplitStats}``
        """
        result: dict[str, SplitStats] = {}
        for split_name in ("train", "val", "test"):
            ann_path = self._paths.split_annotation_path(split_name)
            if ann_path.exists():
                try:
                    coco = load_coco_annotation(ann_path)
                    count = len(coco["images"])
                    ann_count = len(coco["annotations"])
                    ratio = round(count / total_images, 4) if total_images else 0.0
                    result[split_name] = SplitStats(
                        count=count,
                        annotation_count=ann_count,
                        ratio=ratio,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not load split annotation '%s': %s", ann_path, exc
                    )
                    result[split_name] = SplitStats()
            else:
                logger.debug("Split annotation not found: %s (skipping)", ann_path)
                result[split_name] = SplitStats()
        return result


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute and display dataset statistics for the real dataset.

    Reads from the master COCO annotation and any existing split
    annotations, then pretty-prints the results.
    """
    print("=" * 60)
    print("  ScrewMetric — Dataset Statistics Module")
    print("=" * 60)

    try:
        from config import PipelineConfig
        config = PipelineConfig.default()

        print(f"\nAnnotation : {config.paths.default_annotation_file}")
        print(f"Output     : {config.paths.dataset_stats_path}")
        print("\nComputing statistics…\n")

        computer = DatasetStatisticsComputer(config)
        stats = computer.compute()

        print(f"\n{'─' * 40}")
        print(f"  Total images            : {stats.total_images}")
        print(f"  Total annotations       : {stats.total_annotations}")
        print(f"  Missing labels          : {stats.missing_labels}")
        print(f"  Multi-annotated images  : {stats.images_with_multiple_annotations}")
        print(f"  Dataset size on disk    : {stats.dataset_size_human}")
        print(f"\n  Category distribution:")
        for cat, count in stats.category_distribution.items():
            print(f"    {cat}: {count}")
        print(f"\n  Resolution (avg):  {stats.resolution.avg_width} × {stats.resolution.avg_height} px")
        print(f"  BBox count: {stats.bbox.count}  avg area: {stats.bbox.avg_area:.0f} px²")
        print(f"  Polygon count: {stats.polygon.count}  avg pts: {stats.polygon.avg_points_per_polygon}")
        print(f"\n  Splits:")
        for name, s in stats.splits.items():
            print(f"    {name}: {s.count} images ({s.ratio * 100:.1f}%)")
        print(f"{'─' * 40}")

        print(f"\n✅ dataset_statistics.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ dataset_statistics.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
