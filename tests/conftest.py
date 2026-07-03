"""
ScrewMetric — conftest.py (pytest shared fixtures)
====================================================
Provides reusable fixtures for all dataset processing tests.

All fixtures build self-contained synthetic datasets in a
temporary directory so tests run without touching real project data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import pytest
from PIL import Image

# ── Ensure dataset/scripts is importable from the tests directory ────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    PipelineConfig,
    PreviewConfig,
    SplitConfig,
    ValidationConfig,
)


# ---------------------------------------------------------------------------
# Lightweight path provider (duck-typed — not a frozen dataclass)
# ---------------------------------------------------------------------------

class _TmpPaths:
    """Provides the same interface as :class:`~config.PathConfig` but
    backed by a temporary directory so tests never touch real files.

    Args:
        root: Temporary directory root (e.g. pytest's ``tmp_path``).
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def dataset_root(self) -> Path:
        return self._root

    @property
    def raw_images_dir(self) -> Path:
        return self._root / "raw_images"

    @property
    def screw_only_dir(self) -> Path:
        return self._root / "raw_images" / "screw_only"

    @property
    def screw_checkerboard_dir(self) -> Path:
        return self._root / "raw_images" / "screw_checkerboard"

    @property
    def annotations_dir(self) -> Path:
        return self._root / "annotations"

    @property
    def default_annotation_file(self) -> Path:
        return self._root / "annotations" / "instances_default.json"

    @property
    def splits_dir(self) -> Path:
        return self._root / "splits"

    @property
    def train_dir(self) -> Path:
        return self._root / "splits" / "train"

    @property
    def val_dir(self) -> Path:
        return self._root / "splits" / "val"

    @property
    def test_dir(self) -> Path:
        return self._root / "splits" / "test"

    @property
    def undistorted_dir(self) -> Path:
        return self._root / "undistorted_images"

    @property
    def validation_report_path(self) -> Path:
        return self._root / "validation_report.json"

    @property
    def dataset_stats_path(self) -> Path:
        return self._root / "dataset_stats.json"

    @property
    def dataset_report_path(self) -> Path:
        return self._root / "DATASET_REPORT.md"

    @property
    def previews_dir(self) -> Path:
        return self._root / "previews"

    def split_annotation_path(self, split: str) -> Path:
        return self._root / "splits" / split / "annotations" / f"instances_{split}.json"

    def split_images_dir(self, split: str) -> Path:
        return self._root / "splits" / split / "images"


def _make_tmp_config(tmp_path: Path) -> PipelineConfig:
    """Build a :class:`~config.PipelineConfig` whose paths point at ``tmp_path``.

    Because :class:`~config.PipelineConfig` is a frozen dataclass and its
    ``paths`` field type-hint is :class:`~config.PathConfig`, we use
    ``object.__setattr__`` to bypass the frozen guard and inject our
    duck-typed :class:`_TmpPaths` instance.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        A fully configured :class:`~config.PipelineConfig` for testing.
    """
    config = object.__new__(PipelineConfig)
    object.__setattr__(config, "paths", _TmpPaths(tmp_path))
    object.__setattr__(
        config, "split",
        SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42),
    )
    object.__setattr__(
        config, "validation",
        ValidationConfig(verify_image_integrity=True),
    )
    object.__setattr__(
        config, "preview",
        PreviewConfig(thumbnail_size=(64, 64), max_cols=3, max_images=9),
    )
    return config


# ---------------------------------------------------------------------------
# Tiny synthetic COCO dataset builders
# ---------------------------------------------------------------------------

def _make_coco_images(n: int) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic COCO image records."""
    return [
        {
            "id": i + 1,
            "file_name": f"img_{i + 1:03d}.jpg",
            "width": 640,
            "height": 480,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
        for i in range(n)
    ]


def _make_coco_annotations(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate one annotation per image."""
    return [
        {
            "id": img["id"],
            "image_id": img["id"],
            "category_id": 1,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 80.0, 10.0, 80.0]],
            "area": 2800.0,
            "bbox": [10.0, 10.0, 40.0, 70.0],
            "iscrowd": 0,
            "attributes": {"occluded": False},
        }
        for img in images
    ]


def _make_coco(n: int = 10) -> dict[str, Any]:
    """Build a complete minimal COCO dict with ``n`` images."""
    images = _make_coco_images(n)
    return {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": ""},
        "categories": [{"id": 1, "name": "screw", "supercategory": ""}],
        "images": images,
        "annotations": _make_coco_annotations(images),
    }


def _write_real_images(img_dir: Path, image_records: list[dict[str, Any]]) -> None:
    """Write tiny real JPEG files for each image record using Pillow."""
    img_dir.mkdir(parents=True, exist_ok=True)
    for rec in image_records:
        img_path = img_dir / rec["file_name"]
        im = Image.new("RGB", (rec["width"], rec["height"]), color=(128, 64, 32))
        im.save(str(img_path), "JPEG")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> dict[str, Any]:
    """Create a fully synthetic dataset in a temporary directory.

    Returns a dictionary with:
    - ``root``: the dataset root :class:`~pathlib.Path`
    - ``coco``: the COCO annotation dict
    - ``config``: a :class:`~config.PipelineConfig` pointed at this root
    """
    n_images = 10

    # Build directory structure
    ann_dir = tmp_path / "annotations"
    img_dir = tmp_path / "raw_images" / "screw_only"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    # Write COCO annotation
    coco = _make_coco(n=n_images)
    ann_file = ann_dir / "instances_default.json"
    ann_file.write_text(json.dumps(coco), encoding="utf-8")

    # Write JPEG images
    _write_real_images(img_dir, coco["images"])

    config = _make_tmp_config(tmp_path)
    return {"root": tmp_path, "coco": coco, "config": config}


@pytest.fixture()
def tmp_dataset_with_split(tmp_dataset: dict[str, Any]) -> dict[str, Any]:
    """Extend ``tmp_dataset`` by running the splitter so split dirs exist."""
    from dataset_splitter import DatasetSplitter

    splitter = DatasetSplitter(tmp_dataset["config"])
    result = splitter.split()
    tmp_dataset["split_result"] = result
    return tmp_dataset


@pytest.fixture()
def minimal_coco() -> dict[str, Any]:
    """Return a minimal in-memory COCO dict (no filesystem interaction)."""
    return _make_coco(n=5)


@pytest.fixture()
def coco_with_problems() -> dict[str, Any]:
    """Return a COCO dict containing deliberate errors for validation tests."""
    coco = _make_coco(n=4)

    # Duplicate image ID
    coco["images"].append({**coco["images"][0]})

    # Orphan annotation (unknown image_id)
    coco["annotations"].append({
        "id": 999,
        "image_id": 9999,
        "category_id": 1,
        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
        "area": 100.0,
        "bbox": [0, 0, 10, 10],
        "iscrowd": 0,
    })

    # Invalid category ID
    coco["annotations"].append({
        "id": 1000,
        "image_id": 1,
        "category_id": 999,
        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
        "area": 100.0,
        "bbox": [0, 0, 10, 10],
        "iscrowd": 0,
    })
    return coco


# ===========================================================================
# ── CALIBRATION FIXTURES ─────────────────────────────────────────────────────
# ===========================================================================

_CALIB_SCRIPTS = Path(__file__).resolve().parent.parent / "calibration" / "scripts"
if str(_CALIB_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CALIB_SCRIPTS))


def _make_checkerboard_image(
    inner_cols: int = 9,
    inner_rows: int = 6,
    square_px: int = 60,
) -> "np.ndarray":
    """Create a synthetic checkerboard that OpenCV can detect corners in.

    Args:
        inner_cols: Number of inner corners horizontally (board has
            ``inner_cols + 1`` squares wide).
        inner_rows: Number of inner corners vertically.
        square_px: Pixel size of each square.

    Returns:
        BGR numpy array of the checkerboard.
    """
    import numpy as np

    total_cols = inner_cols + 1
    total_rows = inner_rows + 1
    h = total_rows * square_px
    w = total_cols * square_px
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(total_rows):
        for c in range(total_cols):
            if (r + c) % 2 == 0:
                y0, y1 = r * square_px, (r + 1) * square_px
                x0, x1 = c * square_px, (c + 1) * square_px
                board[y0:y1, x0:x1] = 255
    return board


class _TmpCalibPaths:
    """Duck-typed path provider backed by a temporary directory."""

    def __init__(self, root: "Path") -> None:
        self._root = root

    @property
    def calibration_root(self) -> "Path":
        return self._root

    @property
    def images_dir(self) -> "Path":
        return self._root / "images"

    @property
    def output_dir(self) -> "Path":
        return self._root / "output"

    @property
    def undistortion_preview_dir(self) -> "Path":
        return self._root / "output" / "undistortion_preview"

    @property
    def camera_matrix_path(self) -> "Path":
        return self._root / "output" / "camera_matrix.npy"

    @property
    def dist_coeffs_path(self) -> "Path":
        return self._root / "output" / "dist_coeffs.npy"

    @property
    def rotation_vectors_path(self) -> "Path":
        return self._root / "output" / "rotation_vectors.npy"

    @property
    def translation_vectors_path(self) -> "Path":
        return self._root / "output" / "translation_vectors.npy"

    @property
    def reprojection_error_path(self) -> "Path":
        return self._root / "output" / "reprojection_error.json"

    @property
    def validation_report_path(self) -> "Path":
        return self._root / "output" / "validation_report.json"

    @property
    def calibration_report_path(self) -> "Path":
        return self._root / "output" / "calibration_report.json"

    @property
    def camera_parameters_path(self) -> "Path":
        return self._root / "output" / "camera_parameters.yaml"

    @property
    def visualization_path(self) -> "Path":
        return self._root / "output" / "calibration_visualization.png"


def _make_calib_config(tmp_path: "Path") -> "CalibrationConfig":
    """Build a CalibrationConfig backed by tmp_path.

    Uses object.__setattr__ to bypass the frozen dataclass guard,
    injecting a duck-typed path provider (same pattern as dataset tests).

    Args:
        tmp_path: Temporary directory from pytest.

    Returns:
        A fully configured CalibrationConfig for testing.
    """
    from calib_config import (  # type: ignore[import]
        CalibrationConfig,
        CalibrationProcessConfig,
        CheckerboardConfig,
        ValidationConfig,
    )

    config = object.__new__(CalibrationConfig)
    object.__setattr__(config, "paths", _TmpCalibPaths(tmp_path))
    object.__setattr__(
        config, "checkerboard",
        CheckerboardConfig(inner_corners_x=9, inner_corners_y=6, square_size_mm=25.0),
    )
    object.__setattr__(
        config, "validation",
        ValidationConfig(min_valid_images=4, verify_image_integrity=True),
    )
    object.__setattr__(
        config, "process",
        CalibrationProcessConfig(max_preview_images=3),
    )
    return config


def _write_checkerboard_images(
    images_dir: "Path",
    n: int = 15,
    inner_cols: int = 9,
    inner_rows: int = 6,
) -> list["Path"]:
    """Write ``n`` synthetic checkerboard JPEG files.

    Args:
        images_dir: Target directory.
        n: Number of images to write.
        inner_cols: Inner corner columns.
        inner_rows: Inner corner rows.

    Returns:
        List of written image paths.
    """
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    board = _make_checkerboard_image(inner_cols, inner_rows, square_px=60)
    for i in range(n):
        p = images_dir / f"checkerboard_{i + 1:03d}.jpg"
        cv2.imwrite(str(p), board)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Calibration pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_calib_dir(tmp_path: "Path") -> dict:
    """Create a synthetic calibration directory with checkerboard images.

    Returns a dict with:
      - ``root``: The tmp_path root
      - ``config``: CalibrationConfig pointing at tmp_path
      - ``image_paths``: List of written image paths
    """
    config = _make_calib_config(tmp_path)
    image_paths = _write_checkerboard_images(
        tmp_path / "images",
        n=15,
        inner_cols=config.checkerboard.inner_corners_x,
        inner_rows=config.checkerboard.inner_corners_y,
    )
    return {"root": tmp_path, "config": config, "image_paths": image_paths}


@pytest.fixture()
def tmp_calib_with_results(tmp_calib_dir: dict) -> dict:
    """Extend tmp_calib_dir by running full calibration.

    Returns a dict with all tmp_calib_dir keys plus:
      - ``calibration_result``: CalibrationResult from CameraCalibrator
    """
    from calibrate_camera import CameraCalibrator  # type: ignore[import]

    config = tmp_calib_dir["config"]
    calibrator = CameraCalibrator(config)
    result = calibrator.calibrate_from_images(tmp_calib_dir["image_paths"])
    tmp_calib_dir["calibration_result"] = result
    return tmp_calib_dir


@pytest.fixture()
def synthetic_calib_result() -> "CalibrationResult":
    """Return a synthetic CalibrationResult with realistic values.

    Does NOT touch the filesystem. Useful for report and viz tests.
    """
    import numpy as np
    from calibrate_camera import CalibrationResult  # type: ignore[import]

    K = np.array([
        [1500.0, 0.0, 960.0],
        [0.0, 1500.0, 540.0],
        [0.0, 0.0, 1.0],
    ])
    d = np.array([[0.1, -0.2, 0.001, 0.0005, 0.05]])
    return CalibrationResult(
        camera_matrix=K,
        dist_coeffs=d,
        rotation_vectors=[np.zeros((3, 1))] * 5,
        translation_vectors=[np.zeros((3, 1))] * 5,
        mean_reprojection_error=0.312,
        per_image_errors=[0.30, 0.31, 0.32, 0.30, 0.33],
        image_size=(1920, 1440),
        successful_images=[],
        failed_images=["bad_001.jpg"],
        calibration_duration_s=2.14,
    )


@pytest.fixture()
def synthetic_detection_results(tmp_path: "Path") -> list:
    """Return a list of CornerDetectionResult objects with synthetic data.

    Uses images written to tmp_path so the paths exist on disk.
    """
    import cv2
    import numpy as np
    from calibrate_camera import CornerDetectionResult  # type: ignore[import]

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    board = _make_checkerboard_image(9, 6, square_px=60)
    results = []
    for i in range(5):
        p = img_dir / f"cb_{i:03d}.jpg"
        cv2.imwrite(str(p), board)
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        r = CornerDetectionResult(image_path=p)
        r.success = bool(found)
        r.corners = corners
        r.image_size = (board.shape[1], board.shape[0])
        results.append(r)
    return results
