"""
ScrewMetric — Unit Tests for Inference Module
=============================================
Tests inference loading constraints, pre/post-processing utilities,
bounding box coordinates extraction, and NMS handling.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure inference/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "inference") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "inference"))
if str(_PROJECT_ROOT / "models") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "models"))

from model_config import ModelConfig  # type: ignore[import]
from infer import ScrewInferenceEngine, InferenceResult  # type: ignore[import]
import infer_utils as utils  # type: ignore[import]


# ---------------------------------------------------------------------------
# Test Cases — Pre/Post Processing Utilities
# ---------------------------------------------------------------------------

def test_load_image_raises_on_missing_file() -> None:
    """load_image must raise FileNotFoundError if target is missing."""
    with pytest.raises(FileNotFoundError):
        utils.load_image(Path("missing_file_xyz.png"))


def test_load_image_valid_image() -> None:
    """Ensure cv2 loads and resolves image format shape."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)
        loaded = utils.load_image(f.name)
        assert loaded.shape == (100, 100, 3)
        Path(f.name).unlink()


def test_preprocess_image_resize_and_pad() -> None:
    """Verify pre-processing produces a correct square letterboxed canvas."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    processed = utils.preprocess_image(img, target_size=640)
    assert processed.shape == (640, 640, 3)


def test_bounding_box_within_image() -> None:
    """Ensure coordinates calculated from binary mask form standard bbox bounds."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 40:90] = 255
    x, y, w, h = utils.extract_bounding_box(mask)
    assert x == 40
    assert y == 25
    assert w == 50
    assert h == 50

    # Ensure empty mask raises ValueError
    with pytest.raises(ValueError):
        utils.extract_bounding_box(np.zeros((100, 100), dtype=np.uint8))


def test_mask_to_polygon_coordinate_extraction() -> None:
    """Verify binary mask contour translates to flat coordinate polygons."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
    poly = utils.mask_to_polygon(mask)
    assert len(poly) > 0
    # Coordinates must be pairs (x, y)
    assert len(poly) % 2 == 0


def test_apply_nms_keeps_best_boxes() -> None:
    """Ensure NMS eliminates heavily overlapping boxes based on score."""
    boxes = np.array([
        [100, 100, 150, 150],  # Box A (Score 0.9)
        [102, 102, 152, 152],  # Box B (Score 0.8 — Overlaps Box A)
        [300, 300, 350, 350],  # Box C (Score 0.7 — Independent)
    ])
    scores = np.array([0.9, 0.8, 0.7])
    kept = utils.apply_nms(boxes, scores, iou_threshold=0.5)
    # Box B should be suppressed, Box A and Box C should survive
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


# ---------------------------------------------------------------------------
# Test Cases — Inference Engine
# ---------------------------------------------------------------------------

def test_engine_raises_if_no_weights() -> None:
    """Unloaded engine raises RuntimeError on predict calls."""
    config = ModelConfig.default()
    engine = ScrewInferenceEngine(config)
    with pytest.raises(RuntimeError, match="load_model"):
        engine.predict(np.zeros((100, 100, 3), dtype=np.uint8))


def test_engine_load_model_missing_pt_file() -> None:
    """FileNotFoundError must be raised if weights are non-existent."""
    # Direct path override to invalid pt
    from dataclasses import replace
    cfg = ModelConfig.default()
    new_paths = replace(cfg.paths, best_weights_path=Path("non_existent_weights.pt"))
    cfg_override = ModelConfig(
        paths=new_paths,
        training=cfg.training,
        inference=cfg.inference,
    )
    engine = ScrewInferenceEngine(cfg_override)
    with pytest.raises(FileNotFoundError):
        engine.load_model()
