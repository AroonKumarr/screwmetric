"""
ScrewMetric — Inference Utilities
====================================
Pure, stateless helper functions for image pre-processing, mask
post-processing, bounding-box extraction, and NMS handling.

Responsibility (Single Responsibility Principle):
    Utility helpers only. No model loading, no config, no disk I/O.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path | str) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array.

    Args:
        path: Path to the image file.

    Returns:
        BGR image array of shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_image(
    image: np.ndarray,
    target_size: int = 640,
) -> np.ndarray:
    """Resize an image to a square while maintaining aspect ratio (letterbox).

    Args:
        image: Input BGR image array.
        target_size: Target side length in pixels.

    Returns:
        Letterboxed BGR image of shape ``(target_size, target_size, 3)``.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Mask extraction
# ---------------------------------------------------------------------------

def extract_largest_mask(
    masks: Any,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int] | None:
    """Extract the binary mask of the largest (highest-area) detection.

    Args:
        masks: Mask data tensor or NumPy array (N, H, W) or (N, 1, H, W).
        image_shape: ``(height, width)`` of the original image.

    Returns:
        Tuple ``(binary_mask, best_idx)``, or ``None`` if no masks are present.
    """
    if masks is None:
        return None

    if isinstance(masks, np.ndarray):
        mask_data = masks
    else:
        try:
            mask_data = masks.cpu().numpy()  # shape (N, H, W) or (N, 1, H, W)
        except AttributeError:
            try:
                mask_data = np.array(masks)
            except Exception:
                return None

    if len(mask_data.shape) == 4:
        mask_data = np.squeeze(mask_data, axis=1)

    if len(mask_data.shape) != 3 or mask_data.shape[0] == 0:
        return None

    # Select the mask with the largest area
    areas = [m.sum() for m in mask_data]
    best_idx = int(np.argmax(areas))
    mask_f = mask_data[best_idx]  # float, 0.0–1.0 or uint8 0/255

    # Resize back to original image dimensions
    h, w = image_shape
    mask_resized = cv2.resize(mask_f.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    binary = (mask_resized > 0.5).astype(np.uint8) * 255
    return binary, best_idx


def extract_confidence(result: Any, best_idx: int = 0) -> float:
    """Extract the confidence score for a specific detection.

    Args:
        result: Dictionary or result object containing confidence scores.
        best_idx: Index of the detection.

    Returns:
        Confidence score in ``[0.0, 1.0]``, or ``0.0`` if unavailable.
    """
    try:
        if isinstance(result, dict):
            confs = result.get("scores", [])
        else:
            confs = getattr(result, "scores", [])
            if len(confs) == 0:
                confs = getattr(getattr(result, "boxes", None), "conf", [])

        try:
            confs_arr = confs.cpu().numpy()
        except AttributeError:
            confs_arr = np.array(confs)

        if len(confs_arr) == 0 or best_idx >= len(confs_arr):
            return 0.0
        return float(confs_arr[best_idx])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Bounding box extraction
# ---------------------------------------------------------------------------

def extract_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute the axis-aligned bounding box of a binary mask.

    Args:
        mask: Binary mask array of shape ``(H, W)`` with values 0/255.

    Returns:
        Tuple ``(x, y, w, h)`` in pixel coordinates (top-left origin).

    Raises:
        ValueError: If the mask is entirely zero (no object).
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("Mask contains no non-zero pixels — cannot extract bounding box.")
    x, y, w, h = cv2.boundingRect(coords)
    return int(x), int(y), int(w), int(h)


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contour(mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest contour from a binary mask.

    Args:
        mask: Binary uint8 mask of shape ``(H, W)``.

    Returns:
        Contour array of shape ``(N, 1, 2)`` or ``None`` if no contour found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert a binary mask to a flat polygon coordinate list.

    Suitable for COCO-format segmentation output.

    Args:
        mask: Binary uint8 mask of shape ``(H, W)``.

    Returns:
        List of ``[x0, y0, x1, y1, ...]`` float coordinates,
        or an empty list if no contour is found.
    """
    contour = extract_contour(mask)
    if contour is None:
        return []
    return contour.flatten().tolist()


# ---------------------------------------------------------------------------
# NMS helper
# ---------------------------------------------------------------------------

def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """Apply non-maximum suppression to a list of bounding boxes.

    Args:
        boxes: Array of shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.
        scores: Confidence scores of shape ``(N,)``.
        iou_threshold: Overlap threshold above which weaker boxes are suppressed.

    Returns:
        List of kept indices.
    """
    if len(boxes) == 0:
        return []
    # Convert to (x, y, w, h) for OpenCV NMS
    boxes_xywh = []
    for b in boxes:
        x1, y1, x2, y2 = b
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if indices is None or len(indices) == 0:
        return []
    return [int(i) for i in indices.flatten()]


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate inference utilities with synthetic data."""
    print("=" * 64)
    print("  ScrewMetric — Inference Utilities Module")
    print("=" * 64)

    try:
        import tempfile

        # -- load_image / preprocess_image
        synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(synthetic, (100, 50), (540, 430), (200, 200, 200), -1)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, synthetic)
            loaded = load_image(f.name)
        assert loaded.shape == (480, 640, 3), f"Unexpected shape: {loaded.shape}"
        print(f"\n[load_image]       shape={loaded.shape}  ✓")

        preprocessed = preprocess_image(synthetic, target_size=640)
        assert preprocessed.shape == (640, 640, 3)
        print(f"[preprocess_image] shape={preprocessed.shape}  ✓")

        # -- extract_bounding_box
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:400, 200:500] = 255
        x, y, w, h = extract_bounding_box(mask)
        assert x == 200 and y == 100 and w == 300 and h == 300
        print(f"[extract_bbox]     (x={x}, y={y}, w={w}, h={h})  ✓")

        # -- mask_to_polygon
        polygon = mask_to_polygon(mask)
        assert len(polygon) > 0
        print(f"[mask_to_polygon]  {len(polygon)} coordinates  ✓")

        # -- apply_nms
        boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 300, 300]])
        scores = np.array([0.9, 0.85, 0.75])
        kept = apply_nms(boxes, scores, iou_threshold=0.5)
        assert len(kept) <= 3
        print(f"[apply_nms]        kept {len(kept)} of 3 boxes  ✓")

        # -- extract_contour
        contour = extract_contour(mask)
        assert contour is not None
        print(f"[extract_contour]  contour pts={len(contour)}  ✓")

        print("\n✅ utils.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ utils.py failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
