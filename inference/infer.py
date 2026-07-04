"""
ScrewMetric — Inference Engine
================================
Loads a trained YOLOv8-seg model and performs screw segmentation
inference on single images or batches.

Responsibility (Single Responsibility Principle):
    Model loading and inference only. No measurement, no training.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure models/ is on sys.path so model_config can be imported
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

_INFERENCE_DIR = Path(__file__).resolve().parent
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

from model_config import ModelConfig, get_logger  # type: ignore[import]
from infer_utils import (  # type: ignore[import]
    load_image,
    extract_largest_mask,
    extract_confidence,
    extract_bounding_box,
    mask_to_polygon,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Structured output from a single screw detection.

    Attributes:
        mask: Binary H×W mask (uint8, 0/255).
        bounding_box: (x, y, w, h) in pixels relative to original image.
        confidence: Detection confidence in [0.0, 1.0].
        class_name: Predicted class label string.
        image_shape: (height, width) of the input image.
        inference_time_s: Wall-clock inference duration in seconds.
    """

    mask: np.ndarray
    bounding_box: tuple[int, int, int, int]
    confidence: float
    class_name: str
    image_shape: tuple[int, int]
    inference_time_s: float = 0.0

    @property
    def bbox_area_px(self) -> int:
        """Bounding box area in square pixels."""
        _, _, w, h = self.bounding_box
        return w * h

    @property
    def mask_area_px(self) -> int:
        """Number of non-zero pixels in the segmentation mask."""
        return int((self.mask > 0).sum())


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class ScrewInferenceEngine:
    """Loads a trained YOLOv8-seg model and runs screw segmentation.

    Args:
        config: Pipeline model configuration.

    Example::

        engine = ScrewInferenceEngine(ModelConfig.default())
        engine.load_model()
        result = engine.predict(Path("screw.jpg"))
        if result:
            print(result.confidence, result.bounding_box)
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: Any = None  # Ultralytics YOLO object (lazy-loaded)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model weights into memory.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            ImportError: If Ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        weights_path = self._config.paths.best_weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {weights_path}\n"
                "Train the model first: python models/model_trainer.py"
            )

        logger.info("Loading model weights from: %s", weights_path)
        self._model = YOLO(str(weights_path))
        logger.info("Model loaded — type: %s", self._config.training.model_type)

    @property
    def is_loaded(self) -> bool:
        """True if model weights are loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray | Path | str,
    ) -> InferenceResult | None:
        """Run inference on a single image.

        Args:
            image: Either an image path (``Path`` / ``str``) or a BGR
                ``np.ndarray`` of shape ``(H, W, 3)``.

        Returns:
            :class:`InferenceResult` on success, or ``None`` if no screw
            is detected above the confidence threshold.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() first."
            )

        # Resolve image to numpy array
        if isinstance(image, (str, Path)):
            bgr = load_image(Path(image))
        else:
            bgr = image

        image_shape = (bgr.shape[0], bgr.shape[1])

        cfg = self._config.inference
        t0 = time.perf_counter()
        results = self._model.predict(
            source=bgr,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            max_det=cfg.max_detections,
            imgsz=cfg.input_size,
            device=cfg.device,
            verbose=False,
            half=cfg.half_precision,
        )
        elapsed = time.perf_counter() - t0

        if not results:
            logger.warning("Inference returned no results.")
            return None

        result = results[0]

        # Extract mask for the best (largest) detection
        mask = extract_largest_mask(result.masks, image_shape)
        if mask is None:
            logger.info("No segmentation mask found — screw not detected.")
            return None

        # Get confidence of the highest-scoring box
        confidence = extract_confidence(result, best_idx=0)
        if confidence < cfg.confidence_threshold:
            logger.info(
                "Detection confidence %.3f below threshold %.3f",
                confidence, cfg.confidence_threshold,
            )
            return None

        try:
            bbox = extract_bounding_box(mask)
        except ValueError as exc:
            logger.warning("Bounding box extraction failed: %s", exc)
            return None

        # Class label
        class_name = cfg.class_names[0] if cfg.class_names else "screw"
        try:
            cls_idx = int(result.boxes.cls[0].item())
            if cls_idx < len(cfg.class_names):
                class_name = cfg.class_names[cls_idx]
        except (AttributeError, IndexError):
            pass

        logger.info(
            "Detected '%s'  conf=%.3f  bbox=%s  time=%.3fs",
            class_name, confidence, bbox, elapsed,
        )

        return InferenceResult(
            mask=mask,
            bounding_box=bbox,
            confidence=confidence,
            class_name=class_name,
            image_shape=image_shape,
            inference_time_s=round(elapsed, 4),
        )

    def predict_batch(
        self,
        images: list[np.ndarray | Path | str],
    ) -> list[InferenceResult | None]:
        """Run inference on a batch of images.

        Args:
            images: List of image paths or BGR numpy arrays.

        Returns:
            List of :class:`InferenceResult` objects (``None`` for
            images where no screw was detected).
        """
        return [self.predict(img) for img in images]


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the inference engine with a synthetic screw-like image."""
    print("=" * 64)
    print("  ScrewMetric — Inference Engine Module")
    print("=" * 64)

    import tempfile
    import cv2

    try:
        config = ModelConfig.default()

        # -- demonstrate model not loaded error
        engine = ScrewInferenceEngine(config)
        try:
            engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as exc:
            print(f"\n[RuntimeError guard]  ✓  ({exc})")

        # -- demonstrate missing weights error
        try:
            engine.load_model()
        except FileNotFoundError as exc:
            print(f"[FileNotFoundError]   ✓  (weights not found — expected before training)")
        except ImportError as exc:
            print(f"[ImportError]         ✓  ({exc})")

        # -- demonstrate utils work with synthetic mask
        from utils import extract_bounding_box, mask_to_polygon  # type: ignore
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255
        bbox = extract_bounding_box(mask)
        poly = mask_to_polygon(mask)
        print(f"[extract_bbox]        ✓  bbox={bbox}")
        print(f"[mask_to_polygon]     ✓  {len(poly)} coords")

        print("\n✅ infer.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ infer.py failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
