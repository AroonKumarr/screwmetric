"""
ScrewMetric — Inference Engine (Mask R-CNN)
============================================
Loads a trained torchvision Mask R-CNN model and performs screw
segmentation inference on single images or batches.

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

import cv2
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
    """Loads a trained Mask R-CNN model and runs screw segmentation.

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
        self._model: Any = None  # PyTorch Mask R-CNN model (lazy-loaded)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model weights into memory.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            ImportError: If torch/torchvision is not installed.
        """
        # Bug 1 Fix: check weights file existence BEFORE trying to import packages or load model
        weights_path = self._config.paths.best_weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {weights_path}\n"
                "Train the model first: python models/model_trainer.py"
            )

        try:
            import torch
            import torchvision  # type: ignore[import]
            from torchvision.models.detection import (  # type: ignore[import]
                maskrcnn_resnet50_fpn,
                MaskRCNN_ResNet50_FPN_Weights,
            )
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # type: ignore[import]
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "torch and torchvision are required. Run: pip install torch torchvision"
            ) from exc

        logger.info("Loading Mask R-CNN model weights from: %s", weights_path)

        # Re-build model structure matching training
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            trainable_backbone_layers=0,
        )
        num_classes = 2  # background (0) + screw (1)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        # Load weights onto CPU
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        self._model = model
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

        try:
            import torch
            from torchvision.transforms import functional as F
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("PyTorch is required for running inference.")

        # Resolve image to numpy array (BGR)
        if isinstance(image, (str, Path)):
            bgr = load_image(Path(image))
        else:
            bgr = image

        image_shape = (bgr.shape[0], bgr.shape[1])

        # Preprocess: convert BGR NumPy array to PIL RGB -> PyTorch Tensor
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb_img)
        
        cfg = self._config.inference
        # Resize to input_size for faster processing matching training
        size = cfg.input_size
        pil_img_resized = pil_img.resize((size, size))
        tensor_img = F.to_tensor(pil_img_resized)

        t0 = time.perf_counter()
        with torch.no_grad():
            preds = self._model([tensor_img])[0]
        elapsed = time.perf_counter() - t0

        if not preds or "scores" not in preds or len(preds["scores"]) == 0:
            logger.warning("Inference returned no results.")
            return None

        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        masks = preds["masks"].cpu().numpy()
        boxes = preds["boxes"].cpu().numpy()

        # Find valid indices: class is screw (1) and confidence above threshold
        valid_indices = [
            i for i, (score, label) in enumerate(zip(scores, labels))
            if label == 1 and score >= cfg.confidence_threshold
        ]

        if not valid_indices:
            logger.info("No screw detected above confidence threshold.")
            return None

        # Filter predictions
        valid_masks = masks[valid_indices]
        valid_scores = scores[valid_indices]
        valid_boxes = boxes[valid_indices]

        # Resize masks back to original image shape and find the largest
        resized_masks = []
        h, w = image_shape
        for m in valid_masks:
            m_sq = m.squeeze(0)  # shape (size, size)
            m_res = cv2.resize(m_sq, (w, h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(m_res)

        # Extract mask of the largest detection
        mask_tuple = extract_largest_mask(np.stack(resized_masks), image_shape)
        if mask_tuple is None:
            logger.info("No segmentation mask found.")
            return None

        mask, best_idx = mask_tuple

        # Bug 2 Fix: extract confidence using best_idx corresponding to the largest mask area
        actual_global_idx = valid_indices[best_idx]
        confidence = float(scores[actual_global_idx])

        try:
            bbox = extract_bounding_box(mask)
        except ValueError as exc:
            logger.warning("Bounding box extraction failed: %s", exc)
            return None

        # Class label
        class_name = cfg.class_names[0] if cfg.class_names else "screw"

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
    print("  ScrewMetric — Mask R-CNN Inference Engine Module")
    print("=" * 64)

    try:
        config = ModelConfig.default()
        engine = ScrewInferenceEngine(config)

        # -- demonstrate model not loaded error
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
        from infer_utils import extract_bounding_box, mask_to_polygon  # type: ignore
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
