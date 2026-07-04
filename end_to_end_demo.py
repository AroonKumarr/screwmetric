"""
ScrewMetric — End-to-End Metrology Pipeline Demo
===================================================
Orchestrates the entire ScrewMetric pipeline: loads an image, runs
YOLOv8-seg inference to detect/segment a screw, undistorts the contour
points, performs pixel-to-mm transformation, and prints a structured
JSON result. Optionally saves a visualization of the measurement.

Usage:
    python end_to_end_demo.py --image dataset/splits/val/images/screw_002.jpg --distance-mm 300.0

Authors: ScrewMetric Team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add subdirectories to sys.path so we can import modules correctly
_PROJECT_ROOT = Path(__file__).resolve().parent
for sub in ("models", "inference", "measurement"):
    if str(_PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT / sub))

from model_config import ModelConfig, get_logger  # type: ignore[import]
from infer import ScrewInferenceEngine  # type: ignore[import]
from pixel_to_mm import PixelToMMConverter, MeasurementConfig  # type: ignore[import]
from infer_utils import extract_contour  # type: ignore[import]

logger = get_logger("end_to_end_demo")


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def draw_measurement(
    image: np.ndarray,
    mask: np.ndarray,
    length_mm: float,
    diameter_mm: float,
    scale_mm_per_px: float,
) -> np.ndarray:
    """Overlay the segmentation mask, fitted rotated bounding box, and measurements.

    Args:
        image: Original input BGR image.
        mask: Binary mask of the screw.
        length_mm: Calculated length in mm.
        diameter_mm: Calculated diameter in mm.
        scale_mm_per_px: Scaled multiplier.

    Returns:
        Annotated BGR image.
    """
    canvas = image.copy()

    # Draw semi-transparent green overlay for the mask
    overlay = np.zeros_like(canvas)
    overlay[mask > 0] = [0, 255, 0]  # Green mask
    cv2.addWeighted(canvas, 1.0, overlay, 0.4, 0, canvas)

    # Find the contour
    contour = extract_contour(mask)
    if contour is not None:
        # Fit rotated rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Draw red rotated bounding box
        cv2.drawContours(canvas, [box], 0, (0, 0, 255), 3)

        # Draw a line along the major axis and write text
        (cx, cy), (w, h), angle = rect
        cx, cy = int(cx), int(cy)

        # Draw center point
        cv2.circle(canvas, (cx, cy), 6, (255, 255, 0), -1)

        # Annotate text
        text_lines = [
            f"Length: {length_mm:.1f} mm",
            f"Diameter: {diameter_mm:.1f} mm",
            f"Scale: {scale_mm_per_px:.4f} mm/px",
        ]

        # Draw text overlay box
        text_x = min(max(cx - 150, 20), canvas.shape[1] - 400)
        text_y = min(max(cy - 100, 40), canvas.shape[0] - 120)

        # Background rectangle for text readability
        cv2.rectangle(
            canvas,
            (text_x - 10, text_y - 25),
            (text_x + 350, text_y + 90),
            (20, 20, 20),
            -1,
        )

        for i, line in enumerate(text_lines):
            y_pos = text_y + (i * 30)
            cv2.putText(
                canvas,
                line,
                (text_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return canvas


# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: Path,
    distance_mm: float,
    weights_path: Path | None = None,
    output_vis_path: Path | None = None,
) -> int:
    """Run the complete ScrewMetric end-to-end metrology pipeline.

    Args:
        image_path: Path to the screw image to measure.
        distance_mm: Known depth distance from camera plane to screw plane (mm).
        weights_path: Optional path to override the YOLO weights.
        output_vis_path: Optional path to save an annotated visualization image.

    Returns:
        Exit status code (0 for success, non-zero for error).
    """
    logger.info("Starting ScrewMetric End-to-End Pipeline Demo")
    logger.info("Image      : %s", image_path)
    logger.info("Distance   : %.1f mm", distance_mm)

    # 1. Config Initialisation
    try:
        model_cfg = ModelConfig.default()
        if weights_path:
            # Override weights path
            from dataclasses import replace
            new_paths = replace(model_cfg.paths, best_weights_path=weights_path)
            model_cfg = ModelConfig(
                paths=new_paths,
                training=model_cfg.training,
                inference=model_cfg.inference,
            )

        meas_cfg = MeasurementConfig(
            camera_matrix_path=model_cfg.paths.camera_matrix_path,
            dist_coeffs_path=model_cfg.paths.dist_coeffs_path,
            known_distance_mm=distance_mm,
        )
    except Exception as exc:
        logger.error("Configuration loading failed: %s", exc)
        return 1

    # 2. Setup Inference Engine
    try:
        engine = ScrewInferenceEngine(model_cfg)
        engine.load_model()
    except FileNotFoundError as exc:
        logger.error("Model weights not found. Please train the model first.")
        logger.error("Error: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Failed to load inference engine: %s", exc)
        return 2

    # 3. Setup Metrology Converter
    try:
        converter = PixelToMMConverter(meas_cfg)
        converter.load_calibration()
    except FileNotFoundError as exc:
        logger.error("Camera calibration outputs missing. Run calibration first.")
        logger.error("Error: %s", exc)
        return 3
    except Exception as exc:
        logger.error("Failed to load camera calibration: %s", exc)
        return 3

    # 4. Load Image
    try:
        if not image_path.exists():
            logger.error("Image file does not exist: %s", image_path)
            return 4
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            logger.error("Could not load/decode image: %s", image_path)
            return 4
    except Exception as exc:
        logger.error("Error reading image: %s", exc)
        return 4

    # 5. Run Object Detection & Segmentation
    try:
        inference_result = engine.predict(bgr)
        if inference_result is None:
            logger.warning("No screw detected in the image.")
            # Print a standard failure JSON structure
            print(json.dumps({
                "status": "FAILED",
                "error": "No screw detected above confidence threshold."
            }, indent=2))
            return 0
    except Exception as exc:
        logger.error("Segmentation inference failed: %s", exc)
        return 5

    # 6. Run Physical Metrology Calculations
    try:
        measurement = converter.measure(
            mask=inference_result.mask,
            confidence=inference_result.confidence,
        )
    except Exception as exc:
        logger.error("Metrology pixel-to-mm scaling failed: %s", exc)
        return 6

    # 7. Print Output Schema
    output_data = {
        "status": "SUCCESS",
        "length_mm": round(measurement.length_mm, 2),
        "diameter_mm": round(measurement.diameter_mm, 2),
        "confidence": round(measurement.confidence, 4),
        "scale_mm_per_px": round(measurement.scale_mm_per_px, 6),
        "pixel_length": round(measurement.pixel_length, 2),
        "pixel_diameter": round(measurement.pixel_diameter, 2),
        "bounding_box": {
            "x": inference_result.bounding_box[0],
            "y": inference_result.bounding_box[1],
            "w": inference_result.bounding_box[2],
            "h": inference_result.bounding_box[3],
        },
        "method": measurement.method,
    }

    # Output directly to stdout as formatted JSON
    print(json.dumps(output_data, indent=2))

    # 8. Optional: Save Visualization
    if output_vis_path:
        try:
            output_vis_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img = draw_measurement(
                image=bgr,
                mask=inference_result.mask,
                length_mm=measurement.length_mm,
                diameter_mm=measurement.diameter_mm,
                scale_mm_per_px=measurement.scale_mm_per_px,
            )
            cv2.imwrite(str(output_vis_path), vis_img)
            logger.info("Saved annotated visualization to: %s", output_vis_path)
        except Exception as exc:
            logger.error("Failed to save visualization: %s", exc)

    return 0


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the pipeline from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="ScrewMetric End-to-End Metrology Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image containing a screw to measure.",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=300.0,
        help="Known perpendicular distance in mm between camera lens and screw plane.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO segmentation weights best.pt (optional).",
    )
    parser.add_argument(
        "--save-vis",
        type=str,
        default=None,
        help="Path to save annotated visualization image (optional).",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights) if args.weights else None
    vis_path = Path(args.save_vis) if args.save_vis else None

    status = run_pipeline(
        image_path=image_path,
        distance_mm=args.distance_mm,
        weights_path=weights_path,
        output_vis_path=vis_path,
    )
    sys.exit(status)


if __name__ == "__main__":
    main()
