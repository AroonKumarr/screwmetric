"""
ScrewMetric — Screw Length Extraction Utilities
=================================================
Helper functions for extracting the length (major axis) of a detected
screw from a binary segmentation mask.

The core measurement is handled by PixelToMMConverter in pixel_to_mm.py.
This module provides standalone helper functions that can be used
independently of the full pipeline.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_length_px(mask: np.ndarray) -> float:
    """Extract the screw length in pixels from a binary mask.

    Fits a minimum-area rotated rectangle to the largest contour
    and returns the major axis (length direction).

    Args:
        mask: Binary uint8 mask of shape (H, W) where 255 = screw.

    Returns:
        Major axis length in pixels (screw length).

    Raises:
        ValueError: If the mask contains no valid contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask.")
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        x, y, w, h = cv2.boundingRect(largest)
        return float(max(w, h))
    _, (w, h), _ = cv2.minAreaRect(largest)
    return float(max(w, h))


def extract_length_mm(mask: np.ndarray, scale_mm_per_px: float) -> float:
    """Convert the screw length from pixels to millimetres.

    Args:
        mask: Binary uint8 mask.
        scale_mm_per_px: Pixel-to-mm scale factor from camera calibration.

    Returns:
        Screw length in millimetres.
    """
    return extract_length_px(mask) * scale_mm_per_px
