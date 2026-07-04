"""
ScrewMetric Frontend — Utility Helpers
=========================================
Helper functions for model checkups, calibration parameters loader,
and image formatting conversion.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

def get_project_paths() -> dict[str, Path]:
    """Retrieve canonical project paths for weights, calibration data, and dataset."""
    root = Path(__file__).resolve().parent.parent.parent
    return {
        "root": root,
        "weights": root / "models" / "weights" / "best.pt",
        "camera_matrix": root / "calibration" / "output" / "camera_matrix.npy",
        "dist_coeffs": root / "calibration" / "output" / "dist_coeffs.npy",
        "calibration_dir": root / "calibration" / "output",
        "val_images": root / "dataset" / "splits" / "val" / "images",
    }

def check_weights_status() -> dict[str, any]:
    """Check if the YOLOv8 segmentation weights best.pt file is present."""
    paths = get_project_paths()
    w_path = paths["weights"]
    exists = w_path.exists()
    return {
        "exists": exists,
        "path": w_path,
        "size_mb": round(w_path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
    }

def check_calibration_status() -> dict[str, any]:
    """Check if camera calibration parameters have been generated."""
    paths = get_project_paths()
    cm_path = paths["camera_matrix"]
    dc_path = paths["dist_coeffs"]
    
    exists = cm_path.exists() and dc_path.exists()
    info = {
        "exists": exists,
        "camera_matrix": None,
        "dist_coeffs": None,
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0,
    }
    
    if exists:
        try:
            K = np.load(str(cm_path))
            D = np.load(str(dc_path))
            info["camera_matrix"] = K.tolist()
            info["dist_coeffs"] = D.tolist()[0] if D.ndim > 1 else D.tolist()
            info["fx"] = float(K[0, 0])
            info["fy"] = float(K[1, 1])
            info["cx"] = float(K[0, 2])
            info["cy"] = float(K[1, 2])
        except Exception:
            info["exists"] = False
            
    return info

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a BGR NumPy array for OpenCV consumption."""
    rgb_arr = np.array(pil_img.convert("RGB"))
    # RGB to BGR
    return rgb_arr[:, :, ::-1].copy()

def bgr_to_pil(bgr_arr: np.ndarray) -> Image.Image:
    """Convert a BGR NumPy array from OpenCV back to a PIL Image."""
    # BGR to RGB
    rgb_arr = bgr_arr[:, :, ::-1].copy()
    return Image.fromarray(rgb_arr)

def get_sample_images() -> list[Path]:
    """Return a list of paths to valid sample images from the validation set."""
    paths = get_project_paths()
    val_dir = paths["val_images"]
    if val_dir.exists():
        return sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))
    return []
