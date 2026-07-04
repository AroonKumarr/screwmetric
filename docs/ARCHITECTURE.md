# System Architecture (docs/ARCHITECTURE.md)

This document describes the module layout, component design, and end-to-end data flow of the ScrewMetric system.

---

## 🏗️ Core Pipeline Data Flow

The flow of data from a raw image file to computed metric sizes is structured as follows:

```
+--------------------------------------------------------+
|                      INPUT SOURCE                      |
|  - Raw image file (BGR array of shape 4032x3024)       |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|                   INFERENCE STAGE                      |
|  - Resize image to 640x640 tensor                      |
|  - Run Torchvision Mask R-CNN prediction              |
|  - Retrieve bounding boxes, labels, masks, and scores  |
|  - Filter by confidence and select largest mask       |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|                  METROLOGY CONVERSION                  |
|  - Read camera K & D from calibration output           |
|  - Scale camera intrinsics dynamically to 4032x3024    |
|  - Extract contour from mask                           |
|  - Correct distortion (cv2.undistortPoints)            |
|  - Fit min-area rotated rect                           |
|  - Convert major/minor pixel axes to millimetres       |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|                     OUTPUTS CARD                       |
|  - Display length (mm), diameter (mm), and confidence   |
|  - Draw mask, contour, and rotated bbox overlays       |
|  - Save and display history logs                       |
+--------------------------------------------------------+
```

---

## 📂 Module Layout & Responsibilities

The codebase conforms to the **Single Responsibility Principle (SRP)**:

1. **`calibration/` (Calibration Stage):**
   * Computes intrinsic matrix and distortion coefficients from checkerboard frames.
   * Outputs calibration files (`camera_matrix.npy`, `dist_coeffs.npy`).
2. **`dataset/` (Data Stage):**
   * Validates dataset integrity, splits images into train/val/test partitions, and outputs COCO labels.
3. **`models/` (Model Stage):**
   * Configures hyperparameters (`model_config.py`).
   * Manages the training and checkpoint loop (`model_trainer.py`).
4. **`inference/` (Inference Stage):**
   * Loads trained checkpoints (`infer.py`).
   * Evaluates image segmentation.
5. **`measurement/` (Metrology Stage):**
   * Integrates camera calibration arrays with segmentation masks to calculate real-world measurements (`pixel_to_mm.py`).
6. **`app.py` (Presentation Stage):**
   * Connects the pipeline into a Streamlit GUI dashboard.
