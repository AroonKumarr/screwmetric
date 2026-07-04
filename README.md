# ScrewMetric — AI-Powered Screw Dimension Measurement System

> **Production-grade Computer Vision pipeline that segments a screw and measures its real-world length and diameter in millimetres using a calibrated monocular camera.**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Dataset Pipeline](#dataset-pipeline)
6. [Camera Calibration](#camera-calibration)
7. [Model Training](#model-training)
8. [Inference](#inference)
9. [End-to-End Measurement](#end-to-end-measurement)
10. [Testing](#testing)
11. [Expected Outputs](#expected-outputs)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)

---

## Project Overview

**ScrewMetric** is an end-to-end Computer Vision system for the non-contact dimensional inspection of screws using a smartphone or industrial camera.

| Input | Output |
|-------|--------|
| Single JPEG/PNG image of a screw | `length_mm`, `diameter_mm`, confidence, segmentation mask, bounding box (JSON) |

**Pipeline**:
```
Image → Camera Undistortion → YOLOv8-seg Segmentation → Pixel Contour Extraction
     → Lens Distortion Correction → Min-Area Rectangle Fit → Physical Scale (mm)
     → JSON Result + Annotated Visualization
```

---

## Architecture

```
screwmetric/
│
├── dataset/
│   ├── annotations/          COCO JSON master annotation file
│   ├── images/               Raw screw images
│   ├── splits/               train / val / test image splits
│   └── scripts/              Dataset pipeline scripts
│
├── calibration/
│   ├── images/               Checkerboard calibration images
│   ├── output/               camera_matrix.npy, dist_coeffs.npy, YAML, reports
│   └── scripts/              Calibration pipeline scripts
│
├── models/
│   ├── model_config.py       Centralized configuration dataclasses
│   ├── model_trainer.py      YOLOv8-seg training + COCO→YOLO converter
│   ├── configs/dataset.yaml  Auto-generated YOLO dataset descriptor
│   └── weights/best.pt       Trained model checkpoint
│
├── inference/
│   ├── infer.py              ScrewInferenceEngine (loads weights, predicts)
│   └── infer_utils.py        Pre/post-processing helpers
│
├── measurement/
│   └── pixel_to_mm.py        Metrology engine: pinhole scale + undistortion
│
├── tests/                    285 pytest unit and integration tests
├── end_to_end_demo.py        Full pipeline CLI script
├── run_tests.py              Test runner with coloured summary
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process dataset
cd dataset/scripts && python dataset_processor.py && cd ../..

# 3. Train the segmentation model (3 epochs quick test)
python models/model_trainer.py --epochs 3

# 4. Measure a screw
python end_to_end_demo.py \
    --image dataset/splits/val/images/screw_003.jpg \
    --distance-mm 300.0 \
    --save-vis output_visualization.jpg
```

---

## Installation

### Requirements
- Python 3.9+
- macOS / Linux / Windows
- Apple Silicon (CPU training), NVIDIA GPU (CUDA), or CPU

### Install

```bash
git clone https://github.com/AroonKumarr/screwmetric.git
cd screwmetric
pip install -r requirements.txt
```

### Verified Dependency Versions

| Package | Version |
|---------|---------|
| torch | 2.2.1 |
| torchvision | 0.17.1 |
| ultralytics | 8.4.x |
| opencv-python | 4.10.0.84 |
| numpy | 1.26.4 |

> **Note**: numpy must be `<2.0` for binary compatibility with PyTorch 2.2 and OpenCV 4.10.
> MPS (Apple Silicon GPU) is not used for training due to PyTorch 2.2 autocast limitations;
> CPU training is used instead and produces identical model weights.

---

## Dataset Pipeline

```bash
cd dataset/scripts
python dataset_processor.py
```

This runs: Validation → Splitting (70/20/10) → Statistics → Report → Previews

---

## Camera Calibration

```bash
cd calibration/scripts
python calibrate_camera.py
```

Outputs written to `calibration/output/`:
- `camera_matrix.npy` — intrinsic matrix K (3×3)
- `dist_coeffs.npy` — distortion coefficients (1×5)
- `camera_parameters.yaml` — human-readable parameters

---

## Model Training

```bash
# Quick training (3 epochs, for testing)
python models/model_trainer.py --epochs 3

# Full training (recommended: 100 epochs)
python models/model_trainer.py --epochs 100
```

COCO → YOLO label conversion is **automatic** on every training run.

---

## End-to-End Measurement

```bash
python end_to_end_demo.py \
    --image my_screw.jpg \
    --distance-mm 300.0 \
    --save-vis output/result.jpg
```

### JSON Output Schema:

```json
{
  "status": "SUCCESS",
  "length_mm": 45.2,
  "diameter_mm": 8.4,
  "confidence": 0.93,
  "scale_mm_per_px": 0.629,
  "bounding_box": {"x": 120, "y": 340, "w": 80, "h": 250}
}
```

### Measurement Math:

scale = Z / f_avg  where Z = camera-to-screw distance (mm),  f_avg = (fx + fy) / 2

---

## Testing

```bash
python run_tests.py        # All 285 tests
python run_tests.py -v     # Verbose
python run_tests.py -k measurement  # Filter by keyword
```

---

## Configuration

All paths are centralized, config-driven:
- `dataset/scripts/config.py` → `PipelineConfig`
- `models/model_config.py` → `ModelConfig`, `TrainingConfig`, `InferenceConfig`
- `measurement/pixel_to_mm.py` → `MeasurementConfig`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `numpy.dtype size changed` | `pip install "numpy<2"` |
| `RuntimeError: unsupported autocast device_type 'mps'` | Fixed in model_config.py — forces CPU |
| `FileNotFoundError: camera_matrix.npy` | Run `python calibration/scripts/calibrate_camera.py` first |
| No screw detected | Train for more epochs; verify `--distance-mm` is accurate |
