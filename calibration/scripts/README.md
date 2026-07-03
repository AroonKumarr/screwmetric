# Camera Calibration Module

> **ScrewMetric** — Production-quality camera calibration for accurate screw dimension measurement.

## Overview

This module estimates intrinsic camera parameters (camera matrix, distortion coefficients) from a set of checkerboard images. Calibration is a prerequisite for converting pixel distances to millimetres in the Measurement module.

## Module Architecture

```
calibration/
├── images/                    ← Place checkerboard images here
├── output/                    ← All generated artefacts
│   ├── camera_matrix.npy
│   ├── dist_coeffs.npy
│   ├── rotation_vectors.npy
│   ├── translation_vectors.npy
│   ├── reprojection_error.json
│   ├── validation_report.json
│   ├── calibration_report.json
│   ├── camera_parameters.yaml
│   ├── calibration_visualization.png
│   └── undistortion_preview/
└── scripts/
    ├── config.py                      ← All configuration (frozen dataclasses)
    ├── utils.py                       ← Stateless I/O helpers
    ├── capture_calibration_images.py  ← Live capture + iPhone guidance
    ├── calibration_validator.py       ← Pre-calibration validation
    ├── calibrate_camera.py            ← OpenCV calibration engine
    ├── report_generator.py            ← JSON + YAML reports
    ├── visualize_calibration.py       ← Visualizations and undistortion previews
    └── camera_calibration.py          ← CLI orchestrator (entry point)
```

## Quick Start

### Step 1 — Capture calibration images

Print a 10×7 square checkerboard (inner corners: 9×6). Measure the physical size of one square.

**Option A — Live webcam capture:**
```bash
cd calibration/scripts
python capture_calibration_images.py
```

**Option B — iPhone capture (see guidance):**
```bash
python capture_calibration_images.py  # prints detailed instructions
```
Place your images in `calibration/images/` named `checkerboard_001.jpg`, `checkerboard_002.jpg`, etc.

### Step 2 — Run the full pipeline

```bash
cd calibration/scripts
python camera_calibration.py
```

This will automatically:
1. Validate all images
2. Detect checkerboard corners (sub-pixel refined)
3. Estimate camera matrix + distortion coefficients
4. Save all artefacts to `calibration/output/`
5. Generate JSON report + YAML parameters
6. Generate visualizations and undistortion previews

### Step 3 — Customise checkerboard parameters

```bash
python camera_calibration.py --corners-x 8 --corners-y 5 --square-mm 30.0
```

## CLI Reference

```
usage: camera_calibration.py [-h] [--skip-validation] [--skip-calibration]
                              [--skip-report] [--skip-visualization]
                              [--corners-x N] [--corners-y N]
                              [--square-mm MM] [--alpha A]

Options:
  --skip-validation      Skip pre-calibration image validation
  --skip-calibration     Skip the OpenCV calibration step
  --skip-report          Skip JSON/YAML report generation
  --skip-visualization   Skip visualization generation
  --corners-x N          Inner corners along horizontal axis (default: 9)
  --corners-y N          Inner corners along vertical axis (default: 6)
  --square-mm MM         Physical square size in mm (default: 25.0)
  --alpha A              Undistortion alpha: 0.0=crop, 1.0=all (default: 0.0)
```

## Running Individual Modules

Each script is independently executable:

```bash
cd calibration/scripts

# Validation only
python calibration_validator.py

# Calibration only (uses images in calibration/images/)
python calibrate_camera.py

# Report only (loads artefacts from calibration/output/)
python report_generator.py

# Visualization only
python visualize_calibration.py

# Configuration demo
python config.py

# Utilities demo
python utils.py
```

## Output Files

| File | Description |
|------|-------------|
| `camera_matrix.npy` | 3×3 intrinsic camera matrix **K** |
| `dist_coeffs.npy` | Distortion coefficients (k1, k2, p1, p2, k3) |
| `rotation_vectors.npy` | Per-image rotation vectors |
| `translation_vectors.npy` | Per-image translation vectors |
| `reprojection_error.json` | Mean + per-image reprojection errors |
| `validation_report.json` | Pre-calibration image quality report |
| `calibration_report.json` | Full post-calibration report |
| `camera_parameters.yaml` | Human-readable parameters (ROS-compatible) |
| `calibration_visualization.png` | Composite diagnostic image |
| `undistortion_preview/` | Sample images before/after undistortion |

## Camera Parameters YAML Format

```yaml
image_width: 4032
image_height: 3024
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_coeffs:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
focal_length_fx: 2847.31
focal_length_fy: 2851.92
principal_point_cx: 2016.0
principal_point_cy: 1512.0
reprojection_error: 0.3142
square_size_mm: 25.0
date_created: "2024-07-04T00:00:00+00:00"
```

## Testing

Run all calibration tests:
```bash
# From project root
python run_tests.py

# Calibration tests only
python -m pytest tests/test_calibration_validator.py \
                 tests/test_camera_calibration.py \
                 tests/test_visualize_calibration.py \
                 tests/test_report_generator.py \
                 tests/test_capture_calibration_images.py -v
```

## Capture Tips for Best Results

- ✅ Use **20–30 images** covering varied angles
- ✅ Tilt the board **30–45°** in multiple directions
- ✅ Vary **distance** (near, medium, far)
- ✅ Ensure **all board corners are visible** in every frame
- ✅ Use **Timer or Remote Shutter** to avoid blur
- ✅ Shoot in **good, even lighting** (avoid reflections)
- ❌ Do NOT use partial board views
- ❌ Avoid angles greater than 60°
- ❌ Avoid glossy/laminated board surfaces

## Checkerboard Specifications

Default configuration (`config.py`):

| Parameter | Value |
|-----------|-------|
| Inner corners X | 9 |
| Inner corners Y | 6 |
| Total squares | 10 × 7 |
| Square size | 25 mm |

## Integration with Measurement Module

The Measurement module loads camera parameters directly from the generated artefacts:

```python
import numpy as np

camera_matrix = np.load("calibration/output/camera_matrix.npy")
dist_coeffs   = np.load("calibration/output/dist_coeffs.npy")
```

## Design Principles

This module follows SOLID principles throughout:

| Principle | Implementation |
|-----------|---------------|
| **S**ingle Responsibility | Each class has one job (detect, calibrate, report, visualise) |
| **O**pen/Closed | New stages added by extending `CalibrationPipeline` without modification |
| **L**iskov Substitution | Duck-typed path providers enable test injection |
| **I**nterface Segregation | Config split into `CheckerboardConfig`, `ValidationConfig`, `CalibrationProcessConfig` |
| **D**ependency Inversion | All classes receive `CalibrationConfig` — no hardcoded paths |
