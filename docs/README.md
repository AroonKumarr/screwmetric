# ScrewMetric — Metrology and Sizing Pipeline
## Project Overview & Guide

ScrewMetric is an end-to-end computer vision and deep learning metrology system designed to measure the physical dimensions (length and diameter) of hardware screws in millimetres. The system combines camera intrinsic calibration (lens undistortion), instance segmentation via **Torchvision Mask R-CNN**, and pinhole camera math.

```
+------------------+     +------------------------+     +------------------------+
|  Raw Image (BGR)  | --> | Lens Undistortion      | --> | Mask R-CNN Segmenter   |
|  (4032x3024)     |     | (Scaled Camera Matrix) |     | (Torchvision ResNet50) |
+------------------+     +------------------------+     +------------------------+
                                                                     |
                                                                     v
+------------------+     +------------------------+     +------------------------+
| Real-World Sizing| <-- | Rotated Rectangle Fit  | <-- | Undistorted Contour    |
| (Length / Dia)   |     | (Axis Extraction)      |     | (cv2.undistortPoints)  |
+------------------+     +------------------------+     +------------------------+
```

---

## 📂 Repository Directory Tree
```
screwmetric/
├── app.py                     # Streamlit web application dashboard
├── end_to_end_demo.py         # Command Line Interface (CLI) pipeline demo
├── requirements.txt           # Python dependency specifications
├── LICENSE                    # MIT License file
├── calibration/               # Camera calibration scripts and outputs
│   ├── images/                # Checkerboard calibration images
│   └── output/                # K and D matrix files (.npy)
├── dataset/                   # Dataset split, labels, splits, annotations
│   ├── raw_images/            # Collected raw screw photos
│   └── splits/                # Train/Val/Test directories
├── models/                    # Model training configurations, weights, trainer
│   ├── weights/               # Trained weights checkpoint files (.pt)
│   ├── train.py               # Training entry point CLI
│   └── model_trainer.py       # Mask R-CNN training logic
├── inference/                 # Inference engine and pre-processing
│   ├── infer.py               # Prediction class
│   └── infer_utils.py         # stateles image/mask helpers
├── measurement/               # Metrology scale conversion & accuracy check
│   ├── pixel_to_mm.py         # Core conversion logic
│   └── accuracy_validation/   # ground_truth.csv and caliper report
└── docs/                      # Substantive project documentation reports
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/AroonKumarr/screwmetric.git
cd screwmetric
pip install -r requirements.txt
```

### 2. Launching the Web App
Run the Streamlit interactive dashboard:
```bash
streamlit run app.py
```

### 3. Model Sizing Demo
Execute a metrology scan on a single image via the CLI:
```bash
python end_to_end_demo.py --image dataset/splits/val/images/screw_003.jpg --distance 300.0
```

---

## 📊 Performance Summary
* **Instance Segmentation Architecture:** Torchvision Mask R-CNN (ResNet-50-FPN backbone)
* **Average Inference Latency:** ~200ms per image (CPU)
* **Mean Absolute Error (MAE) - Length:** 1.01 mm
* **Mean Absolute Error (MAE) - Diameter:** 0.16 mm
* **Mean Percentage Error (MAPE):** ~3.8%
* **Calibration Reprojection Error:** 4.29 pixels
