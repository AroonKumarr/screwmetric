# Installation & Setup Guide (docs/SETUP.md)

This document details the step-by-step setup, compilation, validation, and execution instructions for the ScrewMetric system.

---

## 📋 Prerequisites
* **Operating System:** macOS, Linux, or Windows 10/11
* **Python Version:** Python 3.10 or 3.11 (Recommended)
* **Hardware Requirements:** 
  * CPU: 4+ Cores
  * GPU: Optional (CUDA compatible for training acceleration, but defaults to CPU which works fine)
  * RAM: 8 GB minimum

---

## 🛠️ Step-by-Step Installation

### 1. Configure Virtual Environment
It is highly recommended to isolate dependencies in a virtual environment:
```bash
# Create venv
python3 -m venv venv

# Activate venv (macOS/Linux)
source venv/bin/activate

# Activate venv (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies
Install all required libraries, including PyTorch, torchvision, OpenCV, and Streamlit:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Run Instructions

### 1. Camera Calibration
To run camera intrinsic calibration, extract the checkerboard corner coordinates, and compute distortion coefficients:
```bash
python calibration/scripts/camera_calibration.py
```
This saves:
* `calibration/output/camera_matrix.npy` (Intrinsic parameter matrix `K`)
* `calibration/output/dist_coeffs.npy` (Distortion parameter vector `D`)
* `calibration/output/reprojection_error.json` (Error details)

### 2. Model Training
To train the Mask R-CNN segmentation model on the COCO dataset splits:
```bash
python models/train.py --epochs 80 --batch_size 2 --lr 0.005
```
This fits the model on train images, runs early stopping based on val loss, and saves:
* `models/weights/best.pt` (Best validation model state dict)
* `models/weights/last.pt` (Last trained epoch state dict)

### 3. Model Sizing (Inference CLI)
Run metrology prediction on an image using the CLI demo:
```bash
python end_to_end_demo.py --image dataset/splits/val/images/screw_006.jpg --distance 300.0
```

### 4. Interactive Web App
Launch the Streamlit graphical user interface:
```bash
streamlit run app.py
```

### 5. Verification Test Suite
Run the test suite to verify module integration and correctness:
```bash
python -m pytest tests/
```

---

## 🔍 Troubleshooting & Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'torch'`
* **Cause:** torch is not installed in the active environment.
* **Solution:** Verify the virtual environment is activated (`source venv/bin/activate`) and run `pip install -r requirements.txt`.

### Issue 2: `FileNotFoundError: Model weights not found at best.pt`
* **Cause:** You tried to run inference before training the model.
* **Solution:** Run training first using `python models/train.py` to produce the `best.pt` checkpoint inside `models/weights/`.
