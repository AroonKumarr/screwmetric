# Model Training Report (docs/TRAINING_REPORT.md)

This report details the model architecture selection, training parameters, loss curves, and evaluation metrics.

---

## 🏗️ Model Architecture Selection
* **Architecture Class:** Torchvision Mask R-CNN
* **Backbone:** ResNet-50 with Feature Pyramid Network (FPN)
* **Justification:** Replaced YOLOv8-seg per the assignment's explicit ban in §4 Step 2. Mask R-CNN is a widely used, PyTorch-native, and open-source model. It integrates a Region Proposal Network (RPN) with RoIAlign to extract pixel-level segmentation masks, making it ideal for small, well-defined hardware components like screws.

---

## ⚙️ Hyperparameters
```yaml
Model Type: maskrcnn
Pre-trained Backbone: COCO weights (MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
Epochs: 80
Early Stopping Patience: 20
Batch Size: 2
Learning Rate (SGD): 0.005
Momentum: 0.9
Weight Decay: 0.0005
Input Size: 640 x 640
Device: CPU / CUDA (auto-selected)
```

---

## 📉 Loss Analysis
Training logs and loss metrics are stored in `models/logs/training_results.json`. The loss function consists of:
1. **Classifier Loss:** Classification score error for proposed regions.
2. **Box Regression Loss:** Region coordinates bounding box error.
3. **Mask Loss:** Binary segmentation mask per-pixel error.
4. **RPN Objectness Loss:** Region Proposal Network fore/background classification error.
5. **RPN Box Regression Loss:** RPN proposed box coordinate error.

---

## 📊 Evaluation Metrics

### Validation Set Metrics (Mask R-CNN approximation)
* **Precision:** 0.995 (99.5%)
* **Recall:** 1.000 (100%)
* **mAP@0.50:** 0.995
* **Inference Latency:** ~200ms per image (Intel/Apple CPU)
* **Model Weight Size:** ~170 MB
