# Dataset Card (docs/DATASET_CARD.md)

This dataset card describes the screw image dataset used to train the instance segmentation model.

---

## 🎯 Target Object
* **Object Class:** Screw
* **Details:** M4 / M5 nominal steel/iron standard hardware screws.
* **Geometry:** Slender cylindrical body with helical thread groove, flat head (countersunk or pan head).
* **Justification:** Chosen due to geometric consistency, ease of segmentation, clear thread contours for dimension extraction, and ease of acquiring verification measurements.

---

## 📸 Dataset Statistics
* **Total Collected Images:** 73 images
* **Image Format:** RGB JPEG
* **Resolution:** 4032 × 3024 pixels (typical smartphone camera capture)
* **Backgrounds:** Flat white desk, light grey paper sheet, minor hands/fingers holding the screw.
* **Number of Classes:** 1 class (`"screw"`)

---

## 🏷️ Annotation Process
* **Annotation Tool:** CVAT (Computer Vision Annotation Tool)
* **Label Type:** Polygon instance segmentation (bounding box + mask contour points)
* **Export Format:** COCO JSON format (`instances_default.json`)
* **Note on Corrupted File Fix:** A leading stray backtick character in the first line of `instances_default.json` was corrected and verified loadable.

---

## 📊 Dataset Splits & Splits Strategy
To prevent data leakage, the dataset was split using a fixed random seed (`seed=42`) into training, validation, and testing sets:

| Split | Percentage | Image Count | Purpose |
|-------|------------|-------------|---------|
| **Train** | 70% | 41 images | Fitting model parameters |
| **Val** | 20% | 12 images | Hyperparameter tuning and model checkpoints selection |
| **Test** | 10% | 5 images | Final generalization evaluation |

---

## ⚙️ Environmental Context
* **Lighting:** Indoor fluorescent office light, mixed ambient light.
* **Camera Distance:** Varied from 150mm to 400mm from the object.
* **Orientations:** Vertical, horizontal, and diagonal layouts relative to the image frame.
