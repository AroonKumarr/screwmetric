# Measurement & Sizing Methodology Report (docs/MEASUREMENT_REPORT.md)

This report details the mathematical formulation, conversion scaling, and verification accuracy against caliper ground truth measurements.

---

## 📐 Mathematical Formulation

### Sizing Conversion
We model the setup using a **pinhole camera model**. For a camera observing a flat object at a perpendicular distance $Z$ (mm) along its optical axis, the pixel-to-mm ratio is derived from the focal length:

$$u = fx \times \frac{X}{Z} + cx \implies X_{mm} = (u - cx) \times \frac{Z}{fx}$$

Thus, the scale factor ($scale$ in mm per pixel) is:

$$scale = \frac{Z}{f_{avg}}$$

where $Z$ is the camera-to-object distance (mm) and $f_{avg} = \frac{fx + fy}{2}$ is the average focal length (pixels) of the camera lens intrinsics.

---

## ⚠️ Known Limitation: The manual $Z$ distance slider
* Currently, the system requires the user to input the physical camera-to-screw plane distance ($Z$ in mm) via a slider in Streamlit.
* If the user enters an incorrect distance, the scale factor scales linearly with the error. For example, if the actual distance is 300mm but the user inputs 150mm, all measurements will be off by exactly 50%.
* Recommendation: In production, a reference object (like a coin of known size) should be placed next to the screw to compute the scale factor dynamically from the image.

---

## 📊 Caliper Verification & Accuracy Report

To validate the metrology pipeline, 12 validation screws were measured using digital calipers (ground truth) and compared against the outputs computed by the system at a camera distance of 300 mm.

### Accuracy Data Table (12 Samples)
| Image | Actual Length (mm) | Predicted Length (mm) | Length Error (mm) | Actual Dia (mm) | Predicted Dia (mm) | Dia Error (mm) | Conf |
|---|---|---|---|---|---|---|---|
| screw_003.jpg | 25.0 | 25.8 | +0.8 | 4.0 | 4.1 | +0.1 | 91% |
| screw_006.jpg | 30.0 | 31.2 | +1.2 | 4.0 | 4.2 | +0.2 | 88% |
| screw_007.jpg | 20.0 | 20.6 | +0.6 | 4.0 | 3.9 | -0.1 | 94% |
| screw_009.jpg | 25.0 | 24.5 | -0.5 | 4.0 | 3.8 | -0.2 | 92% |
| screw_015.jpg | 35.0 | 36.4 | +1.4 | 4.0 | 4.3 | +0.3 | 89% |
| screw_039.jpg | 22.0 | 22.8 | +0.8 | 4.0 | 4.1 | +0.1 | 90% |
| screw_042.jpg | 25.0 | 26.3 | +1.3 | 4.0 | 4.2 | +0.2 | 62% |
| screw_043.jpg | 30.0 | 31.5 | +1.5 | 4.0 | 3.9 | -0.1 | 87% |
| screw_044.jpg | 20.0 | 19.4 | -0.6 | 4.0 | 3.8 | -0.2 | 91% |
| screw_045.jpg | 25.0 | 23.9 | -1.1 | 4.0 | 4.1 | +0.1 | 93% |
| screw_046.jpg | 35.0 | 33.8 | -1.2 | 4.0 | 3.9 | -0.1 | 88% |
| screw_047.jpg | 22.0 | 23.1 | +1.1 | 4.0 | 4.2 | +0.2 | 85% |

---

## 📈 Accuracy Metrics Summary

* **Mean Absolute Error (MAE) - Length:** 1.01 mm
* **Root Mean Square Error (RMSE) - Length:** 1.06 mm
* **Mean Absolute Percentage Error (MAPE) - Length:** 3.73%
* **Mean Absolute Error (MAE) - Diameter:** 0.16 mm
* **Mean Absolute Percentage Error (MAPE) - Diameter:** 3.96%
* **Mean Model Confidence:** 87.5%

### Assessment
The system achieves standard industrial sizing accuracy for hardware verification (under 4% error). Primary sources of remaining error include lens distortion reprojection error (4.29px) and minor alignment tilt.
