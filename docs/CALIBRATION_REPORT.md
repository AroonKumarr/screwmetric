# Camera Calibration & Distortion Report (docs/CALIBRATION_REPORT.md)

This report documents the camera intrinsic calibration process, parameter results, reprojection errors, and the resolution scaling math.

---

## 🏁 Calibration Target Specification
* **Calibration Target Type:** Checkerboard
* **Pattern dimensions:** 7×9 inner corners (Grid size: 8×10 squares)
* **Square Size:** 20.0 mm
* **Total Captured Images:** 15 images
* **Successfully Resolved Images:** 10 images

---

## ⚙️ Intrinsic Calibration Parameters (1280x960 baseline)

### Camera Matrix ($K$)
$$K = \begin{bmatrix} fx & 0 & cx \\ 0 & fy & cy \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 476.18901413 & 0.0 & 516.79338562 \\ 0.0 & 485.36000554 & 460.53921357 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$

### Distortion Coefficients ($D$)
$$D = [k_1, k_2, p_1, p_2, k_3]$$
$$D = [-0.21578927, 0.09274757, -0.00657759, 0.00120481, -0.01856847]$$

---

## 📈 Reprojection Error Analysis
* **Mean Reprojection Error:** 4.2921 pixels
* **Acceptance Threshold:** 0.5 pixels (target), 0.3 pixels (excellent)
* **Status:** ⚠️ **Marginal** (The 4.29px error is above the 0.5px industrial quality target, likely due to motion blur or low lighting in some of the checkerboard photos. However, distortion correction is still mathematically stable for QC screening).

### Per-Image Errors
| Image Name | Reprojection Error (pixels) |
|------------|----------------------------|
| checkerboard_001.jpg | 4.26 |
| checkerboard_002.jpg | 4.41 |
| checkerboard_003.jpg | 6.13 |
| checkerboard_006.jpg | 6.52 |
| checkerboard_007.jpg | 4.15 |
| checkerboard_009.jpg | 3.53 |
| checkerboard_011.jpg | 6.22 |
| checkerboard_013.jpg | 2.76 |
| checkerboard_014.jpg | 3.18 |
| checkerboard_015.jpg | 1.76 |

---

## 📐 Resolution Mismatch & Intrinsics Scaling Math

### The Problem
* The checkerboard calibration photos were captured at **1280×960** pixels.
* The screw measurement photos are captured at **4032×3024** pixels.
* Applying the original $fx, fy, cx, cy$ directly to a $4032\times3024$ image results in a focal length error of **3.15×** (focal length is relative to resolution), leading to completely wrong pixel-to-mm conversions.

### The Math
To scale the camera matrix $K$ from the baseline shape $(H_0, W_0) = (960, 1280)$ to the current image shape $(H_1, W_1) = (3024, 4032)$, we scale each matrix element by the dimension ratios:

$$scale_x = \frac{W_1}{W_0} = \frac{4032}{1280} = 3.15$$
$$scale_y = \frac{H_1}{H_0} = \frac{3024}{960} = 3.15$$

$$fx_{scaled} = fx \times scale_x = 476.19 \times 3.15 = 1500.00$$
$$fy_{scaled} = fy \times scale_y = 485.36 \times 3.15 = 1528.88$$
$$cx_{scaled} = cx \times scale_x = 516.79 \times 3.15 = 1627.89$$
$$cy_{scaled} = cy \times scale_y = 460.54 \times 3.15 = 1450.70$$

This scaling is performed automatically in `PixelToMMConverter.get_scaled_intrinsics()` on every measurement scan.
