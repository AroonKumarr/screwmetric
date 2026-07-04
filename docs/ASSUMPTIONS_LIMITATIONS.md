# Assumptions & Limitations (docs/ASSUMPTIONS_LIMITATIONS.md)

This document outlines the underlying assumptions, operational limitations, and recommendations for improvement in the ScrewMetric metrology system.

---

## 📋 Underlying Assumptions

1. **Fronto-Parallel Plane:**
   * We assume the screw lies in a plane that is strictly perpendicular (fronto-parallel) to the camera's optical axis.
   * Any tilt (out-of-plane rotation) causes foreshortening, reducing the measured length by a factor of $\cos(\theta)$.
2. **Known Distance ($Z$):**
   * Sizing calculations assume the distance $Z$ from the camera lens to the screw plane is known exactly.
   * Any error in depth measurement propagates linearly into the sizing calculation.
3. **Fronto-Flat Calibration Grid:**
   * The calibration checkerboard was assumed flat. Surface deviations in the grid target contribute to reprojection errors.

---

## 🚫 Operational Limitations

1. **Reprojection Error (4.29px):**
   * The camera calibration has a mean reprojection error of 4.29 pixels, which is above the 0.5-pixel standard. This reduces sub-pixel coordinate alignment precision.
2. **Handheld Occlusion:**
   * Holding screws in the fingers during imaging occludes part of the silhouette, distorting the fitted rotated bounding box.
3. **Color Contrast Sensitivity:**
   * Highly reflective or metallic screws under bright sunlight can cause highlights that degrade segmentation mask boundary precision.
   * Dark or oxidized screws on hand skin have lower contrast, reducing YOLO/Mask R-CNN score levels.

---

## 🚀 Recommendations for Production Use

1. **Fiducial Sizing Target:**
   * Place a reference object of known size (e.g. an ArUco marker or circular coin) next to the screw in the same plane. This allows computing the scale factor dynamically without requiring a manual distance input.
2. **High-Accuracy Calibration Target:**
   * Capture 30+ calibration images using a high-contrast ChArUco board under uniform diffused lighting to bring the reprojection error below 0.3 pixels.
3. **Rigid Fixture Stand:**
   * Mount the camera on a rigid vertical fixture stand at a fixed distance from a flat inspection table to guarantee parallel alignment and constant depth.
