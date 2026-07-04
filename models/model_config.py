"""
ScrewMetric — Model Configuration
====================================
Centralised, frozen configuration for training and inference.
All paths, hyperparameters, and tuneable settings live here.

Responsibility (Single Responsibility Principle):
    Only configuration. No I/O, no model code, no business logic.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger for the models module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Resolve the project root relative to this config file."""
    # model_config.py lives in models/ → parent == screwmetric/
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Auto-detect the best available compute device.

    Priority: CUDA → CPU.

    Note: MPS (Apple Silicon) is intentionally skipped because ultralytics
    8.4.x calls ``torch.amp.autocast('mps')`` which raises a RuntimeError in
    PyTorch ≤ 2.3.  Training on CPU produces identical results and avoids
    this upstream incompatibility.  Once ultralytics and PyTorch fully support
    MPS autocast this function can re-enable the MPS branch.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS is skipped: torch.amp.autocast('mps') unsupported in torch<=2.3
        # with ultralytics 8.4.x — fall through to CPU.
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and settings for the YOLOv8-seg training run.

    Attributes:
        model_type: Ultralytics model identifier.
        input_size: Square image size (pixels). Must be divisible by 32.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        patience: Early-stopping patience.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularisation weight.
        warmup_epochs: Epochs of LR warm-up.
        device: Compute device string.
        workers: DataLoader worker threads.
        augment: Enable Ultralytics built-in augmentations.
        save_period: Save a checkpoint every N epochs (-1 = disabled).
    """

    model_type: str = "maskrcnn"
    input_size: int = 640
    epochs: int = 100
    batch_size: int = 8
    patience: int = 30
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    device: str = field(default_factory=_detect_device)
    workers: int = 4
    augment: bool = True
    save_period: int = -1

    def __post_init__(self) -> None:
        valid_types = {
            "maskrcnn", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg",
            "yolov8l-seg", "yolov8x-seg",
        }
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {sorted(valid_types)}, "
                f"got '{self.model_type}'"
            )
        if self.input_size % 32 != 0:
            raise ValueError(
                f"input_size must be a multiple of 32, got {self.input_size}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceConfig:
    """Parameters controlling inference behaviour.

    Attributes:
        confidence_threshold: Minimum detection confidence to accept.
        iou_threshold: NMS IoU threshold.
        max_detections: Maximum detections per image.
        class_names: Ordered list of class names matching training labels.
        input_size: Inference image size (should match training size).
        device: Compute device for inference.
        half_precision: Use FP16 inference (CUDA only).
    """

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 10
    class_names: tuple[str, ...] = ("screw",)
    input_size: int = 640
    device: str = field(default_factory=_detect_device)
    half_precision: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1), "
                f"got {self.confidence_threshold}"
            )
        if not 0.0 < self.iou_threshold < 1.0:
            raise ValueError(
                f"iou_threshold must be in (0, 1), got {self.iou_threshold}"
            )
        if not self.class_names:
            raise ValueError("class_names must not be empty")


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPathConfig:
    """Filesystem paths for the models module.

    Attributes:
        project_root: Root of the entire ScrewMetric project.
    """

    project_root: Path = field(default_factory=_project_root)
    models_dir: Path = field(default=None)
    weights_dir: Path = field(default=None)
    best_weights_path: Path = field(default=None)
    last_weights_path: Path = field(default=None)
    logs_dir: Path = field(default=None)
    configs_dir: Path = field(default=None)
    dataset_yaml_path: Path = field(default=None)
    splits_dir: Path = field(default=None)
    calibration_output_dir: Path = field(default=None)
    camera_matrix_path: Path = field(default=None)
    dist_coeffs_path: Path = field(default=None)

    def __post_init__(self) -> None:
        p_root = self.project_root
        m_dir = self.models_dir if self.models_dir is not None else p_root / "models"
        w_dir = self.weights_dir if self.weights_dir is not None else m_dir / "weights"
        bw_path = self.best_weights_path if self.best_weights_path is not None else w_dir / "best.pt"
        lw_path = self.last_weights_path if self.last_weights_path is not None else w_dir / "last.pt"
        l_dir = self.logs_dir if self.logs_dir is not None else m_dir / "logs"
        c_dir = self.configs_dir if self.configs_dir is not None else m_dir / "configs"
        dy_path = self.dataset_yaml_path if self.dataset_yaml_path is not None else c_dir / "dataset.yaml"
        s_dir = self.splits_dir if self.splits_dir is not None else p_root / "dataset" / "splits"
        co_dir = self.calibration_output_dir if self.calibration_output_dir is not None else p_root / "calibration" / "output"
        cm_path = self.camera_matrix_path if self.camera_matrix_path is not None else co_dir / "camera_matrix.npy"
        dc_path = self.dist_coeffs_path if self.dist_coeffs_path is not None else co_dir / "dist_coeffs.npy"

        # Bypass frozen constraints to populate fields
        object.__setattr__(self, "models_dir", m_dir)
        object.__setattr__(self, "weights_dir", w_dir)
        object.__setattr__(self, "best_weights_path", bw_path)
        object.__setattr__(self, "last_weights_path", lw_path)
        object.__setattr__(self, "logs_dir", l_dir)
        object.__setattr__(self, "configs_dir", c_dir)
        object.__setattr__(self, "dataset_yaml_path", dy_path)
        object.__setattr__(self, "splits_dir", s_dir)
        object.__setattr__(self, "calibration_output_dir", co_dir)
        object.__setattr__(self, "camera_matrix_path", cm_path)
        object.__setattr__(self, "dist_coeffs_path", dc_path)


# ---------------------------------------------------------------------------
# Top-level aggregate configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Aggregates all sub-configurations for the models module.

    Attributes:
        paths: Filesystem path configuration.
        training: Training hyperparameters.
        inference: Inference settings.
    """

    paths: ModelPathConfig = field(default_factory=ModelPathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def default(cls) -> "ModelConfig":
        """Return a config with all default values."""
        return cls(
            paths=ModelPathConfig(),
            training=TrainingConfig(),
            inference=InferenceConfig(),
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate configuration loading and print all key settings."""
    print("=" * 64)
    print("  ScrewMetric — Model Configuration Module")
    print("=" * 64)

    try:
        cfg = ModelConfig.default()

        print("\n[TrainingConfig]")
        print(f"  model_type       : {cfg.training.model_type}")
        print(f"  input_size       : {cfg.training.input_size}")
        print(f"  epochs           : {cfg.training.epochs}")
        print(f"  batch_size       : {cfg.training.batch_size}")
        print(f"  patience         : {cfg.training.patience}")
        print(f"  device           : {cfg.training.device}")
        print(f"  augment          : {cfg.training.augment}")

        print("\n[InferenceConfig]")
        print(f"  confidence_threshold : {cfg.inference.confidence_threshold}")
        print(f"  iou_threshold        : {cfg.inference.iou_threshold}")
        print(f"  class_names          : {cfg.inference.class_names}")
        print(f"  device               : {cfg.inference.device}")

        print("\n[ModelPathConfig]")
        print(f"  project_root     : {cfg.paths.project_root}")
        print(f"  weights_dir      : {cfg.paths.weights_dir}")
        print(f"  best_weights     : {cfg.paths.best_weights_path}")
        print(f"  dataset_yaml     : {cfg.paths.dataset_yaml_path}")
        print(f"  camera_matrix    : {cfg.paths.camera_matrix_path}")
        print(f"  dist_coeffs      : {cfg.paths.dist_coeffs_path}")

        # Validate bad model type raises
        try:
            TrainingConfig(model_type="invalid-model")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

        print("\n✅ model_config.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ model_config.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
