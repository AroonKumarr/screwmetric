"""
ScrewMetric — Model Trainer
=============================
Orchestrates YOLOv8-seg training on the ScrewMetric COCO dataset.

Responsibility (Single Responsibility Principle):
    Dataset preparation, model training, and evaluation only.
    No inference, no measurement, no report generation.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

_MODELS_DIR = Path(__file__).resolve().parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from model_config import ModelConfig, get_logger  # type: ignore[import]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Summary of a completed training run.

    Attributes:
        best_weights_path: Absolute path to the saved best.pt checkpoint.
        epochs_trained: Number of epochs actually trained.
        training_time_s: Total wall-clock training time in seconds.
        map50: mAP@0.50 on the validation split.
        map50_95: mAP@0.50-0.95 on the validation split.
        success: Whether training completed without errors.
        message: Human-readable status message.
    """

    best_weights_path: Path
    epochs_trained: int
    training_time_s: float
    map50: float
    map50_95: float
    success: bool
    message: str = ""


@dataclass
class EvaluationResult:
    """Summary of a model evaluation run.

    Attributes:
        map50: mAP@0.50 on the evaluation split.
        map50_95: mAP@0.50-0.95 on the evaluation split.
        precision: Mean precision across all classes.
        recall: Mean recall across all classes.
        evaluation_time_s: Wall-clock evaluation time in seconds.
    """

    map50: float
    map50_95: float
    precision: float
    recall: float
    evaluation_time_s: float


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Prepares the YOLO dataset descriptor and trains a YOLOv8-seg model.

    Args:
        config: Top-level model configuration.

    Example::

        trainer = ModelTrainer(ModelConfig.default())
        yaml_path = trainer.prepare_dataset_yaml()
        result = trainer.train()
        print(result.map50)
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def convert_coco_to_yolo(self) -> None:
        """Convert split COCO JSON annotations into YOLOv8-seg text label files."""
        splits_dir = self._config.paths.splits_dir

        for split in ("train", "val", "test"):
            coco_path = splits_dir / split / "annotations" / f"instances_{split}.json"
            if not coco_path.exists():
                logger.warning("COCO annotation file not found for split: %s", coco_path)
                continue

            # Create labels directory parallel to images
            labels_dir = splits_dir / split / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            with open(coco_path, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

            # Map image_id to annotations list
            img_to_anns: dict[int, list[dict]] = {}
            for ann in coco_data.get("annotations", []):
                img_id = ann["image_id"]
                img_to_anns.setdefault(img_id, []).append(ann)

            # Convert each image
            converted_count = 0
            for img in coco_data.get("images", []):
                img_id = img["id"]
                file_name = img["file_name"]
                width = img["width"]
                height = img["height"]

                if width <= 0 or height <= 0:
                    continue

                txt_name = Path(file_name).with_suffix(".txt")
                txt_path = labels_dir / txt_name

                lines = []
                anns = img_to_anns.get(img_id, [])
                for ann in anns:
                    seg = ann.get("segmentation", [])
                    class_idx = 0  # Only one class: screw

                    if isinstance(seg, list):
                        for poly in seg:
                            if len(poly) < 6:
                                continue
                            # Normalize coordinates [0, 1]
                            norm_poly = []
                            for i, val in enumerate(poly):
                                if i % 2 == 0:
                                    norm_poly.append(val / width)
                                else:
                                    norm_poly.append(val / height)
                            line_str = f"{class_idx} " + " ".join(f"{v:.6f}" for v in norm_poly)
                            lines.append(line_str)

                # Write YOLO txt label file
                with open(txt_path, "w", encoding="utf-8") as out_f:
                    out_f.write("\n".join(lines) + "\n")
                converted_count += 1

            logger.info("Converted split '%s' (%d files) to YOLO label format.", split, converted_count)

    def prepare_dataset_yaml(self) -> Path:
        """Generate the YOLO-format dataset.yaml descriptor.

        Reads the split directory structure created by the dataset pipeline,
        converts COCO segmentations to YOLO labels, and writes the YAML file.

        Returns:
            Absolute path to the written dataset.yaml.

        Raises:
            FileNotFoundError: If split directories do not exist.
        """
        splits_dir = self._config.paths.splits_dir
        yaml_path = self._config.paths.dataset_yaml_path

        # Validate split directories exist
        for split in ("train", "val", "test"):
            images_dir = splits_dir / split / "images"
            if not images_dir.exists():
                logger.warning(
                    "Split images directory not found: %s  "
                    "(run dataset pipeline first)",
                    images_dir,
                )

        # Run automatic COCO to YOLO labels conversion
        self.convert_coco_to_yolo()

        # Build YAML content
        descriptor = {
            "path": str(splits_dir.resolve()),
            "train": str("train/images"),
            "val": str("val/images"),
            "test": str("test/images"),
            "nc": len(self._config.inference.class_names),
            "names": list(self._config.inference.class_names),
        }

        # Write to disk
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(descriptor, f, default_flow_style=False, sort_keys=False)

        logger.info("Dataset YAML written: %s", yaml_path)
        return yaml_path

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Train the YOLOv8-seg model on the prepared dataset.

        Auto-detects device (CUDA / MPS / CPU).
        Saves best checkpoint to ``models/weights/best.pt``.

        Returns:
            :class:`TrainingResult` with metrics and checkpoint path.

        Raises:
            ImportError: If Ultralytics is not installed.
            FileNotFoundError: If dataset.yaml does not exist.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        yaml_path = self._config.paths.dataset_yaml_path
        if not yaml_path.exists():
            yaml_path = self.prepare_dataset_yaml()

        cfg_t = self._config.training
        weights_dir = self._config.paths.weights_dir
        weights_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting training — model=%s  epochs=%d  device=%s",
            cfg_t.model_type, cfg_t.epochs, cfg_t.device,
        )

        model = YOLO(cfg_t.model_type)  # downloads nano weights on first run

        t0 = time.perf_counter()
        try:
            results = model.train(
                data=str(yaml_path),
                epochs=cfg_t.epochs,
                imgsz=cfg_t.input_size,
                batch=cfg_t.batch_size,
                patience=cfg_t.patience,
                lr0=cfg_t.learning_rate,
                weight_decay=cfg_t.weight_decay,
                warmup_epochs=cfg_t.warmup_epochs,
                device=cfg_t.device,
                workers=cfg_t.workers,
                augment=cfg_t.augment,
                save_period=cfg_t.save_period,
                project=str(self._config.paths.logs_dir),
                name="screwmetric_train",
                exist_ok=True,
                verbose=True,
            )
            elapsed = time.perf_counter() - t0

            # Ultralytics saves best.pt inside the project/name/ directory.
            # Copy it to our canonical weights path.
            train_best = (
                self._config.paths.logs_dir
                / "screwmetric_train"
                / "weights"
                / "best.pt"
            )
            if train_best.exists():
                import shutil
                shutil.copy2(train_best, self._config.paths.best_weights_path)
                logger.info("Best weights copied to: %s", self._config.paths.best_weights_path)

            # Extract metrics from results
            try:
                metrics = results.results_dict
                map50 = float(metrics.get("metrics/mAP50(M)", 0.0))
                map50_95 = float(metrics.get("metrics/mAP50-95(M)", 0.0))
            except Exception:
                map50, map50_95 = 0.0, 0.0

            return TrainingResult(
                best_weights_path=self._config.paths.best_weights_path,
                epochs_trained=int(results.epoch) if hasattr(results, "epoch") else cfg_t.epochs,
                training_time_s=round(elapsed, 2),
                map50=round(map50, 4),
                map50_95=round(map50_95, 4),
                success=True,
                message="Training completed successfully.",
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Training failed: %s", exc, exc_info=True)
            return TrainingResult(
                best_weights_path=self._config.paths.best_weights_path,
                epochs_trained=0,
                training_time_s=round(elapsed, 2),
                map50=0.0,
                map50_95=0.0,
                success=False,
                message=f"Training failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, split: str = "val") -> EvaluationResult:
        """Evaluate the trained model on a dataset split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            :class:`EvaluationResult` with mAP and P/R metrics.

        Raises:
            ImportError: If Ultralytics is not installed.
            FileNotFoundError: If the model weights are not found.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        weights_path = self._config.paths.best_weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. Train the model first."
            )

        model = YOLO(str(weights_path))
        cfg_t = self._config.training
        yaml_path = self._config.paths.dataset_yaml_path

        logger.info("Evaluating on '%s' split...", split)
        t0 = time.perf_counter()

        metrics = model.val(
            data=str(yaml_path),
            imgsz=cfg_t.input_size,
            device=cfg_t.device,
            split=split,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        try:
            d = metrics.results_dict
            map50 = float(d.get("metrics/mAP50(M)", 0.0))
            map50_95 = float(d.get("metrics/mAP50-95(M)", 0.0))
            precision = float(d.get("metrics/precision(M)", 0.0))
            recall = float(d.get("metrics/recall(M)", 0.0))
        except Exception:
            map50 = map50_95 = precision = recall = 0.0

        logger.info(
            "Evaluation — mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
            map50, map50_95, precision, recall,
        )

        return EvaluationResult(
            map50=round(map50, 4),
            map50_95=round(map50_95, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            evaluation_time_s=round(elapsed, 2),
        )


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the model trainer: prepare dataset.yaml and start training."""
    print("=" * 64)
    print("  ScrewMetric — Model Trainer Module")
    print("=" * 64)

    import argparse

    parser = argparse.ArgumentParser(description="ScrewMetric model trainer")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs (default: from config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model type (e.g. yolov8s-seg)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only generate dataset.yaml without training",
    )
    args = parser.parse_args()

    try:
        from model_config import ModelConfig, TrainingConfig, ModelPathConfig, InferenceConfig
        cfg = ModelConfig.default()

        # Apply CLI overrides
        if args.epochs or args.model:
            training_kwargs: dict = {}
            if args.epochs:
                training_kwargs["epochs"] = args.epochs
            if args.model:
                training_kwargs["model_type"] = args.model
            from dataclasses import replace
            new_training = replace(cfg.training, **training_kwargs)
            cfg = ModelConfig(
                paths=cfg.paths,
                training=new_training,
                inference=cfg.inference,
            )

        trainer = ModelTrainer(cfg)

        print("\n[Step 1] Preparing dataset YAML...")
        yaml_path = trainer.prepare_dataset_yaml()
        print(f"  → Written: {yaml_path}")

        with open(yaml_path) as f:
            content = yaml.safe_load(f)
        print(f"  Classes : {content['names']}")
        print(f"  nc      : {content['nc']}")

        if args.dry_run:
            print("\n[Dry run] Skipping training.")
            print("\n✅ model_trainer.py dry-run executed successfully.")
            return

        print(f"\n[Step 2] Training model ({cfg.training.model_type}) ...")
        print(f"  device  : {cfg.training.device}")
        print(f"  epochs  : {cfg.training.epochs}")
        print(f"  batch   : {cfg.training.batch_size}")
        print()

        result = trainer.train()

        print("\n[Training Result]")
        print(f"  success          : {result.success}")
        print(f"  epochs_trained   : {result.epochs_trained}")
        print(f"  training_time_s  : {result.training_time_s:.1f}s")
        print(f"  mAP50            : {result.map50:.4f}")
        print(f"  mAP50-95         : {result.map50_95:.4f}")
        print(f"  best_weights     : {result.best_weights_path}")

        if result.success:
            print("\n✅ model_trainer.py executed successfully.")
        else:
            print(f"\n⚠️  Training completed with issues: {result.message}")

    except Exception as exc:
        print(f"\n❌ model_trainer.py failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
