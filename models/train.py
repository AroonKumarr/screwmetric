"""
ScrewMetric — Model Training CLI Entry Point
=============================================
Thin wrapper that delegates to ModelTrainer in model_trainer.py.
Kept for backward compatibility.

Usage:
    python models/train.py --epochs 100
"""

from model_trainer import main  # type: ignore[import]

if __name__ == "__main__":
    main()
