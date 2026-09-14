"""Model loading and inference API for use by scripts or other applications."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass(frozen=True)
class DetectionConfig:
    """Parameters shared by model loading and prediction."""

    weights: str | Path
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45


class InsulatorDetector:
    """Small reusable wrapper around an Ultralytics YOLO model."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = YOLO(str(config.weights))

    def predict(self, source: Any, *, save: bool = False, **overrides: Any):
        """Run detection while allowing per-call parameter overrides."""
        parameters = {
            "imgsz": self.config.image_size,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "device": "cpu",
            "save": save,
        }
        parameters.update(overrides)
        return self.model.predict(source=source, **parameters)
