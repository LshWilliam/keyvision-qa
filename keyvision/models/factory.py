"""Detector factory with explicit production and smoke backends."""

from __future__ import annotations

from keyvision.config import ModelConfig
from keyvision.models.base import DefectDetector
from keyvision.models.tiny_detector import TinyDefectDetector
from keyvision.models.torchvision_detector import TorchvisionFasterRCNNDetector


def build_detector(config: ModelConfig) -> DefectDetector:
    """Construct the configured known-defect detector."""

    if config.architecture == "tiny":
        return TinyDefectDetector(config.class_names)
    if config.architecture == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return TorchvisionFasterRCNNDetector(config.class_names, config.pretrained)
    raise ValueError(f"Unsupported detector architecture: {config.architecture}")
