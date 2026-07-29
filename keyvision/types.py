"""Shared inference and annotation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One predicted or ground-truth defect in absolute pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    score: float = 1.0


@dataclass
class Prediction:
    """Normalized output shared by all known-defect detector backends."""

    detections: list[Detection] = field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyPrediction:
    """Output from an unknown-anomaly detector."""

    score: float
    heatmap: np.ndarray
    threshold: float
