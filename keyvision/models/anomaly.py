"""Fixture-aligned Gaussian template baseline for unknown anomaly localization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from keyvision.types import AnomalyPrediction


def _feature_map(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    smoothed = resized.filter(ImageFilter.GaussianBlur(radius=1.2))
    rgb = np.asarray(smoothed, dtype=np.float32) / 255.0
    grayscale = rgb.mean(axis=2)
    gradient_y, gradient_x = np.gradient(grayscale)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)[..., None]
    return np.concatenate((rgb, gradient), axis=2)


class GaussianTemplateAnomalyDetector:
    """Model normal pixelwise appearance and local gradients with Gaussian statistics.

    This baseline is intentionally simple and interpretable. It works best for
    camera-fixture-aligned keyboards and is expected to degrade under pose shifts.
    """

    def __init__(self, image_size: tuple[int, int] = (160, 90), threshold: float = 3.0) -> None:
        self.image_size = image_size
        self.threshold = threshold
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, images: Sequence[Image.Image]) -> None:
        """Fit per-location normal feature statistics."""

        if len(images) < 2:
            raise ValueError("At least two normal images are required")
        features = np.stack([_feature_map(image, self.image_size) for image in images])
        self.mean = features.mean(axis=0)
        self.std = np.maximum(features.std(axis=0), 0.03)

    def predict(self, image: Image.Image) -> AnomalyPrediction:
        """Return a robust anomaly score and an image-resolution heatmap."""

        if self.mean is None or self.std is None:
            raise RuntimeError("Anomaly detector must be fitted or loaded before prediction")
        features = _feature_map(image, self.image_size)
        standardized = np.abs(features - self.mean) / self.std
        heatmap_small = np.sqrt(np.mean(standardized**2, axis=2))
        score = float(np.quantile(heatmap_small, 0.995))
        heatmap_image = Image.fromarray(
            np.uint8(np.clip(heatmap_small / max(self.threshold * 2, 1e-6), 0, 1) * 255)
        )
        heatmap = (
            np.asarray(
                heatmap_image.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32
            )
            / 255.0
        )
        return AnomalyPrediction(score=score, heatmap=heatmap, threshold=self.threshold)

    def save(self, path: str | Path) -> None:
        """Persist template statistics as a compressed NumPy archive."""

        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save an unfitted anomaly detector")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            mean=self.mean,
            std=self.std,
            image_size=np.asarray(self.image_size),
            threshold=np.asarray(self.threshold),
        )

    @classmethod
    def load(cls, path: str | Path) -> GaussianTemplateAnomalyDetector:
        """Restore a fitted detector."""

        payload = np.load(path)
        image_size = (
            int(payload["image_size"][0]),
            int(payload["image_size"][1]),
        )
        detector = cls(image_size=image_size, threshold=float(payload["threshold"]))
        detector.mean = payload["mean"]
        detector.std = payload["std"]
        return detector
