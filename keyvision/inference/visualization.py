"""Visualize detections and anomaly heatmaps without OpenCV."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from keyvision.types import Prediction


def draw_prediction(image: Image.Image, prediction: Prediction) -> Image.Image:
    """Draw labeled boxes on a copy of the source image."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for detection in prediction.detections:
        box = tuple(round(value) for value in detection.bbox_xyxy)
        draw.rectangle(box, outline=(30, 230, 120), width=3)
        label = f"{detection.class_name} {detection.score:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill=(15, 90, 55))
        draw.text((box[0], box[1]), label, fill="white")
    return output


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a red-yellow anomaly map over an RGB image."""

    normalized = np.clip(heatmap, 0.0, 1.0)
    colored = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    colored[..., 0] = np.uint8(normalized * 255)
    colored[..., 1] = np.uint8(np.sqrt(normalized) * 170)
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = source * (1 - alpha) + colored.astype(np.float32) * alpha
    blended_pixels: NDArray[np.uint8] = np.asarray(np.clip(blended, 0, 255), dtype=np.uint8)
    return Image.fromarray(blended_pixels)
