"""High-level image inference with portable outputs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from keyvision.config import ProjectConfig
from keyvision.models.factory import build_detector
from keyvision.training.checkpoint import load_checkpoint
from keyvision.types import Detection, Prediction
from keyvision.utils.runtime import resolve_device


class DetectorPredictor:
    """Preprocess images, run a detector, and map boxes back to source pixels."""

    def __init__(self, config: ProjectConfig, checkpoint: str | Path | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.training.device)
        self.model = build_detector(config.model).to(self.device)
        if checkpoint is not None:
            load_checkpoint(checkpoint, self.model, device=self.device)
        self.model.eval()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            (self.config.data.image_size, self.config.data.image_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).to(self.device)

    def predict(self, image: Image.Image) -> Prediction:
        """Run inference on one PIL image."""

        source = image.convert("RGB")
        tensor = self._preprocess(source)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.model.predict_tensors([tensor], self.config.model.score_threshold)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        scale_x = source.width / self.config.data.image_size
        scale_y = source.height / self.config.data.image_size
        detections = []
        for box, label, score in zip(
            output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu(), strict=True
        ):
            class_id = int(label)
            detections.append(
                Detection(
                    bbox_xyxy=(
                        float(box[0] * scale_x),
                        float(box[1] * scale_y),
                        float(box[2] * scale_x),
                        float(box[3] * scale_y),
                    ),
                    class_id=class_id,
                    class_name=self.config.model.class_names[class_id],
                    score=float(score),
                )
            )
        return Prediction(
            detections=detections,
            latency_ms=latency_ms,
            metadata={"device": str(self.device), "architecture": self.config.model.architecture},
        )
