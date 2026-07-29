"""ONNX Runtime inference for exportable KeyVision raw detector outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from keyvision.types import Detection, Prediction


class OnnxTinyPredictor:
    """Preprocess and decode the ONNX smoke detector on CPU or CUDA providers."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: tuple[str, ...],
        image_size: int,
        score_threshold: float,
        use_gpu: bool = False,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the 'deploy' extra for ONNX inference") from exc
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.class_names = class_names
        self.image_size = image_size
        self.score_threshold = score_threshold

    def predict(self, image: Image.Image) -> Prediction:
        """Run ONNX Runtime inference on one image."""

        import time

        source = image.convert("RGB")
        resized = source.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        started = time.perf_counter()
        raw = self.session.run(None, {"images": array})[0][0]
        latency_ms = (time.perf_counter() - started) * 1000
        box = 1.0 / (1.0 + np.exp(-raw[:4]))
        objectness = float(1.0 / (1.0 + np.exp(-raw[4])))
        class_logits = raw[5:] - np.max(raw[5:])
        probabilities = np.exp(class_logits) / np.exp(class_logits).sum()
        class_id = int(np.argmax(probabilities))
        score = objectness * float(probabilities[class_id])
        detections = []
        if score >= self.score_threshold:
            center_x, center_y, width, height = box
            detections.append(
                Detection(
                    bbox_xyxy=(
                        float((center_x - width / 2) * source.width),
                        float((center_y - height / 2) * source.height),
                        float((center_x + width / 2) * source.width),
                        float((center_y + height / 2) * source.height),
                    ),
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    score=score,
                )
            )
        return Prediction(
            detections=detections,
            latency_ms=latency_ms,
            metadata={"provider": self.session.get_providers()[0]},
        )
