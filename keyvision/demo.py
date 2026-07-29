"""Gradio application factory for known and unknown defect inspection."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from PIL import Image

from keyvision.config import load_config
from keyvision.data.synthetic import generate_normal_images
from keyvision.inference.predictor import DetectorPredictor
from keyvision.inference.visualization import draw_prediction, overlay_heatmap
from keyvision.models.anomaly import GaussianTemplateAnomalyDetector


class DemoService:
    """Hold lazily prepared models for Gradio callbacks."""

    def __init__(self, config_path: str, checkpoint: str | None) -> None:
        self.config = load_config(config_path)
        self.detector = DetectorPredictor(self.config, checkpoint)
        self.checkpoint = checkpoint
        self.anomaly = GaussianTemplateAnomalyDetector()
        normal_paths = generate_normal_images("artifacts/demo_normal", count=6)
        normal_images = [Image.open(path).convert("RGB") for path in normal_paths]
        self.anomaly.fit(normal_images)

    def inspect(
        self, image: Image.Image | np.ndarray | None, mode: str
    ) -> tuple[Image.Image | None, dict[str, Any]]:
        """Run the selected branch and return a visualization plus metadata."""

        if image is None:
            return None, {"error": "Upload an image first."}
        source = image if isinstance(image, Image.Image) else Image.fromarray(image)
        if mode == "Known defect detector":
            prediction = self.detector.predict(source)
            details = {
                "branch": "known defect detector",
                "architecture": self.config.model.architecture,
                "checkpoint_loaded": self.checkpoint is not None,
                "latency_ms": round(prediction.latency_ms or 0.0, 3),
                "detections": [
                    {
                        "class": item.class_name,
                        "confidence": round(item.score, 4),
                        "bbox_xyxy": [round(value, 1) for value in item.bbox_xyxy],
                    }
                    for item in prediction.detections
                ],
            }
            return draw_prediction(source, prediction), details
        started = time.perf_counter()
        anomaly_prediction = self.anomaly.predict(source)
        latency_ms = (time.perf_counter() - started) * 1000
        details = {
            "branch": "unknown anomaly detector",
            "method": "fixture-aligned Gaussian template",
            "anomaly_score": round(anomaly_prediction.score, 4),
            "threshold": anomaly_prediction.threshold,
            "is_anomalous": anomaly_prediction.score >= anomaly_prediction.threshold,
            "latency_ms": round(latency_ms, 3),
            "reference_data": "generated synthetic normal examples",
        }
        return overlay_heatmap(source, anomaly_prediction.heatmap), details


def build_demo(config_path: str | None = None, checkpoint: str | None = None) -> Any:
    """Create a local Gradio Blocks application without launching it."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the 'demo' extra to launch the Gradio app") from exc
    resolved_config = config_path or os.environ.get("KEYVISION_CONFIG") or "configs/smoke.yaml"
    resolved_checkpoint = checkpoint or os.getenv("KEYVISION_CHECKPOINT")
    service = DemoService(resolved_config, resolved_checkpoint)
    warning = (
        "A trained checkpoint is loaded."
        if resolved_checkpoint
        else "No checkpoint is loaded; known-defect output uses random smoke-model weights."
    )
    with gr.Blocks(title="KeyVision-QA") as demo:
        gr.Markdown(
            "# KeyVision-QA\n"
            "Known defect detection and unknown anomaly localization for keyboard inspection.\n\n"
            f"> **Demo status:** {warning} Synthetic references are visibly watermarked and "
            "are not production data."
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Keyboard image", sources=["upload", "webcam"])
            output_image = gr.Image(type="pil", label="Inspection result")
        mode = gr.Radio(
            ["Known defect detector", "Unknown anomaly detector"],
            value="Known defect detector",
            label="Inspection branch",
        )
        run_button = gr.Button("Inspect", variant="primary")
        metadata = gr.JSON(label="Inference details")
        run_button.click(
            fn=service.inspect,
            inputs=[input_image, mode],
            outputs=[output_image, metadata],
            api_name="inspect",
        )
    return demo


def launch() -> None:
    """Launch the local application."""

    build_demo().launch()
