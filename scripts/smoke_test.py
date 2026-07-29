"""Run an end-to-end synthetic smoke workflow with truthful result labels."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from keyvision.config import load_config
from keyvision.data.io import load_manifest
from keyvision.data.synthetic import generate_dataset, generate_normal_images
from keyvision.data.validation import validate_dataset
from keyvision.deployment.export_onnx import export_detector
from keyvision.inference.predictor import DetectorPredictor
from keyvision.models.anomaly import GaussianTemplateAnomalyDetector
from keyvision.training.train import train
from keyvision.utils.runtime import write_json


def main() -> None:
    """Execute data, training, inference, anomaly, and ONNX smoke checks."""

    config = load_config("configs/smoke.yaml")
    splits = generate_dataset(config.data.root, count=42, seed=config.training.seed)
    root = Path(config.data.root)
    issues = validate_dataset(root, root / "manifest.jsonl")
    if issues:
        raise RuntimeError(f"Synthetic dataset validation failed: {issues}")
    training_summary = train(config)
    checkpoint = Path(config.training.output_dir) / "best.pt"
    records = load_manifest(root / config.data.test_manifest)
    sample_path = root / records[0].image
    sample = Image.open(sample_path).convert("RGB")
    prediction = DetectorPredictor(config, checkpoint).predict(sample)

    normal_paths = generate_normal_images("artifacts/anomaly_normal", count=6)
    anomaly = GaussianTemplateAnomalyDetector()
    anomaly.fit([Image.open(path).convert("RGB") for path in normal_paths])
    anomaly_prediction = anomaly.predict(sample)
    anomaly.save("artifacts/models/anomaly_template.npz")

    onnx_report = export_detector(
        "configs/smoke.yaml",
        checkpoint,
        "artifacts/models/keyvision_tiny.onnx",
        verify=True,
    )
    report = {
        "scope": "synthetic smoke test; not production performance",
        "splits": splits,
        "dataset_issues": len(issues),
        "training": training_summary,
        "sample_prediction_count": len(prediction.detections),
        "sample_latency_ms": prediction.latency_ms,
        "anomaly_score": anomaly_prediction.score,
        "onnx": onnx_report,
    }
    write_json("artifacts/smoke_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
