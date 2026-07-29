"""Evaluate a checkpoint against a JSONL manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from keyvision.config import load_config
from keyvision.data.io import load_manifest
from keyvision.evaluation.confusion import detection_confusion_matrix
from keyvision.evaluation.metrics import evaluate_detections
from keyvision.evaluation.reporting import export_evaluation
from keyvision.inference.predictor import DetectorPredictor
from keyvision.types import Detection


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.data.root)
    manifest_name = getattr(config.data, f"{args.split}_manifest")
    records = load_manifest(root / manifest_name)
    predictor = DetectorPredictor(config, args.checkpoint)
    predictions = []
    ground_truth = []
    latencies = []
    for record in records:
        image = Image.open(root / record.image).convert("RGB")
        prediction = predictor.predict(image)
        predictions.append(prediction.detections)
        if prediction.latency_ms is not None:
            latencies.append(prediction.latency_ms)
        ground_truth.append(
            [
                Detection(
                    bbox_xyxy=(x, y, x + width, y + height),
                    class_id=annotation.category_id,
                    class_name=annotation.category,
                )
                for annotation in record.annotations
                for x, y, width, height in [annotation.bbox]
            ]
        )
    results = evaluate_detections(predictions, ground_truth, config.model.class_names)
    results["confusion_matrix"] = detection_confusion_matrix(
        predictions, ground_truth, len(config.model.class_names)
    )
    results["result_scope"] = (
        "synthetic smoke test; not representative of production performance"
        if any(record.synthetic for record in records)
        else "user-provided dataset"
    )
    results["images"] = len(records)
    results["latency_ms_mean"] = sum(latencies) / len(latencies) if latencies else None
    results["fps_from_mean_latency"] = (
        1000.0 / results["latency_ms_mean"] if results["latency_ms_mean"] else None
    )
    paths = export_evaluation(results, args.output)
    print(f"Evaluation complete: {results['result_scope']}; outputs={paths}")


if __name__ == "__main__":
    main()
