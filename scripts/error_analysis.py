"""Generate false-positive and false-negative artifacts for one split."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from keyvision.config import load_config
from keyvision.data.io import load_manifest
from keyvision.evaluation.error_analysis import save_error_cases
from keyvision.inference.predictor import DetectorPredictor
from keyvision.types import Detection


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", default="artifacts/error_analysis")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.data.root)
    records = load_manifest(root / getattr(config.data, f"{args.split}_manifest"))
    predictor = DetectorPredictor(config, args.checkpoint)
    images = [Image.open(root / record.image).convert("RGB") for record in records]
    predictions = [predictor.predict(image).detections for image in images]
    ground_truth = [
        [
            Detection(
                bbox_xyxy=(x, y, x + width, y + height),
                class_id=annotation.category_id,
                class_name=annotation.category,
            )
            for annotation in record.annotations
            for x, y, width, height in [annotation.bbox]
        ]
        for record in records
    ]
    report = save_error_cases(
        images,
        [record.image for record in records],
        predictions,
        ground_truth,
        args.output,
    )
    print(report)


if __name__ == "__main__":
    main()
