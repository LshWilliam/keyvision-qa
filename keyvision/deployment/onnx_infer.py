"""Run ONNX Runtime inference on one image."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from keyvision.config import load_config
from keyvision.deployment.onnx_runtime import OnnxTinyPredictor
from keyvision.inference.visualization import draw_prediction


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/onnx_prediction.png")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    predictor = OnnxTinyPredictor(
        args.model,
        config.model.class_names,
        config.data.image_size,
        config.model.score_threshold,
        use_gpu=args.gpu,
    )
    image = Image.open(args.input).convert("RGB")
    prediction = predictor.predict(image)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_prediction(image, prediction).save(output)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "latency_ms": prediction.latency_ms,
                "detections": [asdict(item) for item in prediction.detections],
                "metadata": prediction.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
