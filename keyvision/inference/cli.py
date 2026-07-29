"""Run image, folder, or webcam inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from keyvision.config import load_config
from keyvision.inference.predictor import DetectorPredictor
from keyvision.inference.visualization import draw_prediction

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def infer_path(
    predictor: DetectorPredictor, input_path: Path, output_dir: Path
) -> list[dict[str, object]]:
    """Infer one image or all supported images in a directory."""

    paths = (
        [input_path]
        if input_path.is_file()
        else sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found at {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        prediction = predictor.predict(image)
        draw_prediction(image, prediction).save(output_dir / path.name)
        results.append(
            {
                "image": path.name,
                "latency_ms": prediction.latency_ms,
                "detections": [asdict(detection) for detection in prediction.detections],
            }
        )
    (output_dir / "predictions.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def webcam(predictor: DetectorPredictor, camera_index: int = 0) -> None:
    """Run a local real-time camera loop when OpenCV is installed."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the 'vision' extra for webcam inference") from exc
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            rendered = draw_prediction(image, predictor.predict(image))
            cv2.imshow(
                "KeyVision-QA (press q to quit)",
                cv2.cvtColor(np.asarray(rendered), cv2.COLOR_RGB2BGR),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    parser.add_argument("--output", default="artifacts/predictions")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    predictor = DetectorPredictor(load_config(args.config), args.checkpoint)
    if args.webcam:
        webcam(predictor, args.camera_index)
    elif args.input:
        results = infer_path(predictor, Path(args.input), Path(args.output))
        print(f"Inferred {len(results)} image(s); output={args.output}")
    else:
        parser.error("Provide --input or --webcam")


if __name__ == "__main__":
    main()
