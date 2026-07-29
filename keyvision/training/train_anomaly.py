"""Fit the normal-only anomaly baseline from an image folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from keyvision.models.anomaly import GaussianTemplateAnomalyDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--output", default="artifacts/models/anomaly_template.npz")
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()
    root = Path(args.normal_dir)
    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < 2:
        raise ValueError("At least two normal images are required")
    images = [Image.open(path).convert("RGB") for path in paths]
    detector = GaussianTemplateAnomalyDetector(threshold=args.threshold)
    detector.fit(images)
    detector.save(args.output)
    print(f"Fitted anomaly template from {len(images)} normal image(s): {args.output}")


if __name__ == "__main__":
    main()
