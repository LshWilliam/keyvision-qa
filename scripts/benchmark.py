"""Benchmark PyTorch and ONNX Runtime on an identical synthetic sample."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from keyvision.config import load_config
from keyvision.data.dataset import KeyboardDefectDataset
from keyvision.data.io import load_manifest
from keyvision.evaluation.benchmark import benchmark_model, model_file_size
from keyvision.models.factory import build_detector
from keyvision.training.checkpoint import load_checkpoint
from keyvision.utils.runtime import environment_report, write_json


def _onnx_benchmark(
    model_path: Path,
    image: Image.Image,
    image_size: int,
    repetitions: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    for _ in range(3):
        session.run(None, {"images": array})
    latencies = []
    for _ in range(repetitions):
        started = time.perf_counter()
        session.run(None, {"images": array})
        latencies.append((time.perf_counter() - started) * 1000)
    median = statistics.median(latencies)
    return {
        "device": "cpu",
        "provider": session.get_providers()[0],
        "repetitions": repetitions,
        "latency_ms_median": median,
        "latency_ms_p95": sorted(latencies)[max(0, int(repetitions * 0.95) - 1)],
        "fps_from_median": 1000.0 / median,
        "model_size_bytes": model_path.stat().st_size,
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", default="artifacts/runs/smoke/best.pt")
    parser.add_argument("--onnx", default="artifacts/models/keyvision_tiny.onnx")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.data.root)
    records = load_manifest(root / config.data.test_manifest)
    source_image = Image.open(root / records[0].image).convert("RGB")
    dataset = KeyboardDefectDataset(
        root,
        root / config.data.test_manifest,
        config.data.image_size,
    )
    tensor, _ = dataset[0]
    model = build_detector(config.model)
    load_checkpoint(args.checkpoint, model)
    pytorch_result = benchmark_model(
        model,
        tensor,
        config.model.score_threshold,
        repetitions=args.repetitions,
    )
    pytorch_result["model_size_bytes"] = model_file_size(args.checkpoint)
    report = {
        "scope": "synthetic smoke benchmark; not production performance",
        "input_shape": [1, 3, config.data.image_size, config.data.image_size],
        "environment": environment_report(),
        "pytorch": pytorch_result,
        "onnx_runtime": _onnx_benchmark(
            Path(args.onnx),
            source_image,
            config.data.image_size,
            args.repetitions,
        ),
        "gpu_comparison": (
            "available but not executed by this CPU smoke config"
            if torch.cuda.is_available()
            else "unavailable in the installed PyTorch build"
        ),
    }
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
