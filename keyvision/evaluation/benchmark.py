"""Reproducible inference latency and throughput benchmarking."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

from keyvision.models.base import DefectDetector, count_parameters


@torch.inference_mode()
def benchmark_model(
    model: DefectDetector,
    image: torch.Tensor,
    score_threshold: float,
    warmup: int = 3,
    repetitions: int = 20,
) -> dict[str, Any]:
    """Measure single-image wall latency on the model's current device."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    model.eval()
    device = next(model.parameters()).device
    sample = image.to(device)
    for _ in range(warmup):
        model.predict_tensors([sample], score_threshold)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latencies = []
    for _ in range(repetitions):
        started = time.perf_counter()
        model.predict_tensors([sample], score_threshold)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
    median = statistics.median(latencies)
    return {
        "device": str(device),
        "repetitions": repetitions,
        "latency_ms_median": median,
        "latency_ms_p95": sorted(latencies)[max(0, int(repetitions * 0.95) - 1)],
        "fps_from_median": 1000.0 / median if median > 0 else None,
        "parameters": count_parameters(model),
    }


def model_file_size(path: str | Path | None) -> int | None:
    """Return checkpoint bytes when a model file exists."""

    if path is None:
        return None
    model_path = Path(path)
    return model_path.stat().st_size if model_path.is_file() else None
