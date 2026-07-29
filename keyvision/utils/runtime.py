"""Runtime utilities for deterministic experiments and diagnostics."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a compact project-wide log format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable smoke experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when the installed PyTorch build supports it."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot access CUDA")
    return torch.device(requested)


def environment_report() -> dict[str, Any]:
    """Return serializable environment details without personal paths."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: str | Path, payload: Any) -> None:
    """Write stable, human-readable JSON and create parent directories."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
