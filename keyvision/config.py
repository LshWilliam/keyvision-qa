"""Typed YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths and image preprocessing settings."""

    root: str = "artifacts/synthetic"
    train_manifest: str = "splits/train.jsonl"
    val_manifest: str = "splits/val.jsonl"
    test_manifest: str = "splits/test.jsonl"
    image_size: int = 320
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """Known-defect detector configuration."""

    architecture: str = "tiny"
    num_classes: int = 6
    class_names: tuple[str, ...] = (
        "missing_keycap",
        "misaligned_keycap",
        "print_defect",
        "stain",
        "scratch",
        "foreign_object",
    )
    score_threshold: float = 0.35
    pretrained: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and reproducibility settings."""

    seed: int = 42
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device: str = "auto"
    output_dir: str = "artifacts/runs/default"
    resume: str | None = None


@dataclass(frozen=True)
class ProjectConfig:
    """Complete project configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Load a project YAML file into validated dataclasses."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration must be a mapping")

    data = DataConfig(**_section(raw, "data"))
    model_raw = _section(raw, "model")
    if "class_names" in model_raw:
        model_raw["class_names"] = tuple(model_raw["class_names"])
    model = ModelConfig(**model_raw)
    training = TrainingConfig(**_section(raw, "training"))

    if data.image_size < 32:
        raise ValueError("data.image_size must be at least 32")
    if model.num_classes != len(model.class_names):
        raise ValueError("model.num_classes must equal the number of class_names")
    if not 0.0 <= model.score_threshold <= 1.0:
        raise ValueError("model.score_threshold must be in [0, 1]")
    if training.epochs < 1 or training.batch_size < 1:
        raise ValueError("training.epochs and batch_size must be positive")
    return ProjectConfig(data=data, model=model, training=training)
