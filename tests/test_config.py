from pathlib import Path

import pytest

from keyvision.config import load_config


def test_load_smoke_config() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.model.architecture == "tiny"
    assert config.model.num_classes == len(config.model.class_names)
    assert config.training.device == "cpu"


def test_rejects_invalid_class_count(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "model:\n  num_classes: 2\n  class_names: [only_one]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_classes"):
        load_config(path)
