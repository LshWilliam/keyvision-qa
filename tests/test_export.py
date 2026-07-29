from pathlib import Path

import torch

from keyvision.config import load_config
from keyvision.deployment.export_onnx import export_detector
from keyvision.models.factory import build_detector
from keyvision.training.checkpoint import save_checkpoint


def test_onnx_export_interface(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    model = build_detector(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, optimizer, 0, 1.0, {"scope": "unit test"})
    output = tmp_path / "model.onnx"
    report = export_detector("configs/smoke.yaml", checkpoint, output, verify=False)
    assert output.is_file()
    assert report["status"] == "exported"
    assert report["architecture"] == "tiny"
