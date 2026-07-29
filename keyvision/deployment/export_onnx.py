"""Export the smoke detector to ONNX and verify numerical consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from keyvision.config import load_config
from keyvision.models.factory import build_detector
from keyvision.models.tiny_detector import TinyDefectDetector
from keyvision.training.checkpoint import load_checkpoint
from keyvision.utils.runtime import resolve_device


def export_detector(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    verify: bool = True,
) -> dict[str, Any]:
    """Export raw detector outputs and optionally compare ONNX Runtime."""

    config = load_config(config_path)
    if config.model.architecture != "tiny":
        raise ValueError(
            "The verified export path currently supports architecture='tiny'. "
            "Torchvision detection export requires deployment-specific NMS handling."
        )
    device = resolve_device(config.training.device)
    model = build_detector(config.model).to(device)
    if not isinstance(model, TinyDefectDetector):
        raise TypeError("Expected TinyDefectDetector for the portable export path")
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    sample = torch.rand(
        1,
        3,
        config.data.image_size,
        config.data.image_size,
        device=device,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        output,
        input_names=["images"],
        output_names=["raw_predictions"],
        dynamic_axes={"images": {0: "batch"}, "raw_predictions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report: dict[str, Any] = {
        "status": "exported",
        "architecture": config.model.architecture,
        "opset": 17,
        "output": output.as_posix(),
        "size_bytes": output.stat().st_size,
        "verified": False,
        "max_absolute_difference": None,
    }
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the 'deploy' extra to verify ONNX Runtime") from exc
        with torch.inference_mode():
            torch_output = model(sample).cpu().numpy()
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"images": sample.cpu().numpy()})[0]
        maximum_difference = float(np.max(np.abs(torch_output - onnx_output)))
        report["verified"] = bool(np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5))
        report["max_absolute_difference"] = maximum_difference
        if not report["verified"]:
            raise RuntimeError(f"ONNX consistency check failed: max_abs_diff={maximum_difference}")
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/models/keyvision_tiny.onnx")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    report = export_detector(
        args.config,
        args.checkpoint,
        args.output,
        verify=not args.skip_verify,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
