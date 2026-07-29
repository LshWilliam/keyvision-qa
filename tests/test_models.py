import torch

from keyvision.config import ModelConfig
from keyvision.models.base import DefectDetector
from keyvision.models.factory import build_detector


def test_tiny_model_implements_interface() -> None:
    model = build_detector(ModelConfig())
    assert isinstance(model, DefectDetector)
    image = torch.rand(3, 64, 64)
    target = {"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]), "labels": torch.tensor([1])}
    losses = model.compute_loss([image], [target])
    assert set(losses) == {"objectness", "box", "class"}
    assert all(loss.ndim == 0 for loss in losses.values())
    outputs = model.predict_tensors([image], score_threshold=0.0)
    assert outputs[0]["boxes"].shape == (1, 4)
    assert outputs[0]["labels"].dtype == torch.int64
