from typing import Any

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from keyvision.data.dataset import detection_collate
from keyvision.models.base import DefectDetector
from keyvision.training.train import evaluate_model


class SingleSampleDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    def __init__(self, sample: tuple[Tensor, dict[str, Tensor]]) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        if index != 0:
            raise IndexError(index)
        return self.sample


class FixedDetector(DefectDetector):
    class_names = ("defect",)

    def compute_loss(
        self, images: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        del images, targets
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    @torch.inference_mode()
    def predict_tensors(
        self, images: list[Tensor], score_threshold: float
    ) -> list[dict[str, Tensor]]:
        del score_threshold
        return [
            {
                "boxes": torch.tensor([[2.0, 2.0, 12.0, 12.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            }
            for _ in images
        ]


def test_evaluate_model_produces_validation_metrics() -> None:
    sample: tuple[Tensor, dict[str, Tensor]] = (
        torch.zeros((3, 16, 16)),
        {
            "boxes": torch.tensor([[2.0, 2.0, 12.0, 12.0]]),
            "labels": torch.tensor([0]),
            "image_id": torch.tensor(0),
        },
    )
    loader: DataLoader[Any] = DataLoader(
        SingleSampleDataset(sample), batch_size=1, collate_fn=detection_collate
    )
    metrics = evaluate_model(FixedDetector(), loader, torch.device("cpu"), ("defect",), 0.05)
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
