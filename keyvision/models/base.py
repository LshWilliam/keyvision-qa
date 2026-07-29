"""Unified interface for known-defect detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class DefectDetector(nn.Module, ABC):
    """Backend-neutral training and inference contract."""

    class_names: tuple[str, ...]

    @abstractmethod
    def compute_loss(
        self, images: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        """Return differentiable named losses for a training batch."""

    @abstractmethod
    def predict_tensors(
        self, images: list[Tensor], score_threshold: float
    ) -> list[dict[str, Tensor]]:
        """Return boxes, zero-based labels, and scores for each input tensor."""


def count_parameters(model: nn.Module) -> int:
    """Count trainable and frozen model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())
