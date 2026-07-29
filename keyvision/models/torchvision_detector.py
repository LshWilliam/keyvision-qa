"""Production-oriented Torchvision Faster R-CNN adapter."""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from keyvision.models.base import DefectDetector


class TorchvisionFasterRCNNDetector(DefectDetector):
    """Wrap Faster R-CNN behind the same contract as the smoke detector."""

    def __init__(self, class_names: tuple[str, ...], pretrained: bool = False) -> None:
        super().__init__()
        self.class_names = class_names
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
        self.detector = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            weights_backbone=None if not pretrained else None,
        )
        predictor = self.detector.roi_heads.box_predictor
        if not isinstance(predictor, FastRCNNPredictor):
            raise TypeError("Unexpected Torchvision box predictor type")
        input_features = predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            input_features, len(class_names) + 1
        )

    def compute_loss(
        self, images: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        self.train()
        adjusted = []
        for target in targets:
            adjusted.append({**target, "labels": target["labels"] + 1})
        losses = self.detector(images, adjusted)
        if not isinstance(losses, dict):
            raise RuntimeError("Faster R-CNN did not return training losses")
        return losses

    @torch.inference_mode()
    def predict_tensors(
        self, images: list[Tensor], score_threshold: float
    ) -> list[dict[str, Tensor]]:
        self.eval()
        raw_outputs = self.detector(images)
        results = []
        for output in raw_outputs:
            keep = output["scores"] >= score_threshold
            results.append(
                {
                    "boxes": output["boxes"][keep],
                    "labels": output["labels"][keep] - 1,
                    "scores": output["scores"][keep],
                }
            )
        return results
