"""Small export-friendly detector used only for CI and synthetic smoke tests."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from keyvision.models.base import DefectDetector


class TinyDetectorCore(nn.Module):
    """Predict one normalized box, objectness logit, and class logits per image."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 5 + num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw predictions shaped ``[batch, 5 + classes]``."""

        return self.head(self.features(images).flatten(1))


class TinyDefectDetector(DefectDetector):
    """Trainable smoke detector; not intended as a production accuracy baseline."""

    def __init__(self, class_names: tuple[str, ...]) -> None:
        super().__init__()
        self.class_names = class_names
        self.core = TinyDetectorCore(len(class_names))

    def forward(self, images: Tensor) -> Tensor:
        """Expose exportable raw model outputs."""

        return self.core(images)

    def compute_loss(
        self, images: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        batch = torch.stack(images)
        raw = self.core(batch)
        objectness_targets = torch.tensor(
            [float(target["boxes"].shape[0] > 0) for target in targets],
            dtype=raw.dtype,
            device=raw.device,
        )
        objectness_loss = functional.binary_cross_entropy_with_logits(raw[:, 4], objectness_targets)

        positive_indices = [
            index for index, target in enumerate(targets) if target["boxes"].shape[0] > 0
        ]
        if not positive_indices:
            zero = raw.sum() * 0.0
            return {"objectness": objectness_loss, "box": zero, "class": zero}

        normalized_boxes = []
        labels = []
        height, width = batch.shape[-2:]
        for index in positive_indices:
            box = targets[index]["boxes"][0]
            x1, y1, x2, y2 = box.unbind()
            normalized_boxes.append(
                torch.stack(
                    (
                        (x1 + x2) / (2 * width),
                        (y1 + y2) / (2 * height),
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    )
                )
            )
            labels.append(targets[index]["labels"][0])
        index_tensor = torch.tensor(positive_indices, device=raw.device)
        predicted_boxes = raw[index_tensor, :4].sigmoid()
        box_targets = torch.stack(normalized_boxes).to(raw.device)
        class_targets = torch.stack(labels).to(raw.device)
        return {
            "objectness": objectness_loss,
            "box": functional.smooth_l1_loss(predicted_boxes, box_targets),
            "class": functional.cross_entropy(raw[index_tensor, 5:], class_targets),
        }

    @torch.inference_mode()
    def predict_tensors(
        self, images: list[Tensor], score_threshold: float
    ) -> list[dict[str, Tensor]]:
        self.eval()
        batch = torch.stack(images)
        raw = self.core(batch)
        normalized_boxes = raw[:, :4].sigmoid()
        objectness = raw[:, 4].sigmoid()
        class_probabilities = raw[:, 5:].softmax(dim=1)
        class_scores, class_ids = class_probabilities.max(dim=1)
        scores = objectness * class_scores
        outputs: list[dict[str, Tensor]] = []
        height, width = batch.shape[-2:]
        for index in range(batch.shape[0]):
            if scores[index] < score_threshold:
                outputs.append(
                    {
                        "boxes": torch.empty((0, 4), device=batch.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=batch.device),
                        "scores": torch.empty((0,), device=batch.device),
                    }
                )
                continue
            center_x, center_y, box_width, box_height = normalized_boxes[index]
            x1 = (center_x - box_width / 2).clamp(0, 1) * width
            y1 = (center_y - box_height / 2).clamp(0, 1) * height
            x2 = (center_x + box_width / 2).clamp(0, 1) * width
            y2 = (center_y + box_height / 2).clamp(0, 1) * height
            outputs.append(
                {
                    "boxes": torch.stack((x1, y1, x2, y2)).reshape(1, 4),
                    "labels": class_ids[index].reshape(1),
                    "scores": scores[index].reshape(1),
                }
            )
        return outputs
