"""PyTorch dataset for KeyVision JSONL manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from keyvision.data.io import load_manifest
from keyvision.data.schema import ImageRecord


class KeyboardDefectDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Load images and targets, resizing both to a square model input."""

    def __init__(self, root: str | Path, manifest: str | Path, image_size: int) -> None:
        self.root = Path(root)
        self.records = load_manifest(manifest)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.records[index]
        image = Image.open(self.root / record.image).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        boxes = []
        labels = []
        for annotation in record.annotations:
            x, y, width, height = annotation.bbox
            boxes.append([x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y])
            labels.append(annotation.category_id)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        return image_tensor, {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
        }

    @property
    def image_records(self) -> list[ImageRecord]:
        """Expose immutable record values for evaluation and reporting."""

        return list(self.records)


def detection_collate(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """Collate variable-length detection targets."""

    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
