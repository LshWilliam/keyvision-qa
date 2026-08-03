from pathlib import Path

import numpy as np
import torch
from PIL import Image

from keyvision.data.dataset import KeyboardDefectDataset, detection_collate
from keyvision.data.io import write_manifest
from keyvision.data.schema import Annotation, ImageRecord
from keyvision.data.stats import dataset_statistics
from keyvision.inference.visualization import draw_prediction, overlay_heatmap
from keyvision.types import Detection, Prediction


def test_dataset_resizes_images_boxes_and_reports_statistics(tmp_path: Path) -> None:
    Image.new("RGB", (32, 24), "gray").save(tmp_path / "sample.png")
    record = ImageRecord(
        image="sample.png",
        width=32,
        height=24,
        annotations=(Annotation((2, 2, 8, 8), 0, "defect"),),
        synthetic=True,
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])

    dataset = KeyboardDefectDataset(tmp_path, manifest, image_size=16)
    image, target = dataset[0]
    assert image.shape == (3, 16, 16)
    assert torch.allclose(target["boxes"], torch.tensor([[1.0, 4 / 3, 5.0, 20 / 3]]), atol=1e-5)
    images, targets = detection_collate([(image, target)])
    assert len(images) == len(targets) == 1
    assert dataset.image_records == [record]
    assert dataset_statistics(manifest) == {
        "images": 1,
        "annotations": 1,
        "normal_images": 0,
        "synthetic_images": 1,
        "class_counts": {"defect": 1},
    }


def test_prediction_and_heatmap_visualizations_preserve_size() -> None:
    image = Image.new("RGB", (24, 16), "gray")
    prediction = Prediction(detections=[Detection((2, 2, 12, 10), 0, "defect", 0.9)])
    boxed = draw_prediction(image, prediction)
    heatmap = np.zeros((16, 24), dtype=np.float32)
    heatmap[4:8, 6:12] = 1.0
    overlay = overlay_heatmap(image, heatmap)
    assert boxed.size == image.size
    assert overlay.size == image.size
    assert boxed.tobytes() != image.tobytes()
    assert overlay.tobytes() != image.tobytes()
