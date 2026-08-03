from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from keyvision.models.anomaly import GaussianTemplateAnomalyDetector


def test_anomaly_fit_predict_and_round_trip(tmp_path: Path) -> None:
    images = [
        Image.fromarray(np.full((40, 60, 3), value, dtype=np.uint8)) for value in (90, 92, 94)
    ]
    detector = GaussianTemplateAnomalyDetector(image_size=(30, 20))
    detector.fit(images)
    prediction = detector.predict(images[1])
    assert prediction.heatmap.shape == (40, 60)
    path = tmp_path / "template.npz"
    detector.save(path)
    restored = GaussianTemplateAnomalyDetector.load(path)
    assert restored.predict(images[1]).score == pytest.approx(prediction.score)
