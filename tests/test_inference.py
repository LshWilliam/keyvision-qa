from PIL import Image

from keyvision.config import load_config
from keyvision.inference.predictor import DetectorPredictor


def test_predictor_returns_portable_prediction() -> None:
    predictor = DetectorPredictor(load_config("configs/smoke.yaml"))
    prediction = predictor.predict(Image.new("RGB", (200, 100), "gray"))
    assert prediction.latency_ms is not None
    assert prediction.metadata["device"] == "cpu"
    for detection in prediction.detections:
        assert len(detection.bbox_xyxy) == 4
        assert 0 <= detection.class_id < 6
