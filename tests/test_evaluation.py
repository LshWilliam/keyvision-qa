import pytest

from keyvision.evaluation.confusion import detection_confusion_matrix
from keyvision.evaluation.metrics import box_iou, evaluate_detections
from keyvision.types import Detection


def _detection(box: tuple[float, float, float, float], class_id: int = 0) -> Detection:
    return Detection(box, class_id, f"class_{class_id}", 0.9)


def test_iou_and_perfect_metrics() -> None:
    prediction = [[_detection((0, 0, 10, 10))]]
    target = [[_detection((0, 0, 10, 10))]]
    assert box_iou(prediction[0][0].bbox_xyxy, target[0][0].bbox_xyxy) == pytest.approx(1.0)
    metrics = evaluate_detections(prediction, target, ["class_0"])
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["map50"] == pytest.approx(1.0)
    assert detection_confusion_matrix(prediction, target, 1) == [[1, 0], [0, 0]]


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_detections([], [[]], ["class_0"])
