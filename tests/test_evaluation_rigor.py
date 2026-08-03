import pytest

from keyvision.evaluation.metrics import (
    bootstrap_confidence_intervals,
    evaluate_detections,
)
from keyvision.types import Detection


def _detection(class_id: int, score: float = 0.9) -> Detection:
    return Detection((0, 0, 10, 10), class_id, f"class_{class_id}", score)


def test_absent_classes_do_not_depress_macro_ap() -> None:
    metrics = evaluate_detections(
        [[_detection(0)]],
        [[_detection(0)]],
        ["class_0", "class_1"],
    )
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["evaluated_class_count"] == 1
    assert metrics["absent_class_count"] == 1
    assert metrics["per_class"]["class_1"]["ap50"] is None


def test_absent_class_false_alarm_affects_operational_precision() -> None:
    metrics = evaluate_detections(
        [[_detection(0), _detection(1, 0.8)]],
        [[_detection(0)]],
        ["class_0", "class_1"],
    )
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["counts"] == {"tp": 1, "fp": 1, "fn": 0}


def test_bootstrap_intervals_are_deterministic_for_perfect_predictions() -> None:
    predictions = [[_detection(0)] for _ in range(4)]
    targets = [[_detection(0)] for _ in range(4)]
    first = bootstrap_confidence_intervals(predictions, targets, ["class_0"], samples=25, seed=7)
    second = bootstrap_confidence_intervals(predictions, targets, ["class_0"], samples=25, seed=7)
    assert first == second
    for interval in first["metrics"].values():
        assert interval == {"lower": 1.0, "median": 1.0, "upper": 1.0}
