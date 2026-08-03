from pathlib import Path

from PIL import Image

from keyvision.evaluation.error_analysis import save_error_cases
from keyvision.evaluation.metrics import (
    bootstrap_confidence_intervals,
    evaluate_detections,
)
from keyvision.evaluation.reporting import export_evaluation
from keyvision.types import Detection


def _detection(box: tuple[float, float, float, float], score: float = 0.9) -> Detection:
    return Detection(box, 0, "defect", score)


def test_report_exports_null_ap_and_bootstrap_intervals(tmp_path: Path) -> None:
    predictions = [[_detection((0, 0, 10, 10))]]
    targets = [[_detection((0, 0, 10, 10))]]
    results = evaluate_detections(predictions, targets, ["defect", "absent"])
    results["result_scope"] = "unit test"
    results["confidence_intervals"] = bootstrap_confidence_intervals(
        predictions, targets, ["defect", "absent"], samples=5
    )
    paths = export_evaluation(results, tmp_path / "report")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "95% bootstrap confidence intervals" in markdown
    assert "| absent |" in markdown
    assert "N/A" in markdown
    assert paths["json"].is_file()
    assert paths["csv"].is_file()


def test_error_analysis_writes_ranked_fp_and_fn(tmp_path: Path) -> None:
    image = Image.new("RGB", (40, 40), "gray")
    report = save_error_cases(
        [image],
        ["sample.png"],
        [[_detection((25, 25, 35, 35), 0.8)]],
        [[_detection((2, 2, 12, 12))]],
        tmp_path / "errors",
    )
    assert report["false_positive_count"] == 1
    assert report["false_negative_count"] == 1
    assert len(list((tmp_path / "errors" / "false_positives").glob("*.png"))) == 1
    assert len(list((tmp_path / "errors" / "false_negatives").glob("*.png"))) == 1
