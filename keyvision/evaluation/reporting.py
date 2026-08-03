"""Export evaluation results as JSON, CSV, and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _metric_text(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def export_evaluation(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write equivalent human- and machine-readable evaluation artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "metrics.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for class_name, metrics in results["per_class"].items():
        rows.append(
            {
                "class": class_name,
                **{key: metrics[key] for key in ("precision", "recall", "f1", "ap50", "ap50_95")},
                "support": metrics["support"],
                "predictions": metrics.get("predictions", 0),
            }
        )
    csv_path = root / "per_class_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["class"])
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = root / "metrics.md"
    lines = [
        "# Evaluation Results",
        "",
        "> Generated from an actual execution. Interpret the `result_scope` field before use.",
        "",
        f"- Result scope: **{results.get('result_scope', 'unspecified')}**",
        f"- Precision: {_metric_text(results['precision'])}",
        f"- Recall: {_metric_text(results['recall'])}",
        f"- F1: {_metric_text(results['f1'])}",
        f"- mAP@50: {_metric_text(results['map50'])}",
        f"- mAP@50:95: {_metric_text(results['map50_95'])}",
        f"- AP macro average: {results.get('macro_averaging', 'unspecified')}",
        f"- Evaluated classes: {results.get('evaluated_class_count', 'unspecified')}",
        "",
    ]
    confidence_intervals = results.get("confidence_intervals")
    if confidence_intervals:
        confidence = float(confidence_intervals["confidence"])
        lines.extend(
            [
                f"## {confidence:.0%} bootstrap confidence intervals",
                "",
                f"Image-level resampling, {confidence_intervals['samples']} samples, "
                f"seed {confidence_intervals['seed']}.",
                "",
                "| Metric | Lower | Median | Upper |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, interval in confidence_intervals["metrics"].items():
            lines.append(
                f"| {key} | {_metric_text(interval['lower'])} | "
                f"{_metric_text(interval['median'])} | {_metric_text(interval['upper'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Per-class metrics",
            "",
            "| Class | Precision | Recall | F1 | AP@50 | AP@50:95 | Support | Predictions |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['class']} | {_metric_text(row['precision'])} | "
            f"{_metric_text(row['recall'])} | {_metric_text(row['f1'])} | "
            f"{_metric_text(row['ap50'])} | {_metric_text(row['ap50_95'])} | "
            f"{row['support']} | {row['predictions']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
