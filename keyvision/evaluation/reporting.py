"""Export evaluation results as JSON, CSV, and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


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
        f"- Precision: {results['precision']:.4f}",
        f"- Recall: {results['recall']:.4f}",
        f"- F1: {results['f1']:.4f}",
        f"- mAP@50: {results['map50']:.4f}",
        f"- mAP@50:95: {results['map50_95']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | AP@50 | AP@50:95 | Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['ap50']:.4f} | {row['ap50_95']:.4f} | "
            f"{row['support']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
