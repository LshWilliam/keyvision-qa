"""False-positive and false-negative artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from keyvision.evaluation.metrics import box_iou
from keyvision.types import Detection


def _draw_error(
    image: Image.Image,
    detections: list[Detection],
    color: tuple[int, int, int],
    prefix: str,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for detection in detections:
        draw.rectangle(detection.bbox_xyxy, outline=color, width=4)
        draw.text(
            (detection.bbox_xyxy[0], detection.bbox_xyxy[1]),
            f"{prefix}: {detection.class_name} {detection.score:.2f}",
            fill=color,
        )
    return output


def save_error_cases(
    images: list[Image.Image],
    names: list[str],
    predictions: list[list[Detection]],
    ground_truth: list[list[Detection]],
    output_dir: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Save confidence-sorted false positives and false negatives."""

    root = Path(output_dir)
    fp_dir, fn_dir = root / "false_positives", root / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    false_positives: list[tuple[float, int, Detection]] = []
    false_negatives: list[tuple[int, Detection]] = []

    for image_index, (image_predictions, image_targets) in enumerate(
        zip(predictions, ground_truth, strict=True)
    ):
        matched_targets: set[int] = set()
        for prediction in sorted(image_predictions, key=lambda item: item.score, reverse=True):
            candidates = [
                (box_iou(prediction.bbox_xyxy, target.bbox_xyxy), target_index)
                for target_index, target in enumerate(image_targets)
                if target_index not in matched_targets and target.class_id == prediction.class_id
            ]
            best = max(candidates, default=(0.0, -1))
            if best[0] >= iou_threshold:
                matched_targets.add(best[1])
            else:
                false_positives.append((prediction.score, image_index, prediction))
        for target_index, target in enumerate(image_targets):
            if target_index not in matched_targets:
                false_negatives.append((image_index, target))

    for rank, (_, image_index, detection) in enumerate(
        sorted(false_positives, reverse=True, key=lambda item: item[0]), 1
    ):
        filename = f"{rank:04d}_{Path(names[image_index]).stem}.png"
        _draw_error(images[image_index], [detection], (255, 60, 60), "FP").save(fp_dir / filename)
    for rank, (image_index, detection) in enumerate(false_negatives, 1):
        filename = f"{rank:04d}_{Path(names[image_index]).stem}.png"
        _draw_error(images[image_index], [detection], (255, 190, 30), "FN").save(fn_dir / filename)

    report = {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "sorting": "false positives sorted by descending confidence",
        "output_dir": str(root),
    }
    (root / "report.md").write_text(
        "# Error Analysis\n\n"
        f"- False positives: {len(false_positives)}\n"
        f"- False negatives: {len(false_negatives)}\n"
        "- False positives are sorted by descending model confidence.\n"
        "- Review reflective surfaces, low contrast, small defects, viewpoint, lighting, "
        "occlusion, and domain shift before changing thresholds.\n",
        encoding="utf-8",
    )
    return report
