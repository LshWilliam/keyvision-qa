"""Dependency-light object detection metrics with transparent matching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from keyvision.types import Detection


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute intersection-over-union for two ``xyxy`` boxes."""

    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


def _class_ap(
    predictions: Sequence[Sequence[Detection]],
    ground_truth: Sequence[Sequence[Detection]],
    class_id: int,
    iou_threshold: float,
) -> tuple[float, list[float], list[float]]:
    candidates = sorted(
        (
            (detection.score, image_index, detection)
            for image_index, image_predictions in enumerate(predictions)
            for detection in image_predictions
            if detection.class_id == class_id
        ),
        reverse=True,
        key=lambda item: item[0],
    )
    targets = {
        image_index: [item for item in image_targets if item.class_id == class_id]
        for image_index, image_targets in enumerate(ground_truth)
    }
    target_count = sum(len(items) for items in targets.values())
    if target_count == 0:
        return 0.0, [], []
    matched: dict[int, set[int]] = defaultdict(set)
    true_positives = []
    false_positives = []
    for _, image_index, prediction in candidates:
        best_index = -1
        best_iou = 0.0
        for target_index, target in enumerate(targets[image_index]):
            if target_index in matched[image_index]:
                continue
            overlap = box_iou(prediction.bbox_xyxy, target.bbox_xyxy)
            if overlap > best_iou:
                best_iou, best_index = overlap, target_index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        if is_match:
            matched[image_index].add(best_index)
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / target_count
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return _average_precision(recall, precision), recall.tolist(), precision.tolist()


def evaluate_detections(
    predictions: Sequence[Sequence[Detection]],
    ground_truth: Sequence[Sequence[Detection]],
    class_names: Sequence[str],
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Calculate precision, recall, F1, COCO-style mAP, and per-class results."""

    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same number of images")
    per_class: dict[str, dict[str, Any]] = {}
    all_ap50 = []
    all_coco_ap = []
    total_tp = total_fp = total_fn = 0
    for class_id, class_name in enumerate(class_names):
        ap50, recall_curve, precision_curve = _class_ap(
            predictions, ground_truth, class_id, match_iou
        )
        threshold_aps = [
            _class_ap(predictions, ground_truth, class_id, float(threshold))[0]
            for threshold in np.arange(0.5, 1.0, 0.05)
        ]
        true_count = sum(
            detection.class_id == class_id
            for image_targets in ground_truth
            for detection in image_targets
        )
        prediction_count = sum(
            detection.class_id == class_id
            for image_predictions in predictions
            for detection in image_predictions
        )
        matched = round(recall_curve[-1] * true_count) if recall_curve else 0
        tp = int(matched)
        fp = int(prediction_count - tp)
        fn = int(true_count - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap50,
            "ap50_95": float(np.mean(threshold_aps)),
            "support": true_count,
            "pr_curve": {"recall": recall_curve, "precision": precision_curve},
        }
        all_ap50.append(ap50)
        all_coco_ap.append(float(np.mean(threshold_aps)))
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(np.mean(all_ap50)) if all_ap50 else 0.0,
        "map50_95": float(np.mean(all_coco_ap)) if all_coco_ap else 0.0,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
        "per_class": per_class,
    }
