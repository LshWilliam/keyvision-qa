"""Detection confusion matrix including background misses and false alarms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from keyvision.evaluation.metrics import box_iou
from keyvision.types import Detection


def detection_confusion_matrix(
    predictions: Sequence[Sequence[Detection]],
    ground_truth: Sequence[Sequence[Detection]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> list[list[int]]:
    """Build a matrix with the final row/column representing background."""

    background = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for image_predictions, image_targets in zip(predictions, ground_truth, strict=True):
        unmatched_predictions = set(range(len(image_predictions)))
        for target in image_targets:
            best_index = -1
            best_overlap = 0.0
            for prediction_index in unmatched_predictions:
                overlap = box_iou(image_predictions[prediction_index].bbox_xyxy, target.bbox_xyxy)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = prediction_index
            if best_index >= 0 and best_overlap >= iou_threshold:
                prediction = image_predictions[best_index]
                matrix[target.class_id, prediction.class_id] += 1
                unmatched_predictions.remove(best_index)
            else:
                matrix[target.class_id, background] += 1
        for prediction_index in unmatched_predictions:
            matrix[background, image_predictions[prediction_index].class_id] += 1
    return matrix.tolist()
