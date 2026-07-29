"""Deterministic dataset splitting with class-aware ordering."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from keyvision.data.schema import ImageRecord


def split_records(
    records: Sequence[ImageRecord],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Split records reproducibly while spreading primary classes across splits."""

    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    buckets: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        primary_class = record.annotations[0].category_id if record.annotations else -1
        buckets[primary_class].append(record)

    rng = random.Random(seed)
    splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    for class_id in sorted(buckets):
        group = list(buckets[class_id])
        rng.shuffle(group)
        for index, record in enumerate(group):
            fraction = (index + 0.5) / max(len(group), 1)
            if fraction <= ratios[0]:
                target = "train"
            elif fraction <= ratios[0] + ratios[1]:
                target = "val"
            else:
                target = "test"
            splits[target].append(record)

    for values in splits.values():
        rng.shuffle(values)
    return splits
