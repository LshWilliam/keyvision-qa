"""Deterministic class-aware splitting with optional group isolation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from keyvision.data.schema import ImageRecord

SPLIT_NAMES = ("train", "val", "test")


def _record_groups(records: Sequence[ImageRecord]) -> list[list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        key = record.group_id or f"__image__:{record.image}"
        groups.setdefault(key, []).append(record)
    return list(groups.values())


def _primary_class(records: Sequence[ImageRecord]) -> int:
    counts = Counter(
        annotation.category_id for record in records for annotation in record.annotations
    )
    if not counts:
        return -1
    return min(counts, key=lambda class_id: (-counts[class_id], class_id))


def split_records(
    records: Sequence[ImageRecord],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Split reproducibly while keeping each non-empty ``group_id`` in one split.

    Groups can represent keyboard SKU, production lot, capture session, or a video
    sequence. Assignment is greedy and class-aware; exact sample ratios are not
    guaranteed when a group is larger than a target split.
    """

    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    buckets: dict[int, list[list[ImageRecord]]] = defaultdict(list)
    for group in _record_groups(records):
        buckets[_primary_class(group)].append(group)

    rng = random.Random(seed)
    splits: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    for class_id in sorted(buckets):
        groups = list(buckets[class_id])
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        class_total = sum(len(group) for group in groups)
        targets = {
            name: class_total * ratio for name, ratio in zip(SPLIT_NAMES, ratios, strict=True)
        }
        assigned = {name: 0 for name in SPLIT_NAMES}
        for group in groups:
            target = max(
                SPLIT_NAMES,
                key=lambda name: (
                    (targets[name] - assigned[name]) / max(targets[name], 1.0),
                    -SPLIT_NAMES.index(name),
                ),
            )
            splits[target].extend(group)
            assigned[target] += len(group)

    for values in splits.values():
        rng.shuffle(values)
    return splits
