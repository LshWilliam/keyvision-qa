"""Canonical JSONL annotation schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Annotation:
    """One COCO-style ``xywh`` annotation stored in a manifest record."""

    bbox: tuple[float, float, float, float]
    category_id: int
    category: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Annotation:
        """Parse and minimally type-check an annotation mapping."""

        bbox = payload.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("annotation.bbox must contain four numbers")
        x, y, width, height = (float(value) for value in bbox)
        return cls(
            bbox=(x, y, width, height),
            category_id=int(payload["category_id"]),
            category=str(payload["category"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the annotation into the manifest representation."""

        return {
            "bbox": [round(value, 3) for value in self.bbox],
            "category_id": self.category_id,
            "category": self.category,
        }


@dataclass(frozen=True)
class ImageRecord:
    """One annotated image record."""

    image: str
    width: int
    height: int
    annotations: tuple[Annotation, ...]
    synthetic: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImageRecord:
        """Parse an image record from JSON."""

        raw_annotations = payload.get("annotations", [])
        if not isinstance(raw_annotations, list):
            raise ValueError("annotations must be a list")
        return cls(
            image=str(payload["image"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            annotations=tuple(Annotation.from_dict(item) for item in raw_annotations),
            synthetic=bool(payload.get("synthetic", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record without machine-specific paths."""

        return {
            "image": self.image.replace("\\", "/"),
            "width": self.width,
            "height": self.height,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "synthetic": self.synthetic,
        }
