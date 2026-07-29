"""Dataset and label integrity validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from keyvision.data.io import load_manifest
from keyvision.data.schema import ImageRecord
from keyvision.utils.runtime import write_json


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable dataset problem."""

    severity: str
    code: str
    image: str
    message: str


def validate_record(root: Path, record: ImageRecord) -> list[ValidationIssue]:
    """Validate one manifest record and its image file."""

    issues: list[ValidationIssue] = []
    image_path = root / record.image
    if image_path.is_absolute() and not image_path.resolve().is_relative_to(root.resolve()):
        issues.append(
            ValidationIssue(
                "error", "absolute_path", record.image, "Image must stay under dataset root"
            )
        )
        return issues
    if not image_path.is_file():
        issues.append(
            ValidationIssue("error", "missing_image", record.image, "Image file is missing")
        )
        return issues
    try:
        with Image.open(image_path) as image:
            actual_size = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        issues.append(ValidationIssue("error", "corrupt_image", record.image, str(exc)))
        return issues
    if actual_size != (record.width, record.height):
        issues.append(
            ValidationIssue(
                "error",
                "size_mismatch",
                record.image,
                f"Manifest size {(record.width, record.height)} != image size {actual_size}",
            )
        )
    for annotation in record.annotations:
        x, y, width, height = annotation.bbox
        if width <= 0 or height <= 0:
            issues.append(
                ValidationIssue("error", "invalid_box_size", record.image, str(annotation.bbox))
            )
        if x < 0 or y < 0 or x + width > record.width or y + height > record.height:
            issues.append(
                ValidationIssue("error", "box_out_of_bounds", record.image, str(annotation.bbox))
            )
        if annotation.category_id < 0 or not annotation.category:
            issues.append(
                ValidationIssue(
                    "error", "invalid_category", record.image, str(annotation.category_id)
                )
            )
    return issues


def validate_dataset(root: str | Path, manifest: str | Path) -> list[ValidationIssue]:
    """Validate image existence, readability, dimensions, and bounding boxes."""

    dataset_root = Path(root)
    records = load_manifest(manifest)
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    for record in records:
        if record.image in seen:
            issues.append(
                ValidationIssue(
                    "error", "duplicate_image", record.image, "Duplicate manifest record"
                )
            )
        seen.add(record.image)
        issues.extend(validate_record(dataset_root, record))
    return issues


def main() -> None:
    """CLI entry point for dataset validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", default="artifacts/reports/dataset_validation.json")
    args = parser.parse_args()
    issues = validate_dataset(args.root, args.manifest)
    write_json(args.report, {"valid": not issues, "issues": [asdict(issue) for issue in issues]})
    print(f"Validation complete: {len(issues)} issue(s); report={args.report}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
