from pathlib import Path

from PIL import Image

from keyvision.data.io import write_manifest
from keyvision.data.schema import Annotation, ImageRecord
from keyvision.data.split import split_records
from keyvision.data.validation import validate_dataset


def _record(image: str, group_id: str | None = None, class_id: int = 0) -> ImageRecord:
    return ImageRecord(
        image=image,
        width=32,
        height=24,
        annotations=(Annotation((2, 2, 8, 8), class_id, f"class_{class_id}"),),
        group_id=group_id,
    )


def test_grouped_split_never_leaks_a_group() -> None:
    records = [
        _record(f"images/{group}_{sample}.png", f"session-{group}", group % 3)
        for group in range(9)
        for sample in range(2)
    ]
    splits = split_records(records, seed=9)
    locations: dict[str, set[str]] = {}
    for split_name, values in splits.items():
        for record in values:
            assert record.group_id is not None
            locations.setdefault(record.group_id, set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in locations.values())
    assert sum(len(values) for values in splits.values()) == len(records)


def test_group_id_round_trips_through_schema() -> None:
    record = _record("images/a.png", "sku-01")
    assert ImageRecord.from_dict(record.to_dict()) == record


def test_relative_path_traversal_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [_record("../escape.png")])
    issues = validate_dataset(tmp_path, manifest)
    assert {issue.code for issue in issues} == {"path_outside_root"}


def test_duplicate_image_bytes_are_detected(tmp_path: Path) -> None:
    Image.new("RGB", (32, 24), "gray").save(tmp_path / "a.png")
    Image.new("RGB", (32, 24), "gray").save(tmp_path / "b.png")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [_record("a.png"), _record("b.png")])
    issues = validate_dataset(tmp_path, manifest)
    assert "duplicate_content" in {issue.code for issue in issues}


def test_category_mapping_conflict_is_detected(tmp_path: Path) -> None:
    Image.new("RGB", (32, 24), "black").save(tmp_path / "a.png")
    Image.new("RGB", (32, 24), "white").save(tmp_path / "b.png")
    first = _record("a.png")
    second = ImageRecord(
        image="b.png",
        width=32,
        height=24,
        annotations=(Annotation((2, 2, 8, 8), 0, "different_name"),),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [first, second])
    issues = validate_dataset(tmp_path, manifest)
    assert "category_mapping_conflict" in {issue.code for issue in issues}
