from pathlib import Path

from PIL import Image

from keyvision.data.io import write_manifest
from keyvision.data.schema import Annotation, ImageRecord
from keyvision.data.validation import validate_dataset


def test_validation_accepts_valid_record(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 80), "gray").save(image_dir / "sample.png")
    record = ImageRecord(
        image="images/sample.png",
        width=100,
        height=80,
        annotations=(Annotation((10, 10, 20, 15), 0, "missing_keycap"),),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])
    assert validate_dataset(tmp_path, manifest) == []


def test_validation_reports_out_of_bounds_box(tmp_path: Path) -> None:
    Image.new("RGB", (50, 50), "gray").save(tmp_path / "sample.png")
    record = ImageRecord(
        image="sample.png",
        width=50,
        height=50,
        annotations=(Annotation((45, 45, 10, 10), 0, "missing_keycap"),),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])
    issues = validate_dataset(tmp_path, manifest)
    assert {issue.code for issue in issues} == {"box_out_of_bounds"}
