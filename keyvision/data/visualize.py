"""Create an annotated contact sheet for dataset inspection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

from keyvision.data.io import load_manifest


def create_contact_sheet(
    root: str | Path,
    manifest: str | Path,
    output: str | Path,
    limit: int = 12,
) -> Path:
    """Render boxes and class names on a compact contact sheet."""

    dataset_root = Path(root)
    records = load_manifest(manifest)[:limit]
    if not records:
        raise ValueError("Cannot visualize an empty manifest")
    thumb_size = (320, 180)
    columns = min(3, len(records))
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "white")
    for index, record in enumerate(records):
        image = Image.open(dataset_root / record.image).convert("RGB")
        scale_x, scale_y = thumb_size[0] / image.width, thumb_size[1] / image.height
        image = image.resize(thumb_size)
        draw = ImageDraw.Draw(image)
        for annotation in record.annotations:
            x, y, width, height = annotation.bbox
            box = (
                int(x * scale_x),
                int(y * scale_y),
                int((x + width) * scale_x),
                int((y + height) * scale_y),
            )
            draw.rectangle(box, outline=(0, 255, 120), width=2)
            draw.text((box[0], max(22, box[1] - 11)), annotation.category, fill=(0, 255, 120))
        sheet.paste(image, ((index % columns) * thumb_size[0], (index // columns) * thumb_size[1]))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="assets/synthetic_contact_sheet.png")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    print(create_contact_sheet(args.root, args.manifest, args.output, args.limit))


if __name__ == "__main__":
    main()
