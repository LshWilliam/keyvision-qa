"""Generate an explicitly watermarked synthetic keyboard inspection dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from keyvision.data.io import write_manifest
from keyvision.data.schema import Annotation, ImageRecord
from keyvision.data.split import split_records

CLASS_NAMES = (
    "missing_keycap",
    "misaligned_keycap",
    "print_defect",
    "stain",
    "scratch",
    "foreign_object",
)
KEY_LABELS = tuple("1234567890QWERTYUIOPASDFGHJKLZXCVBNM")


def _keyboard_geometry(width: int, height: int) -> tuple[list[tuple[int, int, int, int]], int, int]:
    rows, columns = 4, 10
    margin_x, margin_y = int(width * 0.08), int(height * 0.20)
    gap = max(3, width // 150)
    key_width = (width - 2 * margin_x - (columns - 1) * gap) // columns
    key_height = (height - 2 * margin_y - (rows - 1) * gap) // rows
    boxes = []
    for row in range(rows):
        for column in range(columns):
            x1 = margin_x + column * (key_width + gap)
            y1 = margin_y + row * (key_height + gap)
            boxes.append((x1, y1, x1 + key_width, y1 + key_height))
    return boxes, key_width, key_height


def _draw_base_keyboard(
    rng: random.Random, width: int, height: int
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    background = rng.randint(26, 44)
    image = Image.new("RGB", (width, height), (background, background + 2, background + 4))
    draw = ImageDraw.Draw(image)
    boxes, _, _ = _keyboard_geometry(width, height)
    font = ImageFont.load_default()
    for index, box in enumerate(boxes):
        shade = rng.randint(70, 100)
        draw.rounded_rectangle(
            box, radius=4, fill=(shade, shade, shade + 3), outline=(140, 140, 145)
        )
        label = KEY_LABELS[index % len(KEY_LABELS)]
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        draw.text((center_x, center_y), label, font=font, fill=(220, 220, 215), anchor="mm")
    draw.rounded_rectangle(
        (int(width * 0.04), int(height * 0.12), int(width * 0.96), int(height * 0.88)),
        radius=12,
        outline=(100, 105, 110),
        width=3,
    )
    return image, boxes


def _apply_defect(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    class_id: int,
    rng: random.Random,
) -> Annotation:
    draw = ImageDraw.Draw(image)
    box = boxes[rng.randrange(len(boxes))]
    x1, y1, x2, y2 = box
    category = CLASS_NAMES[class_id]

    if category == "missing_keycap":
        draw.rounded_rectangle(box, radius=4, fill=(18, 18, 20), outline=(8, 8, 8), width=2)
    elif category == "misaligned_keycap":
        draw.rectangle(box, fill=(35, 35, 38))
        dx, dy = max(3, (x2 - x1) // 8), max(2, (y2 - y1) // 8)
        moved = (x1 + dx, y1 - dy, x2 + dx, y2 - dy)
        draw.rounded_rectangle(moved, radius=4, fill=(90, 90, 94), outline=(190, 120, 60), width=2)
        x1, y1, x2, y2 = moved
    elif category == "print_defect":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.rectangle((cx - 5, cy - 3, cx + 6, cy + 3), fill=(50, 50, 52))
        x1, y1, x2, y2 = cx - 8, cy - 6, cx + 8, cy + 6
    elif category == "stain":
        radius = max(4, min(x2 - x1, y2 - y1) // 4)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(80, 55, 30))
        x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
    elif category == "scratch":
        draw.line((x1 + 3, y2 - 4, x2 - 3, y1 + 4), fill=(230, 225, 210), width=2)
    else:
        radius = max(5, min(x2 - x1, y2 - y1) // 4)
        cx = rng.randint(x1 + radius, x2 - radius)
        cy = rng.randint(y1 + radius, y2 - radius)
        draw.polygon(
            [
                (cx, cy - radius),
                (cx + radius, cy),
                (cx, cy + radius),
                (cx - radius, cy),
            ],
            fill=(190, 45, 50),
        )
        x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius

    return Annotation(
        bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
        category_id=class_id,
        category=category,
    )


def generate_dataset(
    output_dir: str | Path,
    count: int = 42,
    seed: int = 42,
    width: int = 640,
    height: int = 360,
) -> dict[str, int]:
    """Create synthetic images and deterministic train/validation/test manifests."""

    if count < len(CLASS_NAMES) * 3:
        raise ValueError(f"count must be at least {len(CLASS_NAMES) * 3} for class coverage")
    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[ImageRecord] = []

    for index in range(count):
        image, boxes = _draw_base_keyboard(rng, width, height)
        class_id = index % len(CLASS_NAMES)
        annotation = _apply_defect(image, boxes, class_id, rng)
        draw = ImageDraw.Draw(image)
        watermark = "SYNTHETIC EXAMPLE - NOT PRODUCTION DATA"
        draw.rectangle((0, 0, width, 22), fill=(125, 20, 25))
        draw.text((width // 2, 11), watermark, fill="white", anchor="mm")
        name = f"keyboard_{index:04d}.png"
        image.save(image_dir / name)
        records.append(
            ImageRecord(
                image=f"images/{name}",
                width=width,
                height=height,
                annotations=(annotation,),
                synthetic=True,
            )
        )

    splits = split_records(records, seed=seed)
    for split_name, split_values in splits.items():
        write_manifest(root / "splits" / f"{split_name}.jsonl", split_values)
    write_manifest(root / "manifest.jsonl", records)
    return {name: len(values) for name, values in splits.items()}


def generate_normal_images(
    output_dir: str | Path,
    count: int = 12,
    seed: int = 42,
    width: int = 320,
    height: int = 180,
) -> list[Path]:
    """Generate normal-only synthetic images for the anomaly baseline."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[Path] = []
    for index in range(count):
        image, _ = _draw_base_keyboard(rng, width, height)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 18), fill=(25, 75, 120))
        draw.text((width // 2, 9), "SYNTHETIC NORMAL EXAMPLE", fill="white", anchor="mm")
        path = root / f"normal_{index:03d}.png"
        image.save(path)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    """Build the synthetic-data CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/synthetic")
    parser.add_argument("--count", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    splits = generate_dataset(args.output, args.count, args.seed, args.width, args.height)
    print(f"Generated synthetic dataset at {args.output}: {splits}")


if __name__ == "__main__":
    main()
