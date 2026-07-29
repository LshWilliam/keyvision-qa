"""JSONL manifest input and output."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from keyvision.data.schema import ImageRecord


def load_manifest(path: str | Path) -> list[ImageRecord]:
    """Read a UTF-8 JSONL manifest."""

    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    records: list[ImageRecord] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(ImageRecord.from_dict(payload))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid record at {manifest}:{line_number}: {exc}") from exc
    return records


def write_manifest(path: str | Path, records: Iterable[ImageRecord]) -> None:
    """Write portable, sorted-key JSONL records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
