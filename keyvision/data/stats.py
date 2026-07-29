"""Dataset statistics for manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from keyvision.data.io import load_manifest
from keyvision.utils.runtime import write_json


def dataset_statistics(manifest: str | Path) -> dict[str, Any]:
    """Compute class and image-level counts without loading image pixels."""

    records = load_manifest(manifest)
    class_counts = Counter(
        annotation.category for record in records for annotation in record.annotations
    )
    return {
        "images": len(records),
        "annotations": sum(class_counts.values()),
        "normal_images": sum(not record.annotations for record in records),
        "synthetic_images": sum(record.synthetic for record in records),
        "class_counts": dict(sorted(class_counts.items())),
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="artifacts/reports/dataset_stats.json")
    args = parser.parse_args()
    stats = dataset_statistics(args.manifest)
    write_json(args.output, stats)
    print(stats)


if __name__ == "__main__":
    main()
