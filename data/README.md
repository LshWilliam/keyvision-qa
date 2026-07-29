# Data directory

No real keyboard dataset is committed to this repository.

## Recommended layout

```text
data/
├── raw/          # immutable, authorized source images; ignored by Git
├── interim/      # converted labels and QA rejects; ignored by Git
├── processed/    # frozen manifests and transformed data; ignored by Git
└── README.md
```

The runtime examples use `artifacts/synthetic/`, which is also ignored. Only the watermarked contact
sheet in `assets/` is committed.

## Annotation contract

Manifests are UTF-8 JSONL. Each record contains a repository-independent relative image path,
declared image dimensions, zero or more COCO-style absolute `xywh` boxes, and a `synthetic` flag.
Category identifiers are zero-based and must match the configured class order.

Run validation before any split, training, or evaluation:

```bash
python -m keyvision.data.validation --root DATASET_ROOT --manifest DATASET_ROOT/manifest.jsonl
```

Data must be public with a compatible license, captured by the user, or explicitly authorized.
Never place customer, employer, or personally identifying imagery in this repository.

