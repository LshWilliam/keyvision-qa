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

Manifests are UTF-8 JSONL. Each record contains a root-constrained relative image path, declared
dimensions, zero or more COCO-style absolute `xywh` boxes, a `synthetic` flag, and an optional
`group_id`. Use one group for correlated images from the same SKU/lot/session or source video.
Category identifiers are zero-based and must map one-to-one to names. Validation rejects byte-level
duplicates, path traversal, invalid boxes, corrupt images, and category mapping conflicts.

Run validation before any split, training, or evaluation:

```bash
python -m keyvision.data.validation --root DATASET_ROOT --manifest DATASET_ROOT/manifest.jsonl
```

Data must be public with a compatible license, captured by the user, or explicitly authorized.
Never place customer, employer, or personally identifying imagery in this repository.

