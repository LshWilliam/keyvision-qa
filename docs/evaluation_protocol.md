# Evaluation Protocol

This protocol separates engineering smoke evidence from model-quality evidence. A successful command
or exported model is not an accuracy claim.

## 1. Dataset unit and leakage control

The JSONL schema supports an optional `group_id`. Use it for the smallest unit that must remain
isolated, such as a keyboard SKU and lot, a capture session, or a source video. `split_records`
assigns complete groups to one of train, validation, or test while balancing dominant defect classes.
Exact image ratios are secondary to preventing leakage.

Before training, validation checks:

- resolved paths remain under the dataset root, including relative `..` traversal;
- images are readable and match declared dimensions;
- boxes have positive area and remain inside the image;
- duplicate manifest paths and byte-identical images are rejected;
- category IDs and category names form a one-to-one mapping;
- non-null group IDs are not blank.

For real production imagery, also compare perceptual hashes and acquisition metadata. Byte hashing
cannot detect re-encoded near duplicates.

## 2. Train, validation, and test responsibilities

- **Train:** fit weights and optimizer state only.
- **Validation:** select checkpoints, tune score thresholds, and compare architectures.
- **Test:** run once after the model and decision policy are frozen.

`best.pt` is selected by validation mAP@50. `last.pt` is always written for recovery. The validation
score threshold defaults to `0.05` so low-confidence predictions remain available for ranking-based
AP; the operational threshold remains a separate model configuration.

## 3. Detection metrics

The dependency-light evaluator reports:

- operational micro precision, recall, F1, TP, FP, and FN at the supplied prediction threshold;
- 101-point interpolated AP@50;
- AP averaged over IoU thresholds 0.50 through 0.95 in 0.05 steps;
- per-class support, prediction count, PR curves, and a background-aware confusion matrix.

Macro AP includes only classes with at least one ground-truth instance. An absent class is shown with
`null` AP rather than silently lowering the macro average. Predictions for an absent class still
count as false positives in operational precision.

This implementation intentionally does not claim full COCO equivalence: it does not reproduce COCO
area ranges, crowd handling, or max-detection conventions. Before publishing a serious benchmark,
cross-check results with the official COCO API or TorchMetrics' COCO-backed `MeanAveragePrecision`.

## 4. Uncertainty

Use image-level bootstrap intervals for any reported test result:

```bash
python -m keyvision.evaluation.cli \
  --config configs/default.yaml \
  --checkpoint artifacts/runs/fasterrcnn/best.pt \
  --split test \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42
```

The evaluator resamples images with replacement and exports lower, median, and upper bounds for
precision, recall, F1, AP@50, and AP@50:95. Group-level bootstrap should replace image-level
bootstrap when observations within a lot or capture session are correlated.

## 5. Required slices

A production-oriented report should break down results by:

- defect class and bounding-box area;
- keyboard SKU and production lot;
- camera, lens, exposure, and lighting cell;
- viewpoint or registration error;
- normal versus anomalous material/legend variants;
- latency distribution, batch size, warmup, and end-to-end I/O scope.

Report false accepts and false rejects against their business costs. A global mAP does not determine
an operating point for a manufacturing line.

## 6. Current evidence boundary

The checked-in smoke workflow uses 42 watermarked synthetic images and one training epoch. It proves
that generation, validation, grouped-capable splitting, training, validation selection, evaluation,
anomaly scoring, export, and parity checks execute. Its zero detection scores are explicitly retained
and must not be interpreted as model quality.

## References

- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Group-aware splitting concepts](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
- [TorchMetrics detection mAP](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)
