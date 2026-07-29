# Interview guide

## 30-second introduction

KeyVision-QA is an end-to-end industrial vision project for keyboard inspection. It combines a
supervised detector for six known defect types with a normal-only anomaly model for unseen defects.
I built the data contract, validation and reproducible split, unified model interface, resumable
training, transparent detection metrics, confidence-ranked error analysis, ONNX parity gate, batch
and camera inference, Gradio UI, tests, CI, and documentation. Because I had no authorized factory
data, I use visibly watermarked synthetic images only for smoke testing and make no production
accuracy claim.

## Two-minute introduction

The business problem has two different uncertainty sources. Known defects such as missing keycaps,
print damage, stains, scratches, or foreign objects need names and bounding boxes because those map
to quality rules and root-cause actions. At the same time, a closed taxonomy cannot describe every
future failure, so the second branch models normal appearance and returns an anomaly score and
heatmap.

The production-oriented detector adapter uses Faster R-CNN MobileNetV3 FPN behind a common interface.
A small single-defect CNN is deliberately kept as a CI and ONNX fixture; it proves the full
train/checkpoint/infer/export path quickly but is not positioned as a production model. The anomaly
baseline fits position-dependent RGB and gradient statistics from aligned normal images, making its
behavior interpretable but dependent on a stable camera fixture.

The engineering work matters as much as the networks. JSONL manifests use relative paths and
validated boxes. Splits are deterministic and class-aware. Training saves best and last
checkpoints, supports resume, fixes seeds, and records configuration and environment. Evaluation
exports precision, recall, F1, AP@50, AP@50:95, per-class PR data, a confusion matrix, and runtime
measurements. Error analysis preserves false positives and false negatives. ONNX export is accepted
only after numerical comparison with PyTorch. A Gradio app demonstrates both model branches.

The most important integrity choice is what I did not claim: there is no public real keyboard
dataset in this workspace, so synthetic results are explicitly smoke results. My next step would be
an authorized, lot-grouped dataset with real capture metadata and a frozen acceptance-cost
evaluation.

## Three hardest problems

### 1. Separating interface proof from accuracy evidence

An end-to-end demo needs data, but synthetic data can make a weak model look deceptively good. I
solved this by watermarking pixels, using a separate smoke architecture, excluding generated metrics
from production claims, and leaving real benchmark cells unfilled.

### 2. Unifying heterogeneous detector behavior

Faster R-CNN accepts lists of variable-size targets and changes its return type between train and
evaluation. The smoke detector emits a dense tensor. The `DefectDetector` contract separates
`compute_loss` from `predict_tensors`, normalizes labels to zero-based identifiers, and presents one
inference shape to downstream evaluation and deployment.

### 3. Defining a truthful ONNX boundary

Detection export is more than writing a graph: preprocessing, decoding, dynamic axes, NMS, and
runtime operators must agree. The project exports an intentionally narrow raw-output graph, fixes
opset 17, and compares real ONNX Runtime output with PyTorch. Faster R-CNN NMS export is documented
as incomplete rather than implied.

## Why use detection and anomaly detection together?

Detection has semantic precision but a closed label set. Anomaly detection has open-set sensitivity
but weak semantics and can confuse harmless domain variation with a defect. Combining them permits
class-specific automatic actions for calibrated known classes and a review queue for novel
appearance. Fusion must be validated against business costs; simply taking the maximum score can
inflate false rejects.

## Handling class imbalance

- Measure image and instance counts, defect size, and lot distribution per class.
- Split by lot or capture session before rebalancing to prevent leakage.
- Use loss weighting or focal loss only after confirming which errors imbalance causes.
- Oversample rare classes with bounded augmentation, not duplicate adjacent frames across splits.
- Add targeted data for rare capture conditions, not only synthetic object pastes.
- Report macro and per-class metrics alongside aggregate AP.
- Tune thresholds by class when business costs and calibration support it.

## Reducing false positives and false negatives

Start with confidence-ranked error slices, not a global threshold change. For false positives, mine
hard normal examples such as reflections, legends, keyboard edges, and fixtures. For false
negatives, inspect box size, contrast, truncation, occlusion, and annotation consistency. Improve
capture conditions before model complexity when possible. Calibrate scores on validation data and
choose operating points from the cost of false accepts versus false rejects. Keep the test set
frozen.

## Small-defect strategy

- Increase optical resolution before digital upscaling.
- Validate focus, motion blur, and pixel coverage at line speed.
- Use tiled or overlapping inference with duplicate suppression.
- Preserve high-resolution feature maps or add a small-object feature level.
- Audit augmentation so tiny boxes are not erased.
- Stratify recall and AP by box area.
- Compare detector localization against segmentation when scratch shape matters.

## Illumination robustness

First stabilize hardware with diffuse lighting, exposure lock, shielding, and possibly polarization.
Then measure brightness, color temperature, glare, and shift as explicit test slices. Use bounded
photometric augmentation and normalization derived from real capture variation. For anomaly
detection, maintain references by camera and product family and alert on global score drift rather
than silently updating the normal template.

## Generalization evaluation

Create grouped splits by keyboard SKU, production lot, capture date, camera, and lighting cell.
Report within-domain and held-out-domain metrics, calibration error, uncertainty, defect-size
slices, and operational false reject/accept rates. Avoid frame-level random splitting when adjacent
frames share nearly identical products. A temporal or site holdout is stronger evidence than a
larger random test set.

## PyTorch-to-ONNX problems

- Unsupported or version-dependent operators
- Dynamic image shapes and batch axes
- Non-maximum suppression availability and coordinate conventions
- Train/eval branch differences and dropout or batch-normalization state
- Python-side preprocessing or post-processing outside the graph
- Numerical drift from different kernels
- Fixed constants accidentally traced from sample inputs
- Runtime provider differences and quantization sensitivity

The mitigation is a fixed export contract, representative samples, shape tests, ONNX checker,
PyTorch/ORT tolerance checks, decoded-output comparison, and target-device benchmarks.

## CPU versus GPU inference

GPU throughput benefits large batches and parallel convolution, but transfer, kernel launch, and
warmup can dominate batch-one latency. CPU is simpler, cheaper, and often more predictable at the
edge. Compare median and tail latency after warmup on the exact target, include preprocessing and
post-processing, pin thread settings, and measure power and memory. A workstation GPU benchmark
does not predict an embedded deployment.

## Deployment optimization path

1. Profile capture, preprocessing, model, NMS, serialization, and UI separately.
2. Fix the target accuracy and operational error budget.
3. Reduce resolution or use tiling based on defect pixel size.
4. Select a smaller backbone and validate score calibration.
5. Export to ONNX and fuse supported operations.
6. Try FP16 on compatible GPUs and INT8 with representative calibration data.
7. Batch only if latency and line buffering allow it.
8. Add asynchronous capture and bounded queues.
9. Monitor latency, input drift, score drift, and review outcomes.

## Current limitations

There is no authorized real dataset, no production metric, no calibrated decision threshold, and no
camera/PLC integration. The tiny model predicts at most one defect. The anomaly baseline assumes
registration. Faster R-CNN ONNX NMS is not implemented. The current local PyTorch build may be
CPU-only even when physical GPU hardware exists.

## Follow-up improvements

The highest-value work is an explicitly licensed real dataset with lot-based splits, small-defect
tiled detection, and target-hardware quantized inference. Domain adaptation is useful only after a
credible source/target evaluation protocol exists.

## Fifteen likely questions and reference answers

### 1. Why Faster R-CNN instead of YOLO?

I chose a mature Torchvision implementation with clear internals and strong small-object potential
as the production candidate. It is not a claim that Faster R-CNN will win. I would benchmark a
modern one-stage model under the same grouped splits, latency boundary, input resolution, and
annotation policy before selection.

### 2. Why is the tiny model in the repository?

It keeps CI deterministic and inexpensive while exercising optimization, checkpoint resume,
inference decoding, ONNX export, and runtime parity. Its explicit single-object limitation prevents
it from being confused with the production candidate.

### 3. How do you avoid data leakage?

Use relative manifests tied to immutable source IDs, hash or perceptually deduplicate images, and
group splits by lot/session/SKU rather than random frames. Any augmentation happens after splitting.
The current synthetic split is deterministic, but real grouped splitting still needs data metadata.

### 4. Is AP implementation identical to COCO API?

It follows 101-point interpolation over IoU 0.50 to 0.95 for explainability, but edge conventions
can differ. For a publication-quality result I would cross-check predictions with pycocotools and
store both tool versions and converted annotations.

### 5. How would you select the confidence threshold?

On validation data, translate false accepts and false rejects into operational cost, plot the
precision-recall or cost curve per class, and select thresholds under throughput constraints. Never
tune on the final test set.

### 6. What happens when both branches disagree?

Use an explicit policy: high-confidence known defects can trigger the class rule; high anomaly with
no known box enters review; low-confidence disagreement is retained for active learning. The policy
needs calibration and an audit trail.

### 7. How would you validate anomaly detection?

Fit only normal training images, set thresholds on normal validation plus representative anomalies,
and test on held-out lots with both known and novel defect families. Report image AUROC, pixel AUROC,
PRO or localization metrics, and operational false-positive rate—none are claimed here yet.

### 8. How do you handle label noise?

Define a visual annotation guide, double-review ambiguous classes, measure agreement, inspect
high-loss and model-disagreement samples, and distinguish “not visible” from “not annotated.”
Version corrections without moving test cases into training.

### 9. Why store both best and last checkpoints?

`last` is for recovery after interruption. `best` is selected by an explicit validation criterion
for inference. In the smoke loop the current selection uses training loss and is labeled accordingly;
a real experiment must select by validation metric.

### 10. What makes the run reproducible?

Typed YAML, fixed seeds, deterministic split, captured environment, preserved manifests,
checkpoints, commands, and CI. Reproducible does not mean bitwise identical across different GPU
kernels, so hardware and library versions remain part of the report.

### 11. Why no real images?

No authorized dataset was available. Publishing employer, customer, or scraped copyrighted images
would weaken the project. The honest solution is a marked synthetic integration fixture plus a
clear real-data protocol.

### 12. How would you integrate with a production line?

Use triggered acquisition, exposure/focus health checks, bounded queues, versioned model service,
PLC/MES result contracts, reject confirmation, image retention policy, operator review, and
monitoring for uptime, latency, drift, and false decisions. Fail-safe behavior must be defined with
manufacturing engineering.

### 13. What would you monitor?

Input dimensions and exposure, blur, camera availability, preprocessing failures, inference median
and tail latency, queue depth, class/score/box-size distributions, anomaly drift, operator override
rate, and delayed labeled performance by SKU and lot.

### 14. How would quantization be evaluated?

Calibrate INT8 on representative authorized images, compare decoded boxes and scores to FP32,
measure per-class accuracy at the chosen operating point, benchmark warm and cold latency, memory,
and power on the target device, and investigate any small-defect recall loss.

### 15. What part should an interviewer read first?

`keyvision/models/base.py` for the model boundary, `keyvision/data/validation.py` for the data gate,
`keyvision/training/train.py` for reproducibility and resume, `keyvision/evaluation/metrics.py` for
metric reasoning, and `keyvision/deployment/export_onnx.py` for the deployment acceptance gate.

