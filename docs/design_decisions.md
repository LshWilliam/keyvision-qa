# Design decisions

This log records the alternatives considered, not only the final implementation.

## DD-001: Two complementary inspection branches

**Decision:** combine supervised object detection with normal-only anomaly detection.

**Why:** labeled detection produces actionable defect names, while a normal model creates a route
for novel failure modes. A single branch leaves a predictable coverage gap.

**Rejected:** one multiclass image classifier. It cannot localize small defects and gives weak
evidence for operator review.

## DD-002: Faster R-CNN candidate plus a separate smoke detector

**Decision:** expose Torchvision Faster R-CNN MobileNetV3 FPN as the serious detector candidate and
use a tiny single-object CNN for CI and verified ONNX export.

**Why:** Faster R-CNN is a mature variable-object detector, but full CPU training and portable NMS
export are too slow and brittle for every CI run. The smoke model proves interfaces without
pretending to prove accuracy.

**Rejected:** downloading a pretrained YOLO checkpoint during tests. It adds a network dependency,
opaque cache state, and third-party weight terms to the basic validation path.

## DD-003: Portable JSONL manifests

**Decision:** use one record per image with relative paths and absolute `xywh` boxes.

**Why:** JSONL streams well, diffs at record granularity, supports zero or multiple boxes, and is
easy to convert to COCO or YOLO. Relative paths prevent machine identity leakage.

**Rejected:** storing Python pickle annotations. Pickle is unsafe for untrusted data and not
language-neutral.

## DD-004: Fixture-aligned Gaussian anomaly template

**Decision:** model per-location RGB and gradient statistics over normal images.

**Why:** it is interpretable, trains without downloading weights, produces a dense heatmap, and
matches the common industrial assumption of a controlled camera fixture.

**Rejected:** claiming PatchCore without a real feature backbone evaluation. PatchCore is a strong
future baseline, but a nominal implementation plus random features would be misleading.

## DD-005: Explicit synthetic watermark

**Decision:** draw a permanent red banner onto every defect example.

**Why:** filenames and captions can be separated from images. A pixel-level mark reduces the chance
that portfolio screenshots are mistaken for production evidence.

## DD-006: Artifacts stay out of Git

**Decision:** ignore raw datasets, runs, checkpoints, logs, and ONNX files.

**Why:** these are large, frequently regenerated, may have separate rights, and can contain local
metadata. Reproduction commands and small synthetic visualizations belong in Git; binaries do not.

## DD-007: Validation metrics remain transparent

**Decision:** implement IoU matching and 101-point interpolated AP in readable project code.

**Why:** the implementation is testable and explainable in interviews. For a production benchmark,
cross-check against COCO API because edge-case conventions differ.

## DD-008: ONNX export scope is narrow and verified

**Decision:** export the tiny raw-output model with opset 17 and compare ONNX Runtime output.

**Why:** numerical parity is stronger evidence than a successful file write. Faster R-CNN
post-processing and NMS require a target-runtime decision, so the project documents that boundary.

