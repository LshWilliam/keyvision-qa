# Model Card

## Models

### Faster R-CNN MobileNetV3 FPN adapter

- **Role:** production-oriented known-defect candidate
- **Inputs:** RGB keyboard images and labeled defect boxes
- **Outputs:** multiple boxes, class identifiers, and confidence scores
- **Status:** implementation available; real-data training and evaluation not run
- **Weights:** not distributed

### Tiny single-defect detector

- **Role:** CI, training-loop, inference, and ONNX smoke fixture
- **Inputs:** square RGB tensors
- **Outputs:** at most one box, objectness, and class scores
- **Status:** exercised only on watermarked synthetic data
- **Limitation:** not a production accuracy architecture

### Gaussian template anomaly detector

- **Role:** unknown-anomaly baseline for fixed camera fixtures
- **Inputs:** aligned normal images for fitting; arbitrary inspection image for inference
- **Outputs:** scalar anomaly score and heatmap
- **Status:** synthetic smoke coverage only
- **Limitation:** sensitive to pose, registration, illumination, and legitimate appearance variation

## Evaluation

The project exports operational precision/recall/F1, support-aware 101-point AP, per-class PR data,
a confusion matrix, optional image-bootstrap confidence intervals, latency, FPS, parameters, and
file size. Training selects `best.pt` on validation mAP@50 rather than training loss. The readable
AP implementation is not fully COCO-equivalent and serious benchmarks require an official COCO API
cross-check. There are no production metrics in this release.

## Safety and human oversight

Do not use these models as the sole acceptance authority for safety-critical or contractual quality
decisions. Calibrate thresholds against real business costs, retain traceable evidence, monitor
drift, and provide an operator review route for uncertain or novel findings.

## Ethical and privacy considerations

Inspection cameras may capture serial numbers, operator hands, badges, screens, or facility details.
Apply field-of-view minimization, access controls, retention rules, and redaction appropriate to the
deployment jurisdiction and organization.

