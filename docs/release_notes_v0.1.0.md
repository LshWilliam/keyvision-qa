# KeyVision-QA v0.1.0

## Current capabilities

- Validated JSONL data contract and deterministic split
- Watermarked synthetic pipeline examples
- Faster R-CNN adapter and export-friendly smoke detector
- Resumable training and transparent detection evaluation
- Gaussian anomaly heatmaps and confidence-ranked FP/FN analysis
- PyTorch, ONNX Runtime, batch, camera, and Gradio entry points
- Ruff, mypy, pytest, end-to-end smoke test, CI, and Docker

## Run

```bash
python -m pip install -e ".[dev,demo,deploy,vision]"
python scripts/smoke_test.py
python app.py
```

## Limitations

No authorized real dataset or production metric is included. The tiny detector is a CI fixture,
the anomaly model assumes image registration, and verified ONNX export currently covers only the
tiny model.

## Next

Priorities are a licensed real dataset with grouped splits, tiled small-defect detection, domain
adaptation evaluation, and target-device quantization.

