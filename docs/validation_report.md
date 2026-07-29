# Local validation report

**Execution date:** 2026-07-29  
**Scope:** synthetic smoke verification only; not production model performance

## Environment

- Windows 10 build 26200
- Python 3.10.11
- PyTorch 2.10.0+cpu
- Physical GPU detected by `nvidia-smi`: NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB
- CUDA available to installed PyTorch: no

## Commands executed

```bash
python -m pip install -e ".[dev,demo,deploy,vision]"
ruff check .
ruff format --check .
mypy keyvision scripts
pytest --basetemp=artifacts/pytest_tmp
python scripts/smoke_test.py
python -m keyvision.data.stats --manifest artifacts/synthetic/manifest.jsonl
python -m keyvision.data.visualize --root artifacts/synthetic --manifest artifacts/synthetic/manifest.jsonl --output assets/synthetic_contact_sheet.png --limit 12
python -m keyvision.evaluation.cli --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt --split test
python -m keyvision.inference.cli --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt --input artifacts/synthetic/images/keyboard_0000.png
python -m keyvision.deployment.onnx_infer --config configs/smoke.yaml --model artifacts/models/keyvision_tiny.onnx --input artifacts/synthetic/images/keyboard_0000.png
python -m keyvision.training.train_anomaly --normal-dir artifacts/anomaly_normal
python scripts/error_analysis.py --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt
python scripts/benchmark.py --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt --onnx artifacts/models/keyvision_tiny.onnx --repetitions 30
python scripts/check_docs.py
```

The Gradio application was also launched on `127.0.0.1:7861`, returned HTTP 200 with a 20,800-byte
HTML response, and was closed immediately after the health check.

## Quality gates

| Gate | Actual result |
| --- | --- |
| Ruff lint | Passed |
| Ruff format check | Passed, 67 files formatted |
| mypy | Passed, 44 source files |
| pytest | 12 passed in 9.25 seconds |
| Warnings | 2 PyTorch legacy ONNX exporter deprecation warnings |
| Documentation links | 11 Markdown files passed |
| Dataset validation | 0 issues across 42 synthetic images |
| Demo health check | HTTP 200 |
| ONNX parity | Passed; maximum absolute difference 3.73e-08 |

## Synthetic execution result

- Split: 30 train, 6 validation, 6 test
- Training: 1 epoch, batch size 4, CPU, train loss 2.4733848572
- Test detections above 0.20: 0
- Test counts: 0 TP, 0 FP, 6 FN
- Precision, recall, F1, AP@50, AP@50:95: all 0.0
- Interpretation: the end-to-end pipeline executed correctly; the model was not trained enough to
  provide useful detections

## Batch-one forward benchmark

| Backend | Median | P95 | FPS from median | Model size |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CPU | 1.1107 ms | 1.9175 ms | 900.3 | 312,331 bytes |
| ONNX Runtime CPU | 0.41435 ms | 0.5602 ms | 2,413.4 | 99,561 bytes |

These timings cover the tiny smoke model only. They omit capture, rendering, storage, and line
integration. GPU comparison was not run because the installed PyTorch package is CPU-only.

