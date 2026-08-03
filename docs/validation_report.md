# Local validation report

**Execution date:** 2026-07-30

**Scope:** synthetic smoke verification only; not production model performance

## Environment

- Windows 10 build 26200
- Python 3.10.11
- PyTorch 2.10.0+cpu
- Physical GPU detected previously by `nvidia-smi`: NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB
- CUDA available to installed PyTorch: no

## Commands executed

```bash
ruff check .
ruff format --check .
mypy keyvision scripts tests
pytest --basetemp=artifacts/pytest_final_20260730 --cov=keyvision --cov-report=term-missing --cov-fail-under=60
python scripts/check_docs.py
python -m pip check
python -m build
python scripts/smoke_test.py
python scripts/benchmark.py --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt --onnx artifacts/models/keyvision_tiny.onnx --repetitions 30
git diff --check
```

## Quality gates

| Gate | Actual result |
| --- | --- |
| Ruff lint | Passed |
| Ruff format check | Passed, 73 files checked |
| mypy | Passed, 58 source files |
| pytest | 25 passed in 11.86 seconds |
| Coverage | 62.25%; required threshold 60% |
| Warnings | 2 PyTorch legacy ONNX exporter deprecation warnings |
| Documentation links | 13 Markdown files passed |
| Dependency consistency | `pip check` passed |
| Package build | sdist and wheel built without packaging deprecation warnings |
| Dataset validation | 0 issues across 42 synthetic images |
| ONNX parity | Passed; maximum absolute difference 2.24e-08 |

The tests include group-isolated split behavior, duplicate-content and path-traversal rejection,
class-schema conflict detection, absent-class AP semantics, deterministic bootstrap intervals, and
validation-metric checkpoint selection.

## Synthetic execution result

- Split: 30 train, 6 validation, 6 test
- Training: 1 epoch, batch size 4, CPU, train loss 2.5052678585
- Checkpoint selection: `best.pt` selected by validation AP@50; `last.pt` retained for recovery
- Best validation AP@50: 0.0
- Test detections above 0.20: 0
- Interpretation: the end-to-end pipeline executed correctly; one synthetic epoch is deliberately
  insufficient evidence of useful defect detection performance
- ONNX model: 99,561 bytes, export verified

## Batch-one forward benchmark

| Backend | Median | P95 | FPS from median | Model size |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CPU | 0.7435 ms | 0.7595 ms | 1,345.0 | 312,587 bytes |
| ONNX Runtime CPU | 0.3592 ms | 0.4604 ms | 2,784.4 | 99,561 bytes |

These timings cover the tiny smoke model only. They omit capture, rendering, storage, and line
integration. GPU comparison was not run because the installed PyTorch package is CPU-only.
