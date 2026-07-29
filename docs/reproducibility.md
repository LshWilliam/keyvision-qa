# Reproducibility

## Supported workflow

```bash
python --version
python -m pip install -e ".[dev,demo,deploy,vision]"
python -m keyvision.data.synthetic --output artifacts/synthetic --count 42 --seed 42
python -m keyvision.data.validation --root artifacts/synthetic --manifest artifacts/synthetic/manifest.jsonl
python -m keyvision.training.train --config configs/smoke.yaml
python -m keyvision.evaluation.cli --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt
python -m keyvision.deployment.export_onnx --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt
```

The seed is stored in YAML and applied to Python, NumPy, and PyTorch. Data manifests, run summaries,
and ONNX parity reports are generated from commands rather than edited manually.

## Reproducibility boundaries

Exact floating-point results may vary by PyTorch build, CPU instruction set, CUDA/cuDNN version, and
operator implementation. Reproducible splitting and seeded initialization do not guarantee
bitwise-identical accelerator training. Capture the generated `run_summary.json`, dependency
versions, OS, GPU, and commit SHA with any reported experiment.

