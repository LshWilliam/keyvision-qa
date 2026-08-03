# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic
versioning.

## [Unreleased]

### Added

- Group-aware dataset splitting and SHA-256 duplicate-content validation
- Validation-mAP checkpoint selection with deterministic DataLoader worker seeding
- Support-aware AP semantics and optional image-bootstrap confidence intervals
- 25-test suite with a 60% whole-package coverage gate
- Reproducible CI constraints, dependency checks, documentation validation, and package builds


### Planned

- Real, licensed keyboard dataset integration and production-oriented benchmarks
- Quantized edge inference and drift monitoring

## [0.1.0] - 2026-07-29

### Added

- Dataset schema, deterministic split, validation, statistics, visualization, and synthetic generator
- Unified tiny and Faster R-CNN detector adapters with resumable training
- Gaussian template anomaly baseline with heatmaps
- Detection metrics, confusion matrix, PR data, benchmark helpers, and FP/FN analysis
- PyTorch inference, ONNX export/parity validation, ONNX Runtime, camera/batch CLI, and Gradio demo
- Unit tests, end-to-end smoke workflow, CI, Docker, project cards, and interview documentation

