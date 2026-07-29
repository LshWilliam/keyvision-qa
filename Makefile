.PHONY: install lint format type test smoke demo generate validate train evaluate export

install:
	python -m pip install -e ".[dev,demo,deploy,vision]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy keyvision scripts

test:
	pytest

smoke:
	python scripts/smoke_test.py

demo:
	python app.py

generate:
	python -m keyvision.data.synthetic --output artifacts/synthetic --count 42

validate:
	python -m keyvision.data.validation --root artifacts/synthetic --manifest artifacts/synthetic/manifest.jsonl

train:
	python -m keyvision.training.train --config configs/default.yaml

evaluate:
	python -m keyvision.evaluation.cli --config configs/default.yaml --checkpoint artifacts/runs/fasterrcnn/best.pt

export:
	python -m keyvision.deployment.export_onnx --config configs/smoke.yaml --checkpoint artifacts/runs/smoke/best.pt

