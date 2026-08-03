"""Train a configured known-defect detector with validation-based selection."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from keyvision.config import ProjectConfig, load_config
from keyvision.data.dataset import KeyboardDefectDataset, detection_collate
from keyvision.evaluation.metrics import evaluate_detections
from keyvision.models.base import DefectDetector
from keyvision.models.factory import build_detector
from keyvision.training.checkpoint import load_checkpoint, save_checkpoint
from keyvision.types import Detection
from keyvision.utils.runtime import (
    configure_logging,
    environment_report,
    resolve_device,
    seed_everything,
    seed_worker,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def _as_detections(output: dict[str, Tensor], class_names: tuple[str, ...]) -> list[Detection]:
    detections = []
    values = zip(
        output["boxes"].detach().cpu().tolist(),
        output["labels"].detach().cpu().tolist(),
        output["scores"].detach().cpu().tolist(),
        strict=True,
    )
    for box, label, score in values:
        class_id = int(label)
        detections.append(
            Detection(
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=class_id,
                class_name=class_names[class_id],
                score=float(score),
            )
        )
    return detections


def _targets_as_detections(
    target: dict[str, Tensor], class_names: tuple[str, ...]
) -> list[Detection]:
    detections = []
    values = zip(
        target["boxes"].detach().cpu().tolist(),
        target["labels"].detach().cpu().tolist(),
        strict=True,
    )
    for box, label in values:
        class_id = int(label)
        detections.append(
            Detection(
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=class_id,
                class_name=class_names[class_id],
            )
        )
    return detections


@torch.inference_mode()
def evaluate_model(
    model: DefectDetector,
    loader: DataLoader[Any],
    device: torch.device,
    class_names: tuple[str, ...],
    score_threshold: float,
) -> dict[str, Any]:
    """Evaluate one model snapshot on the validation loader."""

    predictions: list[list[Detection]] = []
    ground_truth: list[list[Detection]] = []
    model.eval()
    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        outputs = model.predict_tensors(device_images, score_threshold)
        predictions.extend(_as_detections(output, class_names) for output in outputs)
        ground_truth.extend(_targets_as_detections(target, class_names) for target in targets)
    return evaluate_detections(predictions, ground_truth, class_names)


def train(config: ProjectConfig) -> dict[str, Any]:
    """Train with deterministic loading and validation-mAP checkpoint selection."""

    seed_everything(config.training.seed, config.training.deterministic)
    device = resolve_device(config.training.device)
    root = Path(config.data.root)
    train_dataset = KeyboardDefectDataset(
        root,
        root / config.data.train_manifest,
        config.data.image_size,
    )
    val_dataset = KeyboardDefectDataset(
        root,
        root / config.data.val_manifest,
        config.data.image_size,
    )
    train_generator = torch.Generator().manual_seed(config.training.seed)
    val_generator = torch.Generator().manual_seed(config.training.seed + 1)
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
        generator=val_generator,
        worker_init_fn=seed_worker,
    )
    model = build_detector(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_epoch = 0
    best_train_loss = float("inf")
    best_val_map50 = float("-inf")
    history: list[dict[str, float | int]] = []
    if config.training.resume:
        restored = load_checkpoint(config.training.resume, model, optimizer, device)
        start_epoch = int(restored["epoch"]) + 1
        best_train_loss = float(restored.get("best_loss", best_train_loss))
        metadata = restored.get("metadata", {})
        if isinstance(metadata, dict):
            best_val_map50 = float(metadata.get("best_val_map50", best_val_map50))
            restored_history = metadata.get("history", [])
            if isinstance(restored_history, list):
                history = [dict(item) for item in restored_history if isinstance(item, dict)]
        LOGGER.info("Resumed from epoch %d", start_epoch)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()} for target in targets
            ]
            losses = model.compute_loss(images, targets)
            total_loss = torch.stack(tuple(losses.values())).sum()
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {total_loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            running_loss += float(total_loss.detach())
            batches += 1

        epoch_loss = running_loss / max(batches, 1)
        best_train_loss = min(best_train_loss, epoch_loss)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            config.model.class_names,
            config.training.validation_score_threshold,
        )
        val_map50 = float(val_metrics["map50"])
        is_best = val_map50 > best_val_map50
        if is_best:
            best_val_map50 = val_map50
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
                "val_map50": val_map50,
                "val_map50_95": float(val_metrics["map50_95"]),
            }
        )
        metadata = {
            "config": asdict(config),
            "history": history,
            "selection_metric": "val_map50",
            "best_val_map50": best_val_map50,
        }
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_train_loss, metadata)
        if is_best:
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch, best_train_loss, metadata
            )
        LOGGER.info(
            "epoch=%d train_loss=%.6f val_map50=%.4f val_f1=%.4f",
            epoch + 1,
            epoch_loss,
            val_map50,
            val_metrics["f1"],
        )

    summary: dict[str, Any] = {
        "status": "completed",
        "result_scope": "synthetic smoke test" if "synthetic" in str(root) else "dataset run",
        "device": str(device),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "epochs_completed": len(history),
        "selection_metric": "val_map50",
        "best_train_loss": best_train_loss,
        "best_val_map50": None if best_val_map50 == float("-inf") else best_val_map50,
        "history": history,
        "environment": environment_report(),
        "config": asdict(config),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    configure_logging()
    summary = train(load_config(args.config))
    message = f"Training completed: {summary['result_scope']}; "
    print(message + f"best_val_map50={summary['best_val_map50']}")


if __name__ == "__main__":
    main()
