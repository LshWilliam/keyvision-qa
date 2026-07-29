"""Train a configured known-defect detector with resumable checkpoints."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from keyvision.config import ProjectConfig, load_config
from keyvision.data.dataset import KeyboardDefectDataset, detection_collate
from keyvision.models.factory import build_detector
from keyvision.training.checkpoint import load_checkpoint, save_checkpoint
from keyvision.utils.runtime import (
    configure_logging,
    environment_report,
    resolve_device,
    seed_everything,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def train(config: ProjectConfig) -> dict[str, Any]:
    """Run training and return a truthful, serializable execution summary."""

    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    root = Path(config.data.root)
    dataset = KeyboardDefectDataset(
        root,
        root / config.data.train_manifest,
        config.data.image_size,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
        generator=generator,
    )
    model = build_detector(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_epoch = 0
    best_loss = float("inf")
    if config.training.resume:
        restored = load_checkpoint(config.training.resume, model, optimizer, device)
        start_epoch = int(restored["epoch"]) + 1
        best_loss = float(restored.get("best_loss", best_loss))
        LOGGER.info("Resumed from epoch %d", start_epoch)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
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
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        metadata = {"config": asdict(config), "history": history}
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_loss, metadata)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_loss, metadata)
        LOGGER.info("epoch=%d train_loss=%.6f", epoch + 1, epoch_loss)

    summary: dict[str, Any] = {
        "status": "completed",
        "result_scope": "synthetic smoke test" if "synthetic" in str(root) else "dataset run",
        "device": str(device),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "epochs_completed": len(history),
        "best_train_loss": best_loss,
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
    print(message + f"best_train_loss={summary['best_train_loss']}")


if __name__ == "__main__":
    main()
