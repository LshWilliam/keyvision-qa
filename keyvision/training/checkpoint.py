"""Safe, resumable PyTorch checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_loss: float,
    metadata: dict[str, Any],
) -> None:
    """Save state required to resume an interrupted training run."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        },
        output,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model state and optional optimizer state."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint has an invalid structure")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
