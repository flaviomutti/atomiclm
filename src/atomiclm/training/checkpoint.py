"""
Checkpoint management for training state persistence.

Saves and loads model weights, optimizer state, scheduler state,
and training progress to enable resumable training.
"""

import os
import random
from dataclasses import asdict
from typing import Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    config: Any,
    is_best: bool = False,
) -> None:
    """
    Save training checkpoint.

    Directory structure:
        checkpoint_dir/
            checkpoint_latest.pt       # Always updated
            checkpoint_best.pt         # Best validation loss
            checkpoint_step_N.pt       # Periodic snapshots
            config.json                # Training config
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    rng_state = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
        "rng_state": rng_state,
    }

    # Always save latest
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    # Save best if requested
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)

    # Save periodic snapshot
    snapshot_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, snapshot_path)

    # Save config as readable JSON
    config.to_json(os.path.join(checkpoint_dir, "config.json"))


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> dict:
    """
    Load training checkpoint and restore state.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: Model instance (updated in-place).
        optimizer: Optional optimizer (updated in-place).
        scheduler: Optional scheduler (updated in-place).
        device: Device to map tensors to during loading.

    Returns:
        Dict with keys: global_step, config.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG state for reproducible resume
    if "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        torch.random.set_rng_state(rng["torch_cpu"].cpu())
        if torch.cuda.is_available() and "torch_cuda" in rng:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])

    return {
        "global_step": checkpoint["global_step"],
        "config": checkpoint.get("config"),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return path to checkpoint_latest.pt if it exists, else None."""
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    return latest_path if os.path.exists(latest_path) else None
