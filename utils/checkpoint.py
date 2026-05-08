import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional


def save_checkpoint(
    state: Dict,
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
) -> str:
    """
    Save checkpoint to specified directory.

    Args:
        state: Dictionary containing checkpoint data
            Required keys: 'epoch', 'model_state_dict'
            Optional keys: 'optimizer_state_dict', 'scheduler_state_dict',
                          'best_dice', 'config', etc.
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename (e.g., 'latest.pth', 'epoch_10.pth')
        is_best: If True, also save as 'best.pth'

    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)

    # If this is the best model, also save as best.pth
    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(state, best_path)

    return str(checkpoint_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> Tuple[int, float]:
    """
    Load checkpoint and restore model, optimizer, and scheduler states.

    Args:
        path: Path to checkpoint file
        model: Model to load state into (can be DataParallel, DDP, or regular model)
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map checkpoint tensors to

    Returns:
        Tuple of (start_epoch, best_dice)
            start_epoch: Epoch to resume training from (epoch + 1)
            best_dice: Best validation Dice score from checkpoint

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)

    # Load model state
    # Handle DataParallel, DDP, and regular models
    model_state_dict = checkpoint["model_state_dict"]

    # Check if model is wrapped (DataParallel or DDP) - both have 'module' attribute
    is_model_wrapped = hasattr(model, 'module')
    # Check if checkpoint is from wrapped model (keys start with 'module.')
    is_checkpoint_parallel = any(key.startswith("module.") for key in model_state_dict.keys())

    if is_model_wrapped and not is_checkpoint_parallel:
        # Model is wrapped but checkpoint is not
        # Add 'module.' prefix to checkpoint keys
        model_state_dict = {f"module.{k}": v for k, v in model_state_dict.items()}
    elif not is_model_wrapped and is_checkpoint_parallel:
        # Model is not wrapped but checkpoint is
        # Remove 'module.' prefix from checkpoint keys
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model.load_state_dict(model_state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Extract epoch and best_dice
    epoch = checkpoint.get("epoch", 0)
    best_dice = checkpoint.get("best_dice", 0.0)

    # Start from next epoch
    start_epoch = epoch + 1

    return start_epoch, best_dice


def get_checkpoint_info(path: str) -> Dict:
    """
    Get information from checkpoint without loading into model.

    Args:
        path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    info = {
        "epoch": checkpoint.get("epoch", "Unknown"),
        "best_dice": checkpoint.get("best_dice", "Unknown"),
    }

    # Add config if present
    if "config" in checkpoint:
        info["config"] = checkpoint["config"]

    return info
