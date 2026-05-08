"""
Prediction script for baseline model (UNet3D without hyperbolic).

Usage:
    python eval/pred_baseline.py --config configs/baseline.yaml
    python eval/pred_baseline.py --config configs/baseline.yaml --ckpt epoch_50.pth
"""
import argparse
import os

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import Config
from data.dataset import HyperBodyDataset
from models.unet3d import UNet3D


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default="best.pth", help="Checkpoint filename")
    parser.add_argument("--output", type=str, default="eval/pred/baseline", help="Output directory")
    return parser.parse_args()


def load_model(cfg, ckpt_path, device):
    """Load UNet3D model and checkpoint (baseline without hyperbolic)."""
    model = UNet3D(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        growth_rate=cfg.growth_rate,
        dense_layers=cfg.dense_layers,
        bn_size=cfg.bn_size,
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")

    return model


@torch.no_grad()
def run_inference(model, dataset, output_dir, data_dir, device, use_amp=True):
    """Run inference on all samples and save predictions."""
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Inference"):
        # Get input tensor
        inp, _ = dataset[idx]
        inp = inp.unsqueeze(0).to(device)  # Add batch dim

        # Forward pass with AMP for memory efficiency
        if use_amp and device.type == "cuda":
            with autocast():
                logits = model(inp)
        else:
            logits = model(inp)

        # Get prediction (argmax)
        pred_labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (X, Y, Z)

        # Load original data for metadata
        filename = dataset.filenames[idx]
        original_path = os.path.join(data_dir, filename)
        original_data = np.load(original_path)

        # Save prediction with metadata (NO GT labels)
        output_path = os.path.join(output_dir, filename)
        np.savez_compressed(
            output_path,
            pred_labels=pred_labels.astype(np.int64),
            grid_world_min=original_data["grid_world_min"],
            grid_voxel_size=original_data["grid_voxel_size"],
            original_filename=filename,
        )

    print(f"Saved {len(dataset)} predictions to {output_dir}")


def main():
    args = parse_args()

    # Load config
    cfg = Config.from_yaml(args.config)

    # Determine checkpoint path
    ckpt_path = os.path.join(cfg.checkpoint_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        # Try direct path
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(cfg, ckpt_path, device)

    # Load test dataset
    test_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "test", cfg.volume_size)
    print(f"Test samples: {len(test_dataset)}")

    # Run inference with AMP
    run_inference(model, test_dataset, args.output, cfg.data_dir, device, use_amp=cfg.use_amp)


if __name__ == "__main__":
    main()
