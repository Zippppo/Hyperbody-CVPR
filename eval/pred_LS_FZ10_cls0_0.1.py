"""
Prediction script for L_S_FZ+cls0_0.3 model (BodyNet with hyperbolic embedding + semantic direction).

This model uses:
- hyp_freeze_epochs=10: Text embeddings frozen for first 10 epochs
- hyp_direction_mode="semantic": Uses BioLORD text embedding directions

Usage:
    python eval/pred_LS_FZ10_cls0_0.1.py --config configs/L_S_FZ+cls0_0.1.yaml --output eval/pred/L_S_FZ+cls0_0.1
"""
import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from config import Config
from data.dataset import HyperBodyDataset
from data.organ_hierarchy import load_organ_hierarchy
from models.body_net import BodyNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test set with BodyNet (L_S_FZ+cls0_0.3)")
    parser.add_argument("--config", type=str, default="configs/L_S_FZ+cls0_0.3.yaml", help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default="best.pth", help="Checkpoint filename")
    parser.add_argument("--output", type=str, default="eval/pred/L_S_FZ+cls0_0.3", help="Output directory")
    parser.add_argument("--gpuids", type=int, default=0, help="GPU device ID to use")
    return parser.parse_args()


def load_model(cfg, ckpt_path, device):
    """Load BodyNet model and checkpoint."""
    # Load class depths from tree.json
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

    # Create model
    model = BodyNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        growth_rate=cfg.growth_rate,
        dense_layers=cfg.dense_layers,
        bn_size=cfg.bn_size,
        embed_dim=cfg.hyp_embed_dim,
        curv=cfg.hyp_curv,
        class_depths=class_depths,
        min_radius=cfg.hyp_min_radius,
        max_radius=cfg.hyp_max_radius,
        direction_mode=cfg.hyp_direction_mode,
        text_embedding_path=cfg.hyp_text_embedding_path,
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Handle DDP checkpoint (strip "module." prefix if present)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
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
        inp = inp.unsqueeze(0).to(device)  # Add batch dim: (1, C, D, H, W)

        # Forward pass with AMP for memory efficiency
        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                logits, voxel_emb, label_emb = model(inp)
        else:
            logits, voxel_emb, label_emb = model(inp)

        # Get prediction (argmax over class dimension)
        pred_labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (D, H, W)

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
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpuids}")
    else:
        device = torch.device("cpu")
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
