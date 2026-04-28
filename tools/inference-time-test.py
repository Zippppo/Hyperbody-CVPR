"""
Benchmark forward-pass inference time for BodyNet.

Measures ONLY model(inp) latency (excluding data loading, argmax, saving).
Uses CUDA synchronization for accurate GPU timing.

Usage:

    python benchmark.py --config configs/021201-19.yaml --ckpt hyperbody-best.pth
    python benchmark.py --config configs/021201-19.yaml --ckpt hyperbody-best.pth --warmup 10 --repeat 3 --no-amp
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import ConcatDataset

from config import Config
from data.dataset import HyperBodyDataset
from data.organ_hierarchy import load_organ_hierarchy
from models.body_net import BodyNet


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BodyNet forward-pass latency")
    parser.add_argument("--config", type=str, default="configs/021201-19.yaml", help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default="hyperbody-best.pth", help="Checkpoint filename or path")
    parser.add_argument("--gpuids", type=int, default=0, help="GPU device ID")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat forward pass per sample (timings averaged)")
    parser.add_argument("--num-samples", type=int, default=-1, help="Limit number of samples (-1 = use whole split)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"],
                        help="Dataset split to benchmark ('all' = train+val+test)")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Force enable AMP (autocast fp16)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Force disable AMP")
    parser.set_defaults(amp=None)  # None => follow cfg.use_amp
    return parser.parse_args()


def load_model(cfg, ckpt_path, device):
    """Load BodyNet model and checkpoint (mirrors inference.py)."""
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

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

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Best Dice: {checkpoint.get('best_dice', 'N/A')}")
    return model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_forward(model, inp, device, use_amp, repeat):
    """Return mean forward time (ms) over `repeat` passes for a pre-loaded input."""
    timings = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                _ = model(inp)
        else:
            _ = model(inp)
        sync(device)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings))


@torch.no_grad()
def run_benchmark(model, dataset, device, use_amp, warmup, repeat, num_samples):
    n = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))
    print(f"Benchmark: {n} samples, warmup={warmup}, repeat={repeat}, AMP={use_amp}")

    # Warmup with first sample (pre-loaded to GPU)
    inp0, _ = dataset[0]
    inp0 = inp0.unsqueeze(0).to(device)
    print(f"Input shape: {tuple(inp0.shape)}")
    for i in range(warmup):
        sync(device)
        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                _ = model(inp0)
        else:
            _ = model(inp0)
        sync(device)
    print(f"Warmup done ({warmup} iters)")

    # Timed runs
    per_sample_ms = []
    for idx in range(n):
        inp, _ = dataset[idx]
        inp = inp.unsqueeze(0).to(device)
        ms = time_forward(model, inp, device, use_amp, repeat)
        per_sample_ms.append(ms)
        print(f"  [{idx + 1}/{n}] {ms:.2f} ms")
    return np.array(per_sample_ms)


def report(timings_ms):
    print("\n===== Forward-pass latency =====")
    print(f"  Samples : {len(timings_ms)}")
    print(f"  Mean    : {timings_ms.mean():.2f} ms")
    print(f"  Median  : {np.median(timings_ms):.2f} ms")
    print(f"  Std     : {timings_ms.std():.2f} ms")
    print(f"  Min     : {timings_ms.min():.2f} ms")
    print(f"  Max     : {timings_ms.max():.2f} ms")
    print(f"  P95     : {np.percentile(timings_ms, 95):.2f} ms")
    print(f"  P99     : {np.percentile(timings_ms, 99):.2f} ms")
    print(f"  Throughput : {1000.0 / timings_ms.mean():.2f} samples/s")


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    ckpt_path = os.path.join(cfg.checkpoint_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpuids}")
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    use_amp = cfg.use_amp if args.amp is None else args.amp

    model = load_model(cfg, ckpt_path, device)

    if args.split == "all":
        splits = [HyperBodyDataset(cfg.data_dir, cfg.split_file, s, cfg.volume_size) for s in ("train", "val", "test")]
        dataset = ConcatDataset(splits)
        print(f"Split 'all': {len(dataset)} samples ({', '.join(f'{s}={len(d)}' for s, d in zip(('train','val','test'), splits))})")
    else:
        dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, args.split, cfg.volume_size)
        print(f"Split '{args.split}': {len(dataset)} samples")

    timings = run_benchmark(model, dataset, device, use_amp, args.warmup, args.repeat, args.num_samples)
    report(timings)


if __name__ == "__main__":
    main()
