import argparse
import json
import math
import os
import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm

from config import Config
from data.dataset import HyperBodyDataset
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system, compute_tree_distance_matrix
from models.hyperbolic.embedding_tracker import EmbeddingTracker
from models.body_net import BodyNet
from models.losses import CombinedLoss, compute_class_weights
from models.hyperbolic.lorentz_loss import LorentzRankingLoss, LorentzTreeRankingLoss
from utils.metrics import DiceMetric
from utils.checkpoint import save_checkpoint, load_checkpoint

##CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4  train.py --config configs/lorentz_random.yaml
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(gpu_ids=None):
    """Initialize distributed training if launched with torchrun.

    If gpu_ids is provided AND we are NOT already in a torchrun env,
    set CUDA_VISIBLE_DEVICES so only specified GPUs are visible.

    Returns:
        local_rank: Local rank of this process (0 if not distributed)
    """
    if gpu_ids is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for body segmentation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--gpuids", type=str, default=None, help="GPU IDs (e.g., '0,1')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def setup_logging(log_dir: str, is_main: bool = True):
    """Setup console and file logging.

    Args:
        log_dir: Directory to store log files
        is_main: If True, log to file and console. If False, use NullHandler.
    """
    os.makedirs(log_dir, exist_ok=True)

    handlers = []
    if is_main:
        log_file = os.path.join(log_dir, "training.log")
        handlers.append(logging.FileHandler(log_file))
        handlers.append(logging.StreamHandler())
    else:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def load_precomputed_graph_distance_matrix(graph_distance_path: str, logger: logging.Logger) -> torch.Tensor:
    """Load graph distance matrix for graph-mode sampling and validate path."""
    if not graph_distance_path or not os.path.exists(graph_distance_path):
        raise FileNotFoundError(
            "Graph distance mode requires precomputed graph_distance_matrix. "
            "Run: python scripts/precompute_graph_distance.py --output-dir ... "
            "Then set graph_distance_matrix in config."
        )

    graph_dist_matrix = torch.load(graph_distance_path, map_location="cpu")
    logger.info(f"Loaded graph distance matrix from {graph_distance_path}")
    return graph_dist_matrix.float()


def train_one_epoch(model, loader, seg_criterion, hyp_criterion, hyp_weight, optimizer, device, grad_clip, epoch=0, scaler=None, cfg=None):
    """Train for one epoch.

    Args:
        model: The model to train (BodyNet)
        loader: DataLoader for training data
        seg_criterion: Segmentation loss function (CombinedLoss)
        hyp_criterion: Hyperbolic loss function (LorentzRankingLoss)
        hyp_weight: Weight for hyperbolic loss
        optimizer: Optimizer
        device: Device to use
        grad_clip: Gradient clipping value (0 to disable)
        epoch: Current epoch number (used for DistributedSampler shuffling)
        scaler: Optional GradScaler for AMP training (None to disable AMP)
        cfg: Config object (for hyp_freeze_epochs, hyp_text_grad_clip)

    Returns:
        Tuple of (avg_total_loss, avg_seg_loss, avg_hyp_loss)
    """
    model.train()

    # Set epoch for DistributedSampler (ensures proper shuffling)
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    hyp_loss_sum = 0.0
    num_batches = 0

    # Check if this is the first unfreeze epoch (for extra gradient clipping)
    is_first_unfreeze_epoch = (cfg is not None and
                               cfg.hyp_freeze_epochs > 0 and
                               epoch == cfg.hyp_freeze_epochs)

    # Only show progress bar on main process
    pbar = tqdm(loader, desc="  Train", leave=False, disable=not is_main_process())
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                logits, voxel_emb, label_emb = model(inputs)
                seg_loss = seg_criterion(logits, targets)
                hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
                total_loss = seg_loss + hyp_weight * hyp_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Extra gradient clipping for text embeddings on first unfreeze epoch
            if is_first_unfreeze_epoch:
                raw_model = model.module if hasattr(model, 'module') else model
                nn.utils.clip_grad_norm_(raw_model.label_emb.tangent_embeddings, cfg.hyp_text_grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, voxel_emb, label_emb = model(inputs)
            seg_loss = seg_criterion(logits, targets)
            hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
            total_loss = seg_loss + hyp_weight * hyp_loss

            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Extra gradient clipping for text embeddings on first unfreeze epoch
            if is_first_unfreeze_epoch:
                raw_model = model.module if hasattr(model, 'module') else model
                nn.utils.clip_grad_norm_(raw_model.label_emb.tangent_embeddings, cfg.hyp_text_grad_clip)
            optimizer.step()

        total_loss_sum += total_loss.item()
        seg_loss_sum += seg_loss.item()
        hyp_loss_sum += hyp_loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", seg=f"{seg_loss.item():.4f}", hyp=f"{hyp_loss.item():.4f}")

    n = max(num_batches, 1)
    return total_loss_sum / n, seg_loss_sum / n, hyp_loss_sum / n


@torch.no_grad()
def validate(model, loader, seg_criterion, hyp_criterion, hyp_weight, metric, device):
    """Validate and compute metrics.

    Returns:
        Tuple of (val_total_loss, val_seg_loss, val_hyp_loss, dice_per_class, mean_dice).
    """
    model.eval()
    metric.reset()
    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    hyp_loss_sum = 0.0
    num_batches = 0

    # Only show progress bar on main process
    pbar = tqdm(loader, desc="  Val  ", leave=False, disable=not is_main_process())
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, voxel_emb, label_emb = model(inputs)
        seg_loss = seg_criterion(logits, targets)
        hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        total_loss_sum += total_loss.item()
        seg_loss_sum += seg_loss.item()
        hyp_loss_sum += hyp_loss.item()
        num_batches += 1

        metric.update(logits, targets)
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    # Sync metrics across processes before compute
    if is_distributed():
        metric.sync_across_processes()

    n = max(num_batches, 1)
    dice_per_class, mean_dice, _ = metric.compute()

    return total_loss_sum / n, seg_loss_sum / n, hyp_loss_sum / n, dice_per_class, mean_dice


def main():
    args = parse_args()

    # Load config from YAML or use defaults
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    # Override config with CLI args
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.gpuids is not None:
        cfg.gpu_ids = [int(x) for x in args.gpuids.split(",")]
    if args.resume:
        cfg.resume = args.resume

    # Setup distributed training (passes gpu_ids to set CUDA_VISIBLE_DEVICES)
    local_rank = setup_distributed(gpu_ids=cfg.gpu_ids)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create timestamped log directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = os.path.join(cfg.log_dir, timestamp)

    # Setup logging (only main process logs)
    logger = setup_logging(run_log_dir, is_main=is_main_process())
    logger.info("=" * 60)
    logger.info("Training 3D U-Net with Dense Bottleneck")
    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Config: {cfg}")

    # Log distributed info
    if is_distributed():
        logger.info(f"Distributed training: rank {get_rank()}/{get_world_size()}, local_rank {local_rank}")
    logger.info(f"Device: {device}")

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "train", cfg.volume_size)
    val_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "val", cfg.volume_size)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Compute class weights (with caching)
    class_weights_cache = os.path.join(cfg.checkpoint_dir, "class_weights.pt")
    if os.path.exists(class_weights_cache):
        logger.info(f"Loading cached class weights from {class_weights_cache}")
    else:
        logger.info("Computing class weights from 100 samples (will be cached)...")
    class_weights = compute_class_weights(
        train_dataset, cfg.num_classes, num_samples=100, cache_path=class_weights_cache
    )
    class_weights = class_weights.to(device)
    logger.info(f"Class weights range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")

    # Load organ hierarchy for hyperbolic embeddings
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)
    class_to_system = load_class_to_system(cfg.tree_file, class_names) 

    # Model
    logger.info("Creating model...")
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

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    model = model.to(device)

    # Wrap model with DDP if distributed
    # find_unused_parameters=True is needed when label embeddings are frozen
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        logger.info(f"Using DistributedDataParallel on {get_world_size()} GPUs")

    # Segmentation loss, optimizer, scheduler
    criterion = CombinedLoss(
        num_classes=cfg.num_classes,
        ce_weight=cfg.ce_weight,
        dice_weight=cfg.dice_weight,
        class_weights=class_weights,
        dice_ignore_index=cfg.dice_ignore_index,
    )
    if cfg.dice_ignore_index is not None:
        logger.info(f"Dice loss ignoring class {cfg.dice_ignore_index}")

    # Hyperbolic ranking loss (with Curriculum Negative Mining)
    # Choose loss class based on hyp_distance_mode config
    if cfg.hyp_distance_mode == "tree":
        # Tree-based negative sampling: uses precomputed tree distances
        tree_dist_matrix = compute_tree_distance_matrix(cfg.tree_file, class_names)
        hyp_criterion = LorentzTreeRankingLoss(
            tree_dist_matrix=tree_dist_matrix,
            margin=cfg.hyp_margin,
            curv=cfg.hyp_curv,
            num_samples_per_class=cfg.hyp_samples_per_class,
            num_negatives=cfg.hyp_num_negatives,
            t_start=cfg.hyp_t_start,
            t_end=cfg.hyp_t_end,
            warmup_epochs=cfg.hyp_warmup_epochs,
            curriculum_epochs=cfg.hyp_curriculum_epochs,
        )
        logger.info("Using LorentzTreeRankingLoss (tree distance mode)")
    elif cfg.hyp_distance_mode == "graph":
        # Graph-based negative sampling: load precomputed graph distance matrix.
        graph_dist_matrix = load_precomputed_graph_distance_matrix(
            cfg.graph_distance_matrix,
            logger,
        )
        expected_shape = (cfg.num_classes, cfg.num_classes)
        if tuple(graph_dist_matrix.shape) != expected_shape:
            raise ValueError(
                f"graph_distance_matrix shape {tuple(graph_dist_matrix.shape)} != {expected_shape}"
            )

        hyp_criterion = LorentzTreeRankingLoss(
            tree_dist_matrix=graph_dist_matrix,
            margin=cfg.hyp_margin,
            curv=cfg.hyp_curv,
            num_samples_per_class=cfg.hyp_samples_per_class,
            num_negatives=cfg.hyp_num_negatives,
            t_start=cfg.hyp_t_start,
            t_end=cfg.hyp_t_end,
            warmup_epochs=cfg.hyp_warmup_epochs,
            curriculum_epochs=cfg.hyp_curriculum_epochs,
        )
        logger.info("Using LorentzTreeRankingLoss (graph distance mode)")
    else:
        # Default: Hyperbolic distance-based negative sampling
        hyp_criterion = LorentzRankingLoss(
            margin=cfg.hyp_margin,
            curv=cfg.hyp_curv,
            num_samples_per_class=cfg.hyp_samples_per_class,
            num_negatives=cfg.hyp_num_negatives,
            t_start=cfg.hyp_t_start,
            t_end=cfg.hyp_t_end,
            warmup_epochs=cfg.hyp_warmup_epochs,
            curriculum_epochs=cfg.hyp_curriculum_epochs,
        )
        logger.info(f"Using LorentzRankingLoss (hyperbolic distance mode)")

    # Move hyp_criterion to device (required for registered buffers like tree_dist_matrix)
    hyp_criterion = hyp_criterion.to(device)

    # Separate param groups for visual and text embeddings (differential LR)
    raw_model = model.module if hasattr(model, 'module') else model
    visual_params = [p for n, p in raw_model.named_parameters() if 'label_emb' not in n]
    text_params = [p for n, p in raw_model.named_parameters() if 'label_emb' in n]

    optimizer = optim.Adam([
        {'params': visual_params, 'lr': cfg.lr},
        {'params': text_params, 'lr': cfg.lr * cfg.hyp_text_lr_ratio}
    ], weight_decay=cfg.weight_decay)
    if cfg.lr_scheduler == "cosine":
        cosine_T_max = cfg.epochs - cfg.lr_warmup_epochs
        assert cosine_T_max > 0, f"epochs ({cfg.epochs}) must be > lr_warmup_epochs ({cfg.lr_warmup_epochs})"
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=cfg.lr_eta_min)
        if cfg.lr_warmup_epochs > 0:
            warmup_sched = LinearLR(optimizer, start_factor=1.0, total_iters=cfg.lr_warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[cfg.lr_warmup_epochs])
        else:
            scheduler = cosine_sched
    elif cfg.lr_scheduler == "cosine_multiphase":
        # Two-phase cosine decay then constant:
        #   warmup -> cosine(lr -> phase1_min) -> cosine(phase1_min -> phase2_min) -> constant(phase2_min)
        assert cfg.lr_phase1_end > cfg.lr_warmup_epochs, \
            f"lr_phase1_end ({cfg.lr_phase1_end}) must be > lr_warmup_epochs ({cfg.lr_warmup_epochs})"
        assert cfg.lr_phase2_end > cfg.lr_phase1_end, \
            f"lr_phase2_end ({cfg.lr_phase2_end}) must be > lr_phase1_end ({cfg.lr_phase1_end})"
        base_lr = cfg.lr
        p1_end = cfg.lr_phase1_end
        p2_end = cfg.lr_phase2_end
        p1_min_factor = cfg.lr_phase1_min / base_lr
        p2_min_factor = cfg.lr_phase2_min / base_lr
        warmup_ep = cfg.lr_warmup_epochs

        def _multiphase_lambda(epoch):
            if epoch < warmup_ep:
                return 1.0
            elif epoch < p1_end:
                t = (epoch - warmup_ep) / max(p1_end - warmup_ep, 1)
                return p1_min_factor + 0.5 * (1.0 - p1_min_factor) * (1 + math.cos(math.pi * t))
            elif epoch < p2_end:
                t = (epoch - p1_end) / max(p2_end - p1_end, 1)
                return p2_min_factor + 0.5 * (p1_min_factor - p2_min_factor) * (1 + math.cos(math.pi * t))
            else:
                return p2_min_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_multiphase_lambda)
        logger.info(f"Using cosine_multiphase LR: warmup={warmup_ep}, "
                     f"phase1=[{warmup_ep},{p1_end}) -> {cfg.lr_phase1_min}, "
                     f"phase2=[{p1_end},{p2_end}) -> {cfg.lr_phase2_min}, "
                     f"then constant")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
        )

    # AMP GradScaler (only if use_amp is enabled)
    scaler = GradScaler() if cfg.use_amp else None
    logger.info(f"AMP enabled: {cfg.use_amp}")

    # Metrics
    metric = DiceMetric(num_classes=cfg.num_classes)

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if cfg.resume:
        logger.info(f"Resuming from checkpoint: {cfg.resume}")
        start_epoch, best_dice = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, device=device
        )
        # Load scaler state if available
        if scaler is not None:
            checkpoint = torch.load(cfg.resume, map_location=device)
            if checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.info("Loaded GradScaler state from checkpoint")
        logger.info(f"Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")
        # Warn about freeze state when resuming
        if cfg.hyp_freeze_epochs > 0:
            if start_epoch < cfg.hyp_freeze_epochs:
                logger.info(f"  Note: Text embeddings will remain FROZEN until epoch {cfg.hyp_freeze_epochs}")
            elif start_epoch == cfg.hyp_freeze_epochs:
                logger.info(f"  Note: Resuming at first unfreeze epoch, extra gradient clipping will be applied")

    # TensorBoard writer (only main process)
    writer = SummaryWriter(log_dir=run_log_dir) if is_main_process() else None

    # Embedding tracker (only main process, controlled by config)
    embedding_tracker = None
    if is_main_process() and cfg.track_embeddings:
        # Use checkpoint_dir basename as model_name to avoid conflicts
        model_name = os.path.basename(cfg.checkpoint_dir)
        embedding_tracker = EmbeddingTracker(
            model_name=model_name,
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir="docs/visualizations",
            curv=cfg.hyp_curv
        )
        # Record initial embeddings (epoch 0) before training
        raw_model = model.module if hasattr(model, 'module') else model
        embedding_tracker.on_epoch_end(epoch=0, label_embedding=raw_model.label_emb)
        logger.info(f"EmbeddingTracker initialized, output: {embedding_tracker.output_dir}")

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {cfg.epochs}")
    if cfg.hyp_freeze_epochs > 0:
        if cfg.hyp_freeze_epochs >= cfg.epochs:
            logger.warning(f"hyp_freeze_epochs ({cfg.hyp_freeze_epochs}) >= epochs ({cfg.epochs}), text embeddings will NEVER be unfrozen!")
        else:
            logger.info(f"Text embedding freeze: epochs 0-{cfg.hyp_freeze_epochs - 1}, unfreeze at epoch {cfg.hyp_freeze_epochs}")
            logger.info(f"Text embedding LR ratio: {cfg.hyp_text_lr_ratio}, grad clip on unfreeze: {cfg.hyp_text_grad_clip}")
    logger.info("-" * 60)

    for epoch in range(start_epoch, cfg.epochs):
        # Set epoch for Curriculum Negative Mining (if supported by loss function)
        if hasattr(hyp_criterion, 'set_epoch'):
            hyp_criterion.set_epoch(epoch)

        # Freeze/unfreeze label embeddings based on epoch
        raw_model = model.module if hasattr(model, 'module') else model
        if cfg.hyp_freeze_epochs > 0:
            if epoch < cfg.hyp_freeze_epochs:
                raw_model.label_emb.tangent_embeddings.requires_grad_(False)
                freeze_status = "FROZEN"
            else:
                raw_model.label_emb.tangent_embeddings.requires_grad_(True)
                freeze_status = "UNFROZEN" if epoch > cfg.hyp_freeze_epochs else "UNFROZEN (first epoch, extra grad clip)"
        else:
            freeze_status = "trainable"

        current_lr = optimizer.param_groups[0]["lr"]
        text_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else current_lr
        logger.info(f"Epoch [{epoch + 1}/{cfg.epochs}]  LR: {current_lr:.6f}  TextLR: {text_lr:.6f}  TextEmb: {freeze_status}")

        # Train
        train_total, train_seg, train_hyp = train_one_epoch(
            model, train_loader, criterion, hyp_criterion, cfg.hyp_weight,
            optimizer, device, cfg.grad_clip, epoch=epoch, scaler=scaler, cfg=cfg
        )

        # Validate
        val_total, val_seg, val_hyp, dice_per_class, mean_dice = validate(
            model, val_loader, criterion, hyp_criterion, cfg.hyp_weight, metric, device
        )

        # Update scheduler
        if cfg.lr_scheduler in ("cosine", "cosine_multiphase"):
            scheduler.step()
        else:
            scheduler.step(mean_dice)

        # Log to TensorBoard (only main process)
        if writer:
            writer.add_scalar("Loss/train_total", train_total, epoch)
            writer.add_scalar("Loss/train_seg", train_seg, epoch)
            writer.add_scalar("Loss/train_hyp", train_hyp, epoch)
            writer.add_scalar("Loss/val_total", val_total, epoch)
            writer.add_scalar("Loss/val_seg", val_seg, epoch)
            writer.add_scalar("Loss/val_hyp", val_hyp, epoch)
            writer.add_scalar("Dice/mean", mean_dice, epoch)
            writer.add_scalar("LR", current_lr, epoch)

            # Log AMP loss scale
            if scaler is not None:
                writer.add_scalar("AMP/loss_scale", scaler.get_scale(), epoch)

            # Log key organ Dice scores (if present)
            # These indices depend on the dataset label mapping
            key_organs = {
                "Dice/class_00_background": 0,
                "Dice/class_01": 1,
                "Dice/class_02": 2,
                "Dice/class_03": 3,
                "Dice/class_04": 4,
            }
            for name, idx in key_organs.items():
                writer.add_scalar(name, dice_per_class[idx].item(), epoch)

        # Track embedding evolution (only main process)
        if embedding_tracker is not None:
            raw_model = model.module if hasattr(model, 'module') else model
            embedding_tracker.on_epoch_end(epoch=epoch + 1, label_embedding=raw_model.label_emb)

        # Check if best model
        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice

        # Log epoch summary
        logger.info(
            f"  Train: total={train_total:.4f} seg={train_seg:.4f} hyp={train_hyp:.4f} | "
            f"Val: total={val_total:.4f} seg={val_seg:.4f} hyp={val_hyp:.4f} | "
            f"Dice: {mean_dice:.4f} (best: {best_dice:.4f})"
            f"{' *' if is_best else ''}"
        )

        # Save checkpoint (only main process)
        if is_main_process():
            # Get model state (unwrap DDP if needed)
            model_state = (
                model.module.state_dict()
                if hasattr(model, 'module')  # Works for both DDP and DataParallel
                else model.state_dict()
            )

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "train_loss": train_total,
                "val_loss": val_total,
                "mean_dice": mean_dice,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            }

            # Save latest (every epoch)
            save_checkpoint(checkpoint_state, cfg.checkpoint_dir, "latest.pth", is_best=is_best)

            # Save periodic checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                save_checkpoint(checkpoint_state, cfg.checkpoint_dir, f"epoch_{epoch + 1}.pth")
                logger.info(f"  Saved periodic checkpoint: epoch_{epoch + 1}.pth")

        logger.info("-" * 60)

    if writer:
        writer.close()
    cleanup_distributed()
    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
