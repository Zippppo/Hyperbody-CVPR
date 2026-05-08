from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple
import yaml


@dataclass
class Config:
    # Data
    data_dir: str = "Dataset/voxel_data"
    split_file: str = "Dataset/dataset_split.json"
    tree_file: str = "Dataset/tree.json"
    dataset_info_file: str = "Dataset/dataset_info.json"
    num_classes: int = 70
    voxel_size: float = 4.0
    volume_size: Tuple[int, int, int] = (144, 128, 268)  # X, Y, Z

    # Model
    in_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4 #当前的无用参数，本意是动态调整3DUnet的层数

    # Dense Bottleneck
    growth_rate: int = 32      # channels added per layer
    dense_layers: int = 4      # number of dense layers
    bn_size: int = 4           # 1x1x1 compression factor

    # Hyperbolic
    hyp_embed_dim: int = 32
    hyp_curv: float = 1.0
    hyp_margin: float = 0.4       # Triplet margin
    hyp_samples_per_class: int = 64
    hyp_num_negatives: int = 8    # Negative classes per anchor
    # Curriculum Negative Mining
    hyp_t_start: float = 2.0      # Initial temperature (high = random)
    hyp_t_end: float = 0.1        # Final temperature (low = hard negatives)
    hyp_warmup_epochs: int = 6    # Pure random sampling for first N epochs
    hyp_curriculum_epochs: int = 50  # Epochs for full easy->hard curriculum (decoupled from total epochs)
    hyp_min_radius: float = 0.1   # Shallow organ init norm
    hyp_max_radius: float = 2.0   # Deep organ init norm
    hyp_direction_mode: str = "random"  # "random" or "semantic"
    hyp_text_embedding_path: str = "Dataset/text_embeddings/sat_label_embeddings.pt"
    hyp_freeze_epochs: int = 5  # Freeze label embeddings for first N epochs (0 = no freeze)
    hyp_text_lr_ratio: float = 0.01  # Text embedding LR = base_lr * ratio
    hyp_text_grad_clip: float = 0.1  # Gradient clip for text embeddings (first unfreeze epoch)
    hyp_distance_mode: str = "graph"  # Distance mode for negative sampling: "hyperbolic", "tree", or "graph"

    # Spatial adjacency (for hyp_distance_mode="graph")
    spatial_dilation_radius: int = 3  # Cube dilation radius in voxels
    spatial_lambda: float = 1.0  # Scale factor for spatial edge distance
    spatial_epsilon: float = 0.01  # Prevents division by zero in contact->distance
    spatial_contact_matrix: str = ""  # Path to precomputed contact_matrix.pt (empty = compute)
    graph_distance_matrix: str = "Dataset/graph_distance_matrix.pt"  # Path to precomputed graph_distance_matrix.pt

    # Training
    batch_size: int = 2  # per GPU
    num_workers: int = 0
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # AMP
    use_amp: bool = True

    # Loss
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_ignore_index: Optional[int] = None
    hyp_weight: float = 0.05      # Loss weight

    # LR scheduler
    lr_scheduler: str = "cosine"       # "plateau", "cosine", or "cosine_multiphase"
    lr_patience: int = 10               # plateau only
    lr_factor: float = 0.5              # plateau only
    lr_warmup_epochs: int = 4           # cosine only: linear warmup epochs
    lr_eta_min: float = 1e-6            # cosine only: minimum LR
    # cosine_multiphase: two-phase cosine decay then constant
    lr_phase1_end: int = 0              # end epoch for phase 1 (0 = disabled)
    lr_phase1_min: float = 1e-6         # min LR at end of phase 1
    lr_phase2_end: int = 0              # end epoch for phase 2
    lr_phase2_min: float = 1e-8         # min LR at end of phase 2, then constant

    # Checkpoint
    checkpoint_dir: str = ""
    save_every: int = 10
    log_dir: str = ""

    # GPU
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # Resume
    resume: str = ""

    # Embedding tracking
    track_embeddings: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML file, overriding defaults."""
        cfg = cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)

        if yaml_cfg is None:
            return cfg

        # Get valid field names
        valid_fields = {f.name for f in fields(cls)}

        # Build a map of field name -> expected type for type coercion
        field_types = {f.name: f.type for f in fields(cls)}

        for key, value in yaml_cfg.items():
            if key in valid_fields:
                # Handle tuple conversion for volume_size
                if key == "volume_size" and isinstance(value, list):
                    value = tuple(value)
                # Coerce str to float/int when dataclass expects numeric type
                # (e.g. YAML parses "1e-6" as str, but field expects float)
                elif isinstance(value, str) and field_types.get(key) in ("float", float):
                    value = float(value)
                elif isinstance(value, str) and field_types.get(key) in ("int", int):
                    value = int(value)
                setattr(cfg, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' in YAML, ignored.")

        return cfg

    def to_yaml(self, yaml_path: str) -> None:
        """Save current config to YAML file."""
        cfg_dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Convert tuple to list for YAML
            if isinstance(value, tuple):
                value = list(value)
            cfg_dict[f.name] = value

        with open(yaml_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
