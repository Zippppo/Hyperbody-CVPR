"""
S2I Class Weight Editor

View, modify, and plot class weights saved by tools/s2i_class_weight_calculator.py
for S2I_Dataset (121 classes).

Usage:
    # View weights (ASCII bars), grouped by anatomical category
    python tools/s2i_class_weight_editor.py view S2I_Dataset/s2i_class_weights.pt
    python tools/s2i_class_weight_editor.py view <file> --group ribs
    python tools/s2i_class_weight_editor.py view <file> --all

    # Assign explicit weights to specific classes (repeatable; spec=value)
    #   spec can be: name | idx | a-b range | comma-list
    python tools/s2i_class_weight_editor.py set S2I_Dataset/s2i_class_weights.pt \
        --output checkpoints/class_weight_store/s2i_custom.pt \
        --set liver=2.0 --set 52-75=3.0 --set kidney_left,kidney_right=1.5

    # Multiply a class range by a factor
    python tools/s2i_class_weight_editor.py boost S2I_Dataset/s2i_class_weights.pt \
        --output checkpoints/class_weight_store/s2i_rib_boost.pt \
        --classes 52-75 --multiplier 2.0

    # Save a PNG bar chart (requires matplotlib)
    python tools/s2i_class_weight_editor.py plot S2I_Dataset/s2i_class_weights.pt \
        --output /tmp/s2i_weights.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

DATASET_INFO_PATH = "S2I_Dataset/dataset_info.json"

# Anatomical groups (start, end inclusive). Indices follow dataset_info.json order.
GROUPS: Dict[str, Tuple[int, int]] = {
    "background":  (0, 0),       # inside_body_empty
    "organs":      (1, 25),      # liver .. adrenal_gland_right
    "spine":       (26, 51),     # vertebrae_C1 .. sacrum
    "ribs":        (52, 75),     # rib_left_1 .. rib_right_12
    "other_bones": (76, 88),     # skull, sternum, costal_cartilages, scapulae, claviculae, humeri, hips, femora
    "muscles":     (89, 98),     # gluteus *3 L/R, autochthon L/R, iliopsoas L/R
    "vessels":     (99, 116),    # arteries + veins
    "fat":         (117, 119),
    "skel_muscle": (120, 120),
}


def load_class_names() -> List[str]:
    with open(DATASET_INFO_PATH) as f:
        return json.load(f)["class_names"]


def group_of(idx: int) -> str:
    for name, (lo, hi) in GROUPS.items():
        if lo <= idx <= hi:
            return name
    return ""


def parse_class_spec(spec: str, name_to_idx: Dict[str, int]) -> List[int]:
    """Parse 'idx' / 'name' / 'a-b' / comma-separated combinations into indices."""
    indices: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        # Range like '52-75' (class names use underscores, never hyphens)
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            indices.extend(range(int(lo_s), int(hi_s) + 1))
        elif token.isdigit():
            indices.append(int(token))
        elif token in name_to_idx:
            indices.append(name_to_idx[token])
        else:
            raise ValueError(f"Unknown class spec: {token!r}")
    return indices


def make_bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return " " * width
    filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


def view_weights(weight_path: str, show_all: bool, group_filter: Optional[str]):
    class_names = load_class_names()
    data = torch.load(weight_path, weights_only=True)
    weights = data["weights"]

    print(f"\n{'=' * 90}")
    print(f"File:         {weight_path}")
    print(f"Dataset:      {data.get('dataset', 'unknown')}")
    print(f"Method:       {data.get('method', 'unknown')}")
    print(f"Num samples:  {data.get('num_samples', 'unknown')}")
    print(
        f"Num classes:  {len(weights)}    "
        f"target_ignore_index: {data.get('target_ignore_index', data.get('ignore_index', '-'))}    "
        f"outside_label: {data.get('outside_label', '-')}"
    )
    print(f"{'=' * 90}\n")

    min_w = float(weights.min().item())
    max_w = float(weights.max().item())
    print("Statistics (overall):")
    print(f"  min={min_w:.4f}  max={max_w:.4f}  mean={weights.mean():.4f}  sum={weights.sum():.4f}")
    print(f"  ratio (max/min): {max_w / max(min_w, 1e-12):.2f}\n")

    print("Per-group statistics:")
    print(f"  {'Group':<13} {'Range':<10} {'Min':>8} {'Max':>8} {'Mean':>8}")
    print("  " + "-" * 49)
    for gname, (lo, hi) in GROUPS.items():
        w = weights[lo:hi + 1]
        print(f"  {gname:<13} {f'{lo}-{hi}':<10} {w.min():>8.4f} {w.max():>8.4f} {w.mean():>8.4f}")
    print()

    print(f"{'Idx':<4} {'Class Name':<32} {'Group':<12} {'Weight':>8}  Bar")
    print("-" * 95)
    shown = 0
    for i, (name, w) in enumerate(zip(class_names, weights)):
        g = group_of(i)
        if group_filter and g != group_filter:
            continue
        if not show_all and not group_filter:
            # Default: only highlight rib classes plus extreme weights
            if not (g == "ribs" or w > 1.5 or w < 0.2):
                continue
        bar = make_bar(float(w), max_w, width=30)
        print(f"{i:<4} {name:<32} {g:<12} {w.item():>8.4f}  {bar}")
        shown += 1

    if not show_all and not group_filter:
        print(f"\n(Showed {shown} of {len(weights)} classes. Use --all or --group <name>.)")
        print(f" Groups: {', '.join(GROUPS.keys())}")


def _save(src: dict, weights: torch.Tensor, output_path: str, suffix: str):
    out = {
        "weights": weights.float(),
        "num_classes": src["num_classes"],
        "num_samples": src.get("num_samples", 0),
        "method": f"{src.get('method', 'unknown')} + {suffix}",
        "dataset": src.get("dataset", "S2I_Dataset"),
    }
    for key in ("target_ignore_index", "outside_label", "label_pad_value", "volume_size", "ignore_index"):
        if key in src:
            out[key] = src[key]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(f"\nSaved to: {output_path}")
    print(f"  method tag: {out['method']}")


def set_weights(input_path: str, output_path: str, set_specs: List[str]):
    class_names = load_class_names()
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    data = torch.load(input_path, weights_only=True)
    weights = data["weights"].clone()

    print(f"Editing weights from: {input_path}\n")
    edit_tags: List[str] = []
    for spec in set_specs:
        if "=" not in spec:
            raise ValueError(f"--set expects 'spec=value', got {spec!r}")
        target, value_s = spec.split("=", 1)
        value = float(value_s)
        for idx in parse_class_spec(target, name_to_idx):
            old = weights[idx].item()
            weights[idx] = value
            print(f"  [{idx:>3}] {class_names[idx]:<28} {old:>8.4f} -> {value:.4f}")
        edit_tags.append(f"set({target}={value})")

    _save(data, weights, output_path, ",".join(edit_tags))


def boost_weights(input_path: str, output_path: str, class_spec: str, multiplier: float):
    class_names = load_class_names()
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    data = torch.load(input_path, weights_only=True)
    weights = data["weights"].clone()
    indices = parse_class_spec(class_spec, name_to_idx)

    print(f"Boosting {class_spec} by {multiplier}x  (input: {input_path})\n")
    for idx in indices:
        old = weights[idx].item()
        weights[idx] *= multiplier
        print(f"  [{idx:>3}] {class_names[idx]:<28} {old:>8.4f} -> {weights[idx].item():.4f}")

    _save(data, weights, output_path, f"boost({class_spec},x{multiplier})")


def plot_weights(weight_path: str, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed; pip install matplotlib  (or use 'view')")

    class_names = load_class_names()
    data = torch.load(weight_path, weights_only=True)
    weights = data["weights"].numpy()

    palette = {
        "background": "#888888", "organs": "#4caa8a", "spine": "#4488cc",
        "ribs": "#cc8844", "other_bones": "#aacc88", "muscles": "#cc44cc",
        "vessels": "#cc4444", "fat": "#cccc44", "skel_muscle": "#666666",
    }
    colors = [palette.get(group_of(i), "#888888") for i in range(len(weights))]

    fig, ax = plt.subplots(figsize=(22, 6))
    ax.bar(range(len(weights)), weights, color=colors)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Weight")
    ax.set_title(
        f"S2I class weights ({data.get('method', '?')}, n={data.get('num_samples', '?')})"
    )
    ax.set_xticks(range(0, len(weights), 5))
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.4)
    for _, hi in list(GROUPS.values())[:-1]:
        ax.axvline(hi + 0.5, color="k", lw=0.3, alpha=0.3)
    # Group legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=g) for g, c in palette.items()]
    ax.legend(handles=legend, loc="upper right", ncol=3, fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="S2I Class Weight Editor")
    sub = parser.add_subparsers(dest="command")

    p_view = sub.add_parser("view", help="View weights with ASCII bars")
    p_view.add_argument("weight_file")
    p_view.add_argument("--all", action="store_true", help="Show all 121 classes")
    p_view.add_argument("--group", choices=list(GROUPS.keys()), help="Filter by group")

    p_set = sub.add_parser("set", help="Assign explicit weights to classes")
    p_set.add_argument("input_file")
    p_set.add_argument("--output", "-o", required=True)
    p_set.add_argument(
        "--set", action="append", default=[], dest="set_specs",
        help="'spec=value' (repeatable). spec is name | idx | a-b range | comma-list.",
    )

    p_boost = sub.add_parser("boost", help="Multiply class weights by a factor")
    p_boost.add_argument("input_file")
    p_boost.add_argument("--classes", "-c", required=True, help="e.g. '52-75' or 'liver,spleen'")
    p_boost.add_argument("--multiplier", "-m", type=float, default=2.0)
    p_boost.add_argument("--output", "-o", required=True)

    p_plot = sub.add_parser("plot", help="Save weights as a bar-chart PNG")
    p_plot.add_argument("weight_file")
    p_plot.add_argument("--output", "-o", required=True)

    args = parser.parse_args()
    if args.command == "view":
        view_weights(args.weight_file, args.all, args.group)
    elif args.command == "set":
        if not args.set_specs:
            raise SystemExit("Provide at least one --set 'spec=value'")
        set_weights(args.input_file, args.output, args.set_specs)
    elif args.command == "boost":
        boost_weights(args.input_file, args.output, args.classes, args.multiplier)
    elif args.command == "plot":
        plot_weights(args.weight_file, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
