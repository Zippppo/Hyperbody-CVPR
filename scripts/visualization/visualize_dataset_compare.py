"""Side-by-side Plotly comparison between Dataset/voxel_data and S2I_Dataset/train.

For a single sample (default BDMAP_00000001.npz), the script produces a single
HTML with a 2x2 grid of 3D scenes:

  +-----------------------------+-----------------------------+
  | (1,1) sensor_pc — Dataset   | (1,2) sensor_pc — S2I       |
  +-----------------------------+-----------------------------+
  | (2,1) voxel_labels — Dataset| (2,2) voxel_labels — S2I    |
  +-----------------------------+-----------------------------+

- Top row visualises the body-surface point cloud (体表点云).
- Bottom row visualises the inside-body voxel labels (体内体素), one Scatter3d
  trace per class so that class names appear in the legend / hover text.

Coordinates are converted from voxel indices to world (mm) using
``grid_world_min`` and ``grid_voxel_size`` so left/right scenes share a
common metric frame.

Usage
-----
    conda activate pasco
    python scripts/visualization/visualize_dataset_compare.py \
        --sample BDMAP_00000001.npz

    # Restrict S2I voxels to classes whose name appears in Dataset/voxel_data
    # (68/70 classes match exactly by name).
    python scripts/visualization/visualize_dataset_compare.py \
        --sample BDMAP_00000001.npz --s2i-classes match

    # Same, but additionally fold S2I sub-structures into Dataset's parent class
    # (lung_* -> lung, vertebrae_* -> spine).
    python scripts/visualization/visualize_dataset_compare.py \
        --sample BDMAP_00000001.npz --s2i-classes match-grouped

Outputs to ``outputs/dataset_compare_<sample_stem>[_<mode>].html``.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_A_DIR = PROJECT_ROOT / "Dataset" / "voxel_data"
DATASET_A_INFO = PROJECT_ROOT / "Dataset" / "dataset_info.json"
DATASET_A_LABEL = "Dataset/voxel_data"

DATASET_B_DIR = PROJECT_ROOT / "S2I_Dataset" / "train"
DATASET_B_INFO = PROJECT_ROOT / "S2I_Dataset" / "dataset_info.json"
DATASET_B_LABEL = "S2I_Dataset/train"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Render budget — caps to keep the HTML reasonably small.
# ---------------------------------------------------------------------------
MAX_PC_POINTS = 60_000        # subsample sensor_pc above this count
MAX_VOXELS_PER_CLASS = 4_000  # subsample voxels per class above this count
PC_MARKER_SIZE = 1.2
VOXEL_MARKER_SIZE = 2.4

# Labels we never render as voxels, regardless of dataset.
IGNORE_LABELS = {0, 255}  # 0 = inside_body_empty, 255 = outside_body_background

# When --s2i-classes=match-grouped, fold these S2I sub-structures into the
# corresponding parent class from Dataset/voxel_data. The values are tuples of
# class-name prefixes; any S2I class whose name starts with one of these
# prefixes is re-tagged to the parent name (and rendered with the parent's
# color), so e.g. all "vertebrae_*" voxels merge into a single "spine" trace.
SUBSTRUCTURE_GROUPS: dict[str, tuple[str, ...]] = {
    "lung": ("lung_",),
    "spine": ("vertebrae_",),
}

S2I_MODE_CHOICES = ("all", "match", "match-grouped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_class_names(info_path: Path) -> list[str]:
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def generate_organ_colors(num_classes: int) -> list[str]:
    """Bright, perceptually distinct colors keyed by class index."""
    colors: list[str] = []
    for i in range(num_classes):
        hue = (i * 137.508) % 360 / 360.0
        sat = 0.85 + 0.15 * ((i % 2) / 1.0)
        light = 0.55 + 0.15 * ((i % 3) / 2.0)
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")
    return colors


def voxels_to_world(
    indices_ijk: np.ndarray,
    grid_world_min: np.ndarray,
    grid_voxel_size: np.ndarray,
) -> np.ndarray:
    """Convert (N,3) integer voxel indices to (N,3) world-space centers (mm)."""
    return grid_world_min[None, :] + (indices_ijk.astype(np.float32) + 0.5) * grid_voxel_size[None, :]


def subsample(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= max_n:
        return arr
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx]


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------
def build_pointcloud_trace(
    sensor_pc: np.ndarray,
    name: str,
    legend_group: str,
    rng: np.random.Generator,
) -> go.Scatter3d:
    pc = subsample(sensor_pc, MAX_PC_POINTS, rng)
    # Color by Z to give a soft depth cue
    return go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=pc[:, 2],
        mode="markers",
        marker=dict(
            size=PC_MARKER_SIZE,
            color=pc[:, 2],
            colorscale="Viridis",
            opacity=0.85,
        ),
        name=name,
        legendgroup=legend_group,
        showlegend=True,
        hoverinfo="skip",
    )


def build_voxel_traces_grouped(
    voxel_labels: np.ndarray,
    grid_world_min: np.ndarray,
    grid_voxel_size: np.ndarray,
    label_to_display: dict[int, tuple[str, str]],
    show_legend: bool,
    rng: np.random.Generator,
) -> list[go.Scatter3d]:
    """Build one Scatter3d trace per *display* class.

    ``label_to_display`` maps a raw integer label value found in
    ``voxel_labels`` to a ``(display_name, color)`` pair. Labels missing from
    the dict are dropped entirely. Multiple raw labels mapped to the same
    ``display_name`` are merged into a single trace (used by the
    ``match-grouped`` mode to fold S2I sub-structures into a parent class).
    """
    # Group voxel coordinates per display name.
    grouped_ijk: dict[str, list[np.ndarray]] = {}
    color_for: dict[str, str] = {}
    raw_labels_for: dict[str, list[int]] = {}

    for c in np.unique(voxel_labels):
        c_int = int(c)
        if c_int not in label_to_display:
            continue
        ijk = np.argwhere(voxel_labels == c_int)
        if ijk.shape[0] == 0:
            continue
        display_name, color = label_to_display[c_int]
        grouped_ijk.setdefault(display_name, []).append(ijk)
        color_for[display_name] = color
        raw_labels_for.setdefault(display_name, []).append(c_int)

    traces: list[go.Scatter3d] = []
    for display_name, ijk_list in grouped_ijk.items():
        ijk = np.concatenate(ijk_list, axis=0)
        ijk = subsample(ijk, MAX_VOXELS_PER_CLASS, rng)
        xyz = voxels_to_world(ijk, grid_world_min, grid_voxel_size)

        raw_labels = sorted(set(raw_labels_for[display_name]))
        if len(raw_labels) == 1:
            hover = f"{display_name} (label={raw_labels[0]})"
        else:
            hover = f"{display_name} (labels={raw_labels})"

        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=VOXEL_MARKER_SIZE,
                    color=color_for[display_name],
                    opacity=0.9,
                    symbol="square",
                    line=dict(width=0),
                ),
                name=display_name,
                # Same legendgroup across columns => clicking liver toggles
                # both Dataset's and S2I's liver trace at once.
                legendgroup=display_name,
                showlegend=show_legend,
                hovertext=hover,
                hoverinfo="text",
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Label-to-display mapping
# ---------------------------------------------------------------------------
def make_label_to_display_self(
    class_names: list[str], colors: list[str]
) -> dict[int, tuple[str, str]]:
    """Identity mapping: every (non-ignored) class index to its own name+color."""
    return {
        i: (name, colors[i])
        for i, name in enumerate(class_names)
        if i not in IGNORE_LABELS
    }


def make_label_to_display_s2i(
    class_names_b: list[str],
    class_names_a: list[str],
    colors_a: list[str],
    colors_b: list[str],
    mode: str,
) -> dict[int, tuple[str, str]]:
    """Build the S2I label→display mapping under one of three modes.

    - ``all``: keep all S2I classes; if the class name also exists in Dataset,
      reuse Dataset's color so left/right share the same color per class.
    - ``match``: keep only S2I classes whose name appears in Dataset's 70.
    - ``match-grouped``: like ``match``, plus fold S2I sub-structures into the
      parent Dataset class via ``SUBSTRUCTURE_GROUPS``.
    """
    if mode not in S2I_MODE_CHOICES:
        raise ValueError(f"unknown s2i mode: {mode!r} (choices: {S2I_MODE_CHOICES})")

    name_to_idx_a = {n: i for i, n in enumerate(class_names_a)}
    out: dict[int, tuple[str, str]] = {}

    for i, name in enumerate(class_names_b):
        if i in IGNORE_LABELS:
            continue

        if name in name_to_idx_a:
            # Direct name match: use Dataset's color so columns are color-paired.
            out[i] = (name, colors_a[name_to_idx_a[name]])
            continue

        if mode == "all":
            out[i] = (name, colors_b[i])
        elif mode == "match":
            # drop
            pass
        elif mode == "match-grouped":
            for parent, prefixes in SUBSTRUCTURE_GROUPS.items():
                if parent in name_to_idx_a and any(name.startswith(p) for p in prefixes):
                    out[i] = (parent, colors_a[name_to_idx_a[parent]])
                    break

    return out


# ---------------------------------------------------------------------------
# Per-dataset payload
# ---------------------------------------------------------------------------
def load_sample(npz_path: Path) -> dict:
    data = np.load(str(npz_path))
    payload = {
        "sensor_pc": np.asarray(data["sensor_pc"], dtype=np.float32),
        "voxel_labels": np.asarray(data["voxel_labels"]),
        "grid_world_min": np.asarray(data["grid_world_min"], dtype=np.float32),
        "grid_world_max": np.asarray(data["grid_world_max"], dtype=np.float32),
        "grid_voxel_size": np.asarray(data["grid_voxel_size"], dtype=np.float32),
        "grid_occ_size": np.asarray(data["grid_occ_size"], dtype=np.int64),
    }
    return payload


def scene_range_from_payload(payload: dict, pad: float = 20.0) -> dict:
    """Per-axis world-coord ranges so left/right share a metric frame."""
    lo = payload["grid_world_min"]
    hi = payload["grid_world_max"]
    return dict(
        xaxis=dict(range=[lo[0] - pad, hi[0] + pad], title="X (mm)"),
        yaxis=dict(range=[lo[1] - pad, hi[1] + pad], title="Y (mm)"),
        zaxis=dict(range=[lo[2] - pad, hi[2] + pad], title="Z (mm)"),
        aspectmode="data",
    )


def shared_scene_range(
    payload_a: dict, payload_b: dict, pad: float = 20.0
) -> dict:
    """Compute one shared world-coord range covering both samples."""
    lo = np.minimum(payload_a["grid_world_min"], payload_b["grid_world_min"])
    hi = np.maximum(payload_a["grid_world_max"], payload_b["grid_world_max"])
    return dict(
        xaxis=dict(range=[float(lo[0] - pad), float(hi[0] + pad)], title="X (mm)"),
        yaxis=dict(range=[float(lo[1] - pad), float(hi[1] + pad)], title="Y (mm)"),
        zaxis=dict(range=[float(lo[2] - pad), float(hi[2] + pad)], title="Z (mm)"),
        aspectmode="data",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side Plotly comparison of Dataset/voxel_data vs S2I_Dataset/train.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="BDMAP_00000001.npz",
        help="Sample filename present under both dataset directories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path (default: outputs/dataset_compare_<stem>.html).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for subsampling.",
    )
    parser.add_argument(
        "--s2i-classes",
        type=str,
        choices=S2I_MODE_CHOICES,
        default="all",
        help=(
            "Filter the right-column S2I voxels: "
            "'all' shows every S2I class (default); "
            "'match' shows only S2I classes whose name appears in Dataset's 70 "
            "(68 exact-name matches; 'lung' and 'spine' have no exact match in "
            "S2I and are dropped); "
            "'match-grouped' additionally folds 'lung_*' -> lung and "
            "'vertebrae_*' -> spine so the right column visually mirrors the left."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    sample_a = DATASET_A_DIR / args.sample
    sample_b = DATASET_B_DIR / args.sample
    if not sample_a.exists():
        raise FileNotFoundError(sample_a)
    if not sample_b.exists():
        raise FileNotFoundError(sample_b)

    print(f"[load] {sample_a}")
    payload_a = load_sample(sample_a)
    print(f"[load] {sample_b}")
    payload_b = load_sample(sample_b)

    class_names_a = load_class_names(DATASET_A_INFO)
    class_names_b = load_class_names(DATASET_B_INFO)
    colors_a = generate_organ_colors(len(class_names_a))
    colors_b = generate_organ_colors(len(class_names_b))

    print(
        f"[info] A: pc={payload_a['sensor_pc'].shape}, "
        f"vox={payload_a['voxel_labels'].shape}, classes={len(class_names_a)}"
    )
    print(
        f"[info] B: pc={payload_b['sensor_pc'].shape}, "
        f"vox={payload_b['voxel_labels'].shape}, classes={len(class_names_b)}"
    )

    # ---- Build label-to-display dicts -------------------------------------
    label_to_display_a = make_label_to_display_self(class_names_a, colors_a)
    label_to_display_b = make_label_to_display_s2i(
        class_names_b=class_names_b,
        class_names_a=class_names_a,
        colors_a=colors_a,
        colors_b=colors_b,
        mode=args.s2i_classes,
    )
    n_kept_b = len({d[0] for d in label_to_display_b.values()})
    print(
        f"[info] s2i-classes mode = {args.s2i_classes!r}: "
        f"keeping {len(label_to_display_b)}/{len(class_names_b)} S2I labels "
        f"-> {n_kept_b} display classes"
    )

    s2i_title_suffix = {
        "all": "all classes",
        "match": "name-matched only",
        "match-grouped": "name-matched + grouped",
    }[args.s2i_classes]

    # ---- Build figure ----
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            f"Sensor PC — {DATASET_A_LABEL} ({payload_a['sensor_pc'].shape[0]} pts)",
            f"Sensor PC — {DATASET_B_LABEL} ({payload_b['sensor_pc'].shape[0]} pts)",
            f"Voxel Labels — {DATASET_A_LABEL} (occ={tuple(payload_a['grid_occ_size'].tolist())})",
            f"Voxel Labels — {DATASET_B_LABEL} (occ={tuple(payload_b['grid_occ_size'].tolist())}, {s2i_title_suffix})",
        ),
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    # --- Row 1: sensor point clouds -------------------------------------------
    fig.add_trace(
        build_pointcloud_trace(
            payload_a["sensor_pc"],
            name=f"sensor_pc · {DATASET_A_LABEL}",
            legend_group="pc_A",
            rng=rng,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        build_pointcloud_trace(
            payload_b["sensor_pc"],
            name=f"sensor_pc · {DATASET_B_LABEL}",
            legend_group="pc_B",
            rng=rng,
        ),
        row=1,
        col=2,
    )

    # --- Row 2: interior voxels -----------------------------------------------
    # Same legendgroup string on left+right means clicking a class in the
    # legend toggles both columns at once. To avoid duplicate legend entries
    # we only flag a trace as "show in legend" the first time we encounter
    # its display name; subsequent same-named traces stay invisible in the
    # legend but still toggle together via legendgroup.
    voxel_traces_a = build_voxel_traces_grouped(
        payload_a["voxel_labels"],
        payload_a["grid_world_min"],
        payload_a["grid_voxel_size"],
        label_to_display=label_to_display_a,
        show_legend=True,
        rng=rng,
    )
    voxel_traces_b = build_voxel_traces_grouped(
        payload_b["voxel_labels"],
        payload_b["grid_world_min"],
        payload_b["grid_voxel_size"],
        label_to_display=label_to_display_b,
        show_legend=True,
        rng=rng,
    )

    seen_in_legend: set[str] = set()
    for tr in voxel_traces_a:
        if tr.name in seen_in_legend:
            tr.showlegend = False
        else:
            seen_in_legend.add(tr.name)
        fig.add_trace(tr, row=2, col=1)
    for tr in voxel_traces_b:
        if tr.name in seen_in_legend:
            tr.showlegend = False
        else:
            seen_in_legend.add(tr.name)
        fig.add_trace(tr, row=2, col=2)

    # ---- Layout / scenes -----------------------------------------------------
    shared_range = shared_scene_range(payload_a, payload_b, pad=20.0)
    initial_camera = dict(eye=dict(x=1.6, y=-1.6, z=1.0))

    scene_kwargs = dict(shared_range, camera=initial_camera, bgcolor="white")
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Dataset comparison — {args.sample}</b><br>"
                f"<sup>Top: 体表点云 (sensor_pc) · Bottom: 体内体素 (voxel_labels) · "
                f"left = {DATASET_A_LABEL} · right = {DATASET_B_LABEL} · "
                f"s2i-classes = {args.s2i_classes}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        scene=scene_kwargs,
        scene2=scene_kwargs,
        scene3=scene_kwargs,
        scene4=scene_kwargs,
        width=1600,
        height=1300,
        legend=dict(
            itemsizing="constant",
            groupclick="togglegroup",
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=110, b=10),
    )

    # ---- Write -------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        stem = Path(args.sample).stem
        mode_tag = "" if args.s2i_classes == "all" else f"_{args.s2i_classes.replace('-', '_')}"
        out_path = OUTPUT_DIR / f"dataset_compare_{stem}{mode_tag}.html"
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
