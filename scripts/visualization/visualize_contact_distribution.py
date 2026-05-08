"""Visualize the distribution of contact matrix values.

Produces a combined histogram + KDE density curve for contact ratios
above a threshold, saved as a PNG image.
"""
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default="Dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/visualization")
    parser.add_argument("--threshold", type=float, default=0.1)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contact = torch.load(dataset_dir / "contact_matrix.pt", map_location="cpu").numpy()

    values = contact[contact > args.threshold].flatten()
    if len(values) == 0:
        raise ValueError(f"No contact values above threshold {args.threshold}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        values,
        bins=40,
        density=True,
        color="#4C72B0",
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
        label="Histogram",
    )

    if len(values) > 1:
        kde = gaussian_kde(values, bw_method=0.3)
        x_grid = np.linspace(args.threshold, 1.0, 500)
        kde_y = kde(x_grid)
        ax.plot(x_grid, kde_y, color="#C44E52", linewidth=2.0, label="KDE density")
        ax.fill_between(x_grid, kde_y, alpha=0.15, color="#C44E52")

    ax.set_xlabel("Contact Ratio", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Distribution of Contact Ratios > {args.threshold} "
        f"(N={len(values)}, matrix {contact.shape[0]} x {contact.shape[1]})",
        fontsize=13,
    )
    ax.legend(fontsize=11)

    stats_text = (
        f"count = {len(values)}\n"
        f"min = {values.min():.4f}\n"
        f"median = {np.median(values):.4f}\n"
        f"mean = {values.mean():.4f}\n"
        f"max = {values.max():.4f}"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
        family="monospace",
    )

    ax.set_xlim(args.threshold, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{dataset_dir.name}_contact_matrix_distribution.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
