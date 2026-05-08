"""
Compare training curves between two experiments.
Parses training.log files and generates smoothed loss/dice plots.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── Config ───────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "TreeDistance": "runs/LR-Curriculum-TreeDistance/2026-02-06_23-30-12/training.log",
    "CD-M04 (Graph)": "runs/LR-CD-M04/2026-02-08_10-10-03/training.log",
}

COLORS = {
    "TreeDistance": "#2176AE",
    "CD-M04 (Graph)": "#D7263D",
}

OUTPUT_DIR = Path("docs/visualizations/training_comparison")
EMA_ALPHA = 0.3  # smoothing factor: smaller = smoother


def parse_log(path: str):
    """Extract epoch, train total loss, val total loss, and dice from log."""
    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\].*?"
        r"Train: total=([\d.]+).*?"
        r"Val: total=([\d.]+).*?"
        r"Dice: ([\d.]+)",
        re.DOTALL,
    )
    epochs, train_loss, val_loss, dice = [], [], [], []
    text = Path(path).read_text()

    # Join consecutive lines so multi-line epoch blocks are captured
    lines = text.split("\n")
    joined = ""
    for line in lines:
        if "Epoch [" in line and joined:
            m = pattern.search(joined)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                dice.append(float(m.group(4)))
            joined = line
        else:
            joined += " " + line
    # last block
    m = pattern.search(joined)
    if m:
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        val_loss.append(float(m.group(3)))
        dice.append(float(m.group(4)))

    return (
        np.array(epochs),
        np.array(train_loss),
        np.array(val_loss),
        np.array(dice),
    )


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average."""
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_metric(
    data: dict,
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    filename: str,
    max_epoch: int,
):
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

    for name, d in data.items():
        c = COLORS[name]
        mask = d["epochs"] <= max_epoch
        ep = d["epochs"][mask]

        # Train
        raw = d[train_key][mask]
        sm = ema(raw, EMA_ALPHA)
        ax_train.plot(ep, sm, color=c, linewidth=1.8, label=name)

        # Val
        raw = d[val_key][mask]
        sm = ema(raw, EMA_ALPHA)
        ax_val.plot(ep, sm, color=c, linewidth=1.8, label=name)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel(ylabel)
    ax_train.set_title("Train")
    ax_train.legend(frameon=True, fancybox=False, edgecolor="#cccccc")

    ax_val.set_xlabel("Epoch")
    ax_val.set_title("Validation")
    ax_val.legend(frameon=True, fancybox=False, edgecolor="#cccccc")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, path in EXPERIMENTS.items():
        epochs, train_loss, val_loss, dice = parse_log(path)
        data[name] = {
            "epochs": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": dice,
        }
        print(f"[{name}] {len(epochs)} epochs loaded (epoch {epochs[0]}-{epochs[-1]})")

    max_epoch = min(len(d["epochs"]) for d in data.values())
    print(f"Truncating to {max_epoch} epochs for fair comparison")

    plot_metric(data, "train_loss", "val_loss", "Total Loss", "Total Loss Comparison", "loss_comparison.png", max_epoch)
    plot_metric(data, "dice", "dice", "Mean Dice", "Mean Dice Comparison", "dice_comparison.png", max_epoch)


if __name__ == "__main__":
    main()
