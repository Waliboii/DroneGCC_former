import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "train": "#00a6ff",
    "val": "#ff3b5c",
    "x": "#00c2a8",
    "y": "#a855f7",
    "z": "#ff9f1c",
    "lr": "#64748b",
    "xyz": "#3b82f6",
    "class": "#f97316",
    "best": "#111827",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def smooth(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def style_axis(ax):
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_facecolor("#fbfdff")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=9)
    ax.title.set_color("#0f172a")
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")


def plot_line(ax, epochs, raw, label, color, smooth_window):
    raw = np.asarray(raw, dtype=np.float64)
    ax.plot(epochs, raw, color=color, alpha=0.22, linewidth=1.2)
    ax.plot(epochs, smooth(raw, smooth_window), color=color, linewidth=2.4, label=label)


def annotate_best(ax, epochs, values, label):
    values = np.asarray(values, dtype=np.float64)
    idx = int(np.nanargmin(values))
    epoch = epochs[idx]
    value = values[idx]
    ax.scatter([epoch], [value], color=COLORS["best"], s=32, zorder=5)
    ax.annotate(
        f"{label}: e{int(epoch)} / {value:.4f}",
        xy=(epoch, value),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )


def make_dashboard(rows, out_path, smooth_window):
    epochs = np.array([row["epoch"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Best Model Training Dashboard: Mel+GCC Temporal Transformer (Sequence-Random)",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    ax = axes[0, 0]
    plot_line(ax, epochs, [r["train_loss"] for r in rows], "Train total", COLORS["train"], smooth_window)
    plot_line(ax, epochs, [r["val_loss"] for r in rows], "Validation total", COLORS["val"], smooth_window)
    annotate_best(ax, epochs, [r["val_loss"] for r in rows], "best val loss")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    style_axis(ax)

    ax = axes[0, 1]
    plot_line(ax, epochs, [r["train_xyz_loss"] for r in rows], "Train XYZ", COLORS["train"], smooth_window)
    plot_line(ax, epochs, [r["val_xyz_loss"] for r in rows], "Validation XYZ", COLORS["val"], smooth_window)
    annotate_best(ax, epochs, [r["val_xyz_loss"] for r in rows], "best val XYZ")
    ax.set_title("Localization Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted SmoothL1")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    style_axis(ax)

    ax = axes[0, 2]
    plot_line(ax, epochs, [r["train_class_loss"] for r in rows], "Train class", COLORS["train"], smooth_window)
    plot_line(ax, epochs, [r["val_class_loss"] for r in rows], "Validation class", COLORS["val"], smooth_window)
    annotate_best(ax, epochs, [r["val_class_loss"] for r in rows], "best val class")
    ax.set_title("Classification Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    style_axis(ax)

    ax = axes[1, 0]
    plot_line(ax, epochs, [r["train_mean_3d_error"] for r in rows], "Train 3D error", COLORS["train"], smooth_window)
    plot_line(ax, epochs, [r["val_mean_3d_error"] for r in rows], "Validation 3D error", COLORS["val"], smooth_window)
    annotate_best(ax, epochs, [r["val_mean_3d_error"] for r in rows], "best val 3D")
    ax.set_title("Mean 3D Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Meters")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    style_axis(ax)

    ax = axes[1, 1]
    ax.plot(epochs, smooth([r["val_mse_x"] for r in rows], smooth_window), color=COLORS["x"], linewidth=2.4, label="Validation MSE x")
    ax.plot(epochs, smooth([r["val_mse_y"] for r in rows], smooth_window), color=COLORS["y"], linewidth=2.4, label="Validation MSE y")
    ax.plot(epochs, smooth([r["val_mse_z"] for r in rows], smooth_window), color=COLORS["z"], linewidth=2.4, label="Validation MSE z")
    ax.set_title("Validation Coordinate MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    style_axis(ax)

    ax = axes[1, 2]
    ax.plot(epochs, [r["lr"] for r in rows], color=COLORS["lr"], linewidth=2.4)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_loss_focus(rows, out_path, smooth_window):
    epochs = np.array([row["epoch"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    plot_line(ax, epochs, [r["train_loss"] for r in rows], "Train total loss", COLORS["train"], smooth_window)
    plot_line(ax, epochs, [r["val_loss"] for r in rows], "Validation total loss", COLORS["val"], smooth_window)
    annotate_best(ax, epochs, [r["val_loss"] for r in rows], "best")
    ax.set_title("Train vs Validation Loss", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_step_loss_plot(step_rows, epoch_rows, out_path, smooth_window):
    steps = np.array([row["step"] for row in step_rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Step-Level Training Loss with Validation Epoch Markers",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    panels = [
        ("train_loss", "val_loss", "Total Loss", COLORS["train"]),
        ("train_xyz_loss", "val_xyz_loss", "XYZ Localization Loss", COLORS["xyz"]),
        ("train_class_loss", "val_class_loss", "Classification Loss", COLORS["class"]),
    ]
    steps_per_epoch = max(row["step"] for row in step_rows) / max(row["epoch"] for row in step_rows)

    for ax, (train_key, val_key, title, color) in zip(axes, panels):
        train_values = [row[train_key] for row in step_rows]
        ax.plot(steps, train_values, color=color, alpha=0.16, linewidth=0.8, label="raw train steps")
        ax.plot(steps, smooth(train_values, smooth_window), color=color, linewidth=2.2, label=f"smoothed train ({smooth_window})")

        val_steps = np.array([row["epoch"] * steps_per_epoch for row in epoch_rows], dtype=np.float64)
        val_values = np.array([row[val_key] for row in epoch_rows], dtype=np.float64)
        ax.scatter(val_steps, val_values, color=COLORS["val"], s=22, alpha=0.9, label="validation per epoch", zorder=4)
        ax.plot(val_steps, val_values, color=COLORS["val"], alpha=0.45, linewidth=1.4)

        best_idx = int(np.nanargmin(val_values))
        ax.scatter([val_steps[best_idx]], [val_values[best_idx]], color=COLORS["best"], s=42, zorder=5)
        ax.annotate(
            f"best e{int(epoch_rows[best_idx]['epoch'])}: {val_values[best_idx]:.4f}",
            xy=(val_steps[best_idx], val_values[best_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right")
        style_axis(ax)

    axes[-1].set_xlabel("Training Step")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary(rows, out_path):
    best_loss = min(rows, key=lambda r: r["val_loss"])
    best_3d = min(rows, key=lambda r: r["val_mean_3d_error"])
    best_x = min(rows, key=lambda r: r["val_mse_x"])
    best_y = min(rows, key=lambda r: r["val_mse_y"])
    best_z = min(rows, key=lambda r: r["val_mse_z"])
    lines = [
        "# Pretty Loss Plot Summary",
        "",
        f"- Best validation loss: epoch {int(best_loss['epoch'])}, {best_loss['val_loss']:.6f}",
        f"- Best validation mean 3D error: epoch {int(best_3d['epoch'])}, {best_3d['val_mean_3d_error']:.6f} m",
        f"- Best validation MSE x: epoch {int(best_x['epoch'])}, {best_x['val_mse_x']:.6f}",
        f"- Best validation MSE y: epoch {int(best_y['epoch'])}, {best_y['val_mse_y']:.6f}",
        f"- Best validation MSE z: epoch {int(best_z['epoch'])}, {best_z['val_mse_z']:.6f}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create prettier loss plots for the best run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/experiments/mel_gcc_temporal_transformer_sequence_random_100ep"),
    )
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()

    rows = read_rows(args.run_dir / "epoch_metrics.csv")
    step_rows = read_rows(args.run_dir / "train_steps.csv")
    make_dashboard(rows, args.run_dir / "pretty_training_dashboard.png", args.smooth_window)
    make_loss_focus(rows, args.run_dir / "pretty_loss_focus.png", args.smooth_window)
    make_step_loss_plot(step_rows, rows, args.run_dir / "pretty_step_losses.png", smooth_window=25)
    write_summary(rows, args.run_dir / "pretty_loss_summary.md")
    print(f"Wrote pretty plots to {args.run_dir}")


if __name__ == "__main__":
    main()
