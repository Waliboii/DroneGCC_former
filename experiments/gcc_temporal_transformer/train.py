import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DRONES, MelGccTemporalDataset, build_index, build_temporal_splits, prepare_gcc_cache, write_index
from model import MelGccTemporalTransformer, count_parameters


class WeightedSmoothL1XYZ(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        return (loss * self.weights.view(1, 3)).mean()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(split_items, batch_size, workers):
    train_ds = MelGccTemporalDataset(split_items["train"])
    val_ds = MelGccTemporalDataset(split_items["val"], xyz_mean=train_ds.xyz_mean, xyz_std=train_ds.xyz_std)
    test_ds = MelGccTemporalDataset(split_items["test"], xyz_mean=train_ds.xyz_mean, xyz_std=train_ds.xyz_std)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
    return loaders, train_ds


def metrics_from_batch(pred_norm, logits, cls, target_xyz, mean, std):
    pred_xyz = pred_norm * std + mean
    err = pred_xyz - target_xyz
    return {
        "mse_x_sum": err[:, 0].pow(2).sum().item(),
        "mse_y_sum": err[:, 1].pow(2).sum().item(),
        "mse_z_sum": err[:, 2].pow(2).sum().item(),
        "mae_x_sum": err[:, 0].abs().sum().item(),
        "mae_y_sum": err[:, 1].abs().sum().item(),
        "mae_z_sum": err[:, 2].abs().sum().item(),
        "mean_3d_sum": err.norm(dim=1).sum().item(),
        "correct": (logits.argmax(dim=1) == cls).sum().item(),
        "samples": target_xyz.size(0),
    }


def merge_sums(total, part):
    for key, value in part.items():
        total[key] = total.get(key, 0.0) + value


def finalize(total):
    n = max(int(total.get("samples", 0)), 1)
    return {
        "mse_x": total.get("mse_x_sum", 0.0) / n,
        "mse_y": total.get("mse_y_sum", 0.0) / n,
        "mse_z": total.get("mse_z_sum", 0.0) / n,
        "mae_x": total.get("mae_x_sum", 0.0) / n,
        "mae_y": total.get("mae_y_sum", 0.0) / n,
        "mae_z": total.get("mae_z_sum", 0.0) / n,
        "mean_3d_error": total.get("mean_3d_sum", 0.0) / n,
        "classification_accuracy": total.get("correct", 0.0) / n,
    }


def evaluate(model, loader, device, mean, std, xyz_loss_fn, cls_loss_fn, cls_weight):
    model.eval()
    total = {"loss_sum": 0.0, "xyz_loss_sum": 0.0, "cls_loss_sum": 0.0, "samples": 0}
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    with torch.no_grad():
        for mel_seq, gcc_seq, target_norm, cls, target_xyz in loader:
            mel_seq = mel_seq.to(device)
            gcc_seq = gcc_seq.to(device)
            target_norm = target_norm.to(device)
            cls = cls.to(device)
            target_xyz = target_xyz.to(device)
            pred_norm, logits = model(mel_seq, gcc_seq)
            xyz_loss = xyz_loss_fn(pred_norm, target_norm)
            cls_loss = cls_loss_fn(logits, cls)
            loss = xyz_loss + cls_weight * cls_loss
            batch = mel_seq.size(0)
            total["loss_sum"] += loss.item() * batch
            total["xyz_loss_sum"] += xyz_loss.item() * batch
            total["cls_loss_sum"] += cls_loss.item() * batch
            merge_sums(total, metrics_from_batch(pred_norm, logits, cls, target_xyz, mean_t, std_t))
    metrics = finalize(total)
    n = max(int(total["samples"]), 1)
    metrics.update({"loss": total["loss_sum"] / n, "xyz_loss": total["xyz_loss_sum"] / n, "class_loss": total["cls_loss_sum"] / n})
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sequence_index(path, split_items):
    rows = []
    for split, items in split_items.items():
        for idx, item in enumerate(items):
            target = item["target"]
            rows.append(
                {
                    "split": split,
                    "sequence_id": f"{split}_{idx:06d}",
                    "drone_type": target["drone_type"],
                    "target_sample_id": target["sample_id"],
                    "center_time": target["center_time"],
                    "x": target["x"],
                    "y": target["y"],
                    "z": target["z"],
                    "sequence_sample_ids": "|".join(row["sample_id"] for row in item["sequence"]),
                }
            )
    write_csv(path, rows)


def plot_history(path, rows):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    axes[0, 0].plot(epochs, [r["val_loss"] for r in rows], label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 1].plot(epochs, [r["train_xyz_loss"] for r in rows], label="train")
    axes[0, 1].plot(epochs, [r["val_xyz_loss"] for r in rows], label="val")
    axes[0, 1].set_title("XYZ Loss")
    axes[0, 2].plot(epochs, [r["train_class_loss"] for r in rows], label="train")
    axes[0, 2].plot(epochs, [r["val_class_loss"] for r in rows], label="val")
    axes[0, 2].set_title("Class Loss")
    axes[1, 0].plot(epochs, [r["train_mean_3d_error"] for r in rows], label="train")
    axes[1, 0].plot(epochs, [r["val_mean_3d_error"] for r in rows], label="val")
    axes[1, 0].set_title("Mean 3D Error")
    axes[1, 1].plot(epochs, [r["val_mse_x"] for r in rows], label="x")
    axes[1, 1].plot(epochs, [r["val_mse_y"] for r in rows], label="y")
    axes[1, 1].plot(epochs, [r["val_mse_z"] for r in rows], label="z")
    axes[1, 1].set_title("Validation Coordinate MSE")
    axes[1, 2].plot(epochs, [r["train_classification_accuracy"] for r in rows], label="train")
    axes[1, 2].plot(epochs, [r["val_classification_accuracy"] for r in rows], label="val")
    axes[1, 2].set_title("Classification Accuracy")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_architecture_diagram(path):
    steps = [
        "Build 5 consecutive 2-second windows",
        "Each window: 4-channel mel CNN -> 128-d feature",
        "Each window: 6-channel GCC-PHAT CNN -> 128-d feature",
        "Concatenate + MLP -> 256-d window embedding",
        "2-layer Transformer encoder over 5 window embeddings",
        "Center-window token -> x, y, z, and drone class heads",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title("Mel + GCC Temporal Transformer", fontsize=16, weight="bold", pad=16)
    y = 0.82
    for idx, step in enumerate(steps, start=1):
        ax.text(0.5, y, f"{idx}. {step}", ha="center", va="center", fontsize=11, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef5ff", "edgecolor": "#4779c4"})
        if idx < len(steps):
            ax.annotate("", xy=(0.5, y - 0.075), xytext=(0.5, y - 0.035), arrowprops={"arrowstyle": "->", "lw": 1.5})
        y -= 0.13
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Mel+GCC temporal Transformer.")
    parser.add_argument("--mmaud-root", type=Path, default=Path("MMAUD_Drone"))
    parser.add_argument("--spectrogram-root", type=Path, default=Path("outputs/mmaud_spectrograms_2s"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/experiments/gcc_phat_multitask_2s_weighted_100ep/features"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temporal-split", choices=["sequence_random", "blocked_random"], required=True)
    parser.add_argument("--block-sec", type=float, default=20.0)
    parser.add_argument("--sequence-len", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--class-weight", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--xyz-weights", type=float, nargs=3, default=[1.0, 2.0, 2.5])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_index(args.spectrogram_root, args.mmaud_root, args.feature_root, "random", args.seed)
    write_index(rows, args.out / "dataset_index.csv")
    if not args.skip_cache:
        prepare_gcc_cache(rows, args.mmaud_root, window_sec=args.window_sec)
    split_items = build_temporal_splits(rows, args.temporal_split, sequence_len=args.sequence_len, block_sec=args.block_sec, seed=args.seed)
    write_sequence_index(args.out / "sequence_index.csv", split_items)

    loaders, train_ds = make_loaders(split_items, args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelGccTemporalTransformer(num_classes=len(DRONES)).to(device)
    model_info = {"model_name": f"mel_gcc_temporal_transformer_{args.temporal_split}", **count_parameters(model)}
    with (args.out / "model_size.json").open("w", encoding="utf-8") as handle:
        json.dump(model_info, handle, indent=2)
    write_architecture_diagram(args.out / "architecture.png")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    xyz_loss_fn = WeightedSmoothL1XYZ(args.xyz_weights).to(device)
    cls_loss_fn = nn.CrossEntropyLoss()
    mean_t = torch.tensor(train_ds.xyz_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(train_ds.xyz_std, dtype=torch.float32, device=device)

    step_rows = []
    epoch_rows = []
    best_score = float("inf")
    global_step = 0
    best_path = args.out / "best_model.pt"

    print(f"Device: {device}", flush=True)
    print(f"Temporal split: {args.temporal_split}", flush=True)
    print(f"Model params: {model_info['parameters']:,}, fp32 size: {model_info['model_size_mb_fp32']:.2f} MB", flush=True)
    print(f"Sequences train/val/test: {len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}/{len(loaders['test'].dataset)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        train_sums = {"loss_sum": 0.0, "xyz_loss_sum": 0.0, "cls_loss_sum": 0.0, "samples": 0}
        for batch_idx, (mel_seq, gcc_seq, target_norm, cls, target_xyz) in enumerate(loaders["train"], start=1):
            mel_seq = mel_seq.to(device)
            gcc_seq = gcc_seq.to(device)
            target_norm = target_norm.to(device)
            cls = cls.to(device)
            target_xyz = target_xyz.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm, logits = model(mel_seq, gcc_seq)
            xyz_loss = xyz_loss_fn(pred_norm, target_norm)
            cls_loss = cls_loss_fn(logits, cls)
            loss = xyz_loss + args.class_weight * cls_loss
            loss.backward()
            optimizer.step()

            batch = mel_seq.size(0)
            global_step += 1
            train_sums["loss_sum"] += loss.item() * batch
            train_sums["xyz_loss_sum"] += xyz_loss.item() * batch
            train_sums["cls_loss_sum"] += cls_loss.item() * batch
            merge_sums(train_sums, metrics_from_batch(pred_norm.detach(), logits.detach(), cls, target_xyz, mean_t, std_t))
            step_rows.append({"epoch": epoch, "step": global_step, "batch": batch_idx, "train_loss": loss.item(), "train_xyz_loss": xyz_loss.item(), "train_class_loss": cls_loss.item()})

        scheduler.step()
        train_metrics = finalize(train_sums)
        n_train = max(int(train_sums["samples"]), 1)
        train_metrics.update({"loss": train_sums["loss_sum"] / n_train, "xyz_loss": train_sums["xyz_loss_sum"] / n_train, "class_loss": train_sums["cls_loss_sum"] / n_train})
        val_metrics = evaluate(model, loaders["val"], device, train_ds.xyz_mean, train_ds.xyz_std, xyz_loss_fn, cls_loss_fn, args.class_weight)
        row = {"epoch": epoch, "seconds": time.time() - start, "lr": scheduler.get_last_lr()[0], **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        epoch_rows.append(row)
        if val_metrics["mean_3d_error"] < best_score:
            best_score = val_metrics["mean_3d_error"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "xyz_mean": train_ds.xyz_mean, "xyz_std": train_ds.xyz_std, "args": vars(args), "model_info": model_info}, best_path)
        write_csv(args.out / "train_steps.csv", step_rows)
        write_csv(args.out / "epoch_metrics.csv", epoch_rows)
        plot_history(args.out / "training_curves.png", epoch_rows)
        print(f"epoch {epoch:03d}/{args.epochs}: train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_3d={val_metrics['mean_3d_error']:.3f}m val_acc={val_metrics['classification_accuracy']:.3f}", flush=True)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test"], device, train_ds.xyz_mean, train_ds.xyz_std, xyz_loss_fn, cls_loss_fn, args.class_weight)
    test_metrics["best_epoch"] = checkpoint["epoch"]
    test_metrics.update(model_info)
    with (args.out / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
