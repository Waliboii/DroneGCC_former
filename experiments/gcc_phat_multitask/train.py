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

from data import MelGccDataset, build_index, prepare_gcc_cache, write_index
from model import MelGccFusionNet


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


def make_loaders(rows, batch_size, workers):
    train_ds = MelGccDataset(rows, "train")
    val_ds = MelGccDataset(rows, "val", mean=train_ds.mean, std=train_ds.std)
    test_ds = MelGccDataset(rows, "test", mean=train_ds.mean, std=train_ds.std)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
    return loaders, train_ds.mean, train_ds.std


def metrics_from_batch(pred_norm, logits, target_norm, cls, target_xyz, mean, std):
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
        for mel, gcc, target_norm, cls, target_xyz in loader:
            mel = mel.to(device)
            gcc = gcc.to(device)
            target_norm = target_norm.to(device)
            cls = cls.to(device)
            target_xyz = target_xyz.to(device)
            pred_norm, logits = model(mel, gcc)
            xyz_loss = xyz_loss_fn(pred_norm, target_norm)
            cls_loss = cls_loss_fn(logits, cls)
            loss = xyz_loss + cls_weight * cls_loss
            batch = mel.size(0)
            total["loss_sum"] += loss.item() * batch
            total["xyz_loss_sum"] += xyz_loss.item() * batch
            total["cls_loss_sum"] += cls_loss.item() * batch
            merge_sums(total, metrics_from_batch(pred_norm, logits, target_norm, cls, target_xyz, mean_t, std_t))
    metrics = finalize(total)
    n = max(int(total["samples"]), 1)
    metrics.update(
        {
            "loss": total["loss_sum"] / n,
            "xyz_loss": total["xyz_loss_sum"] / n,
            "class_loss": total["cls_loss_sum"] / n,
        }
    )
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(path, epoch_rows):
    epochs = [row["epoch"] for row in epoch_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in epoch_rows], label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in epoch_rows], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row["train_mean_3d_error"] for row in epoch_rows], label="train")
    axes[0, 1].plot(epochs, [row["val_mean_3d_error"] for row in epoch_rows], label="val")
    axes[0, 1].set_title("Mean 3D Error")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["train_classification_accuracy"] for row in epoch_rows], label="train")
    axes[1, 0].plot(epochs, [row["val_classification_accuracy"] for row in epoch_rows], label="val")
    axes[1, 0].set_title("Classification Accuracy")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [row["val_mse_x"] for row in epoch_rows], label="x")
    axes[1, 1].plot(epochs, [row["val_mse_y"] for row in epoch_rows], label="y")
    axes[1, 1].plot(epochs, [row["val_mse_z"] for row in epoch_rows], label="z")
    axes[1, 1].set_title("Validation Coordinate MSE")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Mel+GCC-PHAT multitask drone model.")
    parser.add_argument("--mmaud-root", type=Path, default=Path("MMAUD_Drone"))
    parser.add_argument("--spectrogram-root", type=Path, default=Path("outputs/mmaud_spectrograms"))
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments/gcc_phat_multitask"))
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--class-weight", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--xyz-weights", type=float, nargs=3, default=[1.0, 1.5, 2.0])
    parser.add_argument("--split-mode", choices=["random", "time"], default="random")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_index(args.spectrogram_root, args.mmaud_root, args.out / "features", args.split_mode, args.seed)
    write_index(rows, args.out / "dataset_index.csv")
    if not args.skip_cache:
        prepare_gcc_cache(rows, args.mmaud_root, window_sec=args.window_sec)

    loaders, mean, std = make_loaders(rows, args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelGccFusionNet(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    xyz_loss_fn = WeightedSmoothL1XYZ(args.xyz_weights).to(device)
    cls_loss_fn = nn.CrossEntropyLoss()

    step_rows = []
    epoch_rows = []
    best_score = float("inf")
    best_path = args.out / "best_model.pt"
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    global_step = 0

    print(f"Device: {device}", flush=True)
    print(f"Samples train/val/test: {len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}/{len(loaders['test'].dataset)}", flush=True)
    print(f"Training for {args.epochs} epochs", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        train_sums = {"loss_sum": 0.0, "xyz_loss_sum": 0.0, "cls_loss_sum": 0.0, "samples": 0}
        for batch_idx, (mel, gcc, target_norm, cls, target_xyz) in enumerate(loaders["train"], start=1):
            mel = mel.to(device)
            gcc = gcc.to(device)
            target_norm = target_norm.to(device)
            cls = cls.to(device)
            target_xyz = target_xyz.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm, logits = model(mel, gcc)
            xyz_loss = xyz_loss_fn(pred_norm, target_norm)
            cls_loss = cls_loss_fn(logits, cls)
            loss = xyz_loss + args.class_weight * cls_loss
            loss.backward()
            optimizer.step()

            batch = mel.size(0)
            global_step += 1
            train_sums["loss_sum"] += loss.item() * batch
            train_sums["xyz_loss_sum"] += xyz_loss.item() * batch
            train_sums["cls_loss_sum"] += cls_loss.item() * batch
            merge_sums(train_sums, metrics_from_batch(pred_norm.detach(), logits.detach(), target_norm, cls, target_xyz, mean_t, std_t))
            step_rows.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "batch": batch_idx,
                    "train_loss": loss.item(),
                    "train_xyz_loss": xyz_loss.item(),
                    "train_class_loss": cls_loss.item(),
                }
            )

        scheduler.step()
        train_metrics = finalize(train_sums)
        n_train = max(int(train_sums["samples"]), 1)
        train_metrics.update(
            {
                "loss": train_sums["loss_sum"] / n_train,
                "xyz_loss": train_sums["xyz_loss_sum"] / n_train,
                "class_loss": train_sums["cls_loss_sum"] / n_train,
            }
        )
        val_metrics = evaluate(model, loaders["val"], device, mean, std, xyz_loss_fn, cls_loss_fn, args.class_weight)
        row = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        epoch_rows.append(row)
        if val_metrics["mean_3d_error"] < best_score:
            best_score = val_metrics["mean_3d_error"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                },
                best_path,
            )
        write_csv(args.out / "train_steps.csv", step_rows)
        write_csv(args.out / "epoch_metrics.csv", epoch_rows)
        plot_history(args.out / "training_curves.png", epoch_rows)
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_3d={val_metrics['mean_3d_error']:.3f}m "
            f"val_acc={val_metrics['classification_accuracy']:.3f}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test"], device, mean, std, xyz_loss_fn, cls_loss_fn, args.class_weight)
    test_metrics["best_epoch"] = checkpoint["epoch"]
    with (args.out / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
