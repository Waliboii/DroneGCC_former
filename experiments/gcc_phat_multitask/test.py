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

from data import MelGccDataset, read_rows
from model import MelGccFusionNet


class WeightedSmoothL1XYZ(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        return (loss * self.weights.view(1, 3)).mean()


def evaluate_with_latency(model, loader, device, mean, std, xyz_loss_fn, cls_loss_fn, class_weight, warmup=5):
    model.eval()
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    total = {
        "loss_sum": 0.0,
        "xyz_loss_sum": 0.0,
        "class_loss_sum": 0.0,
        "samples": 0,
        "correct": 0,
        "mse_x_sum": 0.0,
        "mse_y_sum": 0.0,
        "mse_z_sum": 0.0,
        "mean_3d_sum": 0.0,
    }
    latencies = []
    preds = []
    with torch.no_grad():
        for batch_idx, (mel, gcc, target_norm, cls, target_xyz) in enumerate(loader):
            mel = mel.to(device)
            gcc = gcc.to(device)
            target_norm = target_norm.to(device)
            cls = cls.to(device)
            target_xyz = target_xyz.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred_norm, logits = model(mel, gcc)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if batch_idx >= warmup:
                latencies.append(elapsed / mel.size(0))

            xyz_loss = xyz_loss_fn(pred_norm, target_norm)
            class_loss = cls_loss_fn(logits, cls)
            loss = xyz_loss + class_weight * class_loss
            pred_xyz = pred_norm * std_t + mean_t
            err = pred_xyz - target_xyz
            batch = mel.size(0)
            total["loss_sum"] += loss.item() * batch
            total["xyz_loss_sum"] += xyz_loss.item() * batch
            total["class_loss_sum"] += class_loss.item() * batch
            total["samples"] += batch
            total["correct"] += (logits.argmax(dim=1) == cls).sum().item()
            total["mse_x_sum"] += err[:, 0].pow(2).sum().item()
            total["mse_y_sum"] += err[:, 1].pow(2).sum().item()
            total["mse_z_sum"] += err[:, 2].pow(2).sum().item()
            total["mean_3d_sum"] += err.norm(dim=1).sum().item()

            for i in range(batch):
                preds.append(
                    {
                        "true_class": int(cls[i].item()),
                        "pred_class": int(logits[i].argmax().item()),
                        "true_x": float(target_xyz[i, 0].item()),
                        "true_y": float(target_xyz[i, 1].item()),
                        "true_z": float(target_xyz[i, 2].item()),
                        "pred_x": float(pred_xyz[i, 0].item()),
                        "pred_y": float(pred_xyz[i, 1].item()),
                        "pred_z": float(pred_xyz[i, 2].item()),
                        "error_3d": float(err[i].norm().item()),
                    }
                )

    n = max(total["samples"], 1)
    latency = np.array(latencies, dtype=np.float64)
    return {
        "metrics": {
            "loss": total["loss_sum"] / n,
            "xyz_loss": total["xyz_loss_sum"] / n,
            "class_loss": total["class_loss_sum"] / n,
            "mse_x": total["mse_x_sum"] / n,
            "mse_y": total["mse_y_sum"] / n,
            "mse_z": total["mse_z_sum"] / n,
            "mean_3d_error": total["mean_3d_sum"] / n,
            "classification_accuracy": total["correct"] / n,
            "latency_ms_mean": float(latency.mean() * 1000) if latency.size else None,
            "latency_ms_p50": float(np.percentile(latency, 50) * 1000) if latency.size else None,
            "latency_ms_p95": float(np.percentile(latency, 95) * 1000) if latency.size else None,
            "test_samples": n,
        },
        "predictions": preds,
    }


def plot_confusion(preds, out_path, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in preds:
        matrix[row["true_class"], row["pred_class"]] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Drone Classification Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Test Mel+GCC-PHAT multitask drone model.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/experiments/gcc_phat_multitask"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--class-weight", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    rows = read_rows(args.run_dir / "dataset_index.csv")
    train_ds = MelGccDataset(rows, "train")
    test_ds = MelGccDataset(rows, "test", mean=checkpoint["mean"], std=checkpoint["std"])
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelGccFusionNet(num_classes=5).to(device)
    model.load_state_dict(checkpoint["model"])
    weights = checkpoint.get("args", {}).get("xyz_weights", [1.0, 1.5, 2.0])
    xyz_loss_fn = WeightedSmoothL1XYZ(weights).to(device)
    cls_loss_fn = nn.CrossEntropyLoss()
    result = evaluate_with_latency(
        model, loader, device, checkpoint["mean"], checkpoint["std"], xyz_loss_fn, cls_loss_fn, args.class_weight
    )
    result["metrics"]["best_epoch"] = checkpoint["epoch"]
    with (args.run_dir / "test_metrics_with_latency.json").open("w", encoding="utf-8") as handle:
        json.dump(result["metrics"], handle, indent=2)
    write_predictions(args.run_dir / "test_predictions.csv", result["predictions"])
    plot_confusion(result["predictions"], args.run_dir / "confusion_matrix.png", ["Avata", "M300", "Mavic2", "Mavic3", "Pham4"])
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
