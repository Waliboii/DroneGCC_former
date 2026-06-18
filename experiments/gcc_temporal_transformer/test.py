import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mmaud_audio_localization") not in sys.path:
    sys.path.insert(0, str(ROOT / "mmaud_audio_localization"))

from create_spectrograms import build_mel_filterbank, find_audio_files, load_audio_mono, log_mel_spectrogram, read_audio_start_time
from data import DRONES, MelGccTemporalDataset, build_temporal_splits, compute_gcc_feature, read_rows
from model import MelGccTemporalTransformer


class WeightedSmoothL1XYZ(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        return (loss * self.weights.view(1, 3)).mean()


def evaluate_with_latency(model, loader, device, xyz_mean, xyz_std, xyz_loss_fn, cls_loss_fn, class_weight, warmup=5):
    model.eval()
    mean_t = torch.tensor(xyz_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(xyz_std, dtype=torch.float32, device=device)
    total = {"loss_sum": 0.0, "xyz_loss_sum": 0.0, "class_loss_sum": 0.0, "samples": 0, "correct": 0, "mse_x_sum": 0.0, "mse_y_sum": 0.0, "mse_z_sum": 0.0, "mean_3d_sum": 0.0}
    latencies = []
    preds = []
    with torch.no_grad():
        for batch_idx, (mel_seq, gcc_seq, target_norm, cls, target_xyz) in enumerate(loader):
            mel_seq = mel_seq.to(device)
            gcc_seq = gcc_seq.to(device)
            target_norm = target_norm.to(device)
            cls = cls.to(device)
            target_xyz = target_xyz.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred_norm, logits = model(mel_seq, gcc_seq)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if batch_idx >= warmup:
                latencies.append(elapsed / mel_seq.size(0))

            xyz_loss = xyz_loss_fn(pred_norm, target_norm)
            class_loss = cls_loss_fn(logits, cls)
            loss = xyz_loss + class_weight * class_loss
            pred_xyz = pred_norm * std_t + mean_t
            err = pred_xyz - target_xyz
            batch = mel_seq.size(0)
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
            "processed_forward_latency_ms_mean": float(latency.mean() * 1000) if latency.size else None,
            "processed_forward_latency_ms_p50": float(np.percentile(latency, 50) * 1000) if latency.size else None,
            "processed_forward_latency_ms_p95": float(np.percentile(latency, 95) * 1000) if latency.size else None,
            "test_samples": n,
        },
        "predictions": preds,
    }


def load_all_audio(mmaud_root, sample_rate):
    audio_by_drone = {}
    decode_seconds = {}
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for drone in DRONES:
            start = time.perf_counter()
            audio_dir = mmaud_root / f"{drone}_audio"
            files = find_audio_files(audio_dir)
            audio_start = read_audio_start_time(audio_dir / "audio1_audio_timestamps.csv")
            channels = [load_audio_mono(path, sample_rate, temp_dir) for path in files]
            min_len = min(channel.shape[0] for channel in channels)
            audio_by_drone[drone] = (np.stack([channel[:min_len] for channel in channels], axis=0), audio_start)
            decode_seconds[drone] = time.perf_counter() - start
    return audio_by_drone, decode_seconds


def raw_sequence_latencies(model, items, checkpoint, args, device, max_samples):
    model.eval()
    mel_filterbank = build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels, fmax=args.fmax)
    audio_by_drone, decode_seconds = load_all_audio(args.mmaud_root, args.sample_rate)
    selected = list(items)[:max_samples]
    window_samples = int(round(args.window_sec * args.sample_rate))
    latencies = []
    with torch.no_grad():
        for item in selected:
            start = time.perf_counter()
            mel_seq = []
            gcc_seq = []
            for row in item["sequence"]:
                audio, audio_start = audio_by_drone[row["drone_type"]]
                start_sample = int(round((float(row["start_time"]) - audio_start) * args.sample_rate))
                start_sample = max(0, min(start_sample, audio.shape[1] - window_samples))
                window = audio[:, start_sample : start_sample + window_samples]
                window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)

                mel = []
                for channel in window:
                    mel.append(log_mel_spectrogram(channel, mel_filterbank, args.n_fft, args.stft_hop_length))
                mel = np.stack(mel, axis=0)
                mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
                gcc = compute_gcc_feature(window, n_fft=args.gcc_n_fft, hop=args.gcc_hop, max_tau=args.gcc_max_tau)
                mel_seq.append(F.interpolate(torch.from_numpy(mel).unsqueeze(0), size=(128, 64), mode="bilinear", align_corners=False).squeeze(0))
                gcc_seq.append(F.interpolate(torch.from_numpy(gcc).unsqueeze(0), size=(128, 64), mode="bilinear", align_corners=False).squeeze(0))
            mel_t = torch.stack(mel_seq, dim=0).unsqueeze(0).to(device)
            gcc_t = torch.stack(gcc_seq, dim=0).unsqueeze(0).to(device)
            _ = model(mel_t, gcc_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
    latency = np.array(latencies, dtype=np.float64)
    return {
        "raw_sequence_to_decision_latency_ms_mean": float(latency.mean() * 1000) if latency.size else None,
        "raw_sequence_to_decision_latency_ms_p50": float(np.percentile(latency, 50) * 1000) if latency.size else None,
        "raw_sequence_to_decision_latency_ms_p95": float(np.percentile(latency, 95) * 1000) if latency.size else None,
        "raw_latency_samples": int(latency.size),
        "raw_audio_decode_seconds_total": float(sum(decode_seconds.values())),
        "raw_audio_decode_seconds_by_drone": decode_seconds,
    }


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(preds, out_path):
    matrix = np.zeros((len(DRONES), len(DRONES)), dtype=np.int64)
    for row in preds:
        matrix[row["true_class"], row["pred_class"]] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(DRONES)), labels=DRONES, rotation=35, ha="right")
    ax.set_yticks(range(len(DRONES)), labels=DRONES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Drone Classification Confusion Matrix")
    for i in range(len(DRONES)):
        for j in range(len(DRONES)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prediction_scatter(preds, out_path):
    true = np.array([[p["true_x"], p["true_y"], p["true_z"]] for p in preds], dtype=np.float32)
    pred = np.array([[p["pred_x"], p["pred_y"], p["pred_z"]] for p in preds], dtype=np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, name in enumerate(["x", "y", "z"]):
        axes[i].scatter(true[:, i], pred[:, i], s=14, alpha=0.65)
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        axes[i].plot([lo, hi], [lo, hi], color="black", lw=1)
        axes[i].set_title(f"{name} Prediction")
        axes[i].set_xlabel("True")
        axes[i].set_ylabel("Predicted")
        axes[i].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(path, metrics):
    lines = [
        f"# {metrics['model_name']} Results",
        "",
        f"- Best epoch: {metrics['best_epoch']}",
        f"- Test samples: {metrics['test_samples']}",
        f"- MSE x: {metrics['mse_x']:.6f}",
        f"- MSE y: {metrics['mse_y']:.6f}",
        f"- MSE z: {metrics['mse_z']:.6f}",
        f"- Mean 3D error: {metrics['mean_3d_error']:.6f} m",
        f"- Classification accuracy: {metrics['classification_accuracy']:.6f}",
        f"- Processed sequence forward latency: {metrics['processed_forward_latency_ms_mean']:.6f} ms/sample",
        f"- Raw sequence to decision latency: {metrics['raw_sequence_to_decision_latency_ms_mean']:.6f} ms/sample",
        f"- One-time raw audio decode total: {metrics['raw_audio_decode_seconds_total']:.3f} s",
        f"- Parameters: {metrics['parameters']:,}",
        f"- Estimated fp32 model size: {metrics['model_size_mb_fp32']:.3f} MB",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mel+GCC temporal Transformer.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mmaud-root", type=Path, default=Path("MMAUD_Drone"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--gcc-n-fft", type=int, default=1024)
    parser.add_argument("--gcc-hop", type=int, default=512)
    parser.add_argument("--gcc-max-tau", type=int, default=64)
    parser.add_argument("--raw-latency-samples", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even when CUDA is available.")
    args = parser.parse_args()

    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    run_args = checkpoint["args"]
    rows = read_rows(args.run_dir / "dataset_index.csv")
    split_items = build_temporal_splits(
        rows,
        run_args["temporal_split"],
        sequence_len=run_args.get("sequence_len", 5),
        block_sec=run_args.get("block_sec", 20.0),
        seed=run_args.get("seed", 11),
    )
    test_ds = MelGccTemporalDataset(split_items["test"], xyz_mean=checkpoint["xyz_mean"], xyz_std=checkpoint["xyz_std"])
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}", flush=True)
    model = MelGccTemporalTransformer(num_classes=len(DRONES)).to(device)
    model.load_state_dict(checkpoint["model"])
    weights = run_args.get("xyz_weights", [1.0, 2.0, 2.5])
    xyz_loss_fn = WeightedSmoothL1XYZ(weights).to(device)
    cls_loss_fn = nn.CrossEntropyLoss()
    result = evaluate_with_latency(
        model,
        loader,
        device,
        checkpoint["xyz_mean"],
        checkpoint["xyz_std"],
        xyz_loss_fn,
        cls_loss_fn,
        run_args.get("class_weight", 0.5),
    )
    metrics = result["metrics"]
    metrics.update(raw_sequence_latencies(model, split_items["test"], checkpoint, args, device, args.raw_latency_samples))
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics.update(checkpoint["model_info"])
    with (args.run_dir / "test_metrics_with_latency.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_predictions(args.run_dir / "test_predictions.csv", result["predictions"])
    plot_confusion(result["predictions"], args.run_dir / "confusion_matrix.png")
    plot_prediction_scatter(result["predictions"], args.run_dir / "prediction_scatter.png")
    write_summary(args.run_dir / "summary.md", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
