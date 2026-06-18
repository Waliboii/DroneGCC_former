import csv
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mmaud_audio_localization") not in sys.path:
    sys.path.insert(0, str(ROOT / "mmaud_audio_localization"))

from create_spectrograms import find_audio_files, load_audio_mono, read_audio_start_time


DRONES = ["Avata", "M300", "Mavic2", "Mavic3", "Pham4"]
MIC_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def find_ground_truth_dir(root, drone):
    candidates = sorted((root / drone).glob("**/ground_truth"))
    if not candidates:
        raise FileNotFoundError(f"No ground_truth folder found for {drone}")
    return candidates[0]


def load_ground_truth(root, drone):
    gt_dir = find_ground_truth_dir(root, drone)
    files = sorted(gt_dir.glob("*.npy"), key=lambda p: float(p.stem))
    times = np.array([float(p.stem) for p in files], dtype=np.float64)
    xyz = np.stack([np.load(p).astype(np.float32) for p in files], axis=0)
    return times, xyz


def interp_xyz(times, xyz, query_time):
    return np.array(
        [
            np.interp(query_time, times, xyz[:, 0]),
            np.interp(query_time, times, xyz[:, 1]),
            np.interp(query_time, times, xyz[:, 2]),
        ],
        dtype=np.float32,
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_index(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "drone_type",
        "class_id",
        "split",
        "start_time",
        "center_time",
        "end_time",
        "mel_path",
        "gcc_path",
        "x",
        "y",
        "z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_index(spectrogram_root, mmaud_root, feature_root, split_mode="random", seed=7):
    class_to_id = {drone: idx for idx, drone in enumerate(DRONES)}
    rows = []
    for drone in DRONES:
        metadata_path = spectrogram_root / drone / "metadata.csv"
        gt_times, gt_xyz = load_ground_truth(mmaud_root, drone)
        for row in read_rows(metadata_path):
            center_time = float(row["center_time"])
            if center_time < gt_times[0] or center_time > gt_times[-1]:
                continue
            xyz = interp_xyz(gt_times, gt_xyz, center_time)
            sample_id = row["sample_id"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "drone_type": drone,
                    "class_id": class_to_id[drone],
                    "start_time": float(row["start_time"]),
                    "center_time": center_time,
                    "end_time": float(row["end_time"]),
                    "mel_path": str(spectrogram_root / drone / row["spectrogram_npz"]),
                    "gcc_path": str(feature_root / "gcc_phat" / drone / f"{sample_id}_gcc.npz"),
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                }
            )

    rng = random.Random(seed)
    split_rows = []
    for drone in DRONES:
        drone_rows = sorted([row for row in rows if row["drone_type"] == drone], key=lambda r: r["center_time"])
        if split_mode == "random":
            rng.shuffle(drone_rows)
        n = len(drone_rows)
        for idx, row in enumerate(drone_rows):
            frac = idx / max(n - 1, 1)
            if frac < 0.70:
                split = "train"
            elif frac < 0.85:
                split = "val"
            else:
                split = "test"
            split_rows.append({**row, "split": split})
    return split_rows


def framed_gcc_phat(sig_a, sig_b, n_fft=1024, hop=512, max_tau=64, eps=1e-8):
    length = min(sig_a.shape[0], sig_b.shape[0])
    sig_a = sig_a[:length]
    sig_b = sig_b[:length]
    if length < n_fft:
        pad = n_fft - length
        sig_a = np.pad(sig_a, (0, pad))
        sig_b = np.pad(sig_b, (0, pad))
        length = n_fft

    frames = 1 + (length - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    out = np.zeros((2 * max_tau + 1, frames), dtype=np.float32)
    for idx in range(frames):
        start = idx * hop
        a = sig_a[start : start + n_fft] * window
        b = sig_b[start : start + n_fft] * window
        fa = np.fft.rfft(a, n=n_fft)
        fb = np.fft.rfft(b, n=n_fft)
        cross = fa * np.conj(fb)
        cross /= np.maximum(np.abs(cross), eps)
        corr = np.fft.irfft(cross, n=n_fft)
        corr = np.concatenate([corr[-max_tau:], corr[: max_tau + 1]])
        out[:, idx] = corr.astype(np.float32)
    return out


def compute_gcc_feature(window, n_fft=1024, hop=512, max_tau=64):
    features = []
    for a_idx, b_idx in MIC_PAIRS:
        features.append(framed_gcc_phat(window[a_idx], window[b_idx], n_fft=n_fft, hop=hop, max_tau=max_tau))
    gcc = np.stack(features, axis=0)
    gcc = np.clip(gcc, -1.0, 1.0).astype(np.float32)
    return gcc


def load_drone_audio(mmaud_root, drone, sample_rate):
    audio_dir = mmaud_root / f"{drone}_audio"
    audio_files = find_audio_files(audio_dir)
    timestamp_csv = audio_dir / "audio1_audio_timestamps.csv"
    audio_start = read_audio_start_time(timestamp_csv)
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        channels = [load_audio_mono(path, sample_rate, temp_dir) for path in audio_files]
    min_len = min(channel.shape[0] for channel in channels)
    return np.stack([channel[:min_len] for channel in channels], axis=0), audio_start


def prepare_gcc_cache(rows, mmaud_root, sample_rate=16000, window_sec=1.0, n_fft=1024, hop=512, max_tau=64):
    by_drone = {drone: [] for drone in DRONES}
    for row in rows:
        by_drone[row["drone_type"]].append(row)

    window_samples = int(round(window_sec * sample_rate))
    for drone, drone_rows in by_drone.items():
        if not drone_rows:
            continue
        print(f"Preparing GCC-PHAT cache for {drone}: {len(drone_rows)} windows", flush=True)
        audio, audio_start = load_drone_audio(mmaud_root, drone, sample_rate)
        for idx, row in enumerate(sorted(drone_rows, key=lambda r: r["start_time"])):
            out_path = Path(row["gcc_path"])
            if out_path.exists():
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            start_sample = int(round((float(row["start_time"]) - audio_start) * sample_rate))
            start_sample = max(0, min(start_sample, audio.shape[1] - window_samples))
            window = audio[:, start_sample : start_sample + window_samples]
            window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
            gcc = compute_gcc_feature(window, n_fft=n_fft, hop=hop, max_tau=max_tau)
            np.savez_compressed(out_path, gcc=gcc)
            if idx == 0 or (idx + 1) % 250 == 0 or idx + 1 == len(drone_rows):
                print(f"  {drone}: {idx + 1}/{len(drone_rows)}", flush=True)


class MelGccDataset(Dataset):
    def __init__(self, rows, split, mel_size=(128, 64), gcc_size=(128, 64), mean=None, std=None):
        self.rows = [row for row in rows if row["split"] == split]
        self.mel_size = mel_size
        self.gcc_size = gcc_size
        xyz = np.array([[row["x"], row["y"], row["z"]] for row in self.rows], dtype=np.float32)
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else xyz.mean(axis=0)
        self.std = np.array(std, dtype=np.float32) if std is not None else np.maximum(xyz.std(axis=0), 1e-6)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mel = np.load(row["mel_path"])["spectrogram"].astype(np.float32)
        gcc = np.load(row["gcc_path"])["gcc"].astype(np.float32)

        mel = torch.from_numpy(mel)
        gcc = torch.from_numpy(gcc)
        mel = F.interpolate(mel.unsqueeze(0), size=self.mel_size, mode="bilinear", align_corners=False).squeeze(0)
        gcc = F.interpolate(gcc.unsqueeze(0), size=self.gcc_size, mode="bilinear", align_corners=False).squeeze(0)

        xyz = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
        xyz_norm = (xyz - self.mean) / self.std
        return (
            mel,
            gcc,
            torch.from_numpy(xyz_norm),
            torch.tensor(int(row["class_id"]), dtype=torch.long),
            torch.from_numpy(xyz),
        )
