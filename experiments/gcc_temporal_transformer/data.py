import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DATA_PATH = ROOT / "experiments" / "gcc_phat_multitask" / "data.py"
spec = importlib.util.spec_from_file_location("gcc_phat_multitask_data", BASELINE_DATA_PATH)
baseline_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_data)

DRONES = baseline_data.DRONES
build_index = baseline_data.build_index
prepare_gcc_cache = baseline_data.prepare_gcc_cache
read_rows = baseline_data.read_rows
write_index = baseline_data.write_index
compute_gcc_feature = baseline_data.compute_gcc_feature


def make_all_sequences(rows, sequence_len=5, max_gap_sec=0.35):
    if sequence_len % 2 != 1:
        raise ValueError("sequence_len must be odd.")
    radius = sequence_len // 2
    sequences = []
    for drone in DRONES:
        drone_rows = sorted([row for row in rows if row["drone_type"] == drone], key=lambda r: float(r["center_time"]))
        if len(drone_rows) < sequence_len:
            continue
        t0 = float(drone_rows[0]["center_time"])
        for center_idx in range(radius, len(drone_rows) - radius):
            seq = drone_rows[center_idx - radius : center_idx + radius + 1]
            times = [float(row["center_time"]) for row in seq]
            if max(np.diff(times)) <= max_gap_sec:
                center = drone_rows[center_idx]
                sequences.append(
                    {
                        "sequence": seq,
                        "target": center,
                        "drone_type": drone,
                        "center_time": float(center["center_time"]),
                        "relative_time": float(center["center_time"]) - t0,
                    }
                )
    return sequences


def split_sequence_random(sequences, seed=11):
    rng = random.Random(seed)
    by_drone = {drone: [] for drone in DRONES}
    for item in sequences:
        by_drone[item["drone_type"]].append(item)
    split_items = {"train": [], "val": [], "test": []}
    for drone, items in by_drone.items():
        rng.shuffle(items)
        n = len(items)
        for idx, item in enumerate(items):
            frac = idx / max(n - 1, 1)
            if frac < 0.70:
                split = "train"
            elif frac < 0.85:
                split = "val"
            else:
                split = "test"
            split_items[split].append(item)
    return split_items


def split_blocked_random(sequences, block_sec=20.0, seed=11):
    rng = random.Random(seed)
    by_drone_block = {}
    for item in sequences:
        block = int(item["relative_time"] // block_sec)
        by_drone_block.setdefault((item["drone_type"], block), []).append(item)

    split_items = {"train": [], "val": [], "test": []}
    for drone in DRONES:
        blocks = sorted([block for key_drone, block in by_drone_block if key_drone == drone])
        rng.shuffle(blocks)
        n = len(blocks)
        block_to_split = {}
        for idx, block in enumerate(blocks):
            frac = idx / max(n - 1, 1)
            if frac < 0.70:
                split = "train"
            elif frac < 0.85:
                split = "val"
            else:
                split = "test"
            block_to_split[block] = split
        for block, split in block_to_split.items():
            split_items[split].extend(by_drone_block[(drone, block)])
    return split_items


def build_temporal_splits(rows, split_mode, sequence_len=5, max_gap_sec=0.35, block_sec=20.0, seed=11):
    sequences = make_all_sequences(rows, sequence_len=sequence_len, max_gap_sec=max_gap_sec)
    if split_mode == "sequence_random":
        return split_sequence_random(sequences, seed=seed)
    if split_mode == "blocked_random":
        return split_blocked_random(sequences, block_sec=block_sec, seed=seed)
    raise ValueError(f"Unknown temporal split mode: {split_mode}")


class MelGccTemporalDataset(Dataset):
    def __init__(
        self,
        items,
        mel_size=(128, 64),
        gcc_size=(128, 64),
        xyz_mean=None,
        xyz_std=None,
        cache_features=True,
    ):
        self.items = list(items)
        self.mel_size = mel_size
        self.gcc_size = gcc_size
        self.cache_features = cache_features
        xyz = np.array([[item["target"]["x"], item["target"]["y"], item["target"]["z"]] for item in self.items], dtype=np.float32)
        self.xyz_mean = np.array(xyz_mean, dtype=np.float32) if xyz_mean is not None else xyz.mean(axis=0)
        self.xyz_std = np.array(xyz_std, dtype=np.float32) if xyz_std is not None else np.maximum(xyz.std(axis=0), 1e-6)
        self.feature_cache = {}
        if cache_features:
            self._build_feature_cache()

    def _row_key(self, row):
        return row["sample_id"]

    def _load_row_features(self, row):
        mel = np.load(row["mel_path"])["spectrogram"].astype(np.float32)
        gcc = np.load(row["gcc_path"])["gcc"].astype(np.float32)
        mel_t = torch.from_numpy(mel)
        gcc_t = torch.from_numpy(gcc)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=self.mel_size, mode="bilinear", align_corners=False).squeeze(0)
        gcc_t = F.interpolate(gcc_t.unsqueeze(0), size=self.gcc_size, mode="bilinear", align_corners=False).squeeze(0)
        return mel_t, gcc_t

    def _build_feature_cache(self):
        unique_rows = {}
        for item in self.items:
            for row in item["sequence"]:
                unique_rows[self._row_key(row)] = row
        for key, row in unique_rows.items():
            self.feature_cache[key] = self._load_row_features(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel_seq = []
        gcc_seq = []
        for row in item["sequence"]:
            if self.cache_features:
                mel_t, gcc_t = self.feature_cache[self._row_key(row)]
            else:
                mel_t, gcc_t = self._load_row_features(row)
            mel_seq.append(mel_t)
            gcc_seq.append(gcc_t)

        target = item["target"]
        xyz = np.array([target["x"], target["y"], target["z"]], dtype=np.float32)
        xyz_norm = (xyz - self.xyz_mean) / self.xyz_std
        return (
            torch.stack(mel_seq, dim=0),
            torch.stack(gcc_seq, dim=0),
            torch.from_numpy(xyz_norm.astype(np.float32)),
            torch.tensor(int(target["class_id"]), dtype=torch.long),
            torch.from_numpy(xyz),
        )
