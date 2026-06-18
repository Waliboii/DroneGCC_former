import argparse
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "mmaud_audio_localization") not in sys.path:
    sys.path.insert(0, str(ROOT / "mmaud_audio_localization"))
if str(ROOT / "experiments" / "gcc_temporal_transformer") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments" / "gcc_temporal_transformer"))

from create_spectrograms import build_mel_filterbank, log_mel_spectrogram
from data import compute_gcc_feature
from model import MelGccTemporalTransformer


DRONE_CLASSES = ["Avata", "M300", "Mavic2", "Mavic3", "Pham4"]


def list_devices():
    import sounddevice as sd

    print(sd.query_devices())


def normalize_window(window):
    return (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)


def window_to_features(window, mel_filterbank, args):
    mel_channels = []
    for channel in window:
        mel_channels.append(
            log_mel_spectrogram(
                channel,
                mel_filterbank=mel_filterbank,
                n_fft=args.n_fft,
                hop_length=args.stft_hop_length,
            )
        )
    mel = np.stack(mel_channels, axis=0)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    gcc = compute_gcc_feature(
        window,
        n_fft=args.gcc_n_fft,
        hop=args.gcc_hop,
        max_tau=args.gcc_max_tau,
    )

    mel_t = F.interpolate(
        torch.from_numpy(mel.astype(np.float32)).unsqueeze(0),
        size=(128, 64),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    gcc_t = F.interpolate(
        torch.from_numpy(gcc.astype(np.float32)).unsqueeze(0),
        size=(128, 64),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return mel_t, gcc_t


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MelGccTemporalTransformer(num_classes=len(DRONE_CLASSES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    xyz_mean = torch.tensor(checkpoint["xyz_mean"], dtype=torch.float32, device=device)
    xyz_std = torch.tensor(checkpoint["xyz_std"], dtype=torch.float32, device=device)
    return model, xyz_mean, xyz_std, checkpoint


def direction_from_xyz(xyz, threshold):
    x, _, z = xyz
    horizontal = ""
    vertical = ""

    if x > threshold:
        horizontal = "right"
    elif x < -threshold:
        horizontal = "left"

    if z > threshold:
        vertical = "up"
    elif z < -threshold:
        vertical = "down"

    if vertical and horizontal:
        return f"{vertical} + {horizontal}"
    if vertical:
        return vertical
    if horizontal:
        return horizontal
    return "center / unclear"


def format_prediction(pred_xyz, logits, latency_ms, direction_threshold):
    probs = torch.softmax(logits, dim=1)[0]
    class_id = int(probs.argmax().item())
    confidence = float(probs[class_id].item())
    xyz = pred_xyz[0].detach().cpu().numpy()
    direction = direction_from_xyz(xyz, direction_threshold)
    return (
        f"x={xyz[0]:7.3f} m  y={xyz[1]:7.3f} m  z={xyz[2]:7.3f} m  "
        f"class={DRONE_CLASSES[class_id]} ({confidence:.2%})  "
        f"direction={direction}  latency={latency_ms:.2f} ms"
    )


def run_realtime(args):
    import sounddevice as sd

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, xyz_mean, xyz_std, _ = load_model(args.checkpoint, device)
    mel_filterbank = build_mel_filterbank(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        fmax=args.fmax,
    )

    samples_per_window = int(round(args.window_sec * args.sample_rate))
    samples_per_hop = int(round(args.hop_sec * args.sample_rate))
    ring = np.zeros((args.channels, samples_per_window), dtype=np.float32)
    filled = 0
    since_last = 0
    feature_queue = deque(maxlen=args.sequence_len)
    audio_queue = queue.Queue()

    def callback(indata, frames, _time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    print("Starting microphone stream...")
    print(f"Device: {args.device if args.device is not None else 'default'}")
    print(f"Expected channels: {args.channels}")
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=args.sample_rate,
        channels=args.channels,
        dtype="float32",
        device=args.device,
        blocksize=args.blocksize,
        callback=callback,
    ):
        while True:
            block = audio_queue.get()
            if block.ndim == 1:
                block = block[:, None]
            if block.shape[1] < args.channels:
                raise RuntimeError(f"Microphone returned {block.shape[1]} channels, expected {args.channels}.")
            block = block[:, : args.channels].T

            n = block.shape[1]
            if n >= samples_per_window:
                ring = block[:, -samples_per_window:].astype(np.float32)
                filled = samples_per_window
            else:
                ring = np.roll(ring, -n, axis=1)
                ring[:, -n:] = block
                filled = min(samples_per_window, filled + n)
            since_last += n

            if filled < samples_per_window or since_last < samples_per_hop:
                continue
            since_last = 0

            start_total = time.perf_counter()
            window = normalize_window(ring.copy())
            mel_t, gcc_t = window_to_features(window, mel_filterbank, args)
            feature_queue.append((mel_t, gcc_t))

            if len(feature_queue) < args.sequence_len:
                print(f"warming up sequence buffer {len(feature_queue)}/{args.sequence_len}")
                continue

            mel_seq = torch.stack([item[0] for item in feature_queue], dim=0).unsqueeze(0).to(device)
            gcc_seq = torch.stack([item[1] for item in feature_queue], dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                pred_norm, logits = model(mel_seq, gcc_seq)
                pred_xyz = pred_norm * xyz_std + xyz_mean
                if device.type == "cuda":
                    torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start_total) * 1000.0
            print(format_prediction(pred_xyz, logits, latency_ms, args.direction_threshold), flush=True)


def run_wav(args):
    import wave

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, xyz_mean, xyz_std, _ = load_model(args.checkpoint, device)
    mel_filterbank = build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels, fmax=args.fmax)

    with wave.open(str(args.wav), "rb") as handle:
        if handle.getnchannels() < args.channels:
            raise RuntimeError(f"WAV has {handle.getnchannels()} channels, expected at least {args.channels}.")
        if handle.getframerate() != args.sample_rate:
            raise RuntimeError(f"WAV sample rate is {handle.getframerate()}, expected {args.sample_rate}.")
        raw = handle.readframes(handle.getnframes())
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        audio = audio.reshape(-1, handle.getnchannels())[:, : args.channels].T

    samples_per_window = int(round(args.window_sec * args.sample_rate))
    samples_per_hop = int(round(args.hop_sec * args.sample_rate))
    feature_queue = deque(maxlen=args.sequence_len)
    for start in range(0, audio.shape[1] - samples_per_window + 1, samples_per_hop):
        window = normalize_window(audio[:, start : start + samples_per_window])
        mel_t, gcc_t = window_to_features(window, mel_filterbank, args)
        feature_queue.append((mel_t, gcc_t))
        if len(feature_queue) < args.sequence_len:
            continue
        mel_seq = torch.stack([item[0] for item in feature_queue], dim=0).unsqueeze(0).to(device)
        gcc_seq = torch.stack([item[1] for item in feature_queue], dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            begin = time.perf_counter()
            pred_norm, logits = model(mel_seq, gcc_seq)
            pred_xyz = pred_norm * xyz_std + xyz_mean
            latency_ms = (time.perf_counter() - begin) * 1000.0
        timestamp = (start + samples_per_window / 2) / args.sample_rate
        print(f"t={timestamp:8.3f}s  {format_prediction(pred_xyz, logits, latency_ms, args.direction_threshold)}")


def main():
    parser = argparse.ArgumentParser(description="Realtime drone audio localization from a 4-channel microphone array.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "mel_gcc_temporal_transformer_sequence_random_100ep" / "best_model.pt",
        help="Path to best_model.pt.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default=None, help="sounddevice input device id/name.")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=0.25)
    parser.add_argument("--sequence-len", type=int, default=5)
    parser.add_argument("--direction-threshold", type=float, default=0.25)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--gcc-n-fft", type=int, default=1024)
    parser.add_argument("--gcc-hop", type=int, default=512)
    parser.add_argument("--gcc-max-tau", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wav", type=Path, default=None, help="Optional 4-channel WAV file for offline testing.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.sequence_len != 5:
        print("Warning: model was trained with sequence length 5.", file=sys.stderr)

    if args.wav is not None:
        run_wav(args)
    else:
        run_realtime(args)


if __name__ == "__main__":
    main()
