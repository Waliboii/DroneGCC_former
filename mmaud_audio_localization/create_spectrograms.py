import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image


DRONE_AUDIO_SUFFIX = "_audio"
LOCAL_DEPS = Path(__file__).resolve().parents[1] / "work" / "python_deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))


def hz_to_mel(freq_hz):
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2.0

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_idx in range(1, n_mels + 1):
        left = bins[mel_idx - 1]
        center = bins[mel_idx]
        right = bins[mel_idx + 1]

        if center > left:
            filterbank[mel_idx - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filterbank[mel_idx - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filterbank


def read_audio_start_time(timestamp_csv):
    with timestamp_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    return float(first["timestamp_seconds"])


def decode_mp3_to_wav(mp3_path, wav_path, sample_rate):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        raise RuntimeError(
            "MP3 input requires ffmpeg on PATH. Install ffmpeg or convert the MP3s "
            "to WAV first, then rerun this script on the WAV files."
        )

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    subprocess.run(command, check=True)


def load_wav_mono(wav_path):
    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def load_audio_mono(path, sample_rate, temp_dir):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        audio, sr = load_wav_mono(path)
        if sr != sample_rate:
            raise RuntimeError(
                f"{path} has sample rate {sr}, expected {sample_rate}. "
                "Resample it first or choose --sample-rate to match."
            )
        return audio

    if path.suffix.lower() == ".mp3":
        wav_path = temp_dir / f"{path.stem}_{sample_rate}.wav"
        decode_mp3_to_wav(path, wav_path, sample_rate)
        audio, sr = load_wav_mono(wav_path)
        if sr != sample_rate:
            raise RuntimeError(f"Decoded {path} to unexpected sample rate {sr}.")
        return audio

    raise RuntimeError(f"Unsupported audio extension: {path.suffix}")


def stft_power(audio, n_fft, hop_length):
    if audio.shape[0] < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.shape[0]))

    frame_count = 1 + (audio.shape[0] - n_fft) // hop_length
    if frame_count <= 0:
        frame_count = 1
        audio = np.pad(audio, (0, n_fft - audio.shape[0]))

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_count, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
        writeable=False,
    )
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    return power.T


def log_mel_spectrogram(audio, mel_filterbank, n_fft, hop_length, eps=1e-10):
    power = stft_power(audio, n_fft=n_fft, hop_length=hop_length)
    mel = mel_filterbank @ power
    return np.log10(np.maximum(mel, eps)).astype(np.float32)


def normalize_for_image(spec):
    lo = np.percentile(spec, 1)
    hi = np.percentile(spec, 99)
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip((spec - lo) / (hi - lo), 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def save_preview_png(spec, output_path):
    # spec shape: [channels, mels, frames]. Tile channels vertically.
    tiles = [normalize_for_image(channel) for channel in spec]
    image = np.concatenate(tiles, axis=0)
    image = np.flipud(image)
    Image.fromarray(image, mode="L").save(output_path)


def find_audio_files(audio_dir):
    files = []
    for mic_idx in range(1, 5):
        candidates = sorted(audio_dir.glob(f"audio{mic_idx}_audio.*"))
        candidates = [p for p in candidates if p.suffix.lower() in {".mp3", ".wav"}]
        if not candidates:
            raise FileNotFoundError(f"Missing audio{mic_idx}_audio.mp3/.wav in {audio_dir}")
        files.append(candidates[0])
    return files


def process_drone(drone_name, audio_dir, output_root, args):
    print(f"\n{drone_name}: loading audio from {audio_dir}", flush=True)
    audio_files = find_audio_files(audio_dir)
    timestamp_csv = audio_dir / "audio1_audio_timestamps.csv"
    if not timestamp_csv.exists():
        raise FileNotFoundError(f"Missing timestamp CSV: {timestamp_csv}")

    start_time = read_audio_start_time(timestamp_csv)
    drone_out = output_root / drone_name
    drone_out.mkdir(parents=True, exist_ok=True)

    mel_filterbank = build_mel_filterbank(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        fmax=args.fmax,
    )

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        channels = [
            load_audio_mono(path, args.sample_rate, temp_dir) for path in audio_files
        ]

    min_len = min(channel.shape[0] for channel in channels)
    channels = [channel[:min_len] for channel in channels]

    window_samples = int(round(args.window_sec * args.sample_rate))
    hop_samples = int(round(args.hop_sec * args.sample_rate))
    total_windows = 1 + max(0, (min_len - window_samples) // hop_samples)
    if args.max_windows is not None:
        total_windows = min(total_windows, args.max_windows)

    metadata_path = drone_out / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "drone_type",
                "start_time",
                "center_time",
                "end_time",
                "spectrogram_npz",
                "preview_png",
            ],
        )
        writer.writeheader()

        for idx in range(total_windows):
            start_sample = idx * hop_samples
            end_sample = start_sample + window_samples
            if end_sample > min_len:
                break

            spec_channels = []
            for channel in channels:
                window = channel[start_sample:end_sample]
                window = (window - window.mean()) / (window.std() + 1e-8)
                spec_channels.append(
                    log_mel_spectrogram(
                        window,
                        mel_filterbank=mel_filterbank,
                        n_fft=args.n_fft,
                        hop_length=args.stft_hop_length,
                    )
                )

            spec = np.stack(spec_channels, axis=0)
            spec_min = spec.min()
            spec_max = spec.max()
            spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
            spec = spec.astype(np.float32)

            sample_id = f"{drone_name}_{idx:06d}"
            npz_name = f"{sample_id}.npz"
            png_name = f"{sample_id}.png"
            np.savez_compressed(
                drone_out / npz_name,
                spectrogram=spec,
                drone_type=drone_name,
                start_time=start_time + start_sample / args.sample_rate,
                center_time=start_time + (start_sample + window_samples / 2) / args.sample_rate,
                end_time=start_time + end_sample / args.sample_rate,
            )

            if args.preview_every > 0 and idx % args.preview_every == 0:
                save_preview_png(spec, drone_out / png_name)
                preview_name = png_name
            else:
                preview_name = ""

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "drone_type": drone_name,
                    "start_time": f"{start_time + start_sample / args.sample_rate:.9f}",
                    "center_time": f"{start_time + (start_sample + window_samples / 2) / args.sample_rate:.9f}",
                    "end_time": f"{start_time + end_sample / args.sample_rate:.9f}",
                    "spectrogram_npz": npz_name,
                    "preview_png": preview_name,
                }
            )

            if idx == 0 or (idx + 1) % args.progress_every == 0 or idx + 1 == total_windows:
                print(f"  {idx + 1}/{total_windows} windows", flush=True)

    print(f"{drone_name}: wrote {total_windows} windows to {drone_out}", flush=True)


def discover_drone_audio_dirs(root):
    for audio_dir in sorted(root.glob(f"*{DRONE_AUDIO_SUFFIX}")):
        if audio_dir.is_dir():
            yield audio_dir.name[: -len(DRONE_AUDIO_SUFFIX)], audio_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create 4-channel mel spectrogram windows from MMAUD drone audio."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\Owner\Downloads\MMAUD_Drone"),
        help="Root folder containing <Drone>_audio directories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/mmaud_spectrograms"),
        help="Output directory for spectrogram .npz files and preview .png files.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.25)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--preview-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.hop_sec <= 0:
        raise ValueError("--hop-sec must be positive")
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dirs = list(discover_drone_audio_dirs(args.root))
    if not audio_dirs:
        raise FileNotFoundError(f"No *{DRONE_AUDIO_SUFFIX} folders found under {args.root}")

    print(f"Found {len(audio_dirs)} drone audio folders:", flush=True)
    for drone_name, audio_dir in audio_dirs:
        print(f"  {drone_name}: {audio_dir}", flush=True)

    for drone_name, audio_dir in audio_dirs:
        process_drone(drone_name, audio_dir, args.out, args)

    print(f"\nDone. Spectrogram dataset written to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
