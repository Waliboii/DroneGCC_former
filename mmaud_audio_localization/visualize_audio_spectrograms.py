import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from create_spectrograms import (
    DRONE_AUDIO_SUFFIX,
    build_mel_filterbank,
    discover_drone_audio_dirs,
    load_audio_mono,
    log_mel_spectrogram,
    normalize_for_image,
)


def resize_spectrogram(image, max_width, display_height):
    width = min(image.width, max_width)
    return image.resize((width, display_height), Image.Resampling.LANCZOS)


def inferno_colormap(values):
    stops = np.array(
        [
            [0.0015, 0.0005, 0.0139],
            [0.0874, 0.0446, 0.2248],
            [0.2582, 0.0386, 0.4065],
            [0.4163, 0.0902, 0.4329],
            [0.5783, 0.1480, 0.4044],
            [0.7357, 0.2159, 0.3302],
            [0.8650, 0.3168, 0.2261],
            [0.9553, 0.5000, 0.1237],
            [0.9876, 0.7461, 0.1439],
            [0.9884, 0.9984, 0.6449],
        ],
        dtype=np.float32,
    )
    values = np.clip(values, 0.0, 1.0)
    x = values * (len(stops) - 1)
    left = np.floor(x).astype(np.int32)
    right = np.clip(left + 1, 0, len(stops) - 1)
    frac = (x - left)[..., None]
    rgb = stops[left] * (1.0 - frac) + stops[right] * frac
    return (rgb * 255.0).astype(np.uint8)


def add_title(image, title, subtitle=None, min_width=900):
    title_height = 46 if subtitle else 34
    width = max(image.width, min_width)
    out = Image.new("RGB", (width, image.height + title_height), color=(247, 248, 250))
    out.paste(image.convert("RGB"), (0, title_height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    draw.text((10, 7), title, fill=(18, 24, 38), font=font)
    if subtitle:
        draw.text((10, 27), subtitle, fill=(86, 97, 115), font=small_font)
    return out


def spectrogram_image(spec):
    normalized = normalize_for_image(spec).astype(np.float32) / 255.0
    return Image.fromarray(np.flipud(inferno_colormap(normalized)), mode="RGB")


def save_single_spectrogram(spec, output_path, title, subtitle, max_width):
    image = spectrogram_image(spec)
    image = resize_spectrogram(image, max_width, display_height=220)
    image = add_title(image, title, subtitle)
    image.save(output_path)


def save_combined_spectrogram(specs, output_path, titles, subtitles, max_width):
    images = []
    for spec, title, subtitle in zip(specs, titles, subtitles):
        image = spectrogram_image(spec)
        image = resize_spectrogram(image, max_width, display_height=150)
        images.append(add_title(image, title, subtitle))

    width = max(900, max(image.width for image in images))
    gutter = 12
    header = 56
    height = header + sum(image.height for image in images) + gutter * (len(images) - 1)
    combined = Image.new("RGB", (width, height), color=(247, 248, 250))
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    drone_name = titles[0].split(" mic ")[0] if titles else "Drone"
    draw.text((12, 10), f"{drone_name} - 4 microphone mel spectrograms", fill=(18, 24, 38), font=font)
    draw.text((12, 38), "Frequency increases upward; time moves left to right.", fill=(86, 97, 115), font=small_font)

    y = header
    for image in images:
        combined.paste(image, (0, y))
        y += image.height + gutter

    combined.save(output_path)


def find_audio_files(audio_dir):
    files = []
    for mic_idx in range(1, 5):
        candidates = sorted(audio_dir.glob(f"audio{mic_idx}_audio.*"))
        candidates = [p for p in candidates if p.suffix.lower() in {".mp3", ".wav"}]
        if candidates:
            files.append((mic_idx, candidates[0]))
    return files


def process_audio_file(path, args, temp_dir, mel_filterbank):
    audio = load_audio_mono(path, args.sample_rate, temp_dir)
    if args.max_seconds is not None:
        audio = audio[: int(args.max_seconds * args.sample_rate)]
    audio = (audio - audio.mean()) / (audio.std() + 1e-8)
    return log_mel_spectrogram(
        audio,
        mel_filterbank=mel_filterbank,
        n_fft=args.n_fft,
        hop_length=args.stft_hop_length,
    )


def process_drone(drone_name, audio_dir, output_root, args, mel_filterbank):
    print(f"{drone_name}: {audio_dir}", flush=True)
    drone_out = output_root / drone_name
    drone_out.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"  no audio files found", flush=True)
        return

    specs = []
    titles = []
    subtitles = []

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for mic_idx, path in audio_files:
            print(f"  mic {mic_idx}: {path.name}", flush=True)
            spec = process_audio_file(path, args, temp_dir, mel_filterbank)
            specs.append(spec)
            duration = spec.shape[1] * args.stft_hop_length / args.sample_rate
            title = f"{drone_name} mic {mic_idx}"
            subtitle = f"{path.name} | {spec.shape[0]} mel bins x {spec.shape[1]} frames | approx {duration:.1f}s"
            titles.append(title)
            subtitles.append(subtitle)
            save_single_spectrogram(
                spec,
                drone_out / f"{drone_name}_mic{mic_idx}_spectrogram.png",
                title=title,
                subtitle=subtitle,
                max_width=args.max_width,
            )

    if len(specs) > 1:
        save_combined_spectrogram(
            specs,
            drone_out / f"{drone_name}_all_mics_spectrogram.png",
            titles=titles,
            subtitles=subtitles,
            max_width=args.max_width,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize full-file mel spectrograms for MMAUD drone audio folders."
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
        default=Path("outputs/mmaud_audio_spectrogram_visuals"),
        help="Output folder for spectrogram PNGs.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument(
        "--max-width",
        type=int,
        default=2400,
        help="Maximum PNG width after downscaling the time axis.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optionally visualize only the first N seconds of each file.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dirs = list(discover_drone_audio_dirs(args.root))
    if not audio_dirs:
        raise FileNotFoundError(f"No *{DRONE_AUDIO_SUFFIX} folders found under {args.root}")

    mel_filterbank = build_mel_filterbank(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        fmax=args.fmax,
    )

    print(f"Writing spectrogram visuals to {args.out.resolve()}", flush=True)
    for drone_name, audio_dir in audio_dirs:
        process_drone(drone_name, audio_dir, args.out, args, mel_filterbank)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
