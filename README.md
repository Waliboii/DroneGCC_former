# DroneGCC-Former

**DroneGCC-Former** is a multichannel audio model for drone localization and drone type classification. It uses 4-microphone audio from the MMAUD dataset, converts each 2-second window into both mel spectrogram features and GCC-PHAT spatial features, then uses a temporal Transformer over five consecutive windows to predict the drone position `(x, y, z)` and drone class.

Repository: [Waliboii/DroneGCC_former](https://github.com/Waliboii/DroneGCC_former)

![DroneGCC-Former architecture](docs/mel_gcc_transformer.png)

## What The Model Does

The model estimates:

```text
x coordinate in meters
y coordinate in meters
z coordinate in meters
drone type: Avata, M300, Mavic2, Mavic3, or Pham4
```

The input is a 5-window temporal sequence:

```text
Window t-2, Window t-1, Window t, Window t+1, Window t+2
```

Each window is 2 seconds long and contains 4 synchronized microphone channels.

For each window, the model builds two feature streams:

```text
4-channel log-mel spectrogram -> CNN -> 128-d mel feature
6-channel GCC-PHAT map        -> CNN -> 128-d GCC feature
```

The two 128-dimensional features are concatenated and projected into a 256-dimensional window embedding. Five window embeddings are passed into a 2-layer Transformer Encoder. The center token, corresponding to `Window t`, is used by four output heads:

```text
x regression head
y regression head
z regression head
drone classification head
```

## Best Included Run

The included best run is:

```text
outputs/experiments/mel_gcc_temporal_transformer_sequence_random_100ep
```

Best test metrics:

```text
MSE x: 0.0383
MSE y: 0.1474
MSE z: 0.1183
Mean 3D error: 0.4757 m
Classification accuracy: 1.0000
Processed sequence latency: 1.7660 ms/sample
Raw sequence latency: 395.4680 ms/sample
Parameters: 2,381,768
Model size: 9.09 MB fp32
Best epoch: 100
```


## Repository Layout

```text
DroneGCC_former/
  README.md
  requirements.txt
  .gitignore

  docs/
    mel_gcc_transformer.png

  mmaud_audio_localization/
    create_spectrograms.py
    visualize_audio_spectrograms.py

  experiments/
    gcc_phat_multitask/
      data.py
      model.py
      train.py
      test.py

    gcc_temporal_transformer/
      data.py
      model.py
      train.py
      test.py
      plot_pretty_losses.py
      compare_all_models.py

  scripts/
    realtime_drone_localization.py

  outputs/
    experiments/
      mel_gcc_temporal_transformer_sequence_random_100ep/
        best_model.pt
        epoch_metrics.csv
        train_steps.csv
        test_metrics.json
        test_metrics_with_latency.json
        test_predictions.csv
        model_size.json
        summary.md
        training_curves.png
        training_dashboard.png
        pretty_loss_focus.png
        pretty_step_losses.png
        confusion_matrix.png
        prediction_scatter.png
```

## Files Not Included In Git

The raw MMAUD data and generated feature caches are intentionally ignored by `.gitignore`:

```text
MMAUD_Drone/
outputs/mmaud_spectrograms_2s/
outputs/experiments/gcc_phat_multitask_2s_weighted_100ep/features/
*.mp3
*.wav
*.bag
```

These files can be regenerated from the MMAUD audio. If you want to distribute them, use a separate dataset artifact such as GitHub Releases, Hugging Face Dataset, Zenodo, Google Drive, or OneDrive.

## Setup On Windows

Open PowerShell in the repository folder:

```powershell
cd "D:\drone_localization_gcc_cnn_transformers"
```

If you cloned from GitHub instead:

```powershell
git clone https://github.com/Waliboii/DroneGCC_former.git
cd DroneGCC_former
```

Create a virtual environment:

```powershell
python -m venv .venv
```

If PowerShell blocks activation, you can skip activation and call Python directly with `.venv\Scripts\python.exe`. To activate the environment anyway:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want CUDA GPU support, install the correct PyTorch build for your system from the official PyTorch instructions:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Install FFmpeg

FFmpeg is needed when the MMAUD audio files are MP3. Install it with winget:

```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell after installation, then verify:

```powershell
ffmpeg -version
```

If `ffmpeg` is still not recognized, restart VS Code or your terminal so the updated PATH is loaded.

## Prepare The MMAUD Data

Place the MMAUD audio and ground-truth folders under:

```text
MMAUD_Drone/
```

Expected structure:

```text
MMAUD_Drone/
  Avata_audio/
    audio1_audio.mp3
    audio2_audio.mp3
    audio3_audio.mp3
    audio4_audio.mp3
    audio1_audio_timestamps.csv
    ...
  Avata/
    ground_truth/
      *.npy

  M300_audio/
  M300/ground_truth/

  Mavic2_audio/
  Mavic2/ground_truth/

  Mavic3_audio/
  Mavic3/ground_truth/

  Pham4_audio/
  Pham4/ground_truth/
```

The code expects 4 microphone channels per drone. The class names are fixed as:

```text
Avata, M300, Mavic2, Mavic3, Pham4
```

## Step 1: Create Mel Spectrograms

Run this from the repository root:

```powershell
python mmaud_audio_localization\create_spectrograms.py `
  --root MMAUD_Drone `
  --out outputs\mmaud_spectrograms_2s `
  --window-sec 2.0 `
  --hop-sec 0.25 `
  --sample-rate 16000
```

This creates one `.npz` mel spectrogram file per time window, plus preview images and metadata.

If you are using an explicit Python executable instead of an activated environment:

```powershell
& ".\.venv\Scripts\python.exe" mmaud_audio_localization\create_spectrograms.py `
  --root MMAUD_Drone `
  --out outputs\mmaud_spectrograms_2s `
  --window-sec 2.0 `
  --hop-sec 0.25 `
  --sample-rate 16000
```

## Step 2: Train The Temporal Transformer

Run this command to train the best architecture for 100 epochs:

```powershell
python experiments\gcc_temporal_transformer\train.py `
  --mmaud-root MMAUD_Drone `
  --spectrogram-root outputs\mmaud_spectrograms_2s `
  --feature-root outputs\experiments\gcc_features_temporal_transformer `
  --out outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep `
  --temporal-split sequence_random `
  --epochs 100 `
  --batch-size 32 `
  --workers 0 `
  --lr 2e-4 `
  --class-weight 0.5 `
  --window-sec 2.0 `
  --xyz-weights 1.0 2.0 2.5 `
  --seed 11
```

This training command does three things:

1. Reads the mel spectrogram index from `outputs/mmaud_spectrograms_2s`.
2. Computes GCC-PHAT maps into `--feature-root` if they do not already exist.
3. Trains the Mel+GCC temporal Transformer and stores results in `--out`.

Do not pass `--skip-cache` if you want GCC-PHAT maps to be generated automatically.

The optimizer is AdamW with weight decay `1e-4`. The learning-rate scheduler is cosine annealing over the full training run.

## Step 3: Evaluate The Best Model

Evaluate the trained checkpoint:

```powershell
python experiments\gcc_temporal_transformer\test.py `
  --run-dir outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep `
  --mmaud-root MMAUD_Drone `
  --batch-size 32 `
  --raw-latency-samples 100
```

This writes:

```text
test_metrics_with_latency.json
test_predictions.csv
confusion_matrix.png
prediction_scatter.png
summary.md
```

To force CPU evaluation:

```powershell
python experiments\gcc_temporal_transformer\test.py `
  --run-dir outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep `
  --mmaud-root MMAUD_Drone `
  --batch-size 32 `
  --raw-latency-samples 100 `
  --cpu
```

If FFmpeg is not installed and you only want processed-feature latency, skip raw audio latency:

```powershell
python experiments\gcc_temporal_transformer\test.py `
  --run-dir outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep `
  --batch-size 32 `
  --raw-latency-samples 0
```

## Step 4: Recreate Pretty Training Plots

```powershell
python experiments\gcc_temporal_transformer\plot_pretty_losses.py `
  --run-dir outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep `
  --smooth-window 3
```

This creates or updates:

```text
pretty_training_dashboard.png
pretty_loss_focus.png
pretty_step_losses.png
pretty_loss_summary.md
```

## Run The Included Pretrained Model

The repository includes the best checkpoint here:

```text
outputs/experiments/mel_gcc_temporal_transformer_sequence_random_100ep/best_model.pt
```

If the run folder is present, you can evaluate immediately after installing dependencies and placing `MMAUD_Drone/` in the repo root:

```powershell
python experiments\gcc_temporal_transformer\test.py `
  --run-dir outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep `
  --mmaud-root MMAUD_Drone `
  --batch-size 32 `
  --raw-latency-samples 100
```

## Realtime Or WAV Inference

The realtime script is:

```text
scripts/realtime_drone_localization.py
```

List available audio devices:

```powershell
python scripts\realtime_drone_localization.py --list-devices
```

Run realtime inference with a 4-channel microphone array:

```powershell
python scripts\realtime_drone_localization.py `
  --checkpoint outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep\best_model.pt `
  --device 1 `
  --channels 4
```

Force CPU realtime inference:

```powershell
python scripts\realtime_drone_localization.py `
  --checkpoint outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep\best_model.pt `
  --device 1 `
  --channels 4 `
  --cpu
```

Run offline inference on a 4-channel WAV file:

```powershell
python scripts\realtime_drone_localization.py `
  --checkpoint outputs\experiments\mel_gcc_temporal_transformer_sequence_random_100ep\best_model.pt `
  --wav path\to\four_channel_audio.wav `
  --channels 4
```

The realtime script prints predictions like:

```text
x=  1.234 m  y=  4.567 m  z=  0.890 m  class=Mavic3 (99.12%)  direction=up + right  latency=12.34 ms
```

Direction interpretation uses the predicted `x` and `z` coordinates:

```text
x > 0: right
x < 0: left
z > 0: up
z < 0: down
```

## Important Scripts

Mel spectrogram creation:

```text
mmaud_audio_localization/create_spectrograms.py
```

Spectrogram visualization:

```text
mmaud_audio_localization/visualize_audio_spectrograms.py
```

GCC-PHAT feature computation:

```text
experiments/gcc_phat_multitask/data.py
```

Temporal Transformer training:

```text
experiments/gcc_temporal_transformer/train.py
```

Temporal Transformer evaluation:

```text
experiments/gcc_temporal_transformer/test.py
```

Realtime inference:

```text
scripts/realtime_drone_localization.py
```

## Model Architecture Details

Per 2-second audio window:

1. A 4-channel mel spectrogram is passed through a CNN branch with three convolutional blocks.
2. A 6-channel GCC-PHAT map is passed through a second CNN branch with the same block structure.
3. The mel branch outputs a 128-dimensional feature vector.
4. The GCC-PHAT branch outputs a 128-dimensional feature vector.
5. These are concatenated into a 256-dimensional vector.
6. A fusion MLP produces one 256-dimensional embedding per window.

Across time:

1. Five consecutive window embeddings are stacked into a sequence.
2. Positional embeddings are added.
3. A 2-layer Transformer Encoder with 4 attention heads models temporal context.
4. The center token is selected as the representation of the target time.
5. Separate heads predict `x`, `y`, `z`, and drone class.

## Metric Meaning

Mean 3D error is the average Euclidean distance between predicted and true position:

```text
sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2 + (z_pred - z_true)^2)
```

So a mean 3D error of `0.4757 m` means the prediction is, on average, about 0.48 meters away from the ground-truth target position in 3D space.

## GitHub Upload Notes

The intended GitHub upload includes:

```text
README.md
requirements.txt
.gitignore
docs/mel_gcc_transformer.png
experiments/
mmaud_audio_localization/
scripts/
outputs/experiments/mel_gcc_temporal_transformer_sequence_random_100ep/
```

The intended GitHub upload excludes:

```text
MMAUD_Drone/
outputs/mmaud_spectrograms_2s/
outputs/experiments/gcc_phat_multitask_2s_weighted_100ep/features/
*.mp3
*.wav
*.bag
```

## Citation / Dataset

This project is built around the MMAUD dataset from NTU ARIS:

[https://github.com/ntu-aris/MMAUD](https://github.com/ntu-aris/MMAUD)

Please follow the MMAUD authors' dataset license, citation requirements, and distribution rules when using or sharing the raw data.
