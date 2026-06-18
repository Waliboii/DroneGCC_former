import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    ("CNN_GCC_best_random", "outputs/experiments/gcc_phat_multitask_2s_weighted_100ep", "random windows"),
    ("Transformer_temporal_single_window", "outputs/experiments/transformer_temporal_100ep", "random windows"),
    ("Transformer_patch", "outputs/experiments/transformer_patch_100ep", "random windows"),
    ("Transformer_dual_token", "outputs/experiments/transformer_dual_token_100ep", "random windows"),
    ("CNN_GCC_TDOA_token_transformer", "outputs/experiments/gcc_tdoa_token_transformer_100ep", "random windows"),
    ("CNN_GCC_TDOA_MLP_fusion", "outputs/experiments/gcc_tdoa_mlp_fusion_100ep", "random windows"),
    ("CNN_GCC_TDOA_temporal_transformer", "outputs/experiments/gcc_tdoa_temporal_transformer_cached_100ep", "time split"),
    ("Mel_GCC_temporal_transformer_sequence_random", "outputs/experiments/mel_gcc_temporal_transformer_sequence_random_100ep", "sequence random"),
    ("Mel_GCC_temporal_transformer_blocked_random", "outputs/experiments/mel_gcc_temporal_transformer_blocked_random_100ep", "blocked random"),
]

REFERENCE_MODEL_INFO = {
    "gcc_phat_multitask_2s_weighted_100ep": {
        "parameters": 710728,
        "trainable_parameters": 710728,
        "model_size_mb_fp32": 2.711212158203125,
    }
}


def read_metrics(run_dir):
    run_dir = Path(run_dir)
    for name in ["test_metrics_with_latency.json", "test_metrics.json"]:
        path = run_dir / name
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics.update({k: v for k, v in REFERENCE_MODEL_INFO.get(run_dir.name, {}).items() if k not in metrics})
            return metrics
    return None


def latency(metrics):
    for key in ["processed_forward_latency_ms_mean", "latency_ms_mean"]:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def raw_latency(metrics):
    for key in ["raw_sequence_to_decision_latency_ms_mean", "raw_window_to_decision_latency_ms_mean"]:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def plot(out_path, rows):
    names = [row["name"] for row in rows]
    mean_3d = [row["mean_3d_error"] for row in rows]
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["#4779c4" if "random" in row["split"] else "#b56576" for row in rows]
    ax.bar(range(len(rows)), mean_3d, color=colors)
    ax.set_xticks(range(len(rows)), names, rotation=40, ha="right")
    ax.set_ylabel("Mean 3D Error (m)")
    ax.set_title("Model Comparison")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(mean_3d):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare all MMAUD audio localization model runs.")
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments/all_model_comparison"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, run_dir, split in RUNS:
        metrics = read_metrics(run_dir)
        if metrics is None:
            continue
        rows.append(
            {
                "name": name,
                "split": split,
                "mse_x": metrics.get("mse_x"),
                "mse_y": metrics.get("mse_y"),
                "mse_z": metrics.get("mse_z"),
                "mean_3d_error": metrics.get("mean_3d_error"),
                "classification_accuracy": metrics.get("classification_accuracy"),
                "processed_latency_ms": latency(metrics),
                "raw_latency_ms": raw_latency(metrics),
                "parameters": metrics.get("parameters"),
                "model_size_mb_fp32": metrics.get("model_size_mb_fp32"),
                "best_epoch": metrics.get("best_epoch"),
                "run_dir": run_dir,
            }
        )

    headers = [
        "Model",
        "Split",
        "MSE x",
        "MSE y",
        "MSE z",
        "Mean 3D",
        "Class Acc",
        "Proc Lat ms",
        "Raw Lat ms",
        "Params",
        "Size MB",
        "Best Epoch",
    ]
    lines = ["# Overall Model Comparison", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["split"],
                    fmt(row["mse_x"]),
                    fmt(row["mse_y"]),
                    fmt(row["mse_z"]),
                    fmt(row["mean_3d_error"]),
                    fmt(row["classification_accuracy"]),
                    fmt(row["processed_latency_ms"]),
                    fmt(row["raw_latency_ms"]),
                    f"{int(row['parameters']):,}" if row["parameters"] is not None else "",
                    fmt(row["model_size_mb_fp32"], 2),
                    str(row["best_epoch"] or ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `Proc Lat ms` is prepared tensor/features to decision.",
            "- `Raw Lat ms` is decoded audio window/sequence to decision when that experiment measured it.",
            "- Sequence-random is useful for architecture comparison but can have overlap leakage between neighboring sequences.",
            "- Blocked-random and time split are stricter generalization tests.",
        ]
    )
    (args.out / "overall_results.md").write_text("\n".join(lines), encoding="utf-8")
    plot(args.out / "overall_mean_3d_error.png", rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
