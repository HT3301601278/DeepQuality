from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.cli.process import build_project_env
from deep_quality.config import PROJECT_ROOT
from deep_quality.inference import build_split_info, collect_sequences, evaluate_candidates, load_runtime
from deep_quality.utils import save_json

DELAYS = [0, 2, 4, 6, 8, 10, 12]
ALPHAS = tuple(float(round(value, 1)) for value in np.arange(0.1, 1.01, 0.1))
WINDOW_SIZE = 40
LATENT_DIM = 32
LABEL_RATIO = 1.0


def main() -> None:
    args = parse_args()
    python = args.python or sys.executable
    rows = []
    for delay in DELAYS:
        output_name = f"sddae_r_L{WINDOW_SIZE}_d{delay}_z{LATENT_DIM}_r{LABEL_RATIO:g}"
        checkpoint_path = PROJECT_ROOT / "artifacts" / "checkpoints" / f"{output_name}.pt"
        run_train(python, args.config, delay, output_name)
        run_tune(python, checkpoint_path)
        row = evaluate_checkpoint(checkpoint_path)
        row["quality_delay"] = delay
        row["checkpoint"] = str(checkpoint_path)
        rows.append(row)

    table_path = PROJECT_ROOT / "artifacts" / "tables" / "sddae_delay_postprocess_search.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "quality_delay",
        "checkpoint",
        "raw_test_rmse",
        "raw_test_r2",
        "post_test_rmse",
        "post_test_r2",
        "selected_method",
        "selected_alpha",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda item: (item["post_test_r2"], -item["post_test_rmse"]))
    summary = {
        "window_size": WINDOW_SIZE,
        "latent_dim": LATENT_DIM,
        "label_ratio": LABEL_RATIO,
        "rows": rows,
        "best": best,
    }
    save_json(PROJECT_ROOT / "artifacts" / "tables" / "sddae_delay_postprocess_search.json", summary)
    print(summary)


def run_train(python: str, config: str, delay: int, output_name: str) -> None:
    subprocess.check_call(
        [
            python,
            "-m",
            "deep_quality.cli.train_sddae",
            "--config",
            config,
            "--label-ratio",
            str(LABEL_RATIO),
            "--window-size",
            str(WINDOW_SIZE),
            "--quality-delay",
            str(delay),
            "--latent-dim",
            str(LATENT_DIM),
            "--output-name",
            output_name,
        ],
        cwd=PROJECT_ROOT,
        env=build_project_env(),
    )


def run_tune(python: str, checkpoint_path: Path) -> None:
    subprocess.check_call(
        [
            python,
            "-m",
            "deep_quality.cli.tune_postprocess",
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=PROJECT_ROOT,
        env=build_project_env(),
    )


def evaluate_checkpoint(checkpoint_path: Path) -> dict:
    runtime = load_runtime(checkpoint_path, "cpu")
    config = runtime.config
    prepared = runtime.prepared
    splits = runtime.splits
    split_order = ("train", "val", "test")
    split_info = build_split_info(splits, split_order)
    model = runtime.model
    device = runtime.device
    y_true_all, y_pred_all = collect_sequences(model, splits, config, prepared["scaler"], device, split_order)
    candidates = evaluate_candidates(y_true_all, y_pred_all, split_info, ALPHAS)
    raw = next(item for item in candidates if item.method == "raw")
    best = min(candidates, key=lambda item: item.val_metrics["rmse"])
    return {
        "raw_test_rmse": float(raw.test_metrics["rmse"]),
        "raw_test_r2": float(raw.test_metrics["r2"]),
        "post_test_rmse": float(best.test_metrics["rmse"]),
        "post_test_r2": float(best.test_metrics["r2"]),
        "selected_method": best.method,
        "selected_alpha": "" if best.alpha is None else float(best.alpha),
    }


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--config", default="configs/sddae_single_scale.yaml")
    parser.add_argument("--python")
    return parser.parse_args()


if __name__ == "__main__":
    main()
