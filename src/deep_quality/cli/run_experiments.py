from __future__ import annotations

import argparse
import json
import subprocess
import sys

from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.cli.process import build_project_env
from deep_quality.config import PROJECT_ROOT
from deep_quality.utils import save_metrics_csv
from deep_quality.utils.plots import plot_bar_comparison


def main() -> None:
    args = parse_args()
    python = args.python or sys.executable
    checkpoints_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    baseline_config = "configs/sddae_single_scale.yaml"
    improved_config = "configs/ss_ddfae.yaml"

    baseline_100 = checkpoints_dir / "sddae_r_L5_d12_z16_r1.pt"
    improved_100 = checkpoints_dir / "ss_ddfae_L5_d12_z16_r1.pt"

    run(
        [
            python,
            "-m",
            "deep_quality.cli.train_sddae",
            "--config",
            baseline_config,
            "--label-ratio",
            "1.0",
            "--window-size",
            "5",
            "--latent-dim",
            "16",
            *short_args(args),
        ]
    )
    run(
        [
            python,
            "-m",
            "deep_quality.cli.train_ss_ddfae",
            "--config",
            improved_config,
            "--baseline-checkpoint",
            str(baseline_100),
            "--label-ratio",
            "1.0",
            "--window-size",
            "5",
            "--latent-dim",
            "16",
            *short_args(args, semisupervised=True),
        ]
    )
    run([python, "-m", "deep_quality.cli.simulate_online_inference", "--checkpoint", str(improved_100)])

    for ratio in ["0.2", "0.3", "0.5"]:
        baseline = checkpoints_dir / f"sddae_r_L5_d12_z16_r{float(ratio):g}.pt"
        improved = checkpoints_dir / f"ss_ddfae_L5_d12_z16_r{float(ratio):g}.pt"
        run(
            [
                python,
                "-m",
                "deep_quality.cli.train_sddae",
                "--config",
                baseline_config,
                "--label-ratio",
                ratio,
                "--window-size",
                "5",
                "--latent-dim",
                "16",
                *short_args(args),
            ]
        )
        run(
            [
                python,
                "-m",
                "deep_quality.cli.train_ss_ddfae",
                "--config",
                improved_config,
                "--baseline-checkpoint",
                str(baseline),
                "--label-ratio",
                ratio,
                "--window-size",
                "5",
                "--latent-dim",
                "16",
                *short_args(args, semisupervised=True),
            ]
        )
        run([python, "-m", "deep_quality.cli.simulate_online_inference", "--checkpoint", str(improved)])

    for window_size in ["3", "7"]:
        baseline = checkpoints_dir / f"sddae_r_L{window_size}_d12_z16_r1.pt"
        improved = checkpoints_dir / f"ss_ddfae_L{window_size}_d12_z16_r1.pt"
        run(
            [
                python,
                "-m",
                "deep_quality.cli.train_sddae",
                "--config",
                baseline_config,
                "--label-ratio",
                "1.0",
                "--window-size",
                window_size,
                "--latent-dim",
                "16",
                *short_args(args),
            ]
        )
        run(
            [
                python,
                "-m",
                "deep_quality.cli.train_ss_ddfae",
                "--config",
                improved_config,
                "--baseline-checkpoint",
                str(baseline),
                "--label-ratio",
                "1.0",
                "--window-size",
                window_size,
                "--latent-dim",
                "16",
                *short_args(args, semisupervised=True),
            ]
        )
        run([python, "-m", "deep_quality.cli.simulate_online_inference", "--checkpoint", str(improved)])

    for latent_dim in ["8", "32"]:
        baseline = checkpoints_dir / f"sddae_r_L5_d12_z{latent_dim}_r1.pt"
        improved = checkpoints_dir / f"ss_ddfae_L5_d12_z{latent_dim}_r1.pt"
        run(
            [
                python,
                "-m",
                "deep_quality.cli.train_sddae",
                "--config",
                baseline_config,
                "--label-ratio",
                "1.0",
                "--window-size",
                "5",
                "--latent-dim",
                latent_dim,
                *short_args(args),
            ]
        )
        run(
            [
                python,
                "-m",
                "deep_quality.cli.train_ss_ddfae",
                "--config",
                improved_config,
                "--baseline-checkpoint",
                str(baseline),
                "--label-ratio",
                "1.0",
                "--window-size",
                "5",
                "--latent-dim",
                latent_dim,
                *short_args(args, semisupervised=True),
            ]
        )
        run([python, "-m", "deep_quality.cli.simulate_online_inference", "--checkpoint", str(improved)])

    summarize_results()


def short_args(args: argparse.Namespace, semisupervised: bool = False) -> list[str]:
    if args.quick_epochs is None:
        return []
    if semisupervised:
        return ["--epochs", str(args.quick_epochs)]
    return ["--pretrain-epochs", "1", "--finetune-epochs", str(args.quick_epochs)]


def run(command: list[str]) -> None:
    subprocess.check_call(command, cwd=PROJECT_ROOT, env=build_project_env())


def summarize_results() -> None:
    tables_dir = PROJECT_ROOT / "artifacts" / "tables"
    rows = []
    for path in sorted(tables_dir.glob("*_metrics.json")):
        if path.name.endswith("_eval_metrics.json") or path.name.endswith("_online_metrics.json"):
            continue
        with path.open("r", encoding="utf-8") as file:
            rows.append(json.load(file))
    save_metrics_csv(tables_dir / "summary_metrics.csv", rows)
    label_rows = [
        row
        for row in rows
        if int(row["window_size"]) == 5 and int(row["latent_dim"]) == 16
    ]
    plot_bar_comparison(label_rows, "rmse", PROJECT_ROOT / "artifacts" / "figures" / "label_ratio_rmse.png", "RMSE Comparison")
    plot_bar_comparison(label_rows, "mae", PROJECT_ROOT / "artifacts" / "figures" / "label_ratio_mae.png", "MAE Comparison")


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--python")
    parser.add_argument("--quick-epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
