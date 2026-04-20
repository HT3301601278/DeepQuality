from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.cli.process import build_project_env
from deep_quality.config import PROJECT_ROOT
from deep_quality.utils import save_json, save_metrics_csv

DELAYS = [0, 2, 4, 6, 8, 10, 12]


def main() -> None:
    args = parse_args()
    rows = []
    python = args.python or sys.executable
    for delay in DELAYS:
        output_name = f"sddae_r_L40_d{delay}_z32_r1"
        metrics_path = PROJECT_ROOT / "artifacts" / "tables" / f"{output_name}_metrics.json"
        if not metrics_path.exists():
            run_train(python, args.config, output_name, delay)
        row = load_metrics(metrics_path)
        rows.append(
            {
                **row,
                "delay": delay,
                "test_rmse": row["rmse"],
                "test_mae": row["mae"],
                "test_r2": row["r2"],
            }
        )
    rows.sort(key=lambda item: float(item["test_rmse"]))
    save_metrics_csv(PROJECT_ROOT / "artifacts" / "tables" / "sddae_delay_search.csv", rows)
    save_json(PROJECT_ROOT / "artifacts" / "tables" / "sddae_delay_search.json", rows)
    best = rows[0]
    print(
        f"best_delay={best['delay']} test_rmse={best['test_rmse']} "
        f"test_mae={best['test_mae']} test_r2={best['test_r2']}"
    )


def run_train(python: str, config: str, output_name: str, delay: int) -> None:
    subprocess.check_call(
        [
            python,
            "-m",
            "deep_quality.cli.train_sddae",
            "--config",
            config,
            "--label-ratio",
            "1.0",
            "--window-size",
            "40",
            "--quality-delay",
            str(delay),
            "--latent-dim",
            "32",
            "--output-name",
            output_name,
        ],
        cwd=PROJECT_ROOT,
        env=build_project_env(),
    )


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--config", default="configs/sddae_single_scale.yaml")
    parser.add_argument("--python")
    return parser.parse_args()


if __name__ == "__main__":
    main()
