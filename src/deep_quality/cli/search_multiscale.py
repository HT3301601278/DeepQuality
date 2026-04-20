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

COMBINATIONS = [
    {"scales": "40x1,24x2,12x4", "latent_dim": 50},
    {"scales": "40x1,24x2,12x4", "latent_dim": 52},
]


def main() -> None:
    args = parse_args()
    python = args.python or sys.executable
    rows = []
    for combo in COMBINATIONS:
        output_name = build_output_name(combo)
        metrics_path = PROJECT_ROOT / "artifacts" / "tables" / f"{output_name}_metrics.json"
        if not metrics_path.exists():
            run_train(python, args.config, output_name, combo)
        row = load_metrics(metrics_path)
        rows.append(
            {
                "model": row["model"],
                "scales": combo["scales"],
                "latent_dim": combo["latent_dim"],
                "quality_delay": row["quality_delay"],
                "label_ratio": row["label_ratio"],
                "test_rmse": row["rmse"],
                "test_mae": row["mae"],
                "test_r2": row["r2"],
                "checkpoint": str(PROJECT_ROOT / "artifacts" / "checkpoints" / f"{output_name}.pt"),
            }
        )
    rows.sort(key=lambda item: float(item["test_rmse"]))
    save_metrics_csv(PROJECT_ROOT / "artifacts" / "tables" / "multiscale_sddae_search.csv", rows)
    save_json(PROJECT_ROOT / "artifacts" / "tables" / "multiscale_sddae_search.json", rows)
    best = rows[0]
    print(
        f"best_scales={best['scales']} latent_dim={best['latent_dim']} "
        f"rmse={best['test_rmse']} mae={best['test_mae']} r2={best['test_r2']}"
    )


def run_train(python: str, config: str, output_name: str, combo: dict) -> None:
    subprocess.check_call(
        [
            python,
            "-m",
            "deep_quality.cli.train_sddae",
            "--config",
            config,
            "--label-ratio",
            "1.0",
            "--quality-delay",
            "12",
            "--latent-dim",
            str(combo["latent_dim"]),
            "--scales",
            combo["scales"],
            "--output-name",
            output_name,
        ],
        cwd=PROJECT_ROOT,
        env=build_project_env(),
    )


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_output_name(combo: dict) -> str:
    scale_tag = combo["scales"].replace(",", "-")
    return f"sddae_r_ms{scale_tag}_d12_z{combo['latent_dim']}_r1"


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--config", default="configs/sddae.yaml")
    parser.add_argument("--python")
    return parser.parse_args()


if __name__ == "__main__":
    main()
