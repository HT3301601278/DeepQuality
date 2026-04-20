from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.inference import build_split_info, checkpoint_postprocess_paths, collect_sequences, \
    evaluate_candidates, load_runtime
from deep_quality.utils import save_json

ALPHAS = tuple(float(round(value, 1)) for value in np.arange(0.1, 1.01, 0.1))


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    runtime = load_runtime(checkpoint_path, args.device)
    config = runtime.config
    prepared = runtime.prepared
    splits = runtime.splits
    model = runtime.model
    device = runtime.device

    split_order = ("train", "val", "test")
    split_info = build_split_info(splits, split_order)
    y_true_all, y_pred_all = collect_sequences(model, splits, config, prepared["scaler"], device, split_order)

    candidates = evaluate_candidates(y_true_all, y_pred_all, split_info, ALPHAS)
    best = min(candidates, key=lambda item: item.val_metrics["rmse"])
    rows = [
        {
            "method": item.method,
            "alpha": "" if item.alpha is None else item.alpha,
            "c": "" if item.c is None else item.c,
            "phi": "" if item.phi is None else item.phi,
            "val_rmse": item.val_metrics["rmse"],
            "val_mae": item.val_metrics["mae"],
            "val_r2": item.val_metrics["r2"],
            "selected": item is best,
        }
        for item in candidates
    ]

    checkpoint_table_path, checkpoint_json_path = checkpoint_postprocess_paths(checkpoint_path)
    checkpoint_table_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_method": best.method,
        "best_alpha": None if best.alpha is None else float(best.alpha),
        "best_c": best.c,
        "best_phi": best.phi,
        "val_metrics": best.val_metrics,
        "test_metrics": best.test_metrics,
        "candidates": rows,
    }
    save_json(checkpoint_json_path, summary)
    print(
        {
            "best_method": best.method,
            "best_alpha": None if best.alpha is None else float(best.alpha),
            "best_c": best.c,
            "best_phi": best.phi,
            "val_metrics": best.val_metrics,
            "summary_path": str(checkpoint_json_path),
        }
    )
    print({"test_metrics": best.test_metrics})


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
