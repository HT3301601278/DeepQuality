from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_json(path: str | Path, data: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_metrics_csv(path: str | Path, rows: list[dict]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions_csv(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latency_ms: np.ndarray | None = None,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["index", "y_true", "y_pred"]
        if latency_ms is not None:
            fieldnames.append("latency_ms")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
            row = {"index": index, "y_true": float(true_value), "y_pred": float(pred_value)}
            if latency_ms is not None:
                row["latency_ms"] = float(latency_ms[index])
            writer.writerow(row)
