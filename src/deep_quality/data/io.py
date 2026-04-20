from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_csv_dataset(path: str | Path, target_column: str = "y") -> tuple[np.ndarray, np.ndarray, list[str]]:
    file_path = Path(path)
    with file_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if target_column not in reader.fieldnames:
            raise ValueError(f"未找到目标列：{target_column}")
        feature_names = [name for name in reader.fieldnames if name != target_column]
        rows = []
        targets = []
        for row in reader:
            rows.append([_to_float(row[name]) for name in feature_names])
            targets.append(_to_float(row[target_column]))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32), feature_names


def _to_float(value: str) -> float:
    value = value.strip()
    return float(value) if value else float("nan")
