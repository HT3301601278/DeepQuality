from __future__ import annotations

import numpy as np


def clean_missing_values(
    x: np.ndarray,
    y: np.ndarray,
    max_gap: int = 5,
    max_missing_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.column_stack([x, y])
    keep_mask = np.ones(len(data), dtype=bool)
    row_has_missing = np.isnan(data).any(axis=1)
    start = None
    for index, has_missing in enumerate(row_has_missing):
        if has_missing and start is None:
            start = index
        if (not has_missing or index == len(row_has_missing) - 1) and start is not None:
            end = index if not has_missing else index + 1
            if end - start > max_gap:
                keep_mask[start:end] = False
            start = None
    data = data[keep_mask]
    missing_ratios = np.isnan(data).mean(axis=0)
    invalid_columns = np.where(missing_ratios >= max_missing_ratio)[0]
    if len(invalid_columns) > 0:
        raise ValueError(f"以下列的缺失率超出上限：{invalid_columns.tolist()}")
    interpolated = _interpolate_columns(data)
    return interpolated[:, :-1].astype(np.float32), interpolated[:, -1].astype(np.float32)


def _interpolate_columns(data: np.ndarray) -> np.ndarray:
    output = data.copy()
    indices = np.arange(len(output))
    for column_index in range(output.shape[1]):
        values = output[:, column_index]
        valid_mask = ~np.isnan(values)
        if not valid_mask.all():
            output[:, column_index] = np.interp(indices, indices[valid_mask], values[valid_mask])
    return output
