from __future__ import annotations

import numpy as np


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    weights: np.ndarray | None = None,
    quality_delay: int = 0,
    stride: int = 1,
    end_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted_x = x if weights is None else x * weights.reshape(1, -1)
    if end_indices is None:
        start_end = (window_size - 1) * stride
        end_indices = np.arange(start_end, len(x) - quality_delay, dtype=np.int64)
    windows = []
    current_u = []
    targets = []
    for end in end_indices:
        start = end - (window_size - 1) * stride
        indices = np.arange(start, end + 1, stride, dtype=np.int64)
        windows.append(weighted_x[indices].reshape(-1))
        current_u.append(x[end])
        targets.append(y[end + quality_delay])
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(current_u, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def make_multiscale_windows(
    x: np.ndarray,
    y: np.ndarray,
    scales: list[tuple[int, int]],
    weights: np.ndarray | None = None,
    quality_delay: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if not scales:
        raise ValueError("多尺度配置不能为空")
    start_end = max((window_size - 1) * stride for window_size, stride in scales)
    end_indices = np.arange(start_end, len(x) - quality_delay, dtype=np.int64)
    branch_windows = []
    current_u = None
    targets = None
    for window_size, stride in scales:
        wx, branch_current_u, branch_targets = make_windows(
            x,
            y,
            window_size,
            weights=weights,
            quality_delay=quality_delay,
            stride=stride,
            end_indices=end_indices,
        )
        branch_windows.append(wx)
        if current_u is None:
            current_u = branch_current_u
            targets = branch_targets
    return branch_windows, current_u, targets
