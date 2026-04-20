from __future__ import annotations

import numpy as np


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sample_count = len(x)
    train_end = int(sample_count * train_ratio)
    val_end = train_end + int(sample_count * val_ratio)
    return {
        "train": (x[:train_end], y[:train_end]),
        "val": (x[train_end:val_end], y[train_end:val_end]),
        "test": (x[val_end:], y[val_end:]),
    }


def nested_label_masks(n_samples: int, ratios: list[float], seed: int) -> dict[float, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_samples)
    masks = {}
    for ratio in sorted(ratios):
        count = int(round(n_samples * ratio))
        mask = np.zeros(n_samples, dtype=bool)
        mask[order[:count]] = True
        masks[float(ratio)] = mask
    return masks
