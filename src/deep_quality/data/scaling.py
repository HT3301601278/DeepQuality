from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    x_low: np.ndarray
    x_high: np.ndarray
    y_low: float
    y_high: float

    @classmethod
    def fit(cls, x_train: np.ndarray, y_train: np.ndarray) -> "Standardizer":
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        x_std = np.where(x_std == 0, 1.0, x_std)
        y_std = 1.0 if y_std == 0 else y_std
        return cls(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            x_low=x_mean - 3.0 * x_std,
            x_high=x_mean + 3.0 * x_std,
            y_low=y_mean - 3.0 * y_std,
            y_high=y_mean + 3.0 * y_std,
        )

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, self.x_low, self.x_high)
        return (clipped - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        clipped = np.clip(y, self.y_low, self.y_high)
        return (clipped - self.y_mean) / self.y_std

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_std + self.y_mean


def fit_transform_splits(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Standardizer, dict[str, tuple[np.ndarray, np.ndarray]]]:
    scaler = Standardizer.fit(x_train, y_train)
    splits = {
        "train": (scaler.transform_x(x_train), scaler.transform_y(y_train)),
        "val": (scaler.transform_x(x_val), scaler.transform_y(y_val)),
        "test": (scaler.transform_x(x_test), scaler.transform_y(y_test)),
    }
    return scaler, splits


def compute_correlation_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pearson = np.asarray([_pearson(x[:, index], y) for index in range(x.shape[1])], dtype=np.float32)
    y_rank = _rankdata(y)
    spearman = np.asarray([_pearson(_rankdata(x[:, index]), y_rank) for index in range(x.shape[1])], dtype=np.float32)
    score = (np.abs(pearson) + np.abs(spearman)) / 2.0
    return ((score - score.min()) / (score.max() - score.min() + 1e-8)).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered))
    return float(np.sum(a_centered * b_centered) / (denominator + 1e-8))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks
