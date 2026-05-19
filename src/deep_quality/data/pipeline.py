from __future__ import annotations

import numpy as np

from deep_quality.config.loader import PROJECT_ROOT, parse_scales
from deep_quality.data.cleaning import clean_missing_values
from deep_quality.data.io import load_csv_dataset
from deep_quality.data.scaling import Standardizer, compute_correlation_weights, fit_transform_splits
from deep_quality.data.split import chronological_split, nested_label_masks, split_indices
from deep_quality.data.windowing import make_multiscale_windows, make_windows


def prepare_windowed_data(config: dict) -> dict:
    data_config = config["data"]
    train_ratio = float(data_config["train_ratio"])
    val_ratio = float(data_config["val_ratio"])
    window_size = int(data_config["window_size"])
    quality_delay = int(data_config["quality_delay"])
    label_ratio = float(config["training"]["label_ratio"])
    scales = parse_scales(data_config["scales"])
    split_method = data_config.get("split_method", "chronological")

    x, y, feature_names = load_csv_dataset(PROJECT_ROOT / data_config["path"], data_config["target_column"])
    if split_method == "random_window":
        return _prepare_random_windowed_data(config, x, y, feature_names)
    if split_method != "chronological":
        raise ValueError(f"未知数据划分方法：{split_method}")

    raw_splits = chronological_split(x, y, train_ratio, val_ratio)
    cleaned_splits = {name: clean_missing_values(*split) for name, split in raw_splits.items()}

    scaler, scaled_splits = fit_transform_splits(
        cleaned_splits["train"][0],
        cleaned_splits["train"][1],
        cleaned_splits["val"][0],
        cleaned_splits["val"][1],
        cleaned_splits["test"][0],
        cleaned_splits["test"][1],
    )

    if scales:
        train_x, train_u, train_y = make_multiscale_windows(
            scaled_splits["train"][0],
            scaled_splits["train"][1],
            scales,
            quality_delay=quality_delay,
        )
    else:
        train_x, train_u, train_y = make_windows(
            scaled_splits["train"][0],
            scaled_splits["train"][1],
            window_size,
            quality_delay=quality_delay,
        )

    ratios = sorted(set(float(ratio) for ratio in data_config["label_ratios"]) | {label_ratio})
    label_masks = nested_label_masks(len(train_y), ratios, int(config["seed"]))
    label_mask = label_masks[label_ratio]
    weights = _correlation_weights(data_config, train_u[label_mask], train_y[label_mask])

    splits = {}
    for split_name, (x_split, y_split) in scaled_splits.items():
        if scales:
            window_x, current_u, window_y = make_multiscale_windows(
                x_split,
                y_split,
                scales,
                weights=weights,
                quality_delay=quality_delay,
            )
        else:
            window_x, current_u, window_y = make_windows(
                x_split,
                y_split,
                window_size,
                weights=weights,
                quality_delay=quality_delay,
            )
        splits[split_name] = {
            "x": window_x,
            "current_u": current_u,
            "y": window_y,
        }

    return {
        "feature_names": feature_names,
        "label_mask": label_mask,
        "scaler": scaler,
        "scales": scales,
        "split_method": split_method,
        "splits": splits,
    }


def _prepare_random_windowed_data(
    config: dict,
    x,
    y,
    feature_names: list[str],
) -> dict:
    data_config = config["data"]
    train_ratio = float(data_config["train_ratio"])
    val_ratio = float(data_config["val_ratio"])
    window_size = int(data_config["window_size"])
    quality_delay = int(data_config["quality_delay"])
    label_ratio = float(config["training"]["label_ratio"])
    scales = parse_scales(data_config["scales"])

    clean_x, clean_y = clean_missing_values(x, y)
    raw_window_x, raw_current_u, raw_window_y = _make_window_data(
        clean_x,
        clean_y,
        window_size,
        scales,
        quality_delay,
    )
    indices = split_indices(
        len(raw_window_y),
        train_ratio,
        val_ratio,
        "random_window",
        int(config["seed"]),
    )
    scaler = Standardizer.fit(
        _window_source_rows(raw_window_x, raw_current_u, clean_x.shape[1], indices["train"]),
        raw_window_y[indices["train"]],
    )
    scaled_x = scaler.transform_x(clean_x)
    scaled_y = scaler.transform_y(clean_y)

    _, train_u, train_y = _subset_window_data(
        *_make_window_data(scaled_x, scaled_y, window_size, scales, quality_delay),
        indices["train"],
    )
    ratios = sorted(set(float(ratio) for ratio in data_config["label_ratios"]) | {label_ratio})
    label_masks = nested_label_masks(len(train_y), ratios, int(config["seed"]))
    label_mask = label_masks[label_ratio]
    weights = _correlation_weights(data_config, train_u[label_mask], train_y[label_mask])

    weighted_x, current_u, window_y = _make_window_data(
        scaled_x,
        scaled_y,
        window_size,
        scales,
        quality_delay,
        weights,
    )
    splits = {}
    for split_name, split_index in indices.items():
        window_x, split_current_u, split_y = _subset_window_data(weighted_x, current_u, window_y, split_index)
        splits[split_name] = {
            "x": window_x,
            "current_u": split_current_u,
            "y": split_y,
        }

    return {
        "feature_names": feature_names,
        "label_mask": label_mask,
        "scaler": scaler,
        "scales": scales,
        "split_method": "random_window",
        "splits": splits,
    }


def _make_window_data(x, y, window_size: int, scales: list[tuple[int, int]], quality_delay: int, weights=None):
    if scales:
        return make_multiscale_windows(x, y, scales, weights=weights, quality_delay=quality_delay)
    return make_windows(x, y, window_size, weights=weights, quality_delay=quality_delay)


def _correlation_weights(data_config: dict, current_u, y):
    mode = data_config.get("correlation_weight_mode", "original")
    if mode == "none":
        return None
    if mode == "original":
        return compute_correlation_weights(current_u, y)
    raise ValueError(f"未知相关性加权方式：{mode}")


def _subset_window_data(window_x, current_u, window_y, indices):
    if isinstance(window_x, list):
        subset_x = [part[indices] for part in window_x]
    else:
        subset_x = window_x[indices]
    return subset_x, current_u[indices], window_y[indices]


def _window_source_rows(window_x, current_u, feature_count: int, indices):
    if isinstance(window_x, list):
        rows = [part[indices].reshape(-1, feature_count) for part in window_x]
    else:
        rows = [window_x[indices].reshape(-1, feature_count)]
    rows.append(current_u[indices])
    return np.concatenate(rows, axis=0)
