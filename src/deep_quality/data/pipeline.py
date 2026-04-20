from __future__ import annotations

from deep_quality.config.loader import PROJECT_ROOT, parse_scales
from deep_quality.data.cleaning import clean_missing_values
from deep_quality.data.io import load_csv_dataset
from deep_quality.data.scaling import compute_correlation_weights, fit_transform_splits
from deep_quality.data.split import chronological_split, nested_label_masks
from deep_quality.data.windowing import make_multiscale_windows, make_windows


def prepare_windowed_data(config: dict) -> dict:
    data_config = config["data"]
    train_ratio = float(data_config["train_ratio"])
    val_ratio = float(data_config["val_ratio"])
    window_size = int(data_config["window_size"])
    quality_delay = int(data_config["quality_delay"])
    label_ratio = float(config["training"]["label_ratio"])
    scales = parse_scales(data_config["scales"])

    x, y, feature_names = load_csv_dataset(PROJECT_ROOT / data_config["path"], data_config["target_column"])
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
    weights = compute_correlation_weights(train_u[label_mask], train_y[label_mask])

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
        "splits": splits,
    }
