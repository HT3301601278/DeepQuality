from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from deep_quality.data import prepare_windowed_data
from deep_quality.inference.evaluator import build_model, predict
from deep_quality.training import input_dims, make_supervised_loader


@dataclass
class InferenceRuntime:
    checkpoint_path: Path
    checkpoint: dict
    config: dict
    prepared: dict
    splits: dict
    model: torch.nn.Module
    device: torch.device


def load_runtime(checkpoint_path: str | Path, device: str = "cpu") -> InferenceRuntime:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    runtime_device = torch.device(device)
    checkpoint = torch.load(resolved_path, map_location="cpu")
    config = checkpoint["config"]
    prepared = prepare_windowed_data(config)
    splits = prepared["splits"]
    test_split = splits["test"]
    model = build_model(checkpoint["model"], config, input_dims(test_split["x"]), test_split["current_u"].shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(runtime_device)
    return InferenceRuntime(
        checkpoint_path=resolved_path,
        checkpoint=checkpoint,
        config=config,
        prepared=prepared,
        splits=splits,
        model=model,
        device=runtime_device,
    )


def collect_sequences(
    model,
    splits: dict,
    config: dict,
    scaler,
    device: torch.device,
    order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(config["training"]["batch_size"])
    y_true_all = []
    y_pred_all = []
    for name in order:
        loader = make_supervised_loader(
            splits[name]["x"],
            splits[name]["current_u"],
            splits[name]["y"],
            batch_size,
            False,
        )
        y_true_scaled, y_pred_scaled = predict(model, loader, device)
        y_true_all.append(scaler.inverse_y(y_true_scaled))
        y_pred_all.append(scaler.inverse_y(y_pred_scaled))
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)
