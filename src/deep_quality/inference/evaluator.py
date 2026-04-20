from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from deep_quality.models import SemiSupervisedDynamicDeepFusionAE, SupervisedDynamicDenoisingAE
from deep_quality.training import make_supervised_loader, move_batch_to_device


def build_model(model_name: str, config: dict, input_dim: int | list[int], current_u_dim: int):
    model_config = config["model"]
    training_config = config["training"]
    if model_name == "sddae_r":
        return SupervisedDynamicDenoisingAE(
            input_dim=input_dim,
            latent_dim=int(model_config["latent_dim"]),
            current_u_dim=current_u_dim,
            noise_std=float(training_config["noise_std"]),
            dropout=float(model_config["dropout"]),
            hidden_dims=model_config["hidden_dims"],
        )
    if model_name == "ss_ddfae":
        return SemiSupervisedDynamicDeepFusionAE(
            input_dim=input_dim,
            latent_dim=int(model_config["latent_dim"]),
            current_u_dim=current_u_dim,
            noise_std=float(training_config["noise_std"]),
            dropout=float(model_config["dropout"]),
            hidden_dims=model_config["hidden_dims"],
        )
    raise ValueError(f"未知模型：{model_name}")


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for x, current_u, y in loader:
        output = model(move_batch_to_device(x, device), current_u=current_u.to(device), add_noise=False)
        predictions.append(output["prediction"].detach().cpu().numpy().reshape(-1))
        targets.append(y.numpy().reshape(-1))
    return np.concatenate(targets), np.concatenate(predictions)


def predict_split(model, split: dict, config: dict, device: torch.device, scaler) -> dict[str, np.ndarray]:
    loader = make_supervised_loader(
        split["x"],
        split["current_u"],
        split["y"],
        int(config["training"]["batch_size"]),
        False,
    )
    y_true_scaled, y_pred_scaled = predict(model, loader, device)
    return {
        "y_true": scaler.inverse_y(y_true_scaled),
        "y_pred": scaler.inverse_y(y_pred_scaled),
    }


def checkpoint_path_stem(path: str | Path) -> str:
    return Path(path).stem
