from __future__ import annotations

from typing import Sequence

import torch
from deep_quality.models.common_layers import AttentionFusion, MLPRegressorHead
from deep_quality.models.sddae import SupervisedDynamicDenoisingAE
from torch import Tensor, nn
from torch.nn import functional as functional


class SemiSupervisedDynamicDeepFusionAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        current_u_dim: int = 7,
        noise_std: float = 0.03,
        dropout: float = 0.1,
        hidden_dims: Sequence[int] = (64, 32),
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims 必须包含两个值")

        hidden_dim_1, hidden_dim_2 = hidden_dims
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.current_u_dim = current_u_dim
        self.noise_std = noise_std
        self.dropout = nn.Dropout(dropout)

        self.enc1 = nn.Linear(input_dim, hidden_dim_1)
        self.enc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.enc3 = nn.Linear(hidden_dim_2, latent_dim)

        self.dec1_fc1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dec1_fc2 = nn.Linear(hidden_dim_2, input_dim)
        self.dec2_fc1 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.dec2_fc2 = nn.Linear(hidden_dim_1, input_dim)
        self.dec3_fc1 = nn.Linear(latent_dim, hidden_dim_2)
        self.dec3_fc2 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.dec3_fc3 = nn.Linear(hidden_dim_1, input_dim)

        self.reg1 = MLPRegressorHead(hidden_dim_1 + current_u_dim, (32, 16), 1)
        self.reg2 = MLPRegressorHead(hidden_dim_2 + current_u_dim, (32, 16), 1)
        self.reg3 = MLPRegressorHead(latent_dim + current_u_dim, (16, 8), 1)
        self.attention = AttentionFusion([hidden_dim_1, hidden_dim_2, latent_dim], latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_1 = self.dropout(functional.relu(self.enc1(x)))
        hidden_2 = self.dropout(functional.relu(self.enc2(hidden_1)))
        latent = self.dropout(functional.relu(self.enc3(hidden_2)))
        return hidden_1, hidden_2, latent

    def decode_branch1(self, hidden_1: Tensor) -> Tensor:
        return self.dec1_fc2(self.dropout(functional.relu(self.dec1_fc1(hidden_1))))

    def decode_branch2(self, hidden_2: Tensor) -> Tensor:
        return self.dec2_fc2(self.dropout(functional.relu(self.dec2_fc1(hidden_2))))

    def decode_branch3(self, latent: Tensor) -> Tensor:
        hidden_1 = self.dropout(functional.relu(self.dec3_fc1(latent)))
        hidden_2 = self.dropout(functional.relu(self.dec3_fc2(hidden_1)))
        return self.dec3_fc3(hidden_2)

    def load_from_sddae(self, sddae_model: SupervisedDynamicDenoisingAE) -> None:
        _copy_compatible_parameters(self.enc1, sddae_model.enc1)
        _copy_compatible_parameters(self.enc2, sddae_model.enc2)
        _copy_compatible_parameters(self.enc3, sddae_model.enc3)
        _copy_compatible_parameters(self.dec3_fc1, sddae_model.dec1)
        _copy_compatible_parameters(self.dec3_fc2, sddae_model.dec2)
        _copy_compatible_parameters(self.dec3_fc3, sddae_model.dec3)

    def forward(
        self,
        x: Tensor,
        current_u: Tensor,
        add_noise: bool = False,
        noise_std: float | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        noisy_x = x
        if add_noise:
            std = self.noise_std if noise_std is None else noise_std
            noisy_x = x + std * torch.randn_like(x)
        hidden_1, hidden_2, latent = self.encode(noisy_x)
        reconstructions = [
            self.decode_branch1(hidden_1),
            self.decode_branch2(hidden_2),
            self.decode_branch3(latent),
        ]
        branch_predictions = [
            self.reg1(torch.cat([hidden_1, current_u], dim=1)),
            self.reg2(torch.cat([hidden_2, current_u], dim=1)),
            self.reg3(torch.cat([latent, current_u], dim=1)),
        ]
        _, attention = self.attention([hidden_1, hidden_2, latent])
        prediction = sum(attention[:, index : index + 1] * branch_predictions[index] for index in range(len(branch_predictions)))
        return {
            "reconstructions": reconstructions,
            "branch_predictions": branch_predictions,
            "prediction": prediction,
            "attention": attention,
            "features": [hidden_1, hidden_2, latent],
        }


def _copy_compatible_parameters(target: nn.Module, source: nn.Module) -> None:
    target_state = target.state_dict()
    source_state = source.state_dict()
    for name, tensor in target_state.items():
        if name in source_state and source_state[name].shape == tensor.shape:
            target_state[name] = source_state[name].detach().clone()
    target.load_state_dict(target_state)
