from __future__ import annotations

from typing import Any, Sequence

import torch
from deep_quality.models.common_layers import MLPRegressorHead
from torch import Tensor, nn
from torch.nn import functional as functional


class SupervisedDynamicDenoisingAE(nn.Module):
    def __init__(
        self,
        input_dim: int | Sequence[int],
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
        self.multiscale = not isinstance(input_dim, int)
        self.input_dims = [int(dim) for dim in input_dim] if self.multiscale else [int(input_dim)]
        self.input_dim = sum(self.input_dims) if self.multiscale else self.input_dims[0]
        self.latent_dim = latent_dim
        self.current_u_dim = current_u_dim
        self.noise_std = noise_std
        self.dropout = nn.Dropout(dropout)

        if self.multiscale:
            self.branch_latent_dim = hidden_dim_2
            self.encoders = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "enc1": nn.Linear(dim, hidden_dim_1),
                            "enc2": nn.Linear(hidden_dim_1, hidden_dim_2),
                            "enc3": nn.Linear(hidden_dim_2, self.branch_latent_dim),
                        }
                    )
                    for dim in self.input_dims
                ]
            )
            self.shared_latent = nn.Linear(self.branch_latent_dim * len(self.input_dims), latent_dim)
            self.decoders = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "dec1": nn.Linear(latent_dim, hidden_dim_2),
                            "dec2": nn.Linear(hidden_dim_2, hidden_dim_1),
                            "dec3": nn.Linear(hidden_dim_1, dim),
                        }
                    )
                    for dim in self.input_dims
                ]
            )
            regressor_input_dim = latent_dim + self.branch_latent_dim * len(self.input_dims) + current_u_dim
        else:
            self.enc1 = nn.Linear(self.input_dims[0], hidden_dim_1)
            self.enc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
            self.enc3 = nn.Linear(hidden_dim_2, latent_dim)
            self.dec1 = nn.Linear(latent_dim, hidden_dim_2)
            self.dec2 = nn.Linear(hidden_dim_2, hidden_dim_1)
            self.dec3 = nn.Linear(hidden_dim_1, self.input_dims[0])
            regressor_input_dim = latent_dim + current_u_dim

        self.regressor = MLPRegressorHead(regressor_input_dim, (32, 16), 1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_1 = self.dropout(functional.relu(self.enc1(x)))
        hidden_2 = self.dropout(functional.relu(self.enc2(hidden_1)))
        latent = functional.relu(self.enc3(hidden_2))
        return hidden_1, hidden_2, latent

    def decode(self, latent: Tensor) -> Tensor:
        hidden_1 = self.dropout(functional.relu(self.dec1(latent)))
        hidden_2 = self.dropout(functional.relu(self.dec2(hidden_1)))
        return self.dec3(hidden_2)

    def forward(
        self,
        x: Tensor | Sequence[Tensor],
        current_u: Tensor,
        add_noise: bool = False,
        noise_std: float | None = None,
    ) -> dict[str, Any]:
        if self.multiscale:
            x_parts = _ensure_tensor_sequence(x)
            noisy_parts = _apply_noise(x_parts, add_noise, self.noise_std if noise_std is None else noise_std)
            branch_latents = []
            reconstructions = []
            for part, encoder, decoder in zip(noisy_parts, self.encoders, self.decoders):
                hidden_1 = self.dropout(functional.relu(encoder["enc1"](part)))
                hidden_2 = self.dropout(functional.relu(encoder["enc2"](hidden_1)))
                branch_latent = functional.relu(encoder["enc3"](hidden_2))
                branch_latents.append(branch_latent)
            fused = torch.cat(branch_latents, dim=1)
            latent = functional.relu(self.shared_latent(fused))
            for decoder in self.decoders:
                hidden_1 = self.dropout(functional.relu(decoder["dec1"](latent)))
                hidden_2 = self.dropout(functional.relu(decoder["dec2"](hidden_1)))
                reconstructions.append(decoder["dec3"](hidden_2))
            reconstruction = reconstructions
            prediction = self.regressor(torch.cat([latent, fused, current_u], dim=1))
        else:
            if not isinstance(x, torch.Tensor):
                raise TypeError("单尺度输入必须是 torch.Tensor")
            noisy_x = x
            if add_noise:
                std = self.noise_std if noise_std is None else noise_std
                noisy_x = x + std * torch.randn_like(x)
            _, _, latent = self.encode(noisy_x)
            reconstruction = self.decode(latent)
            prediction = self.regressor(torch.cat([latent, current_u], dim=1))
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "latent": latent,
        }


def _ensure_tensor_sequence(x: Tensor | Sequence[Tensor]) -> list[Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    return list(x)


def _apply_noise(x_parts: list[Tensor], add_noise: bool, noise_std: float) -> list[Tensor]:
    if not add_noise:
        return x_parts
    return [part + noise_std * torch.randn_like(part) for part in x_parts]

