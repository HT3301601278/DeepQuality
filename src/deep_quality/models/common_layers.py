from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class MLPRegressorHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class AttentionFusion(nn.Module):
    def __init__(self, feature_dims: Sequence[int], fusion_dim: int | None = None) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims 不能为空")
        if fusion_dim is None:
            fusion_dim = feature_dims[-1]
        self.projections = nn.ModuleList(nn.Linear(dim, fusion_dim) for dim in feature_dims)
        self.scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, fusion_dim), nn.Tanh(), nn.Linear(fusion_dim, 1))
            for dim in feature_dims
        )

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("features 和 feature_dims 的长度必须一致")
        scores = [scorer(feature) for scorer, feature in zip(self.scorers, features)]
        attention = torch.softmax(torch.cat(scores, dim=1), dim=1)
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        fused = sum(attention[:, index : index + 1] * projected[index] for index in range(len(projected)))
        return fused, attention
