from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from deep_quality.training.datasets import move_batch_to_device
from torch import nn
from torch.utils.data import DataLoader


class SupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        patience: int,
        grad_clip: float,
        noise_std: float,
        lambda_rec: float,
        finetune_lambda_rec: float,
        lambda_sup: float,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.patience = patience
        self.grad_clip = grad_clip
        self.noise_std = noise_std
        self.lambda_rec = lambda_rec
        self.finetune_lambda_rec = finetune_lambda_rec
        self.lambda_sup = lambda_sup

    def pretrain(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> list[dict[str, float]]:
        history = []
        for epoch in range(1, epochs + 1):
            train_stats = self._run_epoch(train_loader, train=True, supervised=False)
            val_stats = self._run_epoch(val_loader, train=False, supervised=False)
            history.append({"epoch": epoch, **_prefix(train_stats, "train"), **_prefix(val_stats, "val")})
        return history

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> list[dict[str, float]]:
        history = []
        best_state = None
        best_rmse = float("inf")
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            train_stats = self._run_epoch(
                train_loader,
                train=True,
                supervised=True,
                lambda_rec=self.finetune_lambda_rec,
            )
            val_stats = self._run_epoch(
                val_loader,
                train=False,
                supervised=True,
                lambda_rec=self.finetune_lambda_rec,
            )
            history.append({"epoch": epoch, **_prefix(train_stats, "train"), **_prefix(val_stats, "val")})
            if val_stats["rmse"] < best_rmse:
                best_rmse = val_stats["rmse"]
                best_state = deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        targets = []
        for x, current_u, y in loader:
            x = move_batch_to_device(x, self.device)
            current_u = current_u.to(self.device)
            output = self.model(x, current_u=current_u, add_noise=False)
            predictions.append(output["prediction"].detach().cpu().numpy().reshape(-1))
            targets.append(y.numpy().reshape(-1))
        return np.concatenate(targets), np.concatenate(predictions)

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        supervised: bool,
        lambda_rec: float | None = None,
    ) -> dict[str, float]:
        self.model.train(train)
        reconstruction_weight = self.lambda_rec if lambda_rec is None else lambda_rec
        totals = {"loss": 0.0, "rec_loss": 0.0, "sup_loss": 0.0, "sse": 0.0}
        count = 0
        for x, current_u, y in loader:
            x = move_batch_to_device(x, self.device)
            current_u = current_u.to(self.device)
            y = y.to(self.device)
            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                output = self.model(x, current_u=current_u, add_noise=train, noise_std=self.noise_std)
                reconstruction_loss = self._reconstruction_loss(output["reconstruction"], x)
                prediction = output["prediction"].view_as(y)
                supervised_loss = nn.functional.mse_loss(prediction, y)
                loss = reconstruction_weight * reconstruction_loss
                if supervised:
                    loss = loss + self.lambda_sup * supervised_loss
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
            batch_size = int(y.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["rec_loss"] += float(reconstruction_loss.detach().cpu()) * batch_size
            totals["sup_loss"] += float(supervised_loss.detach().cpu()) * batch_size
            totals["sse"] += float(torch.sum((prediction.detach() - y) ** 2).cpu())
            count += batch_size
        return {
            "loss": totals["loss"] / count,
            "rec_loss": totals["rec_loss"] / count,
            "sup_loss": totals["sup_loss"] / count,
            "rmse": (totals["sse"] / count) ** 0.5,
        }

    def _reconstruction_loss(self, reconstruction, x):
        if isinstance(reconstruction, list):
            losses = [nn.functional.mse_loss(part, target) for part, target in zip(reconstruction, x)]
            return sum(losses) / len(losses)
        return nn.functional.mse_loss(reconstruction, x)


def _prefix(stats: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}
