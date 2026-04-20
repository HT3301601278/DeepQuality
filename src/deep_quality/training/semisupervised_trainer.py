from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class SemiSupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        patience: int,
        grad_clip: float,
        noise_std: float,
        tau: float,
        ramp_start: int,
        ramp_end: int,
        pseudo_start: int,
        lambda_rec: float,
        lambda_sup_fus: float,
        lambda_sup_aux: float,
        lambda_con: float,
        lambda_pl: float,
        reconstruction_weights: list[float],
        auxiliary_weights: list[float],
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.patience = patience
        self.grad_clip = grad_clip
        self.noise_std = noise_std
        self.tau = tau
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        self.pseudo_start = pseudo_start
        self.lambda_rec = lambda_rec
        self.lambda_sup_fus = lambda_sup_fus
        self.lambda_sup_aux = lambda_sup_aux
        self.lambda_con = lambda_con
        self.lambda_pl = lambda_pl
        self.reconstruction_weights = reconstruction_weights
        self.auxiliary_weights = auxiliary_weights

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> list[dict[str, float]]:
        history = []
        best_state = None
        best_rmse = float("inf")
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            train_stats = self._run_train_epoch(train_loader, epoch)
            val_stats = self.evaluate(val_loader)
            history.append({"epoch": epoch, **_prefix(train_stats, "train"), **_prefix(val_stats, "val")})
            if epoch <= self.ramp_start:
                continue
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
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "sse": 0.0}
        count = 0
        for x, current_u, y in loader:
            x = x.to(self.device)
            current_u = current_u.to(self.device)
            y = y.to(self.device)
            output = self.model(x, current_u=current_u, add_noise=False)
            prediction = output["prediction"].view_as(y)
            loss = nn.functional.mse_loss(prediction, y)
            batch_size = len(x)
            totals["loss"] += float(loss.cpu()) * batch_size
            totals["sse"] += float(torch.sum((prediction - y) ** 2).cpu())
            count += batch_size
        return {"loss": totals["loss"] / count, "rmse": (totals["sse"] / count) ** 0.5}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        targets = []
        for x, current_u, y in loader:
            x = x.to(self.device)
            current_u = current_u.to(self.device)
            output = self.model(x, current_u=current_u, add_noise=False)
            predictions.append(output["prediction"].detach().cpu().numpy().reshape(-1))
            targets.append(y.numpy().reshape(-1))
        return np.concatenate(targets), np.concatenate(predictions)

    def _run_train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "rec_loss": 0.0,
            "sup_loss": 0.0,
            "aux_loss": 0.0,
            "con_loss": 0.0,
            "pl_loss": 0.0,
        }
        count = 0
        for x, current_u, y, labeled in loader:
            x = x.to(self.device)
            current_u = current_u.to(self.device)
            y = y.to(self.device)
            labeled = labeled.to(self.device)
            self.optimizer.zero_grad()
            output_1 = self.model(x, current_u=current_u, add_noise=True, noise_std=self.noise_std)
            output_2 = self.model(x, current_u=current_u, add_noise=True, noise_std=self.noise_std)
            reconstruction_loss = self._reconstruction_loss(output_1, x, output_2)
            if epoch <= self.ramp_start:
                supervised_loss = _zero_like(reconstruction_loss)
                auxiliary_loss = _zero_like(reconstruction_loss)
                consistency_loss = _zero_like(reconstruction_loss)
                pseudo_label_loss = _zero_like(reconstruction_loss)
                loss = self.lambda_rec * reconstruction_loss
            else:
                supervised_loss = _masked_mse(output_1["prediction"], y, labeled)
                auxiliary_loss = self._auxiliary_loss(output_1, y, labeled)
                consistency_loss = nn.functional.mse_loss(output_1["prediction"], output_2["prediction"])
                pseudo_label_loss = self._pseudo_label_loss(output_1, output_2, epoch, labeled)
                ramp = _ramp_up(epoch, self.ramp_start, self.ramp_end)
                loss = (
                    self.lambda_rec * reconstruction_loss
                    + self.lambda_sup_fus * supervised_loss
                    + self.lambda_sup_aux * auxiliary_loss
                    + ramp * (self.lambda_con * consistency_loss + self.lambda_pl * pseudo_label_loss)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            batch_size = len(x)
            for name, value in [
                ("loss", loss),
                ("rec_loss", reconstruction_loss),
                ("sup_loss", supervised_loss),
                ("aux_loss", auxiliary_loss),
                ("con_loss", consistency_loss),
                ("pl_loss", pseudo_label_loss),
            ]:
                totals[name] += float(value.detach().cpu()) * batch_size
            count += batch_size
        return {name: value / count for name, value in totals.items()}

    def _reconstruction_loss(self, output_1: dict, x: torch.Tensor, output_2: dict) -> torch.Tensor:
        losses = []
        for weight, rec_1, rec_2 in zip(self.reconstruction_weights, output_1["reconstructions"], output_2["reconstructions"]):
            losses.append(weight * (nn.functional.mse_loss(rec_1, x) + nn.functional.mse_loss(rec_2, x)) / 2.0)
        return sum(losses)

    def _auxiliary_loss(self, output: dict, y: torch.Tensor, labeled: torch.Tensor) -> torch.Tensor:
        losses = []
        for weight, prediction in zip(self.auxiliary_weights, output["branch_predictions"]):
            losses.append(weight * _masked_mse(prediction, y, labeled))
        return sum(losses)

    def _pseudo_label_loss(self, output_1: dict, output_2: dict, epoch: int, labeled: torch.Tensor) -> torch.Tensor:
        if epoch <= self.pseudo_start:
            return _zero_like(output_1["prediction"])
        prediction_1 = output_1["prediction"]
        prediction_2 = output_2["prediction"]
        unlabeled = ~labeled
        diff = torch.abs(prediction_1.detach() - prediction_2.detach())
        confident = unlabeled & (diff < self.tau)
        if not confident.any():
            return _zero_like(prediction_1)
        pseudo = ((prediction_1.detach() + prediction_2.detach()) / 2.0)[confident]
        return nn.functional.mse_loss(prediction_1[confident], pseudo)


def _masked_mse(prediction: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prediction = prediction.view_as(y)
    if not mask.any():
        return _zero_like(prediction)
    return nn.functional.mse_loss(prediction[mask], y[mask])


def _zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def _ramp_up(epoch: int, start: int, end: int) -> float:
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    progress = (epoch - start) / (end - start)
    return math.exp(-5.0 * (1.0 - progress) ** 2)


def _prefix(stats: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}
