from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SupervisedWindowDataset(Dataset):
    def __init__(self, x: np.ndarray | list[np.ndarray], current_u: np.ndarray, y: np.ndarray):
        if isinstance(x, list):
            self.x = [torch.as_tensor(part, dtype=torch.float32) for part in x]
        else:
            self.x = torch.as_tensor(x, dtype=torch.float32)
        self.current_u = torch.as_tensor(current_u, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        if isinstance(self.x, list):
            return len(self.x[0])
        return len(self.x)

    def __getitem__(self, index: int):
        if isinstance(self.x, list):
            return [part[index] for part in self.x], self.current_u[index], self.y[index]
        return self.x[index], self.current_u[index], self.y[index]


class SemiSupervisedWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, current_u: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.current_u = torch.as_tensor(current_u, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        self.labeled = torch.as_tensor(labeled_mask, dtype=torch.bool).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.current_u[index], self.y[index], self.labeled[index]


def make_supervised_loader(
    x: np.ndarray | list[np.ndarray],
    current_u: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SupervisedWindowDataset(x, current_u, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_semisupervised_loader(
    x: np.ndarray,
    current_u: np.ndarray,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SemiSupervisedWindowDataset(x, current_u, y, labeled_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def move_batch_to_device(x, device):
    if isinstance(x, list):
        return [part.to(device) for part in x]
    return x.to(device)


def input_dims(x) -> int | list[int]:
    if isinstance(x, list):
        return [int(part.shape[1]) for part in x]
    return int(x.shape[1])


def subset_windows(x, mask):
    if isinstance(x, list):
        return [part[mask] for part in x]
    return x[mask]


def row_to_tensor(x, index: int, device: torch.device):
    if isinstance(x, list):
        return [torch.as_tensor(part[index], dtype=torch.float32, device=device).view(1, -1) for part in x]
    return torch.as_tensor(x[index], dtype=torch.float32, device=device).view(1, -1)
