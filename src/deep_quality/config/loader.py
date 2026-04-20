from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: str) -> dict:
    base = _read_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    specific_path = Path(config_path)
    if not specific_path.is_absolute():
        specific_path = PROJECT_ROOT / config_path
    return _deep_update(base, _read_yaml(specific_path))


def apply_overrides(
    config: dict,
    label_ratio: float | None = None,
    window_size: int | None = None,
    quality_delay: int | None = None,
    latent_dim: int | None = None,
    scales: list[tuple[int, int]] | None = None,
    pretrain_epochs: int | None = None,
    finetune_epochs: int | None = None,
    epochs: int | None = None,
) -> dict:
    if label_ratio is not None:
        config["training"]["label_ratio"] = float(label_ratio)
    if window_size is not None:
        config["data"]["window_size"] = int(window_size)
    if quality_delay is not None:
        config["data"]["quality_delay"] = int(quality_delay)
    if latent_dim is not None:
        config["model"]["latent_dim"] = int(latent_dim)
    if scales is not None:
        config["data"]["scales"] = [[int(size), int(stride)] for size, stride in scales]
    if pretrain_epochs is not None:
        config["training"]["pretrain_epochs"] = int(pretrain_epochs)
    if finetune_epochs is not None:
        config["training"]["finetune_epochs"] = int(finetune_epochs)
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    return config


def resolve_device(config: dict):
    import torch

    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_scales(value) -> list[tuple[int, int]]:
    if not value:
        return []
    scales: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, str):
            size, stride = item.lower().split("x", 1)
            scales.append((int(size), int(stride)))
        else:
            size, stride = item
            scales.append((int(size), int(stride)))
    return scales


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
