from __future__ import annotations

from deep_quality.config.loader import parse_scales


def build_checkpoint_name(model_name: str, config: dict, suffix: str | None = None) -> str:
    label_ratio = float(config["training"]["label_ratio"])
    quality_delay = int(config["data"]["quality_delay"])
    latent_dim = int(config["model"]["latent_dim"])
    scales = parse_scales(config["data"]["scales"])
    if scales:
        name = f"{model_name}_ms{build_scale_tag(scales)}_d{quality_delay}_z{latent_dim}_r{label_ratio:g}"
    else:
        window_size = int(config["data"]["window_size"])
        name = f"{model_name}_L{window_size}_d{quality_delay}_z{latent_dim}_r{label_ratio:g}"
    if suffix:
        return f"{name}_{suffix}"
    return name


def build_scale_tag(scales: list[tuple[int, int]] | None) -> str | None:
    if not scales:
        return None
    return "-".join(f"{window_size}x{stride}" for window_size, stride in scales)
