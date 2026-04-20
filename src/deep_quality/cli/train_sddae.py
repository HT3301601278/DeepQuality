from __future__ import annotations

import argparse

import torch
from deep_quality.cli.common import save_checkpoint, save_training_outputs
from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.config import apply_overrides, build_checkpoint_name, build_scale_tag, load_config, resolve_device
from deep_quality.data import prepare_windowed_data
from deep_quality.models import SupervisedDynamicDenoisingAE
from deep_quality.training import SupervisedTrainer, input_dims, make_supervised_loader, subset_windows
from deep_quality.utils import regression_metrics, set_seed


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        args.label_ratio,
        args.window_size,
        args.quality_delay,
        args.latent_dim,
        parse_cli_scales(args.scales),
        args.pretrain_epochs,
        args.finetune_epochs,
    )
    set_seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    prepared = prepare_windowed_data(config)
    splits = prepared["splits"]
    label_mask = prepared["label_mask"]
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])

    pretrain_loader = make_supervised_loader(train_split["x"], train_split["current_u"], train_split["y"], batch_size, True)
    finetune_loader = make_supervised_loader(
        subset_windows(train_split["x"], label_mask),
        train_split["current_u"][label_mask],
        train_split["y"][label_mask],
        batch_size,
        True,
    )
    val_loader = make_supervised_loader(val_split["x"], val_split["current_u"], val_split["y"], batch_size, False)
    test_loader = make_supervised_loader(test_split["x"], test_split["current_u"], test_split["y"], batch_size, False)

    model = SupervisedDynamicDenoisingAE(
        input_dim=input_dims(train_split["x"]),
        latent_dim=int(config["model"]["latent_dim"]),
        current_u_dim=train_split["current_u"].shape[1],
        noise_std=float(training_config["noise_std"]),
        dropout=float(config["model"]["dropout"]),
        hidden_dims=config["model"]["hidden_dims"],
    )
    trainer = SupervisedTrainer(
        model=model,
        device=resolve_device(config),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        patience=int(training_config["patience"]),
        grad_clip=float(training_config["grad_clip"]),
        noise_std=float(training_config["noise_std"]),
        lambda_rec=float(training_config["lambda_rec"]),
        finetune_lambda_rec=float(training_config["finetune_lambda_rec"]),
        lambda_sup=float(training_config["lambda_sup"]),
    )

    history = trainer.pretrain(pretrain_loader, val_loader, int(training_config["pretrain_epochs"]))
    history += trainer.fit(finetune_loader, val_loader, int(training_config["finetune_epochs"]))
    y_true_scaled, y_pred_scaled = trainer.predict(test_loader)
    y_true = prepared["scaler"].inverse_y(y_true_scaled)
    y_pred = prepared["scaler"].inverse_y(y_pred_scaled)
    metrics = regression_metrics(y_true, y_pred)

    output_name = args.output_name or build_checkpoint_name("sddae_r", config)
    checkpoint_path = save_checkpoint(
        config,
        output_name,
        {
            "model": "sddae_r",
            "state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
    )

    metric_row = {
        "model": "sddae_r",
        "label_ratio": float(training_config["label_ratio"]),
        "window_size": max((window_size for window_size, _ in prepared["scales"]), default=int(config["data"]["window_size"])),
        "quality_delay": int(config["data"]["quality_delay"]),
        "latent_dim": int(config["model"]["latent_dim"]),
        "scales": build_scale_tag(prepared["scales"]),
        **metrics,
    }
    save_training_outputs(config, output_name, metric_row, y_true, y_pred, history, "SDDAE-R")
    print({"checkpoint": str(checkpoint_path), **metric_row})


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--config", default="configs/sddae_single_scale.yaml")
    parser.add_argument("--label-ratio", type=float)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--quality-delay", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--scales")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--output-name")
    return parser.parse_args()


def parse_cli_scales(value: str | None) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if value == "":
        return []
    scales = []
    for item in value.split(","):
        window_size, stride = item.strip().lower().split("x", 1)
        scales.append((int(window_size), int(stride)))
    return scales


if __name__ == "__main__":
    main()
