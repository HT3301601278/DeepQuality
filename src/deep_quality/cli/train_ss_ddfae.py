from __future__ import annotations

import argparse

import torch
from deep_quality.cli.common import save_checkpoint, save_training_outputs
from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.config import apply_overrides, build_checkpoint_name, load_config, parse_scales, resolve_device
from deep_quality.data import prepare_windowed_data
from deep_quality.models import SemiSupervisedDynamicDeepFusionAE, SupervisedDynamicDenoisingAE
from deep_quality.training import SemiSupervisedTrainer, input_dims, make_semisupervised_loader, make_supervised_loader
from deep_quality.utils import regression_metrics, set_seed


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        args.label_ratio,
        args.window_size,
        args.quality_delay,
        args.latent_dim,
        epochs=args.epochs,
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

    train_loader = make_semisupervised_loader(
        train_split["x"],
        train_split["current_u"],
        train_split["y"],
        label_mask,
        batch_size,
        True,
    )
    val_loader = make_supervised_loader(val_split["x"], val_split["current_u"], val_split["y"], batch_size, False)
    test_loader = make_supervised_loader(test_split["x"], test_split["current_u"], test_split["y"], batch_size, False)
    device = resolve_device(config)

    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
    baseline_config = baseline_checkpoint["config"]
    baseline_scales = parse_scales(baseline_config["data"]["scales"])
    if baseline_scales:
        raise ValueError(
            "SS-DDFAE 训练只支持单尺度基线 checkpoint。"
            "请先用 configs/sddae_single_scale.yaml 训练基线模型，再把该 checkpoint 传给 --baseline-checkpoint。"
        )
    baseline_model = SupervisedDynamicDenoisingAE(
        input_dim=input_dims(train_split["x"]),
        latent_dim=int(baseline_config["model"]["latent_dim"]),
        current_u_dim=train_split["current_u"].shape[1],
        noise_std=float(baseline_config["training"]["noise_std"]),
        dropout=float(baseline_config["model"]["dropout"]),
        hidden_dims=baseline_config["model"]["hidden_dims"],
    )
    baseline_model.load_state_dict(baseline_checkpoint["state_dict"])

    model = SemiSupervisedDynamicDeepFusionAE(
        input_dim=input_dims(train_split["x"]),
        latent_dim=int(config["model"]["latent_dim"]),
        current_u_dim=train_split["current_u"].shape[1],
        noise_std=float(training_config["noise_std"]),
        dropout=float(config["model"]["dropout"]),
        hidden_dims=config["model"]["hidden_dims"],
    )
    model.load_from_sddae(baseline_model)

    trainer = SemiSupervisedTrainer(
        model=model,
        device=device,
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        patience=int(training_config["patience"]),
        grad_clip=float(training_config["grad_clip"]),
        noise_std=float(training_config["noise_std"]),
        tau=float(training_config["tau"]),
        ramp_start=int(training_config["ramp_start"]),
        ramp_end=int(training_config["ramp_end"]),
        pseudo_start=int(training_config["pseudo_start"]),
        lambda_rec=float(training_config["lambda_rec"]),
        lambda_sup_fus=float(training_config["lambda_sup_fus"]),
        lambda_sup_aux=float(training_config["lambda_sup_aux"]),
        lambda_con=float(training_config["lambda_con"]),
        lambda_pl=float(training_config["lambda_pl"]),
        reconstruction_weights=[float(value) for value in training_config["reconstruction_weights"]],
        auxiliary_weights=[float(value) for value in training_config["auxiliary_weights"]],
    )

    history = trainer.fit(train_loader, val_loader, int(training_config["epochs"]))
    y_true_scaled, y_pred_scaled = trainer.predict(test_loader)
    y_true = prepared["scaler"].inverse_y(y_true_scaled)
    y_pred = prepared["scaler"].inverse_y(y_pred_scaled)
    metrics = regression_metrics(y_true, y_pred)

    output_name = args.output_name or build_checkpoint_name("ss_ddfae", config)
    checkpoint_path = save_checkpoint(
        config,
        output_name,
        {
            "model": "ss_ddfae",
            "state_dict": model.state_dict(),
            "config": config,
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "metrics": metrics,
        },
    )

    metric_row = {
        "model": "ss_ddfae",
        "label_ratio": float(training_config["label_ratio"]),
        "window_size": int(config["data"]["window_size"]),
        "quality_delay": int(config["data"]["quality_delay"]),
        "latent_dim": int(config["model"]["latent_dim"]),
        **metrics,
    }
    save_training_outputs(config, output_name, metric_row, y_true, y_pred, history, "SS-DDFAE")
    print({"checkpoint": str(checkpoint_path), **metric_row})


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--config", default="configs/ss_ddfae.yaml")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--label-ratio", type=float)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--quality-delay", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-name")
    return parser.parse_args()


if __name__ == "__main__":
    main()
