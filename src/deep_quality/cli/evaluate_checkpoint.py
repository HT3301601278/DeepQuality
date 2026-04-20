from __future__ import annotations

import argparse

from deep_quality.cli.common import save_evaluation_outputs
from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.inference import apply_summary_to_test, checkpoint_path_stem, load_postprocess_summary, load_runtime, \
    predict_split
from deep_quality.utils import regression_metrics


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args.checkpoint, args.device)
    checkpoint = runtime.checkpoint
    config = runtime.config
    prepared = runtime.prepared
    splits = runtime.splits
    model = runtime.model
    device = runtime.device

    raw_train = predict_split(model, splits["train"], config, device, prepared["scaler"])
    raw_val = predict_split(model, splits["val"], config, device, prepared["scaler"])
    raw_test = predict_split(model, splits["test"], config, device, prepared["scaler"])
    summary = load_postprocess_summary(args.checkpoint, args.postprocess_config)
    y_pred, postprocess = apply_summary_to_test(
        summary,
        raw_train["y_pred"],
        raw_val["y_pred"],
        raw_test["y_pred"],
        float(raw_train["y_true"][-1]),
    )

    y_true = raw_test["y_true"]
    metrics = regression_metrics(y_true, y_pred)
    raw_metrics = regression_metrics(y_true, raw_test["y_pred"])
    output_name = args.output_name or checkpoint_path_stem(args.checkpoint)
    paths = config["paths"]
    metric_row = {
        "model": checkpoint["model"],
        "label_ratio": float(config["training"]["label_ratio"]),
        "window_size": int(config["data"]["window_size"]),
        "quality_delay": int(config["data"]["quality_delay"]),
        "latent_dim": int(config["model"]["latent_dim"]),
        "postprocess_method": postprocess["method"],
        "raw_rmse": raw_metrics["rmse"],
        "raw_mae": raw_metrics["mae"],
        "raw_r2": raw_metrics["r2"],
        **metrics,
    }
    save_evaluation_outputs(config, output_name, metric_row, y_true, y_pred)
    print(metric_row)


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-name")
    parser.add_argument("--postprocess-config")
    return parser.parse_args()


if __name__ == "__main__":
    main()
