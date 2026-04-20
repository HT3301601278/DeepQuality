from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from deep_quality.cli.common import save_online_outputs
from deep_quality.cli.parsing import ChineseArgumentParser
from deep_quality.inference import (
    apply_postprocess_step,
    build_postprocess_state,
    checkpoint_path_stem,
    load_postprocess_summary,
    load_runtime,
    predict_split,
)
from deep_quality.training import row_to_tensor
from deep_quality.utils import regression_metrics


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args.checkpoint, args.device)
    config = runtime.config
    prepared = runtime.prepared
    splits = runtime.splits
    test_split = splits["test"]
    device = runtime.device
    model = runtime.model
    model.eval()

    raw_train = predict_split(model, splits["train"], config, device, prepared["scaler"])
    raw_val = predict_split(model, splits["val"], config, device, prepared["scaler"])
    summary = load_postprocess_summary(args.checkpoint, args.postprocess_config)
    postprocess_state = build_postprocess_state(
        summary,
        raw_train["y_pred"],
        raw_val["y_pred"],
        float(raw_train["y_true"][-1]),
    )

    predictions = []
    latencies = []
    with torch.no_grad():
        for index, current_u_row in enumerate(test_split["current_u"]):
            x = row_to_tensor(test_split["x"], index, device)
            current_u = torch.as_tensor(current_u_row, dtype=torch.float32, device=device).view(1, -1)
            started = time.perf_counter()
            prediction = model(x, current_u=current_u, add_noise=False)["prediction"]
            prediction_value = float(prediction.detach().cpu().reshape(-1)[0])
            prediction_value = float(prepared["scaler"].inverse_y(np.asarray([prediction_value], dtype=np.float32))[0])
            prediction_value = apply_postprocess_step(prediction_value, postprocess_state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            predictions.append(prediction_value)

    y_pred = np.asarray(predictions, dtype=np.float32)
    y_true = prepared["scaler"].inverse_y(test_split["y"])
    latency_ms = np.asarray(latencies, dtype=np.float32)
    metric_row = regression_metrics(y_true, y_pred)

    output_name = args.output_name or f"{checkpoint_path_stem(args.checkpoint)}_online"
    result = {
        **metric_row,
        "avg_latency_ms": float(latency_ms.mean()),
        "postprocess_method": summary["best_method"],
    }
    save_online_outputs(config, output_name, result, y_true, y_pred, latency_ms)
    print(result)


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-name")
    parser.add_argument("--postprocess-config")
    return parser.parse_args()


if __name__ == "__main__":
    main()
