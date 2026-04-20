from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from deep_quality.config import PROJECT_ROOT
from deep_quality.utils import save_json, save_metrics_csv, save_predictions_csv
from deep_quality.utils.plots import plot_loss_curves, plot_prediction_curve, plot_residuals, plot_scatter


def save_checkpoint(config: dict, output_name: str, payload: dict) -> Path:
    checkpoint_path = _artifact_path(config, "checkpoints_dir", f"{output_name}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def save_training_outputs(
    config: dict,
    output_name: str,
    metric_row: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: list[dict[str, float]],
    title_prefix: str,
) -> None:
    save_json(_artifact_path(config, "tables_dir", f"{output_name}_metrics.json"), metric_row)
    save_metrics_csv(_artifact_path(config, "tables_dir", f"{output_name}_metrics.csv"), [metric_row])
    save_predictions_csv(_artifact_path(config, "tables_dir", f"{output_name}_predictions.csv"), y_true, y_pred)
    plot_loss_curves(history, _artifact_path(config, "figures_dir", f"{output_name}_loss.png"), f"{title_prefix} 损失曲线")
    plot_prediction_curve(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_prediction.png"), f"{title_prefix} 预测曲线")
    plot_scatter(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_scatter.png"), f"{title_prefix} 预测散点图")
    plot_residuals(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_residuals.png"), f"{title_prefix} 残差图")


def save_evaluation_outputs(
    config: dict,
    output_name: str,
    metric_row: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    save_json(_artifact_path(config, "tables_dir", f"{output_name}_eval_metrics.json"), metric_row)
    save_metrics_csv(_artifact_path(config, "tables_dir", f"{output_name}_eval_metrics.csv"), [metric_row])
    save_predictions_csv(_artifact_path(config, "tables_dir", f"{output_name}_eval_predictions.csv"), y_true, y_pred)
    plot_prediction_curve(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_eval_prediction.png"), "预测曲线")
    plot_scatter(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_eval_scatter.png"), "预测散点图")
    plot_residuals(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_eval_residuals.png"), "残差图")


def save_online_outputs(
    config: dict,
    output_name: str,
    metric_row: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latency_ms: np.ndarray,
) -> None:
    save_predictions_csv(_artifact_path(config, "tables_dir", f"{output_name}.csv"), y_true, y_pred, latency_ms)
    save_json(_artifact_path(config, "tables_dir", f"{output_name}_metrics.json"), metric_row)
    plot_prediction_curve(y_true, y_pred, _artifact_path(config, "figures_dir", f"{output_name}_prediction.png"), "在线预测曲线")


def _artifact_path(config: dict, path_key: str, filename: str) -> Path:
    return PROJECT_ROOT / config["paths"][path_key] / filename
