from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BUNDLED_FONT_PATH = _PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"


def _configure_font() -> None:
    if _BUNDLED_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(_BUNDLED_FONT_PATH))
        font_name = font_manager.FontProperties(fname=str(_BUNDLED_FONT_PATH)).get_name()
        matplotlib.rcParams["font.family"] = font_name
        matplotlib.rcParams["font.sans-serif"] = [font_name]
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_font()


def plot_loss_curves(history, output_path, title="损失曲线"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot([item["epoch"] for item in history], [item["train_loss"] for item in history], label="训练")
    plt.plot([item["epoch"] for item in history], [item["val_loss"] for item in history], label="验证")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_prediction_curve(y_true, y_pred, output_path, title="预测曲线"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="真实值")
    plt.plot(y_pred, label="预测值")
    plt.xlabel("样本")
    plt.ylabel("数值")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_scatter(y_true, y_pred, output_path, title="预测散点图"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_residuals(y_true, y_pred, output_path, title="残差图"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 4))
    plt.plot(residuals)
    plt.xlabel("样本")
    plt.ylabel("残差")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_bar_comparison(rows, metric, output_path, title=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["model"] for row in rows]
    values = [row[metric] for row in rows]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
