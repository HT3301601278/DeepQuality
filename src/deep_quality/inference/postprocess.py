from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from deep_quality.config import PROJECT_ROOT
from deep_quality.utils import regression_metrics


@dataclass(frozen=True)
class SplitInfo:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class CandidateResult:
    method: str
    alpha: float | None
    c: float | None
    phi: float | None
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
    selected: bool


def checkpoint_postprocess_paths(checkpoint_path: str | Path) -> tuple[Path, Path]:
    resolved = Path(checkpoint_path).expanduser().resolve()
    stem = resolved.stem
    tables_dir = PROJECT_ROOT / "artifacts" / "tables"
    return tables_dir / f"{stem}_postprocess.csv", tables_dir / f"{stem}_postprocess.json"


def load_postprocess_summary(checkpoint_path: str | Path, summary_path: str | None = None) -> dict:
    resolved = Path(checkpoint_path).expanduser().resolve()
    path = Path(summary_path).expanduser().resolve() if summary_path else checkpoint_postprocess_paths(resolved)[1]
    if not path.exists():
        raise FileNotFoundError(f"未找到后处理摘要：{path}")
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if "checkpoint" not in summary:
        raise ValueError("后处理摘要缺少 checkpoint 字段")
    summary_checkpoint = Path(summary["checkpoint"]).expanduser().resolve()
    if summary_checkpoint != resolved:
        raise ValueError("后处理摘要与当前 checkpoint 不匹配")
    return summary


def build_split_info(splits: dict, order: tuple[str, ...]) -> dict[str, SplitInfo]:
    cursor = 0
    info = {}
    for name in order:
        end = cursor + len(splits[name]["y"])
        info[name] = SplitInfo(name=name, start=cursor, end=end)
        cursor = end
    return info


def ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(values, dtype=np.float64)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def fit_ar1(residual: np.ndarray, prev_residual: float) -> tuple[float, float]:
    if len(residual) == 0:
        return 0.0, 0.0
    x = np.concatenate(([prev_residual], residual[:-1]))
    y = residual
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_postprocess(
    method: str,
    raw_train: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    train_true_last: float,
    alpha: float | None = None,
    c: float | None = None,
    phi: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return raw_val.astype(np.float64, copy=True), raw_test.astype(np.float64, copy=True)
    if method == "ema":
        combined = ema_filter(np.concatenate([raw_train, raw_val, raw_test]), float(alpha))
        val_end = len(raw_train) + len(raw_val)
        return combined[len(raw_train):val_end], combined[val_end:]
    if method == "ar":
        return _apply_ar(raw_train, raw_val, raw_test, train_true_last, float(c), float(phi))
    if method == "ema+ar":
        filtered = ema_filter(np.concatenate([raw_train, raw_val, raw_test]), float(alpha))
        val_end = len(raw_train) + len(raw_val)
        return _apply_ar(
            filtered[:len(raw_train)],
            filtered[len(raw_train):val_end],
            filtered[val_end:],
            train_true_last,
            float(c),
            float(phi),
        )
    raise ValueError(f"未知后处理方法：{method}")


def apply_summary_to_test(
    summary: dict,
    raw_train: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    train_true_last: float,
) -> tuple[np.ndarray, dict]:
    _, post_test = apply_postprocess(
        summary["best_method"],
        raw_train,
        raw_val,
        raw_test,
        train_true_last,
        summary.get("best_alpha"),
        summary.get("best_c"),
        summary.get("best_phi"),
    )
    return post_test, {
        "method": summary["best_method"],
        "alpha": summary.get("best_alpha"),
        "c": summary.get("best_c"),
        "phi": summary.get("best_phi"),
    }


def build_postprocess_state(summary: dict, raw_train: np.ndarray, raw_val: np.ndarray, train_true_last: float) -> dict:
    if summary["best_method"] == "raw":
        return {"method": "raw"}
    method = summary["best_method"]
    state = {
        "method": method,
        "alpha": summary.get("best_alpha"),
        "c": summary.get("best_c"),
        "phi": summary.get("best_phi"),
    }
    if method == "ema":
        filtered = ema_filter(np.concatenate([raw_train, raw_val]), float(summary["best_alpha"]))
        state["filtered_prev"] = float(filtered[-1])
        return state
    if method == "ar":
        prev_residual = float(train_true_last - raw_train[-1])
        for raw_value in raw_val:
            residual_hat = float(summary["best_c"] + summary["best_phi"] * prev_residual)
            prev_residual = residual_hat
        state["residual_prev"] = prev_residual
        return state
    if method == "ema+ar":
        filtered = ema_filter(np.concatenate([raw_train, raw_val]), float(summary["best_alpha"]))
        train_end = len(raw_train)
        prev_residual = float(train_true_last - filtered[train_end - 1])
        for filtered_value in filtered[train_end:]:
            residual_hat = float(summary["best_c"] + summary["best_phi"] * prev_residual)
            prev_residual = residual_hat
        state["filtered_prev"] = float(filtered[-1])
        state["residual_prev"] = prev_residual
        return state
    raise ValueError(f"未知后处理方法：{method}")


def apply_postprocess_step(raw_pred: float, state: dict) -> float:
    method = state["method"]
    if method == "raw":
        return float(raw_pred)
    if method == "ema":
        filtered = float(state["alpha"] * raw_pred + (1.0 - state["alpha"]) * state["filtered_prev"])
        state["filtered_prev"] = filtered
        return filtered
    if method == "ar":
        residual_hat = float(state["c"] + state["phi"] * state["residual_prev"])
        state["residual_prev"] = residual_hat
        return float(raw_pred + residual_hat)
    if method == "ema+ar":
        filtered = float(state["alpha"] * raw_pred + (1.0 - state["alpha"]) * state["filtered_prev"])
        state["filtered_prev"] = filtered
        residual_hat = float(state["c"] + state["phi"] * state["residual_prev"])
        state["residual_prev"] = residual_hat
        return float(filtered + residual_hat)
    raise ValueError(f"未知后处理方法：{method}")


def evaluate_candidates(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    split_info: dict[str, SplitInfo],
    alphas: tuple[float, ...],
) -> list[CandidateResult]:
    train = split_info["train"]
    val = split_info["val"]
    test = split_info["test"]
    raw_train = y_pred_all[train.start:train.end]
    raw_val = y_pred_all[val.start:val.end]
    raw_test = y_pred_all[test.start:test.end]
    train_true_last = float(y_true_all[train.end - 1])
    val_true = y_true_all[val.start:val.end]
    test_true = y_true_all[test.start:test.end]

    candidates = [
        CandidateResult(
            method="raw",
            alpha=None,
            c=None,
            phi=None,
            val_metrics=regression_metrics(val_true, raw_val),
            test_metrics=regression_metrics(test_true, raw_test),
            selected=False,
        )
    ]

    for alpha in alphas:
        val_pred, test_pred = apply_postprocess("ema", raw_train, raw_val, raw_test, train_true_last, alpha=alpha)
        candidates.append(
            CandidateResult(
                method="ema",
                alpha=alpha,
                c=None,
                phi=None,
                val_metrics=regression_metrics(val_true, val_pred),
                test_metrics=regression_metrics(test_true, test_pred),
                selected=False,
            )
        )

    c, phi = fit_ar1(val_true - raw_val, train_true_last - raw_train[-1])
    val_pred, test_pred = _apply_ar(raw_train, raw_val, raw_test, train_true_last, c, phi)
    candidates.append(
        CandidateResult(
            method="ar",
            alpha=None,
            c=c,
            phi=phi,
            val_metrics=regression_metrics(val_true, val_pred),
            test_metrics=regression_metrics(test_true, test_pred),
            selected=False,
        )
    )

    for alpha in alphas:
        ema_train, ema_val_test = _split_ema(raw_train, raw_val, raw_test, alpha)
        ema_val = ema_val_test[:len(raw_val)]
        ema_test = ema_val_test[len(raw_val):]
        c, phi = fit_ar1(val_true - ema_val, train_true_last - ema_train[-1])
        val_pred, test_pred = _apply_ar(ema_train, ema_val, ema_test, train_true_last, c, phi)
        candidates.append(
            CandidateResult(
                method="ema+ar",
                alpha=alpha,
                c=c,
                phi=phi,
                val_metrics=regression_metrics(val_true, val_pred),
                test_metrics=regression_metrics(test_true, test_pred),
                selected=False,
            )
        )

    return candidates


def _apply_ar(
    raw_train: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    train_true_last: float,
    c: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    prev_residual = train_true_last - raw_train[-1]
    val_corrected = np.array(raw_val, dtype=np.float64, copy=True)
    for index in range(len(val_corrected)):
        residual_hat = c + phi * prev_residual
        val_corrected[index] += residual_hat
        prev_residual = residual_hat
    test_corrected = np.array(raw_test, dtype=np.float64, copy=True)
    for index in range(len(test_corrected)):
        residual_hat = c + phi * prev_residual
        test_corrected[index] += residual_hat
        prev_residual = residual_hat
    return val_corrected, test_corrected


def _split_ema(raw_train: np.ndarray, raw_val: np.ndarray, raw_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    combined = ema_filter(np.concatenate([raw_train, raw_val, raw_test]), alpha)
    train_end = len(raw_train)
    return combined[:train_end], combined[train_end:]
