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
    order: int | None = None
    phis: list[float] | None = None
    ridge: float | None = None


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
    c, phis = fit_ar(residual, [prev_residual], order=1, ridge=0.0)
    return c, phis[0]


def fit_ar(
    residual: np.ndarray,
    previous_residuals: list[float],
    order: int,
    ridge: float = 0.0,
) -> tuple[float, list[float]]:
    if len(residual) == 0:
        return 0.0, [0.0] * order

    history = [float(value) for value in previous_residuals] or [0.0]
    rows = []
    targets = []
    for value in residual:
        rows.append([1.0, *_residual_lags(history, order)])
        targets.append(float(value))
        history.append(float(value))

    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if ridge > 0.0:
        penalty = np.diag([0.0, *([1.0] * order)])
        coef = np.linalg.solve(design.T @ design + ridge * penalty, design.T @ target)
    else:
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coef[0]), [float(value) for value in coef[1:]]


def apply_postprocess(
    method: str,
    raw_train: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    train_true_last: float,
    alpha: float | None = None,
    c: float | None = None,
    phi: float | None = None,
    phis: list[float] | None = None,
    order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return raw_val.astype(np.float64, copy=True), raw_test.astype(np.float64, copy=True)
    if method == "ema":
        combined = ema_filter(np.concatenate([raw_train, raw_val, raw_test]), float(alpha))
        val_end = len(raw_train) + len(raw_val)
        return combined[len(raw_train):val_end], combined[val_end:]
    if method == "ar":
        return _apply_ar(raw_val, raw_test, [train_true_last - raw_train[-1]], float(c), _resolve_phis(phi, phis, order))
    if method == "ema+ar":
        filtered = ema_filter(np.concatenate([raw_train, raw_val, raw_test]), float(alpha))
        val_end = len(raw_train) + len(raw_val)
        return _apply_ar(
            filtered[len(raw_train):val_end],
            filtered[val_end:],
            [train_true_last - filtered[len(raw_train) - 1]],
            float(c),
            _resolve_phis(phi, phis, order),
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
        summary.get("best_phis"),
        summary.get("best_order"),
    )
    return post_test, {
        "method": summary["best_method"],
        "alpha": summary.get("best_alpha"),
        "c": summary.get("best_c"),
        "phi": summary.get("best_phi"),
        "phis": summary.get("best_phis"),
        "order": summary.get("best_order"),
        "ridge": summary.get("best_ridge"),
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
        "phis": summary.get("best_phis"),
        "order": summary.get("best_order"),
        "ridge": summary.get("best_ridge"),
    }
    if method == "ema":
        filtered = ema_filter(np.concatenate([raw_train, raw_val]), float(summary["best_alpha"]))
        state["filtered_prev"] = float(filtered[-1])
        return state
    if method == "ar":
        phis = _resolve_phis(summary.get("best_phi"), summary.get("best_phis"), summary.get("best_order"))
        residual_history = [float(train_true_last - raw_train[-1])]
        for _ in raw_val:
            residual_hat = _predict_ar_residual(float(summary["best_c"]), phis, residual_history)
            residual_history.append(residual_hat)
        state["residual_history"] = residual_history[-len(phis):]
        return state
    if method == "ema+ar":
        filtered = ema_filter(np.concatenate([raw_train, raw_val]), float(summary["best_alpha"]))
        train_end = len(raw_train)
        phis = _resolve_phis(summary.get("best_phi"), summary.get("best_phis"), summary.get("best_order"))
        residual_history = [float(train_true_last - filtered[train_end - 1])]
        for _ in filtered[train_end:]:
            residual_hat = _predict_ar_residual(float(summary["best_c"]), phis, residual_history)
            residual_history.append(residual_hat)
        state["filtered_prev"] = float(filtered[-1])
        state["residual_history"] = residual_history[-len(phis):]
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
        phis = _resolve_phis(state.get("phi"), state.get("phis"), state.get("order"))
        residual_history = state.setdefault("residual_history", [float(state.get("residual_prev", 0.0))])
        residual_hat = _predict_ar_residual(float(state["c"]), phis, residual_history)
        residual_history.append(residual_hat)
        state["residual_history"] = residual_history[-len(phis):]
        return float(raw_pred + residual_hat)
    if method == "ema+ar":
        filtered = float(state["alpha"] * raw_pred + (1.0 - state["alpha"]) * state["filtered_prev"])
        state["filtered_prev"] = filtered
        phis = _resolve_phis(state.get("phi"), state.get("phis"), state.get("order"))
        residual_history = state.setdefault("residual_history", [float(state.get("residual_prev", 0.0))])
        residual_hat = _predict_ar_residual(float(state["c"]), phis, residual_history)
        residual_history.append(residual_hat)
        state["residual_history"] = residual_history[-len(phis):]
        return float(filtered + residual_hat)
    raise ValueError(f"未知后处理方法：{method}")


def evaluate_candidates(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    split_info: dict[str, SplitInfo],
    alphas: tuple[float, ...],
    ar_orders: tuple[int, ...] = (1,),
    ridge_values: tuple[float, ...] = (0.0,),
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

    for order in ar_orders:
        for ridge in ridge_values:
            c, phis = fit_ar(val_true - raw_val, [train_true_last - raw_train[-1]], order=order, ridge=ridge)
            val_pred, test_pred = _apply_ar(raw_val, raw_test, [train_true_last - raw_train[-1]], c, phis)
            candidates.append(
                CandidateResult(
                    method="ar",
                    alpha=None,
                    c=c,
                    phi=phis[0],
                    val_metrics=regression_metrics(val_true, val_pred),
                    test_metrics=regression_metrics(test_true, test_pred),
                    selected=False,
                    order=order,
                    phis=phis,
                    ridge=ridge,
                )
            )

    for alpha in alphas:
        ema_train, ema_val_test = _split_ema(raw_train, raw_val, raw_test, alpha)
        ema_val = ema_val_test[:len(raw_val)]
        ema_test = ema_val_test[len(raw_val):]
        for order in ar_orders:
            for ridge in ridge_values:
                c, phis = fit_ar(val_true - ema_val, [train_true_last - ema_train[-1]], order=order, ridge=ridge)
                val_pred, test_pred = _apply_ar(ema_val, ema_test, [train_true_last - ema_train[-1]], c, phis)
                candidates.append(
                    CandidateResult(
                        method="ema+ar",
                        alpha=alpha,
                        c=c,
                        phi=phis[0],
                        val_metrics=regression_metrics(val_true, val_pred),
                        test_metrics=regression_metrics(test_true, test_pred),
                        selected=False,
                        order=order,
                        phis=phis,
                        ridge=ridge,
                    )
                )

    return candidates


def _apply_ar(
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    previous_residuals: list[float],
    c: float,
    phis: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    residual_history = [float(value) for value in previous_residuals] or [0.0]
    val_corrected = np.array(raw_val, dtype=np.float64, copy=True)
    for index in range(len(val_corrected)):
        residual_hat = _predict_ar_residual(c, phis, residual_history)
        val_corrected[index] += residual_hat
        residual_history.append(residual_hat)
    test_corrected = np.array(raw_test, dtype=np.float64, copy=True)
    for index in range(len(test_corrected)):
        residual_hat = _predict_ar_residual(c, phis, residual_history)
        test_corrected[index] += residual_hat
        residual_history.append(residual_hat)
    return val_corrected, test_corrected


def _resolve_phis(phi: float | None, phis: list[float] | None, order: int | None) -> list[float]:
    if phis:
        return [float(value) for value in phis]
    resolved_order = 1 if order is None else int(order)
    first_phi = 0.0 if phi is None else float(phi)
    return [first_phi, *([0.0] * (resolved_order - 1))]


def _predict_ar_residual(c: float, phis: list[float], residual_history: list[float]) -> float:
    lags = _residual_lags(residual_history, len(phis))
    return float(c + sum(phi * lag for phi, lag in zip(phis, lags)))


def _residual_lags(history: list[float], order: int) -> list[float]:
    if not history:
        return [0.0] * order
    return [float(history[-index]) if len(history) >= index else float(history[0]) for index in range(1, order + 1)]


def _split_ema(raw_train: np.ndarray, raw_val: np.ndarray, raw_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    combined = ema_filter(np.concatenate([raw_train, raw_val, raw_test]), alpha)
    train_end = len(raw_train)
    return combined[:train_end], combined[train_end:]
