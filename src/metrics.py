"""Metrics for EnKF and adaptive-R experiments."""
from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(estimate: np.ndarray, truth: np.ndarray, axis: int = -1) -> np.ndarray:
    """Root mean-square error along ``axis``."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.sqrt(np.mean((estimate - truth) ** 2, axis=axis))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns NaN if either input is effectively constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_assimilation(result, true_r: np.ndarray, method: str, N: int, dt_obs: float, seed: int) -> Dict[str, float | int | str]:
    """Build one summary row for a completed EnKF run."""
    forecast_rmse = np.asarray(result.forecast_rmse, dtype=float)
    analysis_rmse = np.asarray(result.analysis_rmse, dtype=float)
    used_r = np.asarray(result.used_r_values, dtype=float)
    true_r = np.asarray(true_r, dtype=float)
    innov = np.asarray(result.innovations, dtype=float)

    mean_forecast = float(np.nanmean(forecast_rmse))
    mean_analysis = float(np.nanmean(analysis_rmse))
    return {
        "method": method,
        "N": int(N),
        "dt_obs": float(dt_obs),
        "seed": int(seed),
        "mean_forecast_rmse": mean_forecast,
        "mean_analysis_rmse": mean_analysis,
        "median_analysis_rmse": float(np.nanmedian(analysis_rmse)),
        "analysis_improvement": mean_forecast - mean_analysis,
        "r_mae": float(np.nanmean(np.abs(used_r - true_r))),
        "r_corr": safe_corr(used_r, true_r),
        "innovation_mean": float(np.nanmean(innov)),
        "innovation_variance": float(np.nanvar(innov)),
    }
