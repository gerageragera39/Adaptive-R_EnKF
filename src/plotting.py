"""Plot helpers for experiments and Streamlit."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rmse_comparison(results: Mapping[str, object], path: Optional[str | Path] = None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method, result in results.items():
        ax.plot(result.obs_times, result.analysis_rmse, label=f"{method} analysis", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Analysis RMSE")
    ax.set_title("EnKF analysis RMSE comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_forecast_analysis_rmse(result, method: str, path: Optional[str | Path] = None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(result.obs_times, result.forecast_rmse, label="forecast/background", alpha=0.85)
    ax.plot(result.obs_times, result.analysis_rmse, label="analysis", alpha=0.9)
    ax.set_xlabel("Time")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Forecast vs analysis RMSE ({method})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_r_tracking(obs_times: np.ndarray, true_r: np.ndarray, results: Mapping[str, object], path: Optional[str | Path] = None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(obs_times, true_r, label="true r(t)", color="black", linewidth=2.5)
    for method, result in results.items():
        ax.plot(obs_times, result.used_r_values, label=f"{method} used r", alpha=0.85)
    ax.set_xlabel("Time")
    ax.set_ylabel("Scalar observation-error variance r")
    ax.set_title("True vs used / estimated observation error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def _combo_labels(combos) -> list[str]:
    return [f"N={N}, dt={dt_obs:g}" for N, dt_obs in combos]


def plot_grid_bar(summary: pd.DataFrame, path: Optional[str | Path] = None):
    """Bar plot for raw grid rows, averaged if repeated seeds are present."""
    if summary.empty:
        raise ValueError("summary dataframe is empty")
    grouped = summary.groupby(["N", "dt_obs", "method"], as_index=False)["mean_analysis_rmse"].mean()
    combos = grouped[["N", "dt_obs"]].drop_duplicates().to_records(index=False)
    methods = list(grouped["method"].unique())
    x = np.arange(len(combos))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for j, method in enumerate(methods):
        vals = []
        for N, dt_obs in combos:
            m = grouped[(grouped["N"] == N) & (grouped["dt_obs"] == dt_obs) & (grouped["method"] == method)]
            vals.append(float(m["mean_analysis_rmse"].iloc[0]) if not m.empty else np.nan)
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(_combo_labels(combos), rotation=35, ha="right")
    ax.set_ylabel("Mean analysis RMSE")
    ax.set_title("Experiment grid summary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_grid_bar_aggregated(aggregated: pd.DataFrame, path: Optional[str | Path] = None):
    """Bar plot for aggregated multi-seed mean analysis RMSE with seed error bars."""
    if aggregated.empty:
        raise ValueError("aggregated dataframe is empty")
    required = {"N", "dt_obs", "method", "mean_analysis_rmse_mean", "mean_analysis_rmse_std"}
    missing = required - set(aggregated.columns)
    if missing:
        raise ValueError(f"aggregated dataframe missing columns: {sorted(missing)}")

    combos = aggregated[["N", "dt_obs"]].drop_duplicates().to_records(index=False)
    methods = list(aggregated["method"].unique())
    x = np.arange(len(combos))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for j, method in enumerate(methods):
        vals = []
        errs = []
        for N, dt_obs in combos:
            m = aggregated[(aggregated["N"] == N) & (aggregated["dt_obs"] == dt_obs) & (aggregated["method"] == method)]
            if m.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(float(m["mean_analysis_rmse_mean"].iloc[0]))
                errs.append(float(m["mean_analysis_rmse_std"].iloc[0]))
        ax.bar(
            x + (j - (len(methods) - 1) / 2) * width,
            vals,
            yerr=errs,
            capsize=3,
            width=width,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(_combo_labels(combos), rotation=35, ha="right")
    ax.set_ylabel("Mean analysis RMSE across seeds")
    ax.set_title("Multi-seed grid summary with standard deviation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_adaptive_improvement_aggregated(aggregated: pd.DataFrame, path: Optional[str | Path] = None):
    """Plot adaptive-vs-fixed analysis RMSE improvement from aggregated rows.

    Positive values mean adaptive has lower mean analysis RMSE than fixed.
    Error bars approximate independent standard deviations across seeds.
    """
    if aggregated.empty:
        raise ValueError("aggregated dataframe is empty")
    required = {"N", "dt_obs", "method", "mean_analysis_rmse_mean", "mean_analysis_rmse_std"}
    missing = required - set(aggregated.columns)
    if missing:
        raise ValueError(f"aggregated dataframe missing columns: {sorted(missing)}")

    rows = []
    for (N, dt_obs), group in aggregated.groupby(["N", "dt_obs"]):
        fixed = group[group["method"] == "fixed"]
        adaptive = group[group["method"] == "adaptive"]
        if fixed.empty or adaptive.empty:
            continue
        fixed_mean = float(fixed["mean_analysis_rmse_mean"].iloc[0])
        adaptive_mean = float(adaptive["mean_analysis_rmse_mean"].iloc[0])
        fixed_std = float(fixed["mean_analysis_rmse_std"].iloc[0])
        adaptive_std = float(adaptive["mean_analysis_rmse_std"].iloc[0])
        rows.append((N, dt_obs, fixed_mean - adaptive_mean, float(np.sqrt(fixed_std**2 + adaptive_std**2))))

    if not rows:
        raise ValueError("Need both fixed and adaptive rows to plot improvement")

    x = np.arange(len(rows))
    vals = [r[2] for r in rows]
    errs = [r[3] for r in rows]
    labels = [f"N={r[0]}, dt={r[1]:g}" for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]
    ax.bar(x, vals, yerr=errs, capsize=3, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Fixed RMSE - adaptive RMSE")
    ax.set_title("Adaptive vs fixed mean analysis RMSE improvement across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_state_heatmap(times: np.ndarray, states: np.ndarray, path: Optional[str | Path] = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(states.T, aspect="auto", origin="lower", extent=[times[0], times[-1], 0, states.shape[1] - 1])
    ax.set_xlabel("Time")
    ax.set_ylabel("State index")
    ax.set_title("Lorenz'96 true-state heatmap")
    fig.colorbar(im, ax=ax, label="x_i")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig
