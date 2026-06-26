"""Plot helpers for experiments and Streamlit."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

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


def plot_grid_bar(summary: pd.DataFrame, path: Optional[str | Path] = None):
    if summary.empty:
        raise ValueError("summary dataframe is empty")
    grouped = summary.groupby(["N", "dt_obs", "method"], as_index=False)["mean_analysis_rmse"].mean()
    labels = [f"N={row.N}, dt={row.dt_obs:g}" for row in grouped[["N", "dt_obs"]].drop_duplicates().itertuples()]
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
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean analysis RMSE")
    ax.set_title("Experiment grid summary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
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
